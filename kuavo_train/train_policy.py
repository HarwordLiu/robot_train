import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
from lerobot.configs.policies import PolicyFeature
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from pathlib import Path
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
from hydra.utils import instantiate
from diffusers.optimization import get_scheduler

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.random_utils import set_seed

from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from kuavo_train.wrapper.dataset.LeRobotDatasetWrapper import CustomLeRobotDataset
from kuavo_train.utils.augmenter import crop_image, resize_image, DeterministicAugmenterColor
from kuavo_train.utils.utils import save_rng_state, load_rng_state
from lerobot.policies.act.modeling_act import ACTPolicy
from diffusers.optimization import get_scheduler
from utils.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig

from functools import partial
from contextlib import nullcontext


def build_augmenter(cfg):
    """Since operations such as cropping and resizing in LeRobot are implemented at the model level 
    rather than at the data level, we provide only RGB image augmentations on the data side here, 
    with support for customization. For more details, refer to configs/policy/diffusion_config.yaml. 
    To define custom transformations, please see utils.transforms.py."""

    img_tf_cfg = ImageTransformsConfig(
        enable=cfg.get("enable", False),
        max_num_transforms=cfg.get("max_num_transforms", 3),
        random_order=cfg.get("random_order", False),
        tfs={}
    )

    # deal tfs part
    if "tfs" in cfg:
        for name, tf_dict in cfg["tfs"].items():
            img_tf_cfg.tfs[name] = ImageTransformConfig(
                weight=tf_dict.get("weight", 1.0),
                type=tf_dict.get("type", "Identity"),
                kwargs=tf_dict.get("kwargs", {}),
            )
    return ImageTransforms(img_tf_cfg)


def build_delta_timestamps(dataset_metadata, policy_cfg):
    """Build delta timestamps for observations and actions."""
    obs_indices = getattr(policy_cfg, "observation_delta_indices", None)
    act_indices = getattr(policy_cfg, "action_delta_indices", None)
    if obs_indices is None and act_indices is None:
        return None

    delta_timestamps = {}
    for key in dataset_metadata.info["features"]:
        if "observation" in key and obs_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in obs_indices]
        elif "action" in key and act_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in act_indices]

    return delta_timestamps if delta_timestamps else None


def build_optimizer_and_scheduler(policy, cfg, total_frames):
    """Return optimizer and scheduler."""
    optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
    # If `max_training_step` is specified, it takes precedence; 
    # otherwise, the value is automatically determined based on `max_epoch`.
    if cfg.training.max_training_step is None:
        updates_per_epoch = (total_frames // (cfg.training.batch_size * cfg.training.accumulation_steps)) + 1
        num_training_steps = cfg.training.max_epoch * updates_per_epoch
    else:
        num_training_steps = cfg.training.max_training_step
    lr_scheduler = policy.config.get_scheduler_preset()
    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler.build(optimizer, num_training_steps)
    else:
        lr_scheduler = get_scheduler(
            name=cfg.training.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )

    # or you can set your optimizer and lr_scheduler here and replace it.
    return optimizer, lr_scheduler


def build_policy_config(cfg, input_features, output_features):
    def _normalize_feature_dict(d: Any) -> dict[str, PolicyFeature]:
        if isinstance(d, DictConfig):
            d = OmegaConf.to_container(d, resolve=True)
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict or DictConfig, got {type(d)}")

        return {
            k: PolicyFeature(**v) if isinstance(v, dict) and not isinstance(v, PolicyFeature) else v
            for k, v in d.items()
        }

    policy_cfg = instantiate(
        cfg.policy,
        input_features=input_features,
        output_features=output_features,
        device=cfg.training.device,
    )
                
    policy_cfg.input_features = _normalize_feature_dict(policy_cfg.input_features)
    policy_cfg.output_features = _normalize_feature_dict(policy_cfg.output_features)
    return policy_cfg

def build_policy(name, policy_cfg, dataset_stats):
    policy = {
        "diffusion": CustomDiffusionPolicyWrapper,
        "act": ACTPolicy,
    }[name](policy_cfg, dataset_stats)
    return policy

def build_policy_config(cfg, input_features, output_features):
    def _normalize_feature_dict(d: Any) -> dict[str, PolicyFeature]:
        if isinstance(d, DictConfig):
            d = OmegaConf.to_container(d, resolve=True)
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict or DictConfig, got {type(d)}")

        return {
            k: PolicyFeature(**v) if isinstance(v, dict) and not isinstance(v, PolicyFeature) else v
            for k, v in d.items()
        }

    policy_cfg = instantiate(
        cfg.policy,
        input_features=input_features,
        output_features=output_features,
        device=cfg.training.device,
    )
                
    policy_cfg.input_features = _normalize_feature_dict(policy_cfg.input_features)
    policy_cfg.output_features = _normalize_feature_dict(policy_cfg.output_features)
    return policy_cfg




@hydra.main(config_path="../configs/policy/", config_name="diffusion_config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.training.seed)

    # Setup output directory
    output_directory = Path(cfg.training.output_directory) / f"run_{cfg.timestamp}"
    output_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_directory))

    device = torch.device(cfg.training.device)

    # Dataset metadata and features
    dataset_metadata = LeRobotDatasetMetadata(cfg.repoid, root=cfg.root)
    print("camera_keys:", dataset_metadata.camera_keys)
    print("Original dataset features:", dataset_metadata.features)

    features = dataset_to_policy_features(dataset_metadata.features)
    input_features = {k: ft for k, ft in features.items() if ft.type is not FeatureType.ACTION}
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}

    print(f"Input features: {input_features}")
    print(f"Output features: {output_features}")

    # instantiate the policy
    policy_cfg = build_policy_config(cfg, input_features, output_features)
    print("policy_cfg", policy_cfg)

    # Build policy
    policy = build_policy(cfg.policy_name, policy_cfg, dataset_stats=dataset_metadata.stats)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(policy, cfg, dataset_metadata.info["total_frames"])
    
    # Initialize AMP GradScaler if use_amp is True
    amp_requested = bool(getattr(cfg.policy, "use_amp", False))
    amp_enabled = amp_requested and device.type == "cuda"

    # autocast context (cuda, or no-op when disabled/non-cuda)
    has_torch_autocast = hasattr(torch, "autocast")
    def make_autocast(enabled: bool):
        if not enabled:
            return nullcontext()
        if device.type == "cuda":
            if has_torch_autocast:
                return torch.autocast(device_type="cuda")
            else:
                from torch.cuda.amp import autocast as cuda_autocast  # noqa
                return cuda_autocast()
        # Fallback: disable on non-cuda to avoid dtype surprises
        return nullcontext()

    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(device=device.type, enabled=amp_enabled)
    # print("scaler", device.type, make_autocast(amp_enabled))
    # Initialize training state variables
    start_epoch = 0
    steps = 0
    best_loss = float('inf')

    # ===== Resume logic (perfect resume for AMP & RNG) =====
    
    if cfg.training.resume and cfg.training.resume_timestamp:
        resume_path = Path(cfg.training.output_directory) / cfg.training.resume_timestamp
        print("Resuming from:", resume_path)
        try:
            # Load RNG state
            load_rng_state(resume_path / "rng_state.pth")
            
            # Load policy
            policy = policy.from_pretrained(resume_path, strict=True)

            """ Warning: using `from_pretrained` creates a new policy instance, 
            so the optimizer must be reinitialized here! """
            optimizer, lr_scheduler = build_optimizer_and_scheduler(policy, cfg, dataset_metadata.info["total_frames"])
            
            # Load optimizer, scheduler, scaler and training state
            checkpoint = torch.load(resume_path / "learning_state.pth", map_location=device)
            optimizer.load_state_dict(checkpoint["optimizer"])
            
            if "lr_scheduler" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            
            if "scaler" in checkpoint and amp_enabled:
                scaler.load_state_dict(checkpoint["scaler"])
            
            if "steps" in checkpoint:
                steps = checkpoint["steps"]
            
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]
            
            if "best_loss" in checkpoint:
                best_loss = checkpoint["best_loss"]
            
            # Copy and load log_event
            for file in resume_path.glob("events.*"):
                shutil.copy(file, output_directory)
                
            print(f"Resumed training from epoch {start_epoch}, step {steps}")
        except Exception as e:
            print("Failed to load checkpoint:", e)
            return
    else:
        print("Training from scratch!")

    policy.train().to(device)
    print(f"Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Using AMP: {amp_enabled}")
    # Build dataset and dataloader
    delta_timestamps = build_delta_timestamps(dataset_metadata, policy_cfg)

    image_transforms = build_augmenter(cfg.training.RGB_Augmenter)
    # 限制使用的episodes数量来控制显存占用
    episodes_to_use = getattr(cfg, 'episodes_to_use', None)
    print(f"🔍 Raw episodes_to_use from config: {episodes_to_use}, type: {type(episodes_to_use)}")
    if episodes_to_use is not None:
        if isinstance(episodes_to_use, int):
            # 如果是数字，转换为range list: int -> [0, int-1]
            episodes_to_use = list(range(episodes_to_use))
        elif hasattr(episodes_to_use, '__len__') and len(episodes_to_use) == 2:
            # 如果是[start, end]格式（包括ListConfig），转换为range list
            start, end = int(episodes_to_use[0]), int(episodes_to_use[1])
            episodes_to_use = list(range(start, end + 1))  # +1因为range是左闭右开
            print(f"🔍 Converted range [{start}, {end}] to {len(episodes_to_use)} episodes")
        elif hasattr(episodes_to_use, '__iter__'):
            # 如果已经是episode列表，直接使用
            episodes_to_use = list(episodes_to_use)
        print(f"🚨 Using limited episodes for memory efficiency: {len(episodes_to_use)} episodes")
    else:
        # 如果没有指定，使用所有episodes
        episodes_to_use = None
        print("Using all available episodes")

    dataset = LeRobotDataset(
        cfg.repoid,
        delta_timestamps=delta_timestamps,
        root=cfg.root,
        episodes=episodes_to_use,  # 添加episodes参数
        image_transforms=image_transforms,
    )

    # Training loop
    for epoch in range(start_epoch, cfg.training.max_epoch):
        dataloader = DataLoader(
            dataset,
            num_workers=cfg.training.num_workers,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            pin_memory=(device.type != "cpu"),
            drop_last=cfg.training.drop_last,
            prefetch_factor=1,
        )

        epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.training.max_epoch}")

        total_loss = 0.0
        for batch in epoch_bar:
            
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            with make_autocast(amp_enabled):
                loss, _ = policy.forward(batch)
            # Scale loss and backward with AMP if enabled
            scaled_loss = loss / cfg.training.accumulation_steps
            
            if amp_enabled:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if steps % cfg.training.accumulation_steps == 0:
                if amp_enabled:
                    # Optionally unscale and clip gradients here if you use clipping
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if steps % cfg.training.log_freq == 0:
                writer.add_scalar("train/loss", scaled_loss.item(), steps)
                writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], steps)
                epoch_bar.set_postfix(loss=f"{scaled_loss.item():.3f}", step=steps, lr=lr_scheduler.get_last_lr()[0])

            steps += 1
            total_loss += scaled_loss.item()
        
        # Update best loss
        if total_loss < best_loss:
            best_loss = total_loss
            # Save best model
            policy.save_pretrained(output_directory / "best")
        # Save checkpoint every N epochs
        if (epoch + 1) % cfg.training.save_freq_epoch == 0:
            policy.save_pretrained(output_directory / f"epoch{epoch+1}")

        # Save last checkpoint (includes AMP scaler & progress for perfect resume)
        # Save last checkpoint
        policy.save_pretrained(output_directory)
        # Save training state including optimizer, scheduler, scaler, and step/epoch info
        checkpoint = {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict() if amp_enabled else None,
            "steps": steps,
            "epoch": epoch + 1,
            "best_loss": best_loss
        }
        torch.save(checkpoint, output_directory / "learning_state.pth")
        save_rng_state(output_directory / "rng_state.pth")

    writer.close()


if __name__ == "__main__":
    main()
