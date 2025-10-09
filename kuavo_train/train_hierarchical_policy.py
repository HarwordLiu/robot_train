# -*- coding: utf-8 -*-
"""
分层人形机器人Diffusion Policy专用训练脚本

这个文件专门用于训练分层架构的Diffusion Policy，与传统train_policy.py分离，
避免干扰现有的训练模式。

使用方法：
python kuavo_train/train_hierarchical_policy.py --config-name=humanoid_diffusion_config
"""

# Ensure custom patches are applied, DON'T REMOVE THIS LINE!
import lerobot_patches.custom_patches
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

# 导入分层架构模块
from kuavo_train.wrapper.policy.humanoid.HumanoidDiffusionPolicy import HumanoidDiffusionPolicy
from kuavo_train.wrapper.dataset.LeRobotDatasetWrapper import CustomLeRobotDataset
from kuavo_train.utils.augmenter import crop_image, resize_image, DeterministicAugmenterColor
from kuavo_train.utils.utils import save_rng_state, load_rng_state
from diffusers.optimization import get_scheduler
from utils.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig

from functools import partial
from contextlib import nullcontext


def build_augmenter(cfg):
    """构建图像增强器 - 复用train_policy.py的实现"""
    img_tf_cfg = ImageTransformsConfig(
        enable=cfg.get("enable", False),
        max_num_transforms=cfg.get("max_num_transforms", 3),
        random_order=cfg.get("random_order", False),
        tfs={}
    )

    if "tfs" in cfg:
        for name, tf_dict in cfg["tfs"].items():
            img_tf_cfg.tfs[name] = ImageTransformConfig(
                weight=tf_dict.get("weight", 1.0),
                type=tf_dict.get("type", "Identity"),
                kwargs=tf_dict.get("kwargs", {}),
            )
    return ImageTransforms(img_tf_cfg)


def build_delta_timestamps(dataset_metadata, policy_cfg):
    """构建delta timestamps - 复用train_policy.py的实现"""
    obs_indices = getattr(policy_cfg, "observation_delta_indices", None)
    act_indices = getattr(policy_cfg, "action_delta_indices", None)
    if obs_indices is None and act_indices is None:
        return None

    delta_timestamps = {}
    for key in dataset_metadata.info["features"]:
        if "observation" in key and obs_indices is not None:
            delta_timestamps[key] = [
                i / dataset_metadata.fps for i in obs_indices]
        elif "action" in key and act_indices is not None:
            delta_timestamps[key] = [
                i / dataset_metadata.fps for i in act_indices]

    return delta_timestamps if delta_timestamps else None


def build_optimizer_and_scheduler(policy, cfg, total_frames):
    """构建优化器和调度器 - 复用train_policy.py的实现"""
    optimizer = policy.config.get_optimizer_preset().build(policy.parameters())

    if cfg.training.max_training_step is None:
        updates_per_epoch = (
            total_frames // (cfg.training.batch_size * cfg.training.accumulation_steps)) + 1
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

    return optimizer, lr_scheduler


def build_policy_config(cfg, input_features, output_features):
    """构建policy配置 - 复用train_policy.py的实现"""
    def _normalize_feature_dict(d: Any) -> dict[str, PolicyFeature]:
        if isinstance(d, DictConfig):
            d = OmegaConf.to_container(d, resolve=True)
        if not isinstance(d, dict):
            raise TypeError(
                "Expected dict or DictConfig, got {}".format(type(d)))

        return {
            k: PolicyFeature(**v) if isinstance(v,
                                                dict) and not isinstance(v, PolicyFeature) else v
            for k, v in d.items()
        }

    policy_cfg = instantiate(
        cfg.policy,
        input_features=input_features,
        output_features=output_features,
        device=cfg.training.device,
    )

    policy_cfg.input_features = _normalize_feature_dict(
        policy_cfg.input_features)
    policy_cfg.output_features = _normalize_feature_dict(
        policy_cfg.output_features)
    return policy_cfg


def build_hierarchical_policy(policy_cfg, dataset_stats):
    """构建分层架构的policy"""
    print("🤖 Building HumanoidDiffusionPolicy with hierarchical architecture...")
    return HumanoidDiffusionPolicy(policy_cfg, dataset_stats)


def run_curriculum_learning_stage(policy, stage_config, dataset, cfg, device, writer, current_step,
                                  optimizer=None, lr_scheduler=None, scaler=None, output_directory=None, amp_enabled=False):
    """运行课程学习的单个阶段"""
    stage_name = stage_config.get("name", "unknown")
    enabled_layers = stage_config.get("layers", [])
    stage_epochs = stage_config.get("epochs", 10)

    # 测试训练模式：使用配置的测试epoch数量
    test_training_mode = cfg.training.get('test_training_mode', False)
    if test_training_mode:
        original_epochs = stage_epochs
        test_epochs = cfg.training.get('test_training_epochs', 1)
        stage_epochs = test_epochs
        print(f"🧪 TEST MODE: Overriding {stage_name} stage epochs from {original_epochs} to {test_epochs}")

    print("🎓 Starting curriculum stage: {} (layers: {}, epochs: {})".format(
        stage_name, enabled_layers, stage_epochs))
    print(f"🔧 课程学习阶段参数:")
    print(f"   - output_directory: {output_directory}")
    print(f"   - optimizer: {optimizer is not None}")
    print(f"   - lr_scheduler: {lr_scheduler is not None}")
    print(f"   - scaler: {scaler is not None}")
    print(f"   - amp_enabled: {amp_enabled}")
    print(f"   - save_freq_epoch: {cfg.training.save_freq_epoch}")

    # 激活指定的层
    if hasattr(policy, 'set_curriculum_stage'):
        policy.set_curriculum_stage(enabled_layers)

    # 为这个阶段创建数据加载器
    dataloader = DataLoader(
        dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        pin_memory=(device.type != "cpu"),
        drop_last=cfg.training.drop_last,
        prefetch_factor=1,
    )

    stage_steps = 0
    best_stage_loss = float('inf')

    for epoch in range(stage_epochs):
        print(f"🚀 开始 Epoch {epoch+1}/{stage_epochs}")
        epoch_bar = tqdm(
            dataloader,
            desc="Stage {} Epoch {}/{}".format(stage_name, epoch+1, stage_epochs),
            dynamic_ncols=True,
            leave=False)

        total_epoch_loss = 0.0
        epoch_samples = 0

        for batch in epoch_bar:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            # 前向传播时包含课程学习信息
            loss, _ = policy.forward(batch, curriculum_info={
                'stage': stage_name,
                'enabled_layers': enabled_layers
            })

            # 记录日志
            if stage_steps % cfg.training.log_freq == 0:
                writer.add_scalar("curriculum/{}/loss".format(stage_name),
                                  loss.item(), current_step + stage_steps)
                epoch_bar.set_postfix(loss="{:.3f}".format(
                    loss.item()), stage=stage_name)

            total_epoch_loss += loss.item()
            epoch_samples += 1
            stage_steps += 1

        print(f"🏁 Epoch {epoch+1} 训练完成，开始保存检查点...")
        # 计算平均epoch损失
        avg_epoch_loss = total_epoch_loss / max(epoch_samples, 1)
        print(
            f"📊 Epoch {epoch+1} 平均损失: {avg_epoch_loss:.4f}, 当前最佳损失: {best_stage_loss:.4f}")

        # 保存最佳模型
        if avg_epoch_loss < best_stage_loss and output_directory is not None:
            print(
                f"🎯 发现更好的模型! 损失从 {best_stage_loss:.4f} 改善到 {avg_epoch_loss:.4f}")
            best_stage_loss = avg_epoch_loss
            best_save_path = output_directory / \
                "curriculum_{}_best".format(stage_name)
            print(f"💾 正在保存最佳模型到: {best_save_path}")
            try:
                policy.save_pretrained(best_save_path)
                save_rng_state(best_save_path / "rng_state.pth")
                print(f"✅ 最佳模型保存成功: {best_save_path}")
            except Exception as e:
                print(f"❌ 最佳模型保存失败: {e}")
        elif avg_epoch_loss >= best_stage_loss:
            print(
                f"📉 损失未改善，跳过最佳模型保存 (当前: {avg_epoch_loss:.4f}, 最佳: {best_stage_loss:.4f})")
        elif output_directory is None:
            print(f"⚠️  output_directory 为 None，跳过最佳模型保存")

        # 定期保存检查点
        print(
            f"🔍 检查定期保存条件: output_directory={output_directory is not None}, epoch={epoch+1}, save_freq_epoch={cfg.training.save_freq_epoch}")
        if output_directory is not None and (epoch + 1) % cfg.training.save_freq_epoch == 0:
            checkpoint_save_path = output_directory / \
                "curriculum_{}_epoch{}".format(stage_name, epoch + 1)
            print(f"💾 正在保存定期检查点到: {checkpoint_save_path}")
            try:
                policy.save_pretrained(checkpoint_save_path)
                save_rng_state(checkpoint_save_path / "rng_state.pth")
                print(f"✅ 定期检查点保存成功: {checkpoint_save_path}")
            except Exception as e:
                print(f"❌ 定期检查点保存失败: {e}")

            # 保存课程学习阶段的详细状态
            if optimizer is not None and lr_scheduler is not None:
                print(f"💾 正在保存分层架构详细状态...")
                try:
                    save_hierarchical_checkpoint(
                        policy, optimizer, lr_scheduler, scaler,
                        current_step + stage_steps, epoch + 1, best_stage_loss,
                        output_directory, amp_enabled
                    )
                    print(f"✅ 分层架构状态保存成功")
                except Exception as e:
                    print(f"❌ 分层架构状态保存失败: {e}")
            else:
                print(f"⚠️  optimizer 或 lr_scheduler 为 None，跳过详细状态保存")
        else:
            print(
                f"⏭️  跳过定期检查点保存 (epoch {epoch+1} 不是 {cfg.training.save_freq_epoch} 的倍数)")

        print(f"✅ Epoch {epoch+1} 检查点保存完成")

    # 测试训练模式：在每个阶段结束后自动保存模型
    if test_training_mode and output_directory is not None:
        test_save_path = output_directory / f"test_stage_{stage_name}_complete"
        print(f"🧪 TEST MODE: Auto-saving stage completion to {test_save_path}")
        try:
            policy.save_pretrained(test_save_path)
            print(f"✅ Test stage model saved successfully: {test_save_path}")
        except Exception as e:
            print(f"❌ Test stage model save failed: {e}")

    print("✅ Completed curriculum stage: {} (best loss: {:.4f})".format(
        stage_name, best_stage_loss))
    return current_step + stage_steps


def save_hierarchical_checkpoint(policy, optimizer, lr_scheduler, scaler, steps, epoch, best_loss,
                                 output_directory, amp_enabled):
    """保存分层架构专用的检查点"""
    print(f"🔧 开始保存分层架构检查点...")
    print(f"   - output_directory: {output_directory}")
    print(f"   - steps: {steps}, epoch: {epoch}, best_loss: {best_loss:.4f}")
    print(f"   - amp_enabled: {amp_enabled}")

    # 保存policy
    print(f"💾 正在保存 policy 到: {output_directory}")
    try:
        policy.save_pretrained(output_directory)
        save_rng_state(output_directory / "rng_state.pth")
        print(f"✅ Policy 保存成功")
    except Exception as e:
        print(f"❌ Policy 保存失败: {e}")
        return

    # 保存分层架构特有的状态
    print(f"💾 正在保存训练状态...")
    try:
        checkpoint = {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict() if amp_enabled else None,
            "steps": steps,
            "epoch": epoch,
            "best_loss": best_loss,
            # 保存分层架构特有的状态
            "hierarchical_stats": policy.get_performance_stats() if hasattr(policy, 'get_performance_stats') else {},
            "layer_states": policy.get_layer_states() if hasattr(policy, 'get_layer_states') else {}
        }

        state_file = output_directory / "hierarchical_learning_state.pth"
        torch.save(checkpoint, state_file)
        print(f"✅ 训练状态保存成功: {state_file}")

        # 保存随机数状态
        rng_file = output_directory / "rng_state.pth"
        save_rng_state(rng_file)
        print(f"✅ 随机数状态保存成功: {rng_file}")

    except Exception as e:
        print(f"❌ 训练状态保存失败: {e}")


@hydra.main(config_path="../configs/policy/", config_name="humanoid_diffusion_config", version_base=None)
def main(cfg: DictConfig):
    """分层架构训练主函数"""
    set_seed(cfg.training.seed)

    # 检查测试训练模式
    test_training_mode = cfg.training.get('test_training_mode', False)

    print("🤖 Hierarchical Humanoid Diffusion Policy Training")
    print("=" * 60)
    print("Config: {}".format(cfg.defaults))
    print("Use hierarchical: {}".format(
        cfg.policy.get('use_hierarchical', False)))

    if test_training_mode:
        test_epochs = cfg.training.get('test_training_epochs', 1)
        print(f"🧪 TEST TRAINING MODE ENABLED - Running {test_epochs} epoch(s) per stage for quick validation")
        print(f"⚡ All curriculum stages will be reduced to {test_epochs} epoch(s)")
        print("💾 Automatic saving enabled after each stage")

    # 验证分层架构配置
    if not cfg.policy.get('use_hierarchical', False):
        print("⚠️  Warning: use_hierarchical is False. Set it to True in config to use hierarchical architecture.")

    # 设置输出目录
    output_directory = Path(cfg.training.output_directory) / \
        "run_{}".format(cfg.timestamp)
    output_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_directory))

    device = torch.device(cfg.training.device)

    # 加载数据集元数据和特征
    dataset_metadata = LeRobotDatasetMetadata(cfg.repoid, root=cfg.root)
    print("Camera keys:", dataset_metadata.camera_keys)
    print("Original dataset features:", dataset_metadata.features)

    features = dataset_to_policy_features(dataset_metadata.features)
    input_features = {k: ft for k, ft in features.items(
    ) if ft.type is not FeatureType.ACTION}
    output_features = {k: ft for k,
                       ft in features.items() if ft.type is FeatureType.ACTION}

    print("Input features: {}".format(input_features))
    print("Output features: {}".format(output_features))

    # 构建分层policy配置
    policy_cfg = build_policy_config(cfg, input_features, output_features)
    print("Policy config:", policy_cfg)

    # 构建分层architecture的policy
    policy = build_hierarchical_policy(
        policy_cfg, dataset_stats=dataset_metadata.stats)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(
        policy, cfg, dataset_metadata.info["total_frames"])

    # AMP支持 - 复用train_policy.py的实现
    amp_requested = bool(getattr(cfg.policy, "use_amp", False))
    amp_enabled = amp_requested and device.type == "cuda"

    has_torch_autocast = hasattr(torch, "autocast")

    def make_autocast(enabled: bool):
        if not enabled:
            return nullcontext()
        if device.type == "cuda":
            if has_torch_autocast:
                return torch.autocast(device_type="cuda")
            else:
                from torch.cuda.amp import autocast as cuda_autocast
                return cuda_autocast()
        return nullcontext()

    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled) if hasattr(
        torch, "amp") else torch.cuda.amp.GradScaler(device=device.type, enabled=amp_enabled)

    # 初始化训练状态
    start_epoch = 0
    steps = 0
    best_loss = float('inf')

    # 恢复训练逻辑 - 专用于分层架构
    if cfg.training.resume and cfg.training.resume_timestamp:
        resume_path = Path(cfg.training.output_directory) / \
            cfg.training.resume_timestamp
        print("Resuming hierarchical training from:", resume_path)
        try:
            load_rng_state(resume_path / "rng_state.pth")
            policy = policy.from_pretrained(resume_path, strict=True)

            optimizer, lr_scheduler = build_optimizer_and_scheduler(
                policy, cfg, dataset_metadata.info["total_frames"])

            # 加载分层架构专用的状态
            checkpoint_file = resume_path / "hierarchical_learning_state.pth"
            if not checkpoint_file.exists():
                checkpoint_file = resume_path / "learning_state.pth"  # 回退到标准文件

            checkpoint = torch.load(checkpoint_file, map_location=device)
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

            # 恢复分层架构特有的状态
            if "layer_states" in checkpoint and hasattr(policy, 'load_layer_states'):
                policy.load_layer_states(checkpoint["layer_states"])

            for file in resume_path.glob("events.*"):
                shutil.copy(file, output_directory)

            print("Resumed hierarchical training from epoch {}, step {}".format(
                start_epoch, steps))
        except Exception as e:
            print("Failed to load hierarchical checkpoint:", e)
            return
    else:
        print("Training hierarchical architecture from scratch!")
        # 初始化optimizer和lr_scheduler（非resume情况）
        optimizer, lr_scheduler = build_optimizer_and_scheduler(
            policy, cfg, dataset_metadata.info["total_frames"])

    policy.train().to(device)
    print("Total parameters: {:,}".format(
        sum(p.numel() for p in policy.parameters())))
    print("Using AMP: {}".format(amp_enabled))

    # 打印分层架构信息
    if hasattr(policy, 'print_architecture_summary'):
        policy.print_architecture_summary()

    # 构建数据集
    delta_timestamps = build_delta_timestamps(dataset_metadata, policy_cfg)
    image_transforms = build_augmenter(cfg.training.RGB_Augmenter)

    # Episode限制逻辑 - 复用train_policy.py的实现
    episodes_to_use = getattr(cfg, 'episodes_to_use', None)
    print("Raw episodes_to_use from config: {}, type: {}".format(
        episodes_to_use, type(episodes_to_use)))
    if episodes_to_use is not None:
        if isinstance(episodes_to_use, int):
            episodes_to_use = list(range(episodes_to_use))
        elif hasattr(episodes_to_use, '__len__') and len(episodes_to_use) == 2:
            start, end = int(episodes_to_use[0]), int(episodes_to_use[1])
            episodes_to_use = list(range(start, end + 1))
            print("Converted range [{}, {}] to {} episodes".format(
                start, end, len(episodes_to_use)))
        elif hasattr(episodes_to_use, '__iter__'):
            episodes_to_use = list(episodes_to_use)
        print("Using limited episodes for memory efficiency: {} episodes".format(
            len(episodes_to_use)))
    else:
        episodes_to_use = None
        print("Using all available episodes")

    dataset = LeRobotDataset(
        cfg.repoid,
        delta_timestamps=delta_timestamps,
        root=cfg.root,
        episodes=episodes_to_use,
        image_transforms=image_transforms,
    )

    # 课程学习支持
    curriculum_config = cfg.get('curriculum_learning', {})
    use_curriculum = curriculum_config.get('enable', False)

    # 检查课程学习配置，支持 'stages' 或 'universal_stages'
    stages_config = curriculum_config.get(
        'stages') or curriculum_config.get('universal_stages')

    if use_curriculum and stages_config:
        print("🎓 Starting curriculum learning with {} stages".format(
            len(stages_config)))

        # 运行课程学习阶段
        current_step = steps
        for stage_name, stage_config in stages_config.items():
            current_step = run_curriculum_learning_stage(
                policy, stage_config, dataset, cfg, device, writer, current_step,
                optimizer, lr_scheduler, scaler, output_directory, amp_enabled
            )

        print("✅ Curriculum learning completed. Starting full training...")
        steps = current_step

    # 主要训练循环
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

        epoch_bar = tqdm(
            dataloader,
            desc="Epoch {}/{}".format(epoch+1, cfg.training.max_epoch),
            dynamic_ncols=True,
            leave=False)

        total_loss = 0.0
        for batch in epoch_bar:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            with make_autocast(amp_enabled):
                loss, hierarchical_info = policy.forward(batch)

            scaled_loss = loss / cfg.training.accumulation_steps

            if amp_enabled:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if steps % cfg.training.accumulation_steps == 0:
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # 记录训练日志 - 包含分层架构特有的信息
            if steps % cfg.training.log_freq == 0:
                writer.add_scalar("train/loss", scaled_loss.item(), steps)
                writer.add_scalar(
                    "train/lr", lr_scheduler.get_last_lr()[0], steps)

                # 记录分层架构特有的统计信息
                if isinstance(hierarchical_info, dict):
                    for key, value in hierarchical_info.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(
                                "hierarchical/{}".format(key), value, steps)

                # 记录各层性能统计
                if hasattr(policy, 'get_performance_stats'):
                    perf_stats = policy.get_performance_stats()
                    for layer_name, stats in perf_stats.items():
                        if isinstance(stats, dict):
                            for stat_name, stat_value in stats.items():
                                if isinstance(stat_value, (int, float)):
                                    writer.add_scalar(
                                        "performance/{}/{}".format(layer_name, stat_name), stat_value, steps)

                epoch_bar.set_postfix(
                    loss="{:.3f}".format(scaled_loss.item()),
                    step=steps,
                    lr=lr_scheduler.get_last_lr()[0]
                )

            steps += 1
            total_loss += scaled_loss.item()

        # 更新最佳损失
        if total_loss < best_loss:
            best_loss = total_loss
            best_path = output_directory / "best"
            policy.save_pretrained(best_path)
            save_rng_state(best_path / "rng_state.pth")

        # 定期保存检查点
        if (epoch + 1) % cfg.training.save_freq_epoch == 0:
            epoch_path = output_directory / "epoch{}".format(epoch+1)
            policy.save_pretrained(epoch_path)
            save_rng_state(epoch_path / "rng_state.pth")

        # 保存最新的分层架构检查点
        save_hierarchical_checkpoint(
            policy, optimizer, lr_scheduler, scaler, steps, epoch + 1, best_loss,
            output_directory, amp_enabled
        )

    writer.close()

    # 训练完成后打印分层架构统计信息
    print("\n🎉 Hierarchical training completed!")
    if hasattr(policy, 'get_performance_stats'):
        final_stats = policy.get_performance_stats()
        print("Final layer performance statistics:")
        for layer_name, stats in final_stats.items():
            print("  {}: {}".format(layer_name, stats))


if __name__ == "__main__":
    main()
