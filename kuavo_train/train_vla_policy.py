#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLA Transformer Policy训练脚本

使用Token化架构训练现代化的VLA策略
支持通过配置文件灵活定义输入维度

使用方法：
python kuavo_train/train_vla_policy.py --config-name=vla_config
"""

# Ensure custom patches are applied
import lerobot_patches.custom_patches

import os
import hydra
from omegaconf import DictConfig
from pathlib import Path
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
from hydra.utils import instantiate
from diffusers.optimization import get_scheduler
import logging

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.random_utils import set_seed

# 导入VLA模块
from kuavo_train.wrapper.policy.vla.VLAPolicyWrapper import VLAPolicyWrapper
from kuavo_train.wrapper.policy.vla.VLAConfigWrapper import VLAConfigWrapper

# 导入工具函数
from kuavo_train.utils.augmenter import crop_image, resize_image
from kuavo_train.utils.utils import save_rng_state, load_rng_state
from kuavo_train.utils.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig


def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('vla_training.log', encoding='utf-8')
        ]
    )
    return logging.getLogger("VLATraining")


def build_augmenter(cfg):
    """构建图像增强器"""
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
    """构建delta timestamps"""
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


def build_policy_config(cfg, input_features, output_features):
    """构建policy配置"""
    from lerobot.configs.policies import PolicyFeature

    def _normalize_feature_dict(d):
        if isinstance(d, DictConfig):
            from omegaconf import OmegaConf
            d = OmegaConf.to_container(d, resolve=True)
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict or DictConfig, got {type(d)}")

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


def build_optimizer_and_scheduler(policy, cfg, total_frames):
    """构建优化器和调度器"""
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


@hydra.main(config_path="../configs/policy/", config_name="vla_config", version_base=None)
def main(cfg: DictConfig):
    """VLA训练主函数"""
    logger = setup_logging()
    set_seed(cfg.training.seed)

    print("🤖 VLA Transformer Policy训练")
    print("=" * 70)
    print(f"任务: {cfg.task}")
    print(f"方法: {cfg.method}")
    print(f"Token维度: {cfg.policy.token_embed_dim}")
    print(f"Transformer深度: {cfg.policy.transformer_depth}")

    # 设置输出目录
    output_directory = Path(
        cfg.training.output_directory) / f"run_{cfg.timestamp}"
    output_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_directory))

    device = torch.device(cfg.training.device)

    # ==================== 加载数据集 ====================
    print("📂 加载数据集...")
    dataset_metadata = LeRobotDatasetMetadata(cfg.repoid, root=cfg.root)
    print(f"Camera keys: {dataset_metadata.camera_keys}")
    print(f"Dataset features: {list(dataset_metadata.features.keys())}")

    # 构建特征
    features = dataset_to_policy_features(dataset_metadata.features)
    input_features = {k: ft for k, ft in features.items(
    ) if ft.type is not FeatureType.ACTION}
    output_features = {k: ft for k,
                       ft in features.items() if ft.type is FeatureType.ACTION}

    # Episode限制
    episodes_to_use = getattr(cfg, 'episodes_to_use', None)
    if episodes_to_use is not None:
        if isinstance(episodes_to_use, int):
            episodes_to_use = list(range(episodes_to_use))
        elif hasattr(episodes_to_use, '__len__') and len(episodes_to_use) == 2:
            start, end = int(episodes_to_use[0]), int(episodes_to_use[1])
            episodes_to_use = list(range(start, end + 1))
            print(
                f"使用episodes范围 [{start}, {end}]: {len(episodes_to_use)}个episodes")
        elif hasattr(episodes_to_use, '__iter__'):
            episodes_to_use = list(episodes_to_use)
        print(f"使用限定episodes: {len(episodes_to_use)}个")
    else:
        print("使用全部episodes")

    # 构建policy配置
    policy_cfg = build_policy_config(cfg, input_features, output_features)

    # 构建图像增强器
    image_transforms = build_augmenter(cfg.training.RGB_Augmenter)

    # 加载数据集
    delta_timestamps = build_delta_timestamps(dataset_metadata, policy_cfg)
    dataset = LeRobotDataset(
        cfg.repoid,
        delta_timestamps=delta_timestamps,
        root=cfg.root,
        episodes=episodes_to_use,
        image_transforms=image_transforms,
    )

    total_frames = dataset_metadata.info["total_frames"]
    dataset_stats = dataset_metadata.stats

    print(f"✅ 数据集加载完成: {total_frames} frames")

    # ==================== 构建Policy ====================
    print("🏗️  构建VLA Policy...")
    policy = VLAPolicyWrapper(policy_cfg, dataset_stats)
    policy.train().to(device)

    print(f"总参数量: {sum(p.numel() for p in policy.parameters()):,}")

    # 构建优化器和调度器
    optimizer, lr_scheduler = build_optimizer_and_scheduler(
        policy, cfg, total_frames)

    # AMP支持
    amp_requested = bool(getattr(cfg.policy, "use_amp", False))
    amp_enabled = amp_requested and device.type == "cuda"

    if hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print(f"使用AMP: {amp_enabled}")

    # 初始化训练状态
    start_epoch = 0
    steps = 0
    best_loss = float('inf')

    # 恢复训练（如果需要）
    if cfg.training.resume and cfg.training.resume_timestamp:
        resume_path = Path(cfg.training.output_directory) / \
            cfg.training.resume_timestamp
        print(f"从检查点恢复训练: {resume_path}")
        try:
            load_rng_state(resume_path / "rng_state.pth")
            policy = policy.from_pretrained(resume_path, strict=True)
            policy = policy.to(device)

            optimizer, lr_scheduler = build_optimizer_and_scheduler(
                policy, cfg, total_frames)

            checkpoint_file = resume_path / "learning_state.pth"
            if checkpoint_file.exists():
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

            for file in resume_path.glob("events.*"):
                shutil.copy(file, output_directory)

            print(f"已恢复训练从epoch {start_epoch}, step {steps}")
        except Exception as e:
            print(f"恢复检查点失败: {e}")
            return
    else:
        print("从头开始训练!")

    # ==================== 训练循环 ====================
    print("🚀 开始训练...")

    from contextlib import nullcontext
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
            desc=f"Epoch {epoch+1}/{cfg.training.max_epoch}",
            dynamic_ncols=True,
            leave=False
        )

        total_loss = 0.0
        for batch in epoch_bar:
            # 将batch移到device
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            # 前向传播
            with make_autocast(amp_enabled):
                loss, _ = policy.forward(batch)

            scaled_loss = loss / cfg.training.accumulation_steps

            # 反向传播
            if amp_enabled:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            steps += 1  # 先增加步数计数器

            # 优化器更新
            if steps % cfg.training.accumulation_steps == 0:
                # 梯度裁剪（防止梯度爆炸）
                if amp_enabled:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # 记录日志
            if steps % cfg.training.log_freq == 0:
                writer.add_scalar("train/loss", scaled_loss.item(), steps)
                writer.add_scalar(
                    "train/lr", lr_scheduler.get_last_lr()[0], steps)

                epoch_bar.set_postfix(
                    loss=f"{scaled_loss.item():.4f}",
                    step=steps,
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}"
                )

            total_loss += scaled_loss.item()

        # Epoch结束统计
        avg_loss = total_loss / len(dataloader)
        print(f"\n📊 Epoch {epoch+1} 平均损失: {avg_loss:.4f}")

        # 更新最佳损失
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_directory / "best"
            policy.save_pretrained(best_path)
            save_rng_state(best_path / "rng_state.pth")
            print(f"✅ 最佳模型已保存: loss={best_loss:.4f}")

        # 定期保存检查点
        if (epoch + 1) % cfg.training.save_freq_epoch == 0:
            epoch_path = output_directory / f"epoch{epoch+1}"
            policy.save_pretrained(epoch_path)
            save_rng_state(epoch_path / "rng_state.pth")

            # 保存训练状态
            checkpoint = {
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict() if amp_enabled else None,
                "steps": steps,
                "epoch": epoch + 1,
                "best_loss": best_loss,
            }
            torch.save(checkpoint, epoch_path / "learning_state.pth")

            print(f"✅ 检查点已保存: epoch {epoch+1}")

    writer.close()

    print("\n🎉 训练完成!")
    print("=" * 70)
    print(f"最佳损失: {best_loss:.4f}")
    print(f"模型保存位置: {output_directory}")


if __name__ == "__main__":
    main()
