# -*- coding: utf-8 -*-
"""
任务特定分层人形机器人Diffusion Policy训练脚本

专门用于任务特定的分层架构训练，支持：
- 任务渐进式添加
- 防遗忘机制
- 任务条件层权重调整
- 智能课程学习策略

使用方法：
python kuavo_train/train_hierarchical_task_specific.py --config-name=humanoid_diffusion_config
"""

import lerobot_patches.custom_patches  # 确保自定义补丁生效，不要删除这行！
from lerobot.configs.policies import PolicyFeature
from typing import Any, Dict, List, Optional, Tuple
import os
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from pathlib import Path
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from tqdm import tqdm
import shutil
from hydra.utils import instantiate
from diffusers.optimization import get_scheduler
import numpy as np
import logging

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.random_utils import set_seed

# 导入分层架构模块
from kuavo_train.wrapper.policy.humanoid.HumanoidDiffusionPolicy import HumanoidDiffusionPolicy
from kuavo_train.wrapper.policy.humanoid.TaskSpecificTrainingManager import TaskSpecificTrainingManager
from kuavo_train.wrapper.dataset.LeRobotDatasetWrapper import CustomLeRobotDataset
from kuavo_train.utils.augmenter import crop_image, resize_image, DeterministicAugmenterColor
from kuavo_train.utils.utils import save_rng_state, load_rng_state
from utils.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig

from functools import partial
from contextlib import nullcontext


def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('task_specific_training.log', encoding='utf-8')
        ]
    )
    return logging.getLogger("TaskSpecificTraining")


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


def build_policy_config(cfg, input_features, output_features):
    """构建policy配置"""
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
    print("🤖 构建任务特定的HumanoidDiffusionPolicy...")
    return HumanoidDiffusionPolicy(policy_cfg, dataset_stats)


def load_task_dataset(task_id: int, cfg: DictConfig, policy_cfg, image_transforms) -> Tuple[Optional[LeRobotDataset], Optional[LeRobotDatasetMetadata]]:
    """加载特定任务的数据集"""
    # 临时修改：直接从root配置读取任务一数据
    if task_id == 1:
        # 使用root配置的数据路径
        task_data_path = cfg.get('root', '')
        if not task_data_path:
            print(f"⚠️  root配置未设置")
            return None, None

        if not os.path.exists(task_data_path):
            print(f"⚠️  root数据路径不存在: {task_data_path}")
            return None, None
    else:
        # 其他任务仍使用原有逻辑
        task_config = cfg.get('task_specific_training', {})
        data_config = task_config.get('data_config', {})

        # 构建任务数据路径
        base_path = data_config.get('base_path', '/robot/data')
        task_dir = data_config.get('task_directories', {}).get(task_id)

        if not task_dir:
            print(f"⚠️  任务{task_id}的数据目录未配置")
            return None, None

        task_data_path = os.path.join(base_path, task_dir)
        if not os.path.exists(task_data_path):
            print(f"⚠️  任务{task_id}数据路径不存在: {task_data_path}")
            return None, None

    try:
        # 加载任务数据集元数据
        task_repoid = f"lerobot/task_{task_id}"
        dataset_metadata = LeRobotDatasetMetadata(
            task_repoid, root=task_data_path)

        # 构建delta timestamps
        delta_timestamps = build_delta_timestamps(dataset_metadata, policy_cfg)

        # 应用episode限制
        max_episodes = cfg.task_specific_training.memory_management.get(
            'max_episodes_per_task', 300)
        episodes_to_use = list(
            range(min(max_episodes, dataset_metadata.info["total_episodes"])))

        # 创建数据集
        dataset = LeRobotDataset(
            task_repoid,
            delta_timestamps=delta_timestamps,
            root=task_data_path,
            episodes=episodes_to_use,
            image_transforms=image_transforms,
        )

        print(f"✅ 任务{task_id}数据集加载成功: {len(episodes_to_use)}个episodes")
        return dataset, dataset_metadata

    except Exception as e:
        print(f"❌ 任务{task_id}数据集加载失败: {e}")
        return None, None


def create_task_specific_dataloader(datasets: Dict[int, LeRobotDataset], task_manager: TaskSpecificTrainingManager,
                                    cfg: DictConfig, device: torch.device) -> DataLoader:
    """创建任务特定的数据加载器"""
    if len(datasets) == 1:
        # 单任务情况
        task_id = list(datasets.keys())[0]
        dataset = datasets[task_id]
        return DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=(device.type != "cpu"),
            drop_last=cfg.training.drop_last,
            prefetch_factor=1,
        )

    # 多任务情况 - 使用加权采样
    sampling_strategy = task_manager.get_task_data_sampling_strategy()
    task_weights = sampling_strategy.get("task_weights", {})

    combined_datasets = []
    sample_weights = []

    for task_id, dataset in datasets.items():
        task_weight = task_weights.get(task_id, 1.0 / len(datasets))
        dataset_size = len(dataset)

        combined_datasets.append(dataset)
        # 为该任务的每个样本分配权重
        sample_weights.extend([task_weight] * dataset_size)

    # 合并数据集
    combined_dataset = ConcatDataset(combined_datasets)

    # 创建加权采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(combined_dataset),
        replacement=True
    )

    return DataLoader(
        combined_dataset,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type != "cpu"),
        drop_last=cfg.training.drop_last,
        prefetch_factor=1,
    )


def run_task_specific_curriculum_stage(policy, stage_config: Dict[str, Any], dataloader: DataLoader,
                                       task_manager: TaskSpecificTrainingManager, cfg: DictConfig,
                                       device: torch.device, writer: SummaryWriter,
                                       optimizer, lr_scheduler, scaler, output_directory, amp_enabled: bool,
                                       current_step: int, start_epoch: int = 0) -> int:
    """运行任务特定的课程学习阶段"""
    stage_name = stage_config.get("name", "unknown")
    enabled_layers = stage_config.get("layers", [])
    stage_epochs = stage_config.get("epochs", 10)
    target_task = stage_config.get("target_task")

    print(f"🎓 开始任务特定课程阶段: {stage_name}")
    print(f"   激活层: {enabled_layers}")
    print(f"   目标任务: {target_task}")
    print(f"   训练轮次: {stage_epochs}")
    print(f"🔧 任务特定课程学习阶段参数:")
    print(f"   - output_directory: {output_directory}")
    print(f"   - optimizer: {optimizer is not None}")
    print(f"   - lr_scheduler: {lr_scheduler is not None}")
    print(f"   - scaler: {scaler is not None}")
    print(f"   - amp_enabled: {amp_enabled}")
    print(f"   - save_freq_epoch: {cfg.training.save_freq_epoch}")

    # 配置policy的课程学习状态
    if hasattr(policy, 'set_curriculum_stage'):
        policy.set_curriculum_stage(enabled_layers)

    # 获取任务特定的层权重
    if target_task and target_task != "all":
        layer_weights = task_manager.get_task_specific_layer_weights(
            target_task)
        if hasattr(policy, 'set_task_layer_weights'):
            policy.set_task_layer_weights(layer_weights)

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

    stage_steps = 0
    total_loss = 0.0
    best_stage_loss = float('inf')

    for epoch in range(start_epoch, stage_epochs):
        print(f"🚀 开始 Epoch {epoch+1}/{stage_epochs}")
        epoch_bar = tqdm(
            dataloader, desc=f"课程阶段 {stage_name} Epoch {epoch+1}/{stage_epochs}")
        epoch_loss = 0.0

        for batch in epoch_bar:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            # 添加任务特定信息到batch
            curriculum_info = {
                'stage': stage_name,
                'enabled_layers': enabled_layers,
                'target_task': target_task
            }

            with make_autocast(amp_enabled):
                loss, hierarchical_info = policy.forward(
                    batch, curriculum_info=curriculum_info)

            scaled_loss = loss / cfg.training.accumulation_steps

            if amp_enabled:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if stage_steps % cfg.training.accumulation_steps == 0:
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # 记录日志
            if stage_steps % cfg.training.log_freq == 0:
                writer.add_scalar(
                    f"curriculum/{stage_name}/loss", scaled_loss.item(), current_step + stage_steps)

                # 记录任务特定信息
                if isinstance(hierarchical_info, dict):
                    for key, value in hierarchical_info.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(
                                f"curriculum/{stage_name}/{key}", value, current_step + stage_steps)

                epoch_bar.set_postfix(
                    loss=f"{scaled_loss.item():.3f}",
                    stage=stage_name,
                    target=target_task
                )

            stage_steps += 1
            total_loss += scaled_loss.item()
            epoch_loss += scaled_loss.item()

        print(f"🏁 Epoch {epoch+1} 训练完成，开始保存检查点...")
        # 计算平均epoch损失
        avg_epoch_loss = epoch_loss / len(dataloader)
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
                print(f"💾 正在保存任务特定详细状态...")
                try:
                    save_task_specific_checkpoint(
                        policy, optimizer, lr_scheduler, scaler,
                        current_step + stage_steps, epoch + 1, best_stage_loss,
                        output_directory, task_manager, amp_enabled
                    )
                    print(f"✅ 任务特定状态保存成功")
                except Exception as e:
                    print(f"❌ 任务特定状态保存失败: {e}")
            else:
                print(f"⚠️  optimizer 或 lr_scheduler 为 None，跳过详细状态保存")
        else:
            print(
                f"⏭️  跳过定期检查点保存 (epoch {epoch+1} 不是 {cfg.training.save_freq_epoch} 的倍数)")

        print(f"✅ Epoch {epoch+1} 检查点保存完成")

    avg_stage_loss = total_loss / \
        stage_steps if stage_steps > 0 else float('inf')
    print(f"✅ 课程阶段 {stage_name} 完成，平均损失: {avg_stage_loss:.4f}")

    return current_step + stage_steps


def save_task_specific_checkpoint(policy, optimizer, lr_scheduler, scaler, steps, epoch, best_loss,
                                  output_directory: Path, task_manager: TaskSpecificTrainingManager,
                                  amp_enabled: bool):
    """保存任务特定的检查点"""
    print(f"🔧 开始保存任务特定检查点...")
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

    # 保存训练状态
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
            "layer_states": policy.get_layer_states() if hasattr(policy, 'get_layer_states') else {},
            # 保存任务特定状态
            "available_tasks": task_manager.available_tasks,
            "current_phase": task_manager.current_training_phase
        }

        state_file = output_directory / "task_specific_learning_state.pth"
        torch.save(checkpoint, state_file)
        print(f"✅ 训练状态保存成功: {state_file}")

        # 保存随机数状态
        rng_file = output_directory / "rng_state.pth"
        save_rng_state(rng_file)
        print(f"✅ 随机数状态保存成功: {rng_file}")

        # 保存任务管理器状态
        task_manager.save_training_state(output_directory, epoch,
                                         policy.get_performance_stats() if hasattr(policy, 'get_performance_stats') else {})
        print(f"✅ 任务管理器状态保存成功")

    except Exception as e:
        print(f"❌ 训练状态保存失败: {e}")


@hydra.main(config_path="../configs/policy/", config_name="humanoid_diffusion_config", version_base=None)
def main(cfg: DictConfig):
    """任务特定训练主函数"""
    logger = setup_logging()
    set_seed(cfg.training.seed)

    print("🎯 任务特定分层人形机器人Diffusion Policy训练")
    print("=" * 70)
    print(f"任务: {cfg.task}")
    print(f"方法: {cfg.method}")
    print(f"使用分层架构: {cfg.policy.get('use_hierarchical', False)}")
    print(
        f"任务特定训练: {cfg.get('task_specific_training', {}).get('enable', False)}")

    # 验证配置
    if not cfg.policy.get('use_hierarchical', False):
        logger.warning("⚠️  use_hierarchical为False。请在配置中设为True以使用分层架构。")

    if not cfg.get('task_specific_training', {}).get('enable', False):
        logger.warning("⚠️  task_specific_training未启用。")

    # 设置输出目录
    output_directory = Path(cfg.training.output_directory) / \
        f"task_specific_run_{cfg.timestamp}"
    output_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_directory))

    device = torch.device(cfg.training.device)

    # 初始化任务管理器
    task_manager = TaskSpecificTrainingManager(cfg)

    # 检测可用任务数据
    task_config = cfg.get('task_specific_training', {})
    available_tasks = task_config.get(
        'data_config', {}).get('task_directories', {})

    print(f"🔍 检测可用任务数据...")
    datasets = {}
    dataset_metadatas = {}

    # 构建图像增强器
    image_transforms = build_augmenter(cfg.training.RGB_Augmenter)

    # 先加载第一个任务以获取基础特征信息
    first_task_loaded = False
    input_features = None
    output_features = None

    # 临时修改：只处理任务一，直接从root配置读取
    task_id = 1
    temp_dataset_metadata = None
    try:
        # 使用root配置的数据路径
        task_data_path = cfg.get('root', '')

        if task_data_path and os.path.exists(task_data_path):
            task_repoid = f"lerobot/task_{task_id}"
            temp_dataset_metadata = LeRobotDatasetMetadata(
                task_repoid, root=task_data_path)

            if not first_task_loaded:
                # 使用第一个任务的特征信息
                features = dataset_to_policy_features(
                    temp_dataset_metadata.features)
                input_features = {k: ft for k, ft in features.items(
                ) if ft.type is not FeatureType.ACTION}
                output_features = {
                    k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
                first_task_loaded = True

            task_manager.register_available_task(
                task_id, temp_dataset_metadata.info["total_episodes"], task_data_path)
            print(
                f"✅ 检测到任务{task_id}数据: {temp_dataset_metadata.info['total_episodes']}个episodes")
        else:
            print(f"⚠️  root数据路径不存在或未配置: {task_data_path}")

    except Exception as e:
        print(f"⚠️  任务{task_id}数据检测失败: {e}")

    if not task_manager.available_tasks:
        logger.error("❌ 没有检测到可用的任务数据")
        return

    # 验证训练准备状态
    ready, issues = task_manager.validate_training_readiness()
    if not ready:
        logger.error("❌ 训练准备未完成:")
        for issue in issues:
            logger.error(f"   - {issue}")
        return

    # 打印训练计划
    task_manager.print_training_plan()

    # 构建policy配置（使用第一个任务的特征）
    policy_cfg = build_policy_config(cfg, input_features, output_features)
    print(f"Policy配置: {policy_cfg}")

    # 构建分层架构的policy（使用第一个可用任务的统计信息）
    first_task_id = task_manager.available_tasks[0]
    first_task_dataset, first_task_metadata = load_task_dataset(
        first_task_id, cfg, policy_cfg, image_transforms)

    if first_task_dataset is None:
        logger.error(f"❌ 无法加载任务{first_task_id}数据")
        return

    datasets[first_task_id] = first_task_dataset
    dataset_metadatas[first_task_id] = first_task_metadata

    policy = build_hierarchical_policy(policy_cfg, first_task_metadata.stats)

    # 计算总帧数（用于优化器配置）
    total_frames = sum(metadata.info["total_frames"]
                       for metadata in dataset_metadatas.values())
    optimizer, lr_scheduler = build_optimizer_and_scheduler(
        policy, cfg, total_frames)

    # AMP支持
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
        torch, "amp") else torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # 初始化训练状态
    start_epoch = 0
    steps = 0
    best_loss = float('inf')

    # 恢复训练逻辑
    if cfg.training.resume and cfg.training.resume_timestamp:
        resume_path = Path(cfg.training.output_directory) / \
            cfg.training.resume_timestamp
        print(f"从检查点恢复任务特定训练: {resume_path}")
        try:
            load_rng_state(resume_path / "rng_state.pth")
            policy = policy.from_pretrained(resume_path, strict=True)

            optimizer, lr_scheduler = build_optimizer_and_scheduler(
                policy, cfg, total_frames)

            checkpoint_file = resume_path / "task_specific_learning_state.pth"
            if not checkpoint_file.exists():
                checkpoint_file = resume_path / "hierarchical_learning_state.pth"
            if not checkpoint_file.exists():
                checkpoint_file = resume_path / "learning_state.pth"

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

            # 恢复任务特定状态
            if "available_tasks" in checkpoint:
                for task_id in checkpoint["available_tasks"]:
                    if task_id not in task_manager.available_tasks:
                        task_manager.available_tasks.append(task_id)

            if "current_phase" in checkpoint:
                task_manager.current_training_phase = checkpoint["current_phase"]

            # 恢复任务管理器状态
            state_file = resume_path / "task_training_state.json"
            if state_file.exists():
                task_manager.load_training_state(state_file)

            for file in resume_path.glob("events.*"):
                shutil.copy(file, output_directory)

            print(f"已恢复任务特定训练从epoch {start_epoch}, step {steps}")
        except Exception as e:
            print(f"恢复检查点失败: {e}")
            return
    else:
        print("从头开始任务特定训练!")

    policy.train().to(device)
    print(f"总参数量: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"使用AMP: {amp_enabled}")

    # 打印分层架构信息
    if hasattr(policy, 'print_architecture_summary'):
        policy.print_architecture_summary()

    # 获取当前的课程学习配置
    curriculum_stages = task_manager.get_current_curriculum_stages()
    use_curriculum = bool(curriculum_stages)

    if use_curriculum:
        print(f"🎓 启动任务特定课程学习，共{len(curriculum_stages)}个阶段")

        # 创建数据加载器
        dataloader = create_task_specific_dataloader(
            datasets, task_manager, cfg, device)

        # 运行课程学习阶段
        current_step = steps
        for stage_name, stage_config in curriculum_stages.items():
            current_step = run_task_specific_curriculum_stage(
                policy, stage_config, dataloader, task_manager, cfg, device, writer,
                optimizer, lr_scheduler, scaler, output_directory, amp_enabled, current_step, start_epoch
            )

        print("✅ 任务特定课程学习完成。开始完整训练...")
        steps = current_step

    # 主要训练循环
    print("🚀 开始主要训练循环...")
    for epoch in range(start_epoch, cfg.training.max_epoch):
        dataloader = create_task_specific_dataloader(
            datasets, task_manager, cfg, device)

        epoch_bar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{cfg.training.max_epoch}")

        total_loss = 0.0
        for batch in epoch_bar:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            # 获取任务特定的损失权重
            task_loss_weights = task_manager.get_task_specific_loss_weights(
                batch)

            with make_autocast(amp_enabled):
                loss, hierarchical_info = policy.forward(
                    batch, task_weights=task_loss_weights)

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

            # 记录训练日志
            if steps % cfg.training.log_freq == 0:
                writer.add_scalar("train/loss", scaled_loss.item(), steps)
                writer.add_scalar(
                    "train/lr", lr_scheduler.get_last_lr()[0], steps)

                # 记录任务特定统计信息
                if isinstance(hierarchical_info, dict):
                    for key, value in hierarchical_info.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(
                                f"task_specific/{key}", value, steps)

                # 记录层性能统计
                if hasattr(policy, 'get_performance_stats'):
                    perf_stats = policy.get_performance_stats()
                    for layer_name, stats in perf_stats.items():
                        if isinstance(stats, dict):
                            for stat_name, stat_value in stats.items():
                                if isinstance(stat_value, (int, float)):
                                    writer.add_scalar(
                                        f"performance/{layer_name}/{stat_name}", stat_value, steps)

                epoch_bar.set_postfix(
                    loss=f"{scaled_loss.item():.3f}",
                    step=steps,
                    phase=task_manager.current_training_phase,
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
            epoch_path = output_directory / f"epoch{epoch+1}"
            policy.save_pretrained(epoch_path)
            save_rng_state(epoch_path / "rng_state.pth")

        # 保存最新的任务特定检查点
        save_task_specific_checkpoint(
            policy, optimizer, lr_scheduler, scaler, steps, epoch + 1, best_loss,
            output_directory, task_manager, amp_enabled
        )

    writer.close()

    # 训练完成后打印统计信息
    print("\n🎉 任务特定训练完成!")
    print("=" * 70)

    if hasattr(policy, 'get_performance_stats'):
        final_stats = policy.get_performance_stats()
        print("最终层性能统计:")
        for layer_name, stats in final_stats.items():
            print(f"  {layer_name}: {stats}")

    # 打印任务训练总结
    training_summary = task_manager.get_training_summary()
    print(f"\n📊 训练总结:")
    print(f"   训练阶段: {training_summary['current_phase']}")
    print(f"   处理任务: {len(training_summary['available_tasks'])}")
    print(f"   总episodes: {training_summary['total_episodes']}")
    print(f"   最终损失: {best_loss:.4f}")


if __name__ == "__main__":
    main()
