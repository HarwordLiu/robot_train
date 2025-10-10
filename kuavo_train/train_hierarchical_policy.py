# -*- coding: utf-8 -*-
"""
分层人形机器人Diffusion Policy统一训练脚本

支持两种训练模式：
1. 基础分层训练：使用配置文件直接定义的课程学习阶段
2. 任务特定训练：使用TaskSpecificTrainingManager管理多任务场景

模式选择由 task_specific_training.enable 配置决定

使用方法：
# 基础模式
python kuavo_train/train_hierarchical_policy.py --config-name=humanoid_diffusion_config

# 任务特定模式（在配置中设置 task_specific_training.enable=True）
python kuavo_train/train_hierarchical_policy.py --config-name=humanoid_diffusion_config
"""

# Ensure custom patches are applied, DON'T REMOVE THIS LINE!
import lerobot_patches.custom_patches
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
            logging.FileHandler('hierarchical_training.log', encoding='utf-8')
        ]
    )
    return logging.getLogger("HierarchicalTraining")


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
    print("🤖 构建HumanoidDiffusionPolicy分层架构...")
    return HumanoidDiffusionPolicy(policy_cfg, dataset_stats)


def load_task_dataset(task_id: int, cfg: DictConfig, policy_cfg, image_transforms) -> Tuple[Optional[LeRobotDataset], Optional[LeRobotDatasetMetadata]]:
    """加载特定任务的数据集（任务特定模式使用）"""
    # 任务1直接从root配置读取
    if task_id == 1:
        task_data_path = cfg.get('root', '')
        if not task_data_path:
            print(f"⚠️  root配置未设置")
            return None, None

        if not os.path.exists(task_data_path):
            print(f"⚠️  root数据路径不存在: {task_data_path}")
            return None, None
    else:
        # 其他任务使用task_specific_training配置
        task_config = cfg.get('task_specific_training', {})
        data_config = task_config.get('data_config', {})

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
        task_repoid = f"lerobot/task_{task_id}"
        dataset_metadata = LeRobotDatasetMetadata(
            task_repoid, root=task_data_path)

        delta_timestamps = build_delta_timestamps(dataset_metadata, policy_cfg)

        max_episodes = cfg.task_specific_training.memory_management.get(
            'max_episodes_per_task', 300)
        episodes_to_use = list(
            range(min(max_episodes, dataset_metadata.info["total_episodes"])))

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
    """创建任务特定的数据加载器（多任务加权采样）"""
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
        sample_weights.extend([task_weight] * dataset_size)

    combined_dataset = ConcatDataset(combined_datasets)

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


def run_curriculum_learning_stage(policy, stage_config, dataset, cfg, device, writer, current_step,
                                  optimizer=None, lr_scheduler=None, scaler=None, output_directory=None,
                                  amp_enabled=False, task_manager=None, dataloader=None):
    """
    运行课程学习的单个阶段

    支持两种模式：
    1. 基础模式：dataset参数传入，内部创建dataloader
    2. 任务特定模式：dataloader参数传入，使用task_manager
    """
    stage_name = stage_config.get("name", "unknown")
    enabled_layers = stage_config.get("layers", [])
    stage_epochs = stage_config.get("epochs", 10)
    target_task = stage_config.get("target_task")  # 任务特定模式使用

    # 测试训练模式
    test_training_mode = cfg.training.get('test_training_mode', False)
    if test_training_mode:
        original_epochs = stage_epochs
        test_epochs = cfg.training.get('test_training_epochs', 1)
        stage_epochs = test_epochs
        print(f"🧪 TEST MODE: Overriding {stage_name} stage epochs from {original_epochs} to {test_epochs}")

    print(f"🎓 开始课程学习阶段: {stage_name}")
    print(f"   激活层: {enabled_layers}")
    print(f"   训练轮次: {stage_epochs}")
    if target_task:
        print(f"   目标任务: {target_task}")

    # 激活指定的层
    if hasattr(policy, 'set_curriculum_stage'):
        policy.set_curriculum_stage(enabled_layers)

    # 任务特定模式：配置任务层权重
    if task_manager and target_task and target_task != "all":
        layer_weights = task_manager.get_task_specific_layer_weights(target_task)
        if hasattr(policy, 'set_task_layer_weights'):
            policy.set_task_layer_weights(layer_weights)

    # 创建数据加载器（如果未提供）
    if dataloader is None:
        dataloader = DataLoader(
            dataset,
            num_workers=cfg.training.num_workers,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            pin_memory=(device.type != "cpu"),
            drop_last=cfg.training.drop_last,
            prefetch_factor=1,
        )

    # AMP autocast
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
    best_stage_loss = float('inf')

    for epoch in range(stage_epochs):
        print(f"🚀 开始 Epoch {epoch+1}/{stage_epochs}")
        epoch_bar = tqdm(
            dataloader,
            desc=f"Stage {stage_name} Epoch {epoch+1}/{stage_epochs}",
            dynamic_ncols=True,
            leave=False)

        total_epoch_loss = 0.0
        epoch_samples = 0

        for batch in epoch_bar:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            # 构建课程学习信息
            curriculum_info = {
                'stage': stage_name,
                'enabled_layers': enabled_layers
            }
            if target_task:
                curriculum_info['target_task'] = target_task

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
                writer.add_scalar(f"curriculum/{stage_name}/loss",
                                  scaled_loss.item(), current_step + stage_steps)

                # 记录分层信息
                if isinstance(hierarchical_info, dict):
                    for key, value in hierarchical_info.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(
                                f"curriculum/{stage_name}/{key}", value, current_step + stage_steps)

                epoch_bar.set_postfix(
                    loss=f"{scaled_loss.item():.3f}",
                    stage=stage_name
                )

            total_epoch_loss += scaled_loss.item()
            epoch_samples += 1
            stage_steps += 1

        # 计算平均epoch损失
        avg_epoch_loss = total_epoch_loss / max(epoch_samples, 1)
        print(f"📊 Epoch {epoch+1} 平均损失: {avg_epoch_loss:.4f}, 最佳损失: {best_stage_loss:.4f}")

        # 保存最佳模型
        if avg_epoch_loss < best_stage_loss and output_directory is not None:
            print(f"🎯 发现更好的模型! {best_stage_loss:.4f} → {avg_epoch_loss:.4f}")
            best_stage_loss = avg_epoch_loss
            best_save_path = output_directory / f"curriculum_{stage_name}_best"
            try:
                policy.save_pretrained(best_save_path)
                save_rng_state(best_save_path / "rng_state.pth")
                print(f"✅ 最佳模型保存成功: {best_save_path}")
            except Exception as e:
                print(f"❌ 最佳模型保存失败: {e}")

        # 定期保存检查点
        if output_directory is not None and (epoch + 1) % cfg.training.save_freq_epoch == 0:
            checkpoint_save_path = output_directory / f"curriculum_{stage_name}_epoch{epoch + 1}"
            try:
                policy.save_pretrained(checkpoint_save_path)
                save_rng_state(checkpoint_save_path / "rng_state.pth")
                print(f"✅ 定期检查点保存成功: {checkpoint_save_path}")
            except Exception as e:
                print(f"❌ 定期检查点保存失败: {e}")

            # 保存详细状态
            if optimizer is not None and lr_scheduler is not None:
                try:
                    save_hierarchical_checkpoint(
                        policy, optimizer, lr_scheduler, scaler,
                        current_step + stage_steps, epoch + 1, best_stage_loss,
                        output_directory, amp_enabled, task_manager
                    )
                    print(f"✅ 分层架构状态保存成功")
                except Exception as e:
                    print(f"❌ 分层架构状态保存失败: {e}")

    # 测试训练模式：自动保存
    if test_training_mode and output_directory is not None:
        test_save_path = output_directory / f"test_stage_{stage_name}_complete"
        print(f"🧪 TEST MODE: Auto-saving stage completion to {test_save_path}")
        try:
            policy.save_pretrained(test_save_path)
            print(f"✅ Test stage model saved successfully")
        except Exception as e:
            print(f"❌ Test stage model save failed: {e}")

    print(f"✅ 课程阶段 {stage_name} 完成，最佳损失: {best_stage_loss:.4f}")
    return current_step + stage_steps


def save_hierarchical_checkpoint(policy, optimizer, lr_scheduler, scaler, steps, epoch, best_loss,
                                 output_directory, amp_enabled, task_manager=None):
    """保存分层架构检查点（支持任务特定模式）"""
    print(f"🔧 开始保存分层架构检查点...")
    print(f"   - steps: {steps}, epoch: {epoch}, best_loss: {best_loss:.4f}")

    # 保存policy
    try:
        policy.save_pretrained(output_directory)
        save_rng_state(output_directory / "rng_state.pth")
        print(f"✅ Policy 保存成功")
    except Exception as e:
        print(f"❌ Policy 保存失败: {e}")
        return

    # 保存训练状态
    try:
        checkpoint = {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict() if amp_enabled else None,
            "steps": steps,
            "epoch": epoch,
            "best_loss": best_loss,
            "hierarchical_stats": policy.get_performance_stats() if hasattr(policy, 'get_performance_stats') else {},
            "layer_states": policy.get_layer_states() if hasattr(policy, 'get_layer_states') else {}
        }

        # 任务特定模式：保存任务管理器状态
        if task_manager is not None:
            checkpoint["available_tasks"] = task_manager.available_tasks
            checkpoint["current_phase"] = task_manager.current_training_phase
            state_file = output_directory / "task_specific_learning_state.pth"
        else:
            state_file = output_directory / "hierarchical_learning_state.pth"

        torch.save(checkpoint, state_file)
        print(f"✅ 训练状态保存成功: {state_file}")

        # 任务特定模式：保存任务管理器详细状态
        if task_manager is not None:
            task_manager.save_training_state(output_directory, epoch,
                                             policy.get_performance_stats() if hasattr(policy, 'get_performance_stats') else {})
            print(f"✅ 任务管理器状态保存成功")

    except Exception as e:
        print(f"❌ 训练状态保存失败: {e}")


@hydra.main(config_path="../configs/policy/", config_name="humanoid_diffusion_config", version_base=None)
def main(cfg: DictConfig):
    """统一分层架构训练主函数"""
    logger = setup_logging()
    set_seed(cfg.training.seed)

    # 检查训练模式
    use_task_specific = cfg.get('task_specific_training', {}).get('enable', False)
    test_training_mode = cfg.training.get('test_training_mode', False)

    print("🤖 分层人形机器人Diffusion Policy训练")
    print("=" * 70)
    print(f"任务: {cfg.task}")
    print(f"方法: {cfg.method}")
    print(f"分层架构: {cfg.policy.get('use_hierarchical', False)}")
    print(f"训练模式: {'任务特定模式' if use_task_specific else '基础模式'}")

    if test_training_mode:
        test_epochs = cfg.training.get('test_training_epochs', 1)
        print(f"🧪 测试模式启用 - 每个阶段运行{test_epochs}个epoch")

    # 验证配置
    if not cfg.policy.get('use_hierarchical', False):
        logger.warning("⚠️  use_hierarchical为False，建议设为True以使用分层架构")

    # 设置输出目录
    if use_task_specific:
        output_directory = Path(cfg.training.output_directory) / f"task_specific_run_{cfg.timestamp}"
    else:
        output_directory = Path(cfg.training.output_directory) / f"run_{cfg.timestamp}"

    output_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_directory))

    device = torch.device(cfg.training.device)

    # 初始化任务管理器（任务特定模式）
    task_manager = None
    if use_task_specific:
        task_manager = TaskSpecificTrainingManager(cfg)

    # =================
    # 数据集加载
    # =================
    datasets = {}
    dataset_metadatas = {}
    image_transforms = build_augmenter(cfg.training.RGB_Augmenter)

    if use_task_specific:
        # 任务特定模式：加载任务数据
        print("🔍 任务特定模式：检测可用任务数据...")

        # 临时：只处理任务1，从root配置读取
        task_id = 1
        task_data_path = cfg.get('root', '')

        if task_data_path and os.path.exists(task_data_path):
            try:
                task_repoid = f"lerobot/task_{task_id}"
                temp_dataset_metadata = LeRobotDatasetMetadata(task_repoid, root=task_data_path)

                task_manager.register_available_task(
                    task_id, temp_dataset_metadata.info["total_episodes"], task_data_path)
                print(f"✅ 检测到任务{task_id}数据: {temp_dataset_metadata.info['total_episodes']}个episodes")
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

        task_manager.print_training_plan()

        # 加载第一个任务的数据集（获取特征）
        first_task_id = task_manager.available_tasks[0]
        temp_dataset_metadata = LeRobotDatasetMetadata(
            f"lerobot/task_{first_task_id}", root=task_data_path)

        features = dataset_to_policy_features(temp_dataset_metadata.features)
        input_features = {k: ft for k, ft in features.items() if ft.type is not FeatureType.ACTION}
        output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}

        # 构建policy配置
        policy_cfg = build_policy_config(cfg, input_features, output_features)

        # 加载实际数据集
        first_dataset, first_metadata = load_task_dataset(
            first_task_id, cfg, policy_cfg, image_transforms)

        if first_dataset is None:
            logger.error(f"❌ 无法加载任务{first_task_id}数据")
            return

        datasets[first_task_id] = first_dataset
        dataset_metadatas[first_task_id] = first_metadata

        # 计算总帧数
        total_frames = sum(metadata.info["total_frames"] for metadata in dataset_metadatas.values())
        dataset_stats = first_metadata.stats

    else:
        # 基础模式：直接加载数据集
        print("📂 基础模式：加载数据集...")
        dataset_metadata = LeRobotDatasetMetadata(cfg.repoid, root=cfg.root)
        print("Camera keys:", dataset_metadata.camera_keys)
        print("Original dataset features:", dataset_metadata.features)

        features = dataset_to_policy_features(dataset_metadata.features)
        input_features = {k: ft for k, ft in features.items() if ft.type is not FeatureType.ACTION}
        output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}

        # 构建policy配置
        policy_cfg = build_policy_config(cfg, input_features, output_features)

        # Episode限制
        episodes_to_use = getattr(cfg, 'episodes_to_use', None)
        if episodes_to_use is not None:
            if isinstance(episodes_to_use, int):
                episodes_to_use = list(range(episodes_to_use))
            elif hasattr(episodes_to_use, '__len__') and len(episodes_to_use) == 2:
                start, end = int(episodes_to_use[0]), int(episodes_to_use[1])
                episodes_to_use = list(range(start, end + 1))
                print(f"使用episodes范围 [{start}, {end}]: {len(episodes_to_use)}个episodes")
            elif hasattr(episodes_to_use, '__iter__'):
                episodes_to_use = list(episodes_to_use)
            print(f"使用限定episodes: {len(episodes_to_use)}个")
        else:
            print("使用全部episodes")

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

    # =================
    # 构建Policy
    # =================
    policy = build_hierarchical_policy(policy_cfg, dataset_stats)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(policy, cfg, total_frames)

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
        resume_path = Path(cfg.training.output_directory) / cfg.training.resume_timestamp
        print(f"从检查点恢复训练: {resume_path}")
        try:
            load_rng_state(resume_path / "rng_state.pth")
            policy = policy.from_pretrained(resume_path, strict=True)

            optimizer, lr_scheduler = build_optimizer_and_scheduler(policy, cfg, total_frames)

            # 查找检查点文件
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

            # 恢复分层架构状态
            if "layer_states" in checkpoint and hasattr(policy, 'load_layer_states'):
                policy.load_layer_states(checkpoint["layer_states"])

            # 恢复任务特定状态
            if use_task_specific and task_manager:
                if "available_tasks" in checkpoint:
                    for task_id in checkpoint["available_tasks"]:
                        if task_id not in task_manager.available_tasks:
                            task_manager.available_tasks.append(task_id)
                if "current_phase" in checkpoint:
                    task_manager.current_training_phase = checkpoint["current_phase"]

                state_file = resume_path / "task_training_state.json"
                if state_file.exists():
                    task_manager.load_training_state(state_file)

            for file in resume_path.glob("events.*"):
                shutil.copy(file, output_directory)

            print(f"已恢复训练从epoch {start_epoch}, step {steps}")
        except Exception as e:
            print(f"恢复检查点失败: {e}")
            return
    else:
        print("从头开始训练!")

    policy.train().to(device)
    print(f"总参数量: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"使用AMP: {amp_enabled}")

    # 打印分层架构信息
    if hasattr(policy, 'print_architecture_summary'):
        policy.print_architecture_summary()

    # =================
    # 课程学习
    # =================
    if use_task_specific:
        # 任务特定模式：使用TaskManager的课程学习配置
        curriculum_stages = task_manager.get_current_curriculum_stages()
        use_curriculum = bool(curriculum_stages)
    else:
        # 基础模式：从配置读取课程学习阶段
        curriculum_config = cfg.policy.hierarchical.get('curriculum_learning', {}) if hasattr(
            cfg.policy, 'hierarchical') else {}
        use_curriculum = curriculum_config.get('enable', False)
        stages_config = curriculum_config.get('stages') or curriculum_config.get('universal_stages')
        curriculum_stages = stages_config if use_curriculum else {}

    if use_curriculum and curriculum_stages:
        print(f"🎓 启动课程学习，共{len(curriculum_stages)}个阶段")

        current_step = steps
        for stage_name, stage_config in curriculum_stages.items():
            if use_task_specific:
                # 任务特定模式：使用任务特定数据加载器
                dataloader = create_task_specific_dataloader(datasets, task_manager, cfg, device)
                current_step = run_curriculum_learning_stage(
                    policy, stage_config, None, cfg, device, writer, current_step,
                    optimizer, lr_scheduler, scaler, output_directory, amp_enabled,
                    task_manager=task_manager, dataloader=dataloader
                )
            else:
                # 基础模式：使用单一数据集
                current_step = run_curriculum_learning_stage(
                    policy, stage_config, dataset, cfg, device, writer, current_step,
                    optimizer, lr_scheduler, scaler, output_directory, amp_enabled
                )

        print("✅ 课程学习完成，开始完整训练...")
        steps = current_step

    # =================
    # 主训练循环
    # =================
    print("🚀 开始主要训练循环...")
    for epoch in range(start_epoch, cfg.training.max_epoch):
        if use_task_specific:
            dataloader = create_task_specific_dataloader(datasets, task_manager, cfg, device)
        else:
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
            leave=False)

        total_loss = 0.0
        for batch in epoch_bar:
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            # 任务特定模式：获取任务权重
            if use_task_specific and task_manager:
                task_loss_weights = task_manager.get_task_specific_loss_weights(batch)
                with make_autocast(amp_enabled):
                    loss, hierarchical_info = policy.forward(batch, task_weights=task_loss_weights)
            else:
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

            # 记录训练日志
            if steps % cfg.training.log_freq == 0:
                writer.add_scalar("train/loss", scaled_loss.item(), steps)
                writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], steps)

                # 记录分层架构统计信息
                if isinstance(hierarchical_info, dict):
                    for key, value in hierarchical_info.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(f"hierarchical/{key}", value, steps)

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

        # 保存最新的分层架构检查点
        save_hierarchical_checkpoint(
            policy, optimizer, lr_scheduler, scaler, steps, epoch + 1, best_loss,
            output_directory, amp_enabled, task_manager
        )

    writer.close()

    # 训练完成后打印统计信息
    print("\n🎉 训练完成!")
    print("=" * 70)

    if hasattr(policy, 'get_performance_stats'):
        final_stats = policy.get_performance_stats()
        print("最终层性能统计:")
        for layer_name, stats in final_stats.items():
            print(f"  {layer_name}: {stats}")

    # 任务特定模式：打印任务训练总结
    if use_task_specific and task_manager:
        training_summary = task_manager.get_training_summary()
        print(f"\n📊 任务训练总结:")
        print(f"   训练阶段: {training_summary['current_phase']}")
        print(f"   处理任务: {len(training_summary['available_tasks'])}")
        print(f"   总episodes: {training_summary['total_episodes']}")

    print(f"   最终损失: {best_loss:.4f}")


if __name__ == "__main__":
    main()
