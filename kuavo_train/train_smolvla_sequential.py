#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmolVLA顺序多任务训练脚本

实现SmolVLA的顺序Fine-tuning策略：
- Stage 1: 预训练 → 任务1模型
- Stage 2: 任务1模型 → 任务2模型
- Stage 3: 任务2模型 → 任务3模型
- Stage 4: 任务3模型 → 任务4模型（最终多任务模型）

防遗忘技术：
- Replay Buffer: 混合之前任务的数据
- Lower Learning Rate: 逐步降低学习率
- Freeze Layers: 冻结VLM部分层
- Multi-task Validation: 验证所有之前任务

使用方法：
    # 训练任务1
    python kuavo_train/train_smolvla_sequential.py \\
        --config-path=../configs/policy \\
        --config-name=smolvla_sequential_base \\
        task=tasks/task1_moving_grasp

    # 训练任务2（自动从任务1继续）
    python kuavo_train/train_smolvla_sequential.py \\
        --config-path=../configs/policy \\
        --config-name=smolvla_sequential_base \\
        task=tasks/task2_weighing
"""

# Ensure custom patches are applied FIRST before any lerobot imports
import lerobot_patches.custom_patches

import random
from kuavo_train.utils.augmenter import DeterministicAugmenterColor
from kuavo_train.utils.utils import save_rng_state, load_rng_state
from kuavo_train.wrapper.policy.smolvla.SmolVLAConfigWrapper import SmolVLAConfigWrapper
from kuavo_train.wrapper.policy.smolvla.SmolVLAPolicyWrapper import SmolVLAPolicyWrapper
from lerobot.configs.types import FeatureType
from lerobot.utils.random_utils import set_seed
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import torch.nn as nn
import torch
from typing import Optional, Dict, Any
import json
from functools import partial
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra

import os
# 消除tokenizers fork警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# 导入SmolVLA模块

# 导入训练状态保存/加载工具

# 导入数据增强工具


def setup_logging():
    """设置日志系统"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                'smolvla_sequential_training.log', encoding='utf-8')
        ]
    )
    return logging.getLogger("SmolVLASequentialTraining")


def load_task_config(cfg_root: Path, task_id: int) -> DictConfig:
    """
    加载指定任务的配置

    Args:
        cfg_root: 配置文件根目录
        task_id: 任务ID (1-4)

    Returns:
        任务配置对象
    """
    task_files = {
        1: "task1_moving_grasp.yaml",
        2: "task2_weighing.yaml",
        3: "task3_placement.yaml",
        4: "task4_sorting.yaml",
    }

    task_file = cfg_root / "tasks" / task_files[task_id]
    if not task_file.exists():
        raise FileNotFoundError(f"Task config file not found: {task_file}")

    task_cfg = OmegaConf.load(task_file)
    return task_cfg


class ReplayDatasetManager:
    """
    管理Replay Buffer的类

    在训练任务N时，混合之前任务1到N-1的数据，防止灾难性遗忘
    """

    def __init__(self, cfg: DictConfig, current_task_id: int, cfg_root: Path, dataset_fps: int):
        self.cfg = cfg
        self.current_task_id = current_task_id
        self.cfg_root = cfg_root
        self.dataset_fps = dataset_fps
        self.replay_datasets = {}  # task_id -> dataset
        self.replay_weights = {}   # task_id -> weight

    def load_replay_tasks(self):
        """加载所有需要replay的任务数据"""
        if self.current_task_id == 1:
            # 任务1不需要replay
            return {}, {}

        # 获取当前stage的replay配置
        stage_key = f"stage{self.current_task_id}_replay"
        replay_config = self.cfg.sequential.get(stage_key, {})

        if not replay_config:
            print(
                f"⚠️  No replay config found for stage {self.current_task_id}")
            return {}, {}

        print(f"\n📦 Loading Replay Buffer for Stage {self.current_task_id}")
        print("="*70)

        # 构建delta_timestamps配置 (用于加载action chunks)
        chunk_size = self.cfg.policy.chunk_size
        delta_timestamps = {
            "observation.state": [0],  # 只取当前帧
            # 未来chunk_size帧
            "action": [i / self.dataset_fps for i in range(chunk_size)],
        }

        for task_key, weight in replay_config.items():
            if 'task' in task_key:
                task_id = int(task_key.replace('task', ''))

                # 只加载之前的任务
                if task_id < self.current_task_id:
                    print(
                        f"  Loading Task {task_id} (weight: {weight:.1%})...")

                    # 加载任务配置
                    task_cfg = load_task_config(self.cfg_root, task_id)

                    # 加载数据集（使用delta_timestamps）
                    dataset = LeRobotDataset(
                        task_cfg.task.data.repoid,
                        root=task_cfg.task.data.root,
                        episodes=list(range(
                            task_cfg.task.data.episodes_to_use[0],
                            task_cfg.task.data.episodes_to_use[1] + 1
                        )),
                        delta_timestamps=delta_timestamps
                    )

                    self.replay_datasets[task_id] = dataset
                    self.replay_weights[task_id] = weight

                    print(
                        f"    ✅ Loaded {len(dataset)} frames from Task {task_id}")

        print("="*70 + "\n")
        return self.replay_datasets, self.replay_weights


def pad_tensor_to_target_dim(tensor, target_dim: int):
    """
    将tensor或numpy array从实际维度填充到目标维度

    Args:
        tensor: 输入tensor (torch.Tensor或numpy.ndarray)，形状为 [..., actual_dim]
        target_dim: 目标维度

    Returns:
        填充后的tensor，类型与输入相同
    """
    import numpy as np

    actual_dim = tensor.shape[-1]
    if actual_dim == target_dim:
        return tensor
    elif actual_dim < target_dim:
        # 填充0到目标维度
        pad_size = target_dim - actual_dim
        pad_shape = list(tensor.shape[:-1]) + [pad_size]

        if isinstance(tensor, torch.Tensor):
            # torch.Tensor: 使用torch.zeros
            pad_tensor = torch.zeros(
                pad_shape, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad_tensor], dim=-1)
        elif isinstance(tensor, np.ndarray):
            # numpy.ndarray: 使用np.zeros
            pad_array = np.zeros(pad_shape, dtype=tensor.dtype)
            return np.concatenate([tensor, pad_array], axis=-1)
        else:
            raise TypeError(f"Unsupported tensor type: {type(tensor)}")
    else:
        # 截断到目标维度（不应该发生，但以防万一）
        return tensor[..., :target_dim]


def pad_dataset_stats(dataset_stats: Dict[str, Dict],
                      target_action_dim: int = 32,
                      target_state_dim: int = 32) -> Dict[str, Dict]:
    """
    将dataset_stats中的action和state统计信息填充到目标维度

    对于mean：填充0
    对于std：填充1（这样归一化时填充部分不会被改变）

    Args:
        dataset_stats: 数据集统计信息字典 (可以是torch.Tensor或numpy.ndarray)
        target_action_dim: 目标action维度
        target_state_dim: 目标state维度

    Returns:
        填充后的dataset_stats
    """
    import numpy as np

    def pad_with_ones(tensor, target_dim):
        """填充1到目标维度（用于std）"""
        actual_dim = tensor.shape[-1]
        if actual_dim >= target_dim:
            return tensor

        pad_size = target_dim - actual_dim
        pad_shape = list(tensor.shape[:-1]) + [pad_size]

        if isinstance(tensor, torch.Tensor):
            pad_tensor = torch.ones(
                pad_shape, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad_tensor], dim=-1)
        elif isinstance(tensor, np.ndarray):
            pad_array = np.ones(pad_shape, dtype=tensor.dtype)
            return np.concatenate([tensor, pad_array], axis=-1)
        else:
            raise TypeError(f"Unsupported tensor type: {type(tensor)}")

    padded_stats = {}

    for key, stats_dict in dataset_stats.items():
        if 'action' in key.lower():
            # 填充action相关统计
            padded_stats[key] = {}
            for stat_name, stat_tensor in stats_dict.items():
                if stat_name == 'mean':
                    # mean填充0
                    padded_stats[key][stat_name] = pad_tensor_to_target_dim(
                        stat_tensor, target_action_dim)
                elif stat_name == 'std':
                    # std填充1（避免除0，且不改变填充部分的值）
                    padded_stats[key][stat_name] = pad_with_ones(
                        stat_tensor, target_action_dim)
                else:
                    # 其他统计信息（如min, max）也需要填充
                    padded_stats[key][stat_name] = pad_tensor_to_target_dim(
                        stat_tensor, target_action_dim)

        elif 'state' in key.lower() or 'observation.state' in key:
            # 填充state相关统计
            padded_stats[key] = {}
            for stat_name, stat_tensor in stats_dict.items():
                if stat_name == 'mean':
                    padded_stats[key][stat_name] = pad_tensor_to_target_dim(
                        stat_tensor, target_state_dim)
                elif stat_name == 'std':
                    padded_stats[key][stat_name] = pad_with_ones(
                        stat_tensor, target_state_dim)
                else:
                    padded_stats[key][stat_name] = pad_tensor_to_target_dim(
                        stat_tensor, target_state_dim)
        else:
            # 不是action或state，直接复制
            padded_stats[key] = stats_dict

    return padded_stats


def create_lerobot_dataset_with_deltas(
    repo_id: str,
    root: str,
    episodes: list,
    delta_timestamps: Dict[str, list]
) -> LeRobotDataset:
    """
    创建LeRobotDataset并配置delta_timestamps以加载action chunks

    Args:
        repo_id: Dataset repository ID
        root: Dataset root path
        episodes: List of episode indices
        delta_timestamps: Delta timestamps配置，例如：
            {
                "observation.state": [0],  # 当前帧
                "action": [i/fps for i in range(50)]  # 未来50帧
            }

    Returns:
        配置好的LeRobotDataset
    """
    return LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps
    )


def create_dataloader_with_language(
    dataset: LeRobotDataset,
    language_instruction: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    drop_last: bool = False,
    target_action_dim: int = 32,
    target_state_dim: int = 32,
    use_augmentation: bool = True,
    augmentation_prob: float = 0.5
) -> DataLoader:
    """
    创建包含language instruction的DataLoader，并自动填充action/state维度

    Args:
        dataset: LeRobot数据集
        language_instruction: 任务的language instruction
        batch_size: batch大小
        num_workers: worker数量
        pin_memory: 是否pin memory
        drop_last: 是否丢弃最后一个batch
        target_action_dim: 目标action维度（默认32，与SmolVLA预训练一致）
        target_state_dim: 目标state维度（默认32，与SmolVLA预训练一致）
        use_augmentation: 是否使用数据增强
        augmentation_prob: 数据增强概率

    Returns:
        DataLoader
    """

    # 创建数据增强器
    augmenter = DeterministicAugmenterColor() if use_augmentation else None

    def collate_fn_with_language(batch):
        """为batch添加language instruction并填充action/state维度"""
        # 使用默认collate
        from torch.utils.data._utils.collate import default_collate
        batch_dict = default_collate(batch)

        # 添加task字段
        batch_size = batch_dict[list(batch_dict.keys())[0]].shape[0]
        batch_dict['task'] = [language_instruction] * batch_size

        # 数据增强（50%概率应用）
        if augmenter is not None and random.random() < augmentation_prob:
            augmenter.set_random_params()
            for key in batch_dict.keys():
                if 'image' in key.lower() and isinstance(batch_dict[key], torch.Tensor):
                    # 应用图像增强
                    batch_dict[key] = augmenter.apply_augment_sequence(
                        batch_dict[key])

        # 填充action和state维度（从Kuavo的16维到SmolVLA的32维）
        for key in batch_dict.keys():
            if isinstance(batch_dict[key], torch.Tensor):
                if 'action' in key.lower():
                    # 填充action维度
                    batch_dict[key] = pad_tensor_to_target_dim(
                        batch_dict[key], target_action_dim)
                elif 'state' in key.lower() or 'observation.state' in key:
                    # 填充state维度
                    batch_dict[key] = pad_tensor_to_target_dim(
                        batch_dict[key], target_state_dim)

        return batch_dict

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn_with_language,
        prefetch_factor=2,
        persistent_workers=True,
    )


def create_mixed_dataloader(
    cfg: DictConfig,
    task_cfg: DictConfig,
    replay_manager: Optional[ReplayDatasetManager] = None,
    dataset_fps: int = 10
) -> DataLoader:
    """
    创建混合了replay数据的DataLoader

    Args:
        cfg: 基础配置
        task_cfg: 当前任务配置
        replay_manager: Replay数据管理器
        dataset_fps: 数据集的fps（从metadata读取）

    Returns:
        混合数据的DataLoader
    """
    task_id = task_cfg.task.id
    language_instruction = task_cfg.task.language_instruction

    # 构建delta_timestamps配置 (用于加载action chunks)
    chunk_size = cfg.policy.chunk_size
    delta_timestamps = {
        "observation.state": [0],  # 只取当前帧
        # 未来chunk_size帧
        "action": [i / dataset_fps for i in range(chunk_size)],
    }

    print(f"📐 Dataset delta_timestamps configuration:")
    print(f"   - Dataset FPS: {dataset_fps}")
    print(f"   - observation.state: current frame only")
    print(
        f"   - action: {chunk_size} future frames ({chunk_size/dataset_fps:.2f}s @ {dataset_fps}fps)")

    # 当前任务数据集（使用delta_timestamps）
    current_dataset = LeRobotDataset(
        task_cfg.task.data.repoid,
        root=task_cfg.task.data.root,
        episodes=list(range(
            task_cfg.task.data.episodes_to_use[0],
            task_cfg.task.data.episodes_to_use[1] + 1
        )),
        delta_timestamps=delta_timestamps
    )

    print(f"📊 Current Task {task_id} Dataset: {len(current_dataset)} frames")

    # 如果是第一个任务或不使用replay，直接返回
    if task_id == 1 or not cfg.sequential.use_replay_buffer:
        return create_dataloader_with_language(
            current_dataset,
            language_instruction,
            cfg.training.batch_size,
            cfg.training.num_workers,
            pin_memory=(cfg.training.device != 'cpu'),
            drop_last=cfg.training.drop_last
        )

    # 混合replay数据
    if replay_manager is None:
        raise ValueError("replay_manager is required for task > 1")

    # 创建混合数据集
    # 注意：每个任务需要自己的language instruction
    all_datasets = [(current_dataset, language_instruction)]

    for replay_task_id, replay_dataset in replay_manager.replay_datasets.items():
        replay_task_cfg = load_task_config(
            Path(cfg.hydra.run.dir).parent.parent.parent / "configs/policy", replay_task_id)
        replay_language = replay_task_cfg.task.language_instruction
        all_datasets.append((replay_dataset, replay_language))
        print(
            f"📦 Adding Task {replay_task_id} replay: {len(replay_dataset)} frames")

    # 为每个数据集创建单独的dataloader，然后轮流采样
    # 简化版本：直接concatenate datasets
    # 注意：这里每个dataset都需要保留自己的language instruction

    class MixedDataset(torch.utils.data.Dataset):
        """混合多个数据集，每个数据集保留自己的language instruction"""

        def __init__(self, datasets_with_language):
            self.datasets_with_language = datasets_with_language
            self.lengths = [len(ds) for ds, _ in datasets_with_language]
            self.total_length = sum(self.lengths)

            # 计算每个数据集的采样概率（基于replay weights）
            stage_key = f"stage{task_id}_replay"
            replay_config = cfg.sequential.get(stage_key, {})

            self.weights = []
            for i, (ds, _) in enumerate(datasets_with_language):
                if i == 0:
                    # 当前任务的weight
                    task_key = f"task{task_id}"
                    weight = replay_config.get(task_key, 1.0)
                else:
                    # Replay任务的weight
                    task_key = f"task{i}"  # i对应replay_task_id
                    weight = replay_config.get(task_key, 0.1)
                self.weights.append(weight)

            # 归一化weights
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]

        def __len__(self):
            return self.total_length

        def __getitem__(self, idx):
            # 根据weights随机选择一个dataset
            import random
            dataset_idx = random.choices(
                range(len(self.datasets_with_language)), weights=self.weights, k=1)[0]
            dataset, language = self.datasets_with_language[dataset_idx]

            # 从该dataset随机选择一个样本
            sample_idx = random.randint(0, len(dataset) - 1)
            sample = dataset[sample_idx]

            # 添加language instruction
            sample['task'] = language

            return sample

    mixed_dataset = MixedDataset(all_datasets)

    print(f"📊 Mixed Dataset: {len(mixed_dataset)} frames (with replay)")
    print(f"   Weights: {mixed_dataset.weights}")

    # 创建数据增强器
    augmenter = DeterministicAugmenterColor()

    def collate_fn_with_padding(batch):
        """collate函数：处理mixed dataset的batch并填充维度"""
        from torch.utils.data._utils.collate import default_collate

        # batch中的每个sample已经有'task'字段了
        # 先提取所有非'task'字段进行collate
        tasks = [sample.pop('task') for sample in batch]

        # 使用默认collate处理其他字段
        batch_dict = default_collate(batch)

        # 添加task字段回去
        batch_dict['task'] = tasks

        # 数据增强（50%概率应用）
        if random.random() < 0.5:
            augmenter.set_random_params()
            for key in batch_dict.keys():
                if 'image' in key.lower() and isinstance(batch_dict[key], torch.Tensor):
                    # 应用图像增强
                    batch_dict[key] = augmenter.apply_augment_sequence(
                        batch_dict[key])

        # 填充action和state维度
        target_action_dim = cfg.policy.max_action_dim
        target_state_dim = cfg.policy.max_state_dim

        for key in batch_dict.keys():
            if isinstance(batch_dict[key], torch.Tensor):
                if 'action' in key.lower():
                    batch_dict[key] = pad_tensor_to_target_dim(
                        batch_dict[key], target_action_dim)
                elif 'state' in key.lower() or 'observation.state' in key:
                    batch_dict[key] = pad_tensor_to_target_dim(
                        batch_dict[key], target_state_dim)

        return batch_dict

    return DataLoader(
        mixed_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True,
        pin_memory=(cfg.training.device != 'cpu'),
        drop_last=cfg.training.drop_last,
        collate_fn=collate_fn_with_padding,
        prefetch_factor=1
    )


def validate_all_tasks(
    policy: SmolVLAPolicyWrapper,
    cfg: DictConfig,
    current_task_id: int,
    device: torch.device,
    cfg_root: Path,
    dataset_fps: int = 10
) -> Dict[int, float]:
    """
    验证所有之前的任务（检测遗忘）

    Args:
        policy: SmolVLA策略
        cfg: 配置
        current_task_id: 当前任务ID
        device: 设备
        cfg_root: 配置根目录

    Returns:
        validation_results: {task_id: avg_loss}
    """
    print("\n" + "="*70)
    print(f"🔍 Multi-Task Validation (Tasks 1-{current_task_id})")
    print("="*70)

    policy.eval()
    validation_results = {}

    for task_id in range(1, current_task_id + 1):
        print(f"\n📊 Validating Task {task_id}...")

        # 加载任务配置
        task_cfg = load_task_config(cfg_root, task_id)

        # 加载验证集（使用前N个episodes作为验证，避免与训练数据完全分离）
        # 注意：这是快速验证方法，使用训练数据的子集
        num_val_episodes = cfg.training.validation_episodes

        # 从训练episodes中选择前N个作为验证
        train_episode_start = task_cfg.task.data.episodes_to_use[0]
        train_episode_end = task_cfg.task.data.episodes_to_use[1]

        # 验证用前N个episodes
        val_episode_end = min(train_episode_start +
                              num_val_episodes - 1, train_episode_end)
        val_episodes = list(range(train_episode_start, val_episode_end + 1))

        # 确保不超过num_val_episodes
        val_episodes = val_episodes[:num_val_episodes]

        # 构建delta_timestamps配置
        chunk_size = cfg.policy.chunk_size
        delta_timestamps = {
            "observation.state": [0],
            "action": [i / dataset_fps for i in range(chunk_size)],
        }

        val_dataset = LeRobotDataset(
            task_cfg.task.data.repoid,
            root=task_cfg.task.data.root,
            episodes=val_episodes,
            delta_timestamps=delta_timestamps
        )

        val_loader = create_dataloader_with_language(
            val_dataset,
            task_cfg.task.language_instruction,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers // 2,
            pin_memory=(device.type != 'cpu'),
            drop_last=False
        )

        # 验证
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Task {task_id} Validation", leave=False):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                loss, _ = policy.forward(batch)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / \
            num_batches if num_batches > 0 else float('inf')
        validation_results[task_id] = avg_loss

        print(f"  Task {task_id} Validation Loss: {avg_loss:.4f}")

    # 分析遗忘情况
    if current_task_id > 1:
        print("\n⚠️  Forgetting Analysis:")
        for task_id in range(1, current_task_id):
            loss = validation_results[task_id]
            # 简单的阈值判断
            if loss < 0.7:
                status = "✅ Well Retained"
            elif loss < 1.0:
                status = "⚠️  Slight Degradation"
            else:
                status = "❌ Significant Forgetting"

            print(f"  Task {task_id}: {status} (loss={loss:.4f})")

    print("="*70 + "\n")

    policy.train()
    return validation_results


@hydra.main(config_path="../configs/policy/", config_name="smolvla_sequential_base", version_base=None)
def main(cfg: DictConfig):
    """主训练流程"""

    # 设置 HuggingFace 镜像源以提高下载速度
    import os

    # 从配置读取 HF endpoint，如果没有配置则使用默认镜像源
    hf_endpoint = cfg.get('hf_endpoint', 'https://hf-mirror.com')
    if hf_endpoint:
        os.environ['HF_ENDPOINT'] = hf_endpoint
        print(f"✅ 已设置 HuggingFace 下载源: {hf_endpoint}")
    else:
        print("ℹ️  使用默认 HuggingFace Hub: https://huggingface.co")

    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

    logger = setup_logging()
    set_seed(cfg.training.seed)

    # 加载任务配置
    # 从Hydra配置获取任务名称（支持两种格式：tasks/task1_moving_grasp 或 task1_moving_grasp）
    task_param = cfg.get('task', 'task1_moving_grasp')
    if task_param.startswith('tasks/'):
        task_param = task_param.replace('tasks/', '')

    # 动态加载任务配置
    cfg_root = Path(__file__).parent.parent / "configs/policy"
    task_cfg = load_task_config(cfg_root, int(
        task_param.split('_')[0].replace('task', '')))
    task_id = task_cfg.task.id
    task_name = task_cfg.task.name

    # 设置task字段用于路径（格式：task{id}_{name}，如task1_moving_grasp）
    cfg.task = f"task{task_id}_{task_name}"

    print("\n" + "="*70)
    print(f"🤖 SmolVLA Sequential Training - Stage {task_id}")
    print("="*70)
    print(f"Task ID: {task_id}")
    print(f"Task Name: {task_name}")
    print(f"Description: {task_cfg.task.description}")
    print(f"Language: {task_cfg.task.language_instruction}")
    print("="*70 + "\n")

    # 设置输出目录（与其他策略一致的格式）
    # 格式: outputs/train/{task}/{method}/run_{timestamp}
    # 展开: outputs/train/task1_moving_grasp/smolvla_sequential/run_20251011_123456
    output_directory = Path(
        cfg.training.output_directory) / f"run_{cfg.timestamp}"
    output_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_directory))

    print(f"📁 Output Directory: {output_directory}")
    print(f"📅 Timestamp: {cfg.timestamp}\n")

    device = torch.device(cfg.training.device)

    # ==================== 加载数据集元信息 ====================
    print("📂 Loading Dataset Metadata...")
    dataset_metadata = LeRobotDatasetMetadata(
        task_cfg.task.data.repoid,
        root=task_cfg.task.data.root
    )

    # 获取数据集fps（用于配置delta_timestamps）
    dataset_fps = dataset_metadata.fps
    print(f"📊 Dataset FPS: {dataset_fps}")

    # 构建features
    features = dataset_to_policy_features(dataset_metadata.features)
    input_features = {k: ft for k, ft in features.items(
    ) if ft.type is not FeatureType.ACTION}
    output_features = {k: ft for k,
                       ft in features.items() if ft.type is FeatureType.ACTION}

    dataset_stats = dataset_metadata.stats

    # 填充dataset_stats到目标维度（Kuavo 16维 → SmolVLA 32维）
    print("📐 Padding dataset_stats to match SmolVLA dimensions (16D → 32D)...")
    dataset_stats = pad_dataset_stats(
        dataset_stats,
        target_action_dim=cfg.policy.max_action_dim,
        target_state_dim=cfg.policy.max_state_dim
    )
    print("✅ Dataset stats padded successfully")

    # ==================== 构建Policy配置 ====================
    from hydra.utils import instantiate

    policy_cfg = instantiate(
        cfg.policy,
        input_features=input_features,
        output_features=output_features,
        device=device,
    )

    # Override learning rate from task config
    if hasattr(task_cfg.task.training, 'policy'):
        policy_cfg.optimizer_lr = task_cfg.task.training.policy.optimizer_lr
        policy_cfg.scheduler_warmup_steps = task_cfg.task.training.policy.scheduler_warmup_steps
        policy_cfg.scheduler_decay_steps = task_cfg.task.training.policy.scheduler_decay_steps

    # ==================== 加载/创建模型 ====================
    if task_cfg.task.training.resume_from == 'pretrained':
        # Stage 1: 从HuggingFace预训练加载
        print(
            f"\n📂 Loading pretrained SmolVLA from {task_cfg.task.training.pretrained_path}")
        policy = SmolVLAPolicyWrapper.from_pretrained(
            task_cfg.task.training.pretrained_path,
            config=policy_cfg,
            dataset_stats=dataset_stats
        )

    elif task_cfg.task.training.resume_from == 'task':
        # Stage 2+: 从上一个任务继续
        prev_task_id = task_cfg.task.training.resume_task_id
        resume_path = task_cfg.task.training.resume_path

        print(f"\n📂 Loading from Task {prev_task_id}: {resume_path}")
        policy = SmolVLAPolicyWrapper.from_pretrained(
            resume_path,
            config=policy_cfg,
            dataset_stats=dataset_stats
        )
        print(
            f"✅ Successfully loaded Task {prev_task_id} model for sequential training")

    else:
        # 从头训练（不推荐）
        print("\n⚠️  Training from scratch (not recommended for sequential training)")
        policy = SmolVLAPolicyWrapper(policy_cfg, dataset_stats)

    policy = policy.to(device)

    policy.train()

    # ==================== 准备数据 ====================
    # 加载replay buffer（如果需要）
    replay_manager = None
    if task_id > 1 and cfg.sequential.use_replay_buffer:
        cfg_root = Path(__file__).parent.parent / "configs/policy"
        replay_manager = ReplayDatasetManager(
            cfg, task_id, cfg_root, dataset_fps)
        replay_manager.load_replay_tasks()

    # 创建dataloader（传递dataset_fps）
    dataloader = create_mixed_dataloader(
        cfg, task_cfg, replay_manager, dataset_fps)

    # ==================== 构建优化器 ====================
    optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
    lr_scheduler = policy.config.get_scheduler_preset().build(
        optimizer,
        num_training_steps=task_cfg.task.training.max_epoch * len(dataloader)
    )

    print(f"\n🎯 Training Configuration:")
    print(f"   Epochs: {task_cfg.task.training.max_epoch}")
    print(f"   Batch Size: {cfg.training.batch_size}")
    print(f"   Learning Rate: {policy_cfg.optimizer_lr}")
    print(f"   Steps per Epoch: {len(dataloader)}")
    print(
        f"   Total Steps: {task_cfg.task.training.max_epoch * len(dataloader)}")

    # ==================== 训练循环 ====================
    print("\n🚀 Starting Training...")
    print("="*70 + "\n")

    best_loss = float('inf')

    for epoch in range(task_cfg.task.training.max_epoch):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{task_cfg.task.training.max_epoch}")
        print(f"{'='*70}")

        # 训练
        policy.train()
        total_loss = 0.0
        num_batches = 0

        epoch_bar = tqdm(
            dataloader,
            desc=f"Training Epoch {epoch+1}",
            dynamic_ncols=True,
            leave=False
        )

        for batch in epoch_bar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward
            loss, _ = policy.forward(batch)

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                max_norm=policy_cfg.optimizer_grad_clip_norm
            )

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1

            epoch_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}"
            )

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # TensorBoard logging
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], epoch)

        # 多任务验证
        if (epoch + 1) % cfg.training.validation_freq_epoch == 0 and cfg.training.get('validate_all_previous_tasks', False):
            cfg_root = Path(__file__).parent.parent / "configs/policy"
            validation_results = validate_all_tasks(
                policy, cfg, task_id, device, cfg_root, dataset_fps)

            # Log validation results
            for val_task_id, val_loss in validation_results.items():
                writer.add_scalar(
                    f"validation/task{val_task_id}_loss", val_loss, epoch)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_directory / "best"
            policy.save_pretrained(best_path)
            save_rng_state(best_path / "rng_state.pth")

            # 保存训练状态（用于完美恢复训练）
            checkpoint = {
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_loss": best_loss
            }
            torch.save(checkpoint, best_path / "learning_state.pth")

            print(f"✅ Best model saved: loss={best_loss:.4f}")

        # 定期保存
        if (epoch + 1) % cfg.training.save_freq_epoch == 0:
            epoch_path = output_directory / f"epoch{epoch+1}"
            policy.save_pretrained(epoch_path)
            save_rng_state(epoch_path / "rng_state.pth")

            # 保存训练状态
            checkpoint = {
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_loss": best_loss
            }
            torch.save(checkpoint, epoch_path / "learning_state.pth")

            print(f"✅ Checkpoint saved: epoch {epoch+1}")

    writer.close()

    # ==================== 保存最终状态 ====================
    # 保存最终模型和训练状态（用于完美恢复或继续训练）
    print("\n💾 Saving final model and training state...")
    policy.save_pretrained(output_directory)
    save_rng_state(output_directory / "rng_state.pth")

    # 保存最终训练状态
    final_checkpoint = {
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": task_cfg.task.training.max_epoch,
        "best_loss": best_loss
    }
    torch.save(final_checkpoint, output_directory / "learning_state.pth")
    print("✅ Final model and training state saved")

    # ==================== 最终验证 ====================
    print("\n" + "="*70)
    print("🎯 Final Multi-Task Validation")
    print("="*70)

    cfg_root = Path(__file__).parent.parent / "configs/policy"
    final_results = validate_all_tasks(
        policy, cfg, task_id, device, cfg_root, dataset_fps)

    # 保存训练结果
    results_file = output_directory / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'task_id': task_id,
            'task_name': task_name,
            'description': task_cfg.task.description,
            'language_instruction': task_cfg.task.language_instruction,
            'best_loss': best_loss,
            'final_validation': {str(k): v for k, v in final_results.items()},
            'training_epochs': task_cfg.task.training.max_epoch,
            'learning_rate': policy_cfg.optimizer_lr,
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ Task {task_id} Training Completed!")
    print(f"{'='*70}")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Model saved to: {output_directory}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
