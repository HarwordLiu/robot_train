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

# Ensure custom patches are applied
import lerobot_patches.custom_patches

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from functools import partial
import json
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.random_utils import set_seed
from lerobot.configs.types import FeatureType

# 导入SmolVLA模块
from kuavo_train.wrapper.policy.smolvla.SmolVLAPolicyWrapper import SmolVLAPolicyWrapper
from kuavo_train.wrapper.policy.smolvla.SmolVLAConfigWrapper import SmolVLAConfigWrapper


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

    def __init__(self, cfg: DictConfig, current_task_id: int, cfg_root: Path):
        self.cfg = cfg
        self.current_task_id = current_task_id
        self.cfg_root = cfg_root
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

        for task_key, weight in replay_config.items():
            if 'task' in task_key:
                task_id = int(task_key.replace('task', ''))

                # 只加载之前的任务
                if task_id < self.current_task_id:
                    print(
                        f"  Loading Task {task_id} (weight: {weight:.1%})...")

                    # 加载任务配置
                    task_cfg = load_task_config(self.cfg_root, task_id)

                    # 加载数据集
                    dataset = LeRobotDataset(
                        task_cfg.task.data.repoid,
                        root=task_cfg.task.data.root,
                        episodes=list(range(
                            task_cfg.task.data.episodes_to_use[0],
                            task_cfg.task.data.episodes_to_use[1] + 1
                        ))
                    )

                    self.replay_datasets[task_id] = dataset
                    self.replay_weights[task_id] = weight

                    print(
                        f"    ✅ Loaded {len(dataset)} frames from Task {task_id}")

        print("="*70 + "\n")
        return self.replay_datasets, self.replay_weights


def create_dataloader_with_language(
    dataset: LeRobotDataset,
    language_instruction: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    drop_last: bool = False,
    target_action_dim: int = 16
) -> DataLoader:
    """
    创建包含language instruction的DataLoader

    Args:
        dataset: LeRobot数据集
        language_instruction: 任务的language instruction
        batch_size: batch大小
        num_workers: worker数量
        pin_memory: 是否pin memory
        drop_last: 是否丢弃最后一个batch

    Returns:
        DataLoader
    """

    def collate_fn_with_language(batch):
        """为batch添加language instruction并适配动作维度"""
        # 使用默认collate
        from torch.utils.data._utils.collate import default_collate
        batch_dict = default_collate(batch)

        # 添加task字段
        batch_size = batch_dict[list(batch_dict.keys())[0]].shape[0]
        batch_dict['task'] = [language_instruction] * batch_size

        # 适配动作维度：如果动作是16维但需要32维，进行填充
        for key, value in batch_dict.items():
            if isinstance(value, torch.Tensor) and 'action' in key.lower():
                if value.shape[-1] == 16 and target_action_dim > 16:
                    # 填充动作维度
                    if value.shape[-1] < target_action_dim:
                        padding_size = target_action_dim - value.shape[-1]
                        padding = torch.zeros(
                            *value.shape[:-1], padding_size, dtype=value.dtype)
                        batch_dict[key] = torch.cat([value, padding], dim=-1)

        return batch_dict

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn_with_language,
        prefetch_factor=1
    )


def create_mixed_dataloader(
    cfg: DictConfig,
    task_cfg: DictConfig,
    replay_manager: Optional[ReplayDatasetManager] = None
) -> DataLoader:
    """
    创建混合了replay数据的DataLoader

    Args:
        cfg: 基础配置
        task_cfg: 当前任务配置
        replay_manager: Replay数据管理器

    Returns:
        混合数据的DataLoader
    """
    task_id = task_cfg.task.id
    language_instruction = task_cfg.task.language_instruction

    # 当前任务数据集
    current_dataset = LeRobotDataset(
        task_cfg.task.data.repoid,
        root=task_cfg.task.data.root,
        episodes=list(range(
            task_cfg.task.data.episodes_to_use[0],
            task_cfg.task.data.episodes_to_use[1] + 1
        ))
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
            drop_last=cfg.training.drop_last,
            target_action_dim=cfg.training.target_action_dim
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

    return DataLoader(
        mixed_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True,
        pin_memory=(cfg.training.device != 'cpu'),
        drop_last=cfg.training.drop_last,
        prefetch_factor=1
    )


def validate_all_tasks(
    policy: SmolVLAPolicyWrapper,
    cfg: DictConfig,
    current_task_id: int,
    device: torch.device,
    cfg_root: Path
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

        # 加载验证集（使用最后N个episodes）
        num_val_episodes = cfg.training.validation_episodes
        total_episodes = task_cfg.task.data.episodes_to_use[1] + 1
        val_start = max(0, total_episodes - num_val_episodes)

        val_dataset = LeRobotDataset(
            task_cfg.task.data.repoid,
            root=task_cfg.task.data.root,
            episodes=list(range(val_start, total_episodes))
        )

        val_loader = create_dataloader_with_language(
            val_dataset,
            task_cfg.task.language_instruction,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers // 2,
            pin_memory=(device.type != 'cpu'),
            drop_last=False,
            target_action_dim=cfg.training.target_action_dim
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
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

    # 可选：设置其他镜像源
    # os.environ['HF_ENDPOINT'] = 'https://huggingface.co'  # 官方源
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   # 中国镜像源

    logger = setup_logging()
    set_seed(cfg.training.seed)

    # 加载任务配置
    # 从Hydra配置获取任务名称
    task_name = cfg.get('task', 'tasks/task1_moving_grasp')
    if task_name.startswith('tasks/'):
        task_name = task_name.replace('tasks/', '')

    # 动态加载任务配置
    cfg_root = Path(__file__).parent.parent / "configs/policy"
    task_cfg = load_task_config(cfg_root, int(
        task_name.split('_')[0].replace('task', '')))
    task_id = task_cfg.task.id
    task_name = task_cfg.task.name

    print("\n" + "="*70)
    print(f"🤖 SmolVLA Sequential Training - Stage {task_id}")
    print("="*70)
    print(f"Task ID: {task_id}")
    print(f"Task Name: {task_name}")
    print(f"Description: {task_cfg.task.description}")
    print(f"Language: {task_cfg.task.language_instruction}")
    print("="*70 + "\n")

    # 设置输出目录
    output_directory = Path(cfg.training.output_directory) / \
        f"task{task_id}_{task_name}"
    output_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_directory))

    device = torch.device(cfg.training.device)

    # ==================== 加载数据集元信息 ====================
    print("📂 Loading Dataset Metadata...")
    dataset_metadata = LeRobotDatasetMetadata(
        task_cfg.task.data.repoid,
        root=task_cfg.task.data.root
    )

    # 构建features
    features = dataset_to_policy_features(dataset_metadata.features)
    input_features = {k: ft for k, ft in features.items(
    ) if ft.type is not FeatureType.ACTION}
    output_features = {k: ft for k,
                       ft in features.items() if ft.type is FeatureType.ACTION}

    dataset_stats = dataset_metadata.stats

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

    # 适配动作维度：如果数据是16维但模型是32维，需要填充数据
    if policy.config.max_action_dim > 16:
        print(
            f"\n🔧 Adapting action dimensions: Data=16, Model={policy.config.max_action_dim}")
        print("   Padding action data to match model dimensions...")

    policy.train()

    # ==================== 准备数据 ====================
    # 加载replay buffer（如果需要）
    replay_manager = None
    if task_id > 1 and cfg.sequential.use_replay_buffer:
        cfg_root = Path(__file__).parent.parent / "configs/policy"
        replay_manager = ReplayDatasetManager(cfg, task_id, cfg_root)
        replay_manager.load_replay_tasks()

    # 创建dataloader
    dataloader = create_mixed_dataloader(cfg, task_cfg, replay_manager)

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
        if (epoch + 1) % cfg.training.validation_freq_epoch == 0:
            cfg_root = Path(__file__).parent.parent / "configs/policy"
            validation_results = validate_all_tasks(
                policy, cfg, task_id, device, cfg_root)

            # Log validation results
            for val_task_id, val_loss in validation_results.items():
                writer.add_scalar(
                    f"validation/task{val_task_id}_loss", val_loss, epoch)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_directory / "best"
            policy.save_pretrained(best_path)
            print(f"✅ Best model saved: loss={best_loss:.4f}")

        # 定期保存
        if (epoch + 1) % cfg.training.save_freq_epoch == 0:
            epoch_path = output_directory / f"epoch{epoch+1}"
            policy.save_pretrained(epoch_path)
            print(f"✅ Checkpoint saved: epoch {epoch+1}")

    writer.close()

    # ==================== 最终验证 ====================
    print("\n" + "="*70)
    print("🎯 Final Multi-Task Validation")
    print("="*70)

    cfg_root = Path(__file__).parent.parent / "configs/policy"
    final_results = validate_all_tasks(policy, cfg, task_id, device, cfg_root)

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
