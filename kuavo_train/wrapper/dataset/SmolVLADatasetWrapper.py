"""
SmolVLA优化的Dataset包装器

将数据增强和向量填充操作从collate函数移到worker进程中的__getitem__方法，
减少主线程阻塞，提升DataLoader性能。
"""
import torch
import random
from typing import Dict, Optional
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from kuavo_train.utils.augmenter import DeterministicAugmenterColor


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


class SmolVLADatasetWrapper(torch.utils.data.Dataset):
    """
    包装LeRobotDataset，在worker进程中完成数据增强和向量填充

    优势：
    1. 数据增强在worker进程中执行，不阻塞主线程
    2. 向量填充在worker进程中执行，减少collate函数工作量
    3. 每个worker独立进行数据增强，增加随机性
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        language_instruction: str,
        target_action_dim: int = 32,
        target_state_dim: int = 32,
        use_augmentation: bool = True,
        augmentation_prob: float = 0.5,
    ):
        """
        Args:
            dataset: LeRobot数据集
            language_instruction: 任务的language instruction
            target_action_dim: 目标action维度
            target_state_dim: 目标state维度
            use_augmentation: 是否使用数据增强
            augmentation_prob: 数据增强概率
        """
        self.dataset = dataset
        self.language_instruction = language_instruction
        self.target_action_dim = target_action_dim
        self.target_state_dim = target_state_dim
        self.use_augmentation = use_augmentation
        self.augmentation_prob = augmentation_prob

        # 为每个worker创建独立的数据增强器
        # 注意：这会在worker进程中初始化，确保每个worker有独立的随机状态
        self.augmenter = DeterministicAugmenterColor() if use_augmentation else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict:
        """
        获取单个样本，在worker进程中完成所有处理

        包括：
        1. 从LeRobotDataset加载数据
        2. 添加language instruction
        3. 数据增强（如果启用）
        4. 向量填充（action和state维度）
        """
        # 从底层数据集加载数据
        item = self.dataset[idx]

        # 添加task字段
        item['task'] = self.language_instruction

        # 数据增强（在worker进程中执行，50%概率应用）
        if self.augmenter is not None and random.random() < self.augmentation_prob:
            self.augmenter.set_random_params()
            for key in item.keys():
                if 'image' in key.lower() and isinstance(item[key], torch.Tensor):
                    # 应用图像增强
                    item[key] = self.augmenter.apply_augment_sequence(
                        item[key])

        # 填充action和state维度（从Kuavo的16维到SmolVLA的32维）
        for key in item.keys():
            if isinstance(item[key], torch.Tensor):
                if 'action' in key.lower():
                    # 填充action维度
                    item[key] = pad_tensor_to_target_dim(
                        item[key], self.target_action_dim)
                elif 'state' in key.lower() or 'observation.state' in key:
                    # 填充state维度
                    item[key] = pad_tensor_to_target_dim(
                        item[key], self.target_state_dim)

        return item


class SmolVLAMixedDatasetWrapper(torch.utils.data.Dataset):
    """
    包装多个数据集（用于replay buffer场景），在worker进程中完成所有处理
    """

    def __init__(
        self,
        datasets_with_language: list,
        weights: list,
        target_action_dim: int = 32,
        target_state_dim: int = 32,
        use_augmentation: bool = True,
        augmentation_prob: float = 0.5,
    ):
        """
        Args:
            datasets_with_language: [(dataset, language_instruction), ...] 列表
            weights: 每个数据集的采样权重（已归一化）
            target_action_dim: 目标action维度
            target_state_dim: 目标state维度
            use_augmentation: 是否使用数据增强
            augmentation_prob: 数据增强概率
        """
        self.datasets_with_language = datasets_with_language
        self.weights = weights
        self.lengths = [len(ds) for ds, _ in datasets_with_language]
        self.total_length = sum(self.lengths)
        self.target_action_dim = target_action_dim
        self.target_state_dim = target_state_dim
        self.use_augmentation = use_augmentation
        self.augmentation_prob = augmentation_prob

        # 为每个worker创建独立的数据增强器
        self.augmenter = DeterministicAugmenterColor() if use_augmentation else None

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx) -> Dict:
        """
        根据weights随机选择一个dataset，然后随机选择一个样本
        在worker进程中完成所有处理
        """
        # 根据weights随机选择一个dataset
        dataset_idx = random.choices(
            range(len(self.datasets_with_language)),
            weights=self.weights,
            k=1
        )[0]
        dataset, language = self.datasets_with_language[dataset_idx]

        # 从该dataset随机选择一个样本
        sample_idx = random.randint(0, len(dataset) - 1)
        item = dataset[sample_idx]

        # 添加language instruction
        item['task'] = language

        # 数据增强（在worker进程中执行，50%概率应用）
        if self.augmenter is not None and random.random() < self.augmentation_prob:
            self.augmenter.set_random_params()
            for key in item.keys():
                if 'image' in key.lower() and isinstance(item[key], torch.Tensor):
                    # 应用图像增强
                    item[key] = self.augmenter.apply_augment_sequence(
                        item[key])

        # 填充action和state维度
        for key in item.keys():
            if isinstance(item[key], torch.Tensor):
                if 'action' in key.lower():
                    item[key] = pad_tensor_to_target_dim(
                        item[key], self.target_action_dim)
                elif 'state' in key.lower() or 'observation.state' in key:
                    item[key] = pad_tensor_to_target_dim(
                        item[key], self.target_state_dim)

        return item
