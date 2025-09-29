# -*- coding: utf-8 -*-
"""
Mock观测环境模块

提供模拟观测数据的功能，用于离线测试
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import lerobot_patches.custom_patches

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict

from kuavo_train.wrapper.dataset.LeRobotDatasetWrapper import CustomLeRobotDataset

class MockObservationEnvironment:
    """
    Mock观测环境

    从lerobot数据集创建模拟的观测环境，用于离线评估
    """

    def __init__(self, config):
        """
        初始化Mock环境

        Args:
            config: 评估配置
        """
        self.config = config
        self.device = torch.device(config.common.device)

        # 图像处理配置
        self.image_size = getattr(config, 'image_size', [480, 640])
        self.depth_range = getattr(config, 'depth_range', [0, 1000])

        # 数据集
        self.dataset = None
        self.current_episode = 0
        self.current_step = 0

        # 观测历史 (用于模拟时序)
        self.observation_history = []
        self.max_history_length = 10

        # 统计信息
        self.stats = {
            'total_observations': 0,
            'episodes_loaded': 0,
            'observation_types': defaultdict(int)
        }

    def load_dataset(self, repo_id: str, root: str, episodes: List[int]) -> None:
        """
        加载数据集

        Args:
            repo_id: 数据集ID
            root: 数据根目录
            episodes: episode列表
        """
        print(f"Loading mock environment dataset: {repo_id}")

        self.dataset = CustomLeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
        )

        self.stats['episodes_loaded'] = len(episodes)
        print(f"Loaded {len(self.dataset)} samples from {len(episodes)} episodes")

    def get_observation_from_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        从数据样本提取观测数据

        Args:
            sample: 数据样本

        Returns:
            观测数据字典
        """
        observation = {}

        for key, value in sample.items():
            if key.startswith('observation.'):
                # 确保数据在正确设备上
                if isinstance(value, torch.Tensor):
                    observation[key] = value.to(self.device)

                    # 统计观测类型
                    obs_type = self._get_observation_type(key)
                    self.stats['observation_types'][obs_type] += 1
                else:
                    # 转换为tensor
                    observation[key] = torch.tensor(value).to(self.device)

        self.stats['total_observations'] += 1
        return observation

    def _get_observation_type(self, key: str) -> str:
        """获取观测类型"""
        if 'image' in key.lower() or 'cam' in key.lower():
            return 'image'
        elif 'depth' in key.lower():
            return 'depth'
        elif 'state' in key.lower():
            return 'state'
        else:
            return 'other'

    def preprocess_observation(self, observation: Dict[str, torch.Tensor],
                             add_noise: bool = False) -> Dict[str, torch.Tensor]:
        """
        预处理观测数据

        Args:
            observation: 原始观测数据
            add_noise: 是否添加噪声

        Returns:
            预处理后的观测数据
        """
        processed_obs = {}

        for key, value in observation.items():
            processed_value = value.clone()

            # 根据观测类型进行预处理
            obs_type = self._get_observation_type(key)

            if obs_type == 'image':
                processed_value = self._preprocess_image(processed_value, add_noise)
            elif obs_type == 'depth':
                processed_value = self._preprocess_depth(processed_value, add_noise)
            elif obs_type == 'state':
                processed_value = self._preprocess_state(processed_value, add_noise)

            processed_obs[key] = processed_value

        return processed_obs

    def _preprocess_image(self, image: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        """
        预处理图像观测

        Args:
            image: 图像tensor [C, H, W] 或 [B, C, H, W]
            add_noise: 是否添加噪声

        Returns:
            预处理后的图像
        """
        # 确保图像在[0, 1]范围内
        if image.max() > 1.0:
            image = image.float() / 255.0

        # 调整大小
        if image.dim() == 3:  # [C, H, W]
            image = image.unsqueeze(0)  # [1, C, H, W]

        if image.shape[-2:] != tuple(self.image_size):
            image = torch.nn.functional.interpolate(
                image, size=self.image_size, mode='bilinear', align_corners=False
            )

        # 添加噪声
        if add_noise:
            noise_std = 0.01
            noise = torch.randn_like(image) * noise_std
            image = torch.clamp(image + noise, 0, 1)

        return image

    def _preprocess_depth(self, depth: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        """
        预处理深度观测

        Args:
            depth: 深度tensor
            add_noise: 是否添加噪声

        Returns:
            预处理后的深度
        """
        # 裁剪深度范围
        depth = torch.clamp(depth, self.depth_range[0], self.depth_range[1])

        # 归一化到[0, 1]
        depth_range = self.depth_range[1] - self.depth_range[0]
        depth = (depth - self.depth_range[0]) / depth_range

        # 确保维度正确
        if depth.dim() == 2:  # [H, W]
            depth = depth.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif depth.dim() == 3:  # [1, H, W]
            depth = depth.unsqueeze(0)  # [1, 1, H, W]

        # 调整大小
        if depth.shape[-2:] != tuple(self.image_size):
            depth = torch.nn.functional.interpolate(
                depth, size=self.image_size, mode='nearest'
            )

        # 添加噪声
        if add_noise:
            noise_std = 0.02
            noise = torch.randn_like(depth) * noise_std
            depth = torch.clamp(depth + noise, 0, 1)

        return depth

    def _preprocess_state(self, state: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        """
        预处理状态观测

        Args:
            state: 状态tensor
            add_noise: 是否添加噪声

        Returns:
            预处理后的状态
        """
        # 确保状态是float类型
        state = state.float()

        # 确保维度正确
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [1, state_dim]

        # 添加噪声
        if add_noise:
            noise_std = 0.01
            noise = torch.randn_like(state) * noise_std
            state = state + noise

        return state

    def create_observation_sequence(self, episode_data: List[Dict[str, Any]],
                                  sequence_length: int = 1) -> List[Dict[str, torch.Tensor]]:
        """
        创建观测序列

        Args:
            episode_data: episode数据
            sequence_length: 序列长度

        Returns:
            观测序列
        """
        observations = []

        for i, sample in enumerate(episode_data):
            obs = self.get_observation_from_sample(sample)
            obs = self.preprocess_observation(obs)

            # 如果需要序列，添加历史观测
            if sequence_length > 1 and len(self.observation_history) >= sequence_length - 1:
                # 创建观测序列
                obs_sequence = self.observation_history[-(sequence_length-1):] + [obs]

                # 合并序列观测
                combined_obs = self._combine_observation_sequence(obs_sequence)
                observations.append(combined_obs)
            else:
                observations.append(obs)

            # 更新历史
            self.observation_history.append(obs)
            if len(self.observation_history) > self.max_history_length:
                self.observation_history.pop(0)

        return observations

    def _combine_observation_sequence(self, obs_sequence: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        合并观测序列

        Args:
            obs_sequence: 观测序列

        Returns:
            合并后的观测
        """
        combined = {}

        # 获取所有观测键
        all_keys = set()
        for obs in obs_sequence:
            all_keys.update(obs.keys())

        # 合并每个键的数据
        for key in all_keys:
            values = []
            for obs in obs_sequence:
                if key in obs:
                    values.append(obs[key])
                else:
                    # 如果某个时刻缺失此观测，用零填充
                    if values:
                        zero_value = torch.zeros_like(values[0])
                        values.append(zero_value)

            if values:
                # 在时序维度上堆叠
                combined[key] = torch.stack(values, dim=1)  # [batch, seq_len, ...]

        return combined

    def simulate_real_time_observation(self, observation: Dict[str, torch.Tensor],
                                     delay_ms: float = 0) -> Dict[str, torch.Tensor]:
        """
        模拟实时观测（添加延迟等）

        Args:
            observation: 观测数据
            delay_ms: 延迟毫秒数

        Returns:
            模拟实时观测
        """
        import time

        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

        # 可以在这里添加更多实时模拟逻辑
        # 比如：观测丢失、数据损坏等

        return observation

    def add_observation_noise(self, observation: Dict[str, torch.Tensor],
                            noise_config: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        添加观测噪声

        Args:
            observation: 观测数据
            noise_config: 噪声配置 {'image': 0.01, 'depth': 0.02, 'state': 0.01}

        Returns:
            添加噪声后的观测
        """
        noisy_obs = {}

        for key, value in observation.items():
            obs_type = self._get_observation_type(key)
            noise_std = noise_config.get(obs_type, 0.0)

            if noise_std > 0:
                noise = torch.randn_like(value) * noise_std
                noisy_value = value + noise

                # 根据观测类型进行适当的裁剪
                if obs_type in ['image', 'depth']:
                    noisy_value = torch.clamp(noisy_value, 0, 1)

                noisy_obs[key] = noisy_value
            else:
                noisy_obs[key] = value

        return noisy_obs

    def validate_observation(self, observation: Dict[str, torch.Tensor]) -> bool:
        """
        验证观测数据的有效性

        Args:
            observation: 观测数据

        Returns:
            是否有效
        """
        try:
            for key, value in observation.items():
                # 检查tensor有效性
                if not isinstance(value, torch.Tensor):
                    print(f"Invalid observation type for {key}: {type(value)}")
                    return False

                # 检查是否包含NaN或Inf
                if torch.isnan(value).any() or torch.isinf(value).any():
                    print(f"NaN or Inf detected in observation {key}")
                    return False

                # 检查维度合理性
                obs_type = self._get_observation_type(key)
                if obs_type == 'image' and value.dim() < 3:
                    print(f"Invalid image dimensions for {key}: {value.shape}")
                    return False
                elif obs_type == 'depth' and value.dim() < 2:
                    print(f"Invalid depth dimensions for {key}: {value.shape}")
                    return False
                elif obs_type == 'state' and value.dim() < 1:
                    print(f"Invalid state dimensions for {key}: {value.shape}")
                    return False

            return True

        except Exception as e:
            print(f"Observation validation failed: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """获取环境统计信息"""
        return {
            'total_observations': self.stats['total_observations'],
            'episodes_loaded': self.stats['episodes_loaded'],
            'observation_types': dict(self.stats['observation_types']),
            'dataset_size': len(self.dataset) if self.dataset else 0,
            'current_episode': self.current_episode,
            'current_step': self.current_step,
            'history_length': len(self.observation_history)
        }

    def reset(self) -> None:
        """重置环境状态"""
        self.current_episode = 0
        self.current_step = 0
        self.observation_history.clear()

        # 重置统计（保留数据集相关统计）
        episodes_loaded = self.stats['episodes_loaded']
        self.stats = {
            'total_observations': 0,
            'episodes_loaded': episodes_loaded,
            'observation_types': defaultdict(int)
        }