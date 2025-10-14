#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmolVLA数据增强策略

针对Kuavo机器人任务的数据增强：
1. 视觉增强：颜色抖动、随机裁剪、模糊（已有，保持）
2. 状态/动作增强：工作空间扰动、精细操作噪声增强
"""

import torch
import numpy as np
from typing import Dict, Optional


class StateActionAugmentation:
    """
    状态和动作的数据增强

    目标：提升模型对边界位置和精细操作的泛化能力
    """

    def __init__(
        self,
        boundary_augment_prob: float = 0.3,
        boundary_noise_std: float = 0.05,
        fine_motion_augment_prob: float = 0.5,
        fine_motion_noise_std: float = 0.01,
        arm_indices: Dict[str, list] = None,
    ):
        """
        Args:
            boundary_augment_prob: 边界增强概率
            boundary_noise_std: 边界噪声标准差（弧度）
            fine_motion_augment_prob: 精细操作增强概率
            fine_motion_noise_std: 精细操作噪声标准差
            arm_indices: {'left': [0,1,2,...], 'right': [8,9,10,...]}
        """
        self.boundary_augment_prob = boundary_augment_prob
        self.boundary_noise_std = boundary_noise_std
        self.fine_motion_augment_prob = fine_motion_augment_prob
        self.fine_motion_noise_std = fine_motion_noise_std

        # 默认Kuavo双臂配置
        if arm_indices is None:
            arm_indices = {
                'left': list(range(0, 8)),
                'right': list(range(8, 16))
            }
        self.arm_indices = arm_indices

    def augment_boundary_state(
        self,
        state: torch.Tensor,
        arm: str = 'left'
    ) -> torch.Tensor:
        """
        增强边界位置的状态数据

        通过添加噪声模拟边界位置的变化，提升泛化能力

        Args:
            state: [batch, state_dim] 或 [state_dim]
            arm: 'left' 或 'right'

        Returns:
            增强后的state
        """
        if torch.rand(1).item() > self.boundary_augment_prob:
            return state

        augmented_state = state.clone()
        arm_idx = self.arm_indices[arm]

        # 为左臂添加边界噪声（模拟物体在传送带边缘）
        if len(state.shape) == 1:
            noise = torch.randn(len(arm_idx)) * self.boundary_noise_std
            augmented_state[arm_idx] = augmented_state[arm_idx] + noise
        else:
            noise = torch.randn(state.shape[0], len(arm_idx)) * self.boundary_noise_std
            augmented_state[:, arm_idx] = augmented_state[:, arm_idx] + noise

        # 限制在合理范围内（假设关节限位±π）
        augmented_state = torch.clamp(augmented_state, -np.pi, np.pi)

        return augmented_state

    def augment_fine_motion_action(
        self,
        action: torch.Tensor,
        is_fine_motion: bool = None
    ) -> torch.Tensor:
        """
        增强精细操作阶段的动作数据

        在放置阶段添加小噪声，迫使模型学习更鲁棒的精细控制

        Args:
            action: [batch, chunk_size, action_dim] 或 [chunk_size, action_dim]
            is_fine_motion: 是否是精细操作阶段（如果为None，自动判断）

        Returns:
            增强后的action
        """
        if torch.rand(1).item() > self.fine_motion_augment_prob:
            return action

        # 自动判断是否是精细操作（通过action幅度）
        if is_fine_motion is None:
            action_magnitude = torch.norm(action, dim=-1).mean()
            is_fine_motion = action_magnitude < 0.1  # 小于0.1rad认为是精细操作

        if not is_fine_motion:
            return action

        # 添加小噪声
        noise = torch.randn_like(action) * self.fine_motion_noise_std
        augmented_action = action + noise

        return augmented_action

    def __call__(
        self,
        batch: Dict[str, torch.Tensor],
        augment_state: bool = True,
        augment_action: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        对整个batch进行增强

        Args:
            batch: 包含'observation.state'和'action'的字典
            augment_state: 是否增强状态
            augment_action: 是否增强动作

        Returns:
            增强后的batch
        """
        augmented_batch = batch.copy()

        # 增强状态（针对左臂边界问题）
        if augment_state and 'observation.state' in batch:
            augmented_batch['observation.state'] = self.augment_boundary_state(
                batch['observation.state'],
                arm='left'
            )

        # 增强动作（针对放置精度问题）
        if augment_action and 'action' in batch:
            augmented_batch['action'] = self.augment_fine_motion_action(
                batch['action']
            )

        return augmented_batch


class SmolVLAAugmentationWrapper:
    """
    SmolVLA完整增强流程

    将视觉增强（已有）和状态/动作增强结合
    """

    def __init__(
        self,
        enable_state_action_aug: bool = True,
        boundary_augment_prob: float = 0.3,
        fine_motion_augment_prob: float = 0.5,
    ):
        self.enable_state_action_aug = enable_state_action_aug

        if enable_state_action_aug:
            self.state_action_aug = StateActionAugmentation(
                boundary_augment_prob=boundary_augment_prob,
                fine_motion_augment_prob=fine_motion_augment_prob
            )

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        应用所有增强

        注意：视觉增强已经在LeRobot的dataset中处理，这里只处理状态/动作
        """
        if self.enable_state_action_aug:
            batch = self.state_action_aug(batch)

        return batch
