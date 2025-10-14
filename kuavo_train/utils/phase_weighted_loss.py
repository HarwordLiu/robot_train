#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务阶段加权Loss

核心思路：放置阶段的loss权重更高，迫使模型更关注精细操作
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class PhaseWeightedLoss(nn.Module):
    """
    根据任务阶段动态调整loss权重

    任务阶段识别：
    - 抓取前（靠近）：gripper打开，物体未接触
    - 抓取中：gripper关闭
    - 移动中：gripper关闭，末端运动
    - 放置中：gripper从关闭到打开，动作幅度小（精细操作）
    """

    def __init__(
        self,
        approach_weight: float = 0.8,
        grasp_weight: float = 1.0,
        transport_weight: float = 1.0,
        placement_weight: float = 2.5,  # 放置阶段权重最高
        gripper_indices: list = None,
        fine_motion_threshold: float = 0.05,  # 精细操作阈值
    ):
        super().__init__()
        self.approach_weight = approach_weight
        self.grasp_weight = grasp_weight
        self.transport_weight = transport_weight
        self.placement_weight = placement_weight
        self.fine_motion_threshold = fine_motion_threshold

        # Kuavo夹爪索引（默认最后两维）
        if gripper_indices is None:
            gripper_indices = [14, 15]
        self.gripper_indices = gripper_indices

    def detect_phase(
        self,
        action: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        检测当前是什么阶段

        Args:
            action: [batch, chunk_size, action_dim]
            state: [batch, state_dim]

        Returns:
            phase_weights: [batch, chunk_size]
        """
        batch_size, chunk_size, action_dim = action.shape

        # 1. 计算动作幅度（判断是否是精细操作）
        # 只看前端位置维度（不包括gripper）
        position_action = action[:, :, :self.gripper_indices[0]]
        action_magnitude = torch.norm(position_action, dim=-1)  # [batch, chunk_size]

        # 2. 检测gripper状态（开/关）
        # 从state中提取gripper状态
        gripper_state = state[:, self.gripper_indices[0]]  # [batch]

        # 3. 判断阶段
        is_fine_motion = action_magnitude < self.fine_motion_threshold  # [batch, chunk_size]
        is_gripper_closed = gripper_state > 0.5  # [batch]

        # 展开gripper状态到chunk维度
        is_gripper_closed = is_gripper_closed.unsqueeze(1).expand(-1, chunk_size)

        # 阶段判断逻辑
        phase_weights = torch.ones_like(action_magnitude)

        # 放置阶段：精细操作 + gripper关闭 → 最高权重
        is_placement = is_fine_motion & is_gripper_closed
        phase_weights = torch.where(is_placement,
                                     torch.tensor(self.placement_weight, device=action.device),
                                     phase_weights)

        # 靠近阶段：gripper打开 + 不是精细操作 → 较低权重
        is_approach = (~is_gripper_closed) & (~is_fine_motion)
        phase_weights = torch.where(is_approach,
                                     torch.tensor(self.approach_weight, device=action.device),
                                     phase_weights)

        # 抓取/移动阶段：gripper关闭 + 不是精细操作 → 标准权重
        is_transport = is_gripper_closed & (~is_fine_motion)
        phase_weights = torch.where(is_transport,
                                     torch.tensor(self.transport_weight, device=action.device),
                                     phase_weights)

        return phase_weights

    def forward(
        self,
        predicted_action: torch.Tensor,
        target_action: torch.Tensor,
        state: torch.Tensor,
        base_loss: torch.Tensor
    ) -> torch.Tensor:
        """
        计算加权loss

        Args:
            predicted_action: [batch, chunk_size, action_dim]
            target_action: [batch, chunk_size, action_dim]
            state: [batch, state_dim]
            base_loss: [batch, chunk_size] 基础loss（如MSE）

        Returns:
            weighted_loss: 标量
        """
        # 检测阶段并获取权重
        phase_weights = self.detect_phase(target_action, state)  # [batch, chunk_size]

        # 应用权重
        weighted_loss = base_loss * phase_weights

        # 返回平均loss
        return weighted_loss.mean()


def compute_base_mse_loss(
    predicted_action: torch.Tensor,
    target_action: torch.Tensor,
    reduction: str = 'none'
) -> torch.Tensor:
    """
    计算基础MSE loss（不做reduction）

    Args:
        predicted_action: [batch, chunk_size, action_dim]
        target_action: [batch, chunk_size, action_dim]
        reduction: 'none', 'mean', 'sum'

    Returns:
        loss: [batch, chunk_size] if reduction='none'
    """
    # 计算每个样本、每个时间步的MSE
    mse = torch.nn.functional.mse_loss(
        predicted_action,
        target_action,
        reduction='none'
    )  # [batch, chunk_size, action_dim]

    # 沿action_dim求平均
    mse = mse.mean(dim=-1)  # [batch, chunk_size]

    if reduction == 'mean':
        return mse.mean()
    elif reduction == 'sum':
        return mse.sum()
    else:
        return mse


# ==================== 使用示例 ====================

def example_usage():
    """演示如何使用PhaseWeightedLoss"""

    # 创建加权loss模块
    phase_loss = PhaseWeightedLoss(
        approach_weight=0.8,
        grasp_weight=1.0,
        transport_weight=1.0,
        placement_weight=2.5,  # 放置阶段权重2.5倍
    )

    # 模拟数据
    batch_size = 32
    chunk_size = 50
    action_dim = 16
    state_dim = 16

    predicted = torch.randn(batch_size, chunk_size, action_dim)
    target = torch.randn(batch_size, chunk_size, action_dim)
    state = torch.randn(batch_size, state_dim)

    # 1. 计算基础loss
    base_loss = compute_base_mse_loss(predicted, target, reduction='none')

    # 2. 应用阶段加权
    weighted_loss = phase_loss(predicted, target, state, base_loss)

    print(f"Base loss shape: {base_loss.shape}")
    print(f"Weighted loss (scalar): {weighted_loss.item():.4f}")

    return weighted_loss


if __name__ == "__main__":
    example_usage()
