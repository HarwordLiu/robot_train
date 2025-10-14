#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理时Action后处理

在部署阶段对模型输出进行后处理，提升执行精度：
1. 平滑滤波：减少抖动
2. 精细操作增益调整：放大精细动作的幅度
3. 工作空间限制：防止越界
4. 速度限制：保证安全
"""

import numpy as np
import torch
from typing import Optional, List
from collections import deque


class ActionSmoother:
    """
    动作平滑滤波器

    使用指数移动平均(EMA)平滑action，减少高频抖动
    """

    def __init__(
        self,
        action_dim: int,
        alpha: float = 0.3,
        history_size: int = 5
    ):
        """
        Args:
            action_dim: 动作维度
            alpha: EMA平滑系数（0-1），越小越平滑，但响应越慢
            history_size: 历史缓冲大小
        """
        self.action_dim = action_dim
        self.alpha = alpha
        self.history = deque(maxlen=history_size)
        self.smoothed_action = None

    def reset(self):
        """重置滤波器"""
        self.history.clear()
        self.smoothed_action = None

    def smooth(self, action: np.ndarray) -> np.ndarray:
        """
        平滑action

        Args:
            action: [action_dim]

        Returns:
            smoothed_action: [action_dim]
        """
        if self.smoothed_action is None:
            # 第一帧，直接使用
            self.smoothed_action = action.copy()
        else:
            # EMA平滑
            self.smoothed_action = (
                self.alpha * action +
                (1 - self.alpha) * self.smoothed_action
            )

        self.history.append(action.copy())
        return self.smoothed_action.copy()


class FineTuningGainAdjuster:
    """
    精细操作增益调整器

    在精细操作阶段（放置），适当放大action幅度，提升精度
    """

    def __init__(
        self,
        fine_motion_threshold: float = 0.05,
        fine_motion_gain: float = 1.5,
        gripper_index: int = 14
    ):
        """
        Args:
            fine_motion_threshold: 精细操作判断阈值（rad）
            fine_motion_gain: 精细操作增益（>1放大，<1缩小）
            gripper_index: gripper状态的索引
        """
        self.fine_motion_threshold = fine_motion_threshold
        self.fine_motion_gain = fine_motion_gain
        self.gripper_index = gripper_index

    def adjust(
        self,
        action: np.ndarray,
        current_state: np.ndarray
    ) -> np.ndarray:
        """
        根据当前状态调整action

        Args:
            action: [action_dim]
            current_state: [state_dim] 当前机器人状态

        Returns:
            adjusted_action: [action_dim]
        """
        # 检测是否是精细操作阶段
        gripper_closed = current_state[self.gripper_index] > 0.5

        # 计算action幅度（不包括gripper）
        position_action = action[:self.gripper_index]
        action_magnitude = np.linalg.norm(position_action)

        # 精细操作判断：gripper关闭 + 小幅度动作
        is_fine_motion = gripper_closed and action_magnitude < self.fine_motion_threshold

        if is_fine_motion:
            # 放大精细操作的幅度
            adjusted_action = action.copy()
            adjusted_action[:self.gripper_index] *= self.fine_motion_gain
            return adjusted_action
        else:
            return action


class WorkspaceLimiter:
    """
    工作空间限制器

    防止关节角度超出物理限位
    """

    def __init__(
        self,
        joint_limits: Optional[List[tuple]] = None,
        default_limit: float = np.pi
    ):
        """
        Args:
            joint_limits: 每个关节的限位 [(min, max), ...]
            default_limit: 默认限位（如果未指定）
        """
        if joint_limits is None:
            # Kuavo默认16个关节，限位±π
            joint_limits = [(-default_limit, default_limit)] * 16

        self.joint_limits = np.array(joint_limits)

    def limit(
        self,
        action: np.ndarray,
        current_state: np.ndarray
    ) -> np.ndarray:
        """
        限制action，防止超出工作空间

        Args:
            action: [action_dim] 增量动作
            current_state: [state_dim] 当前关节角度

        Returns:
            limited_action: [action_dim]
        """
        # 计算执行action后的新状态
        next_state = current_state + action

        # 限制在工作空间内
        next_state = np.clip(
            next_state,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1]
        )

        # 计算修正后的action
        limited_action = next_state - current_state

        return limited_action


class VelocityLimiter:
    """
    速度限制器

    限制关节速度，保证安全
    """

    def __init__(
        self,
        max_velocity: float = 0.2,  # rad/s
        control_frequency: float = 10.0  # Hz
    ):
        """
        Args:
            max_velocity: 最大关节速度（rad/s）
            control_frequency: 控制频率（Hz）
        """
        self.max_velocity = max_velocity
        self.max_step = max_velocity / control_frequency

    def limit(self, action: np.ndarray) -> np.ndarray:
        """
        限制action速度

        Args:
            action: [action_dim] 增量动作

        Returns:
            limited_action: [action_dim]
        """
        # 计算action幅度
        action_magnitude = np.linalg.norm(action)

        if action_magnitude > self.max_step:
            # 缩放到最大允许步长
            limited_action = action * (self.max_step / action_magnitude)
        else:
            limited_action = action

        return limited_action


class ActionPostProcessor:
    """
    完整的Action后处理流程

    集成所有后处理模块，按顺序应用
    """

    def __init__(
        self,
        action_dim: int = 16,
        enable_smoothing: bool = True,
        enable_fine_gain: bool = True,
        enable_workspace_limit: bool = True,
        enable_velocity_limit: bool = True,
        smooth_alpha: float = 0.3,
        fine_motion_gain: float = 1.5,
        max_velocity: float = 0.2,
        control_frequency: float = 10.0
    ):
        """
        Args:
            action_dim: 动作维度
            enable_smoothing: 启用平滑
            enable_fine_gain: 启用精细操作增益
            enable_workspace_limit: 启用工作空间限制
            enable_velocity_limit: 启用速度限制
            smooth_alpha: 平滑系数
            fine_motion_gain: 精细操作增益
            max_velocity: 最大速度
            control_frequency: 控制频率
        """
        self.action_dim = action_dim
        self.enable_smoothing = enable_smoothing
        self.enable_fine_gain = enable_fine_gain
        self.enable_workspace_limit = enable_workspace_limit
        self.enable_velocity_limit = enable_velocity_limit

        # 初始化各模块
        if enable_smoothing:
            self.smoother = ActionSmoother(action_dim, alpha=smooth_alpha)

        if enable_fine_gain:
            self.gain_adjuster = FineTuningGainAdjuster(
                fine_motion_gain=fine_motion_gain)

        if enable_workspace_limit:
            self.workspace_limiter = WorkspaceLimiter()

        if enable_velocity_limit:
            self.velocity_limiter = VelocityLimiter(
                max_velocity=max_velocity,
                control_frequency=control_frequency
            )

    def reset(self):
        """重置所有状态"""
        if self.enable_smoothing:
            self.smoother.reset()

    def process(
        self,
        raw_action: np.ndarray,
        current_state: np.ndarray
    ) -> np.ndarray:
        """
        完整后处理流程

        处理顺序：
        1. 精细操作增益调整（放大精细动作）
        2. 平滑滤波（减少抖动）
        3. 工作空间限制（防止越界）
        4. 速度限制（保证安全）

        Args:
            raw_action: [action_dim] 模型原始输出
            current_state: [state_dim] 当前机器人状态

        Returns:
            processed_action: [action_dim] 处理后的action
        """
        action = raw_action.copy()

        # 1. 精细操作增益调整
        if self.enable_fine_gain:
            action = self.gain_adjuster.adjust(action, current_state)

        # 2. 平滑滤波
        if self.enable_smoothing:
            action = self.smoother.smooth(action)

        # 3. 工作空间限制
        if self.enable_workspace_limit:
            action = self.workspace_limiter.limit(action, current_state)

        # 4. 速度限制
        if self.enable_velocity_limit:
            action = self.velocity_limiter.limit(action)

        return action


# ==================== 使用示例 ====================

def example_usage():
    """演示如何使用ActionPostProcessor"""

    # 创建后处理器
    postprocessor = ActionPostProcessor(
        action_dim=16,
        enable_smoothing=True,
        enable_fine_gain=True,
        enable_workspace_limit=True,
        enable_velocity_limit=True,
        smooth_alpha=0.3,
        fine_motion_gain=1.5,  # 精细操作放大1.5倍
        max_velocity=0.2,
        control_frequency=10.0
    )

    # 模拟推理循环
    print("🔧 Action Post-Processing Demo")
    print("="*60)

    for step in range(10):
        # 模拟模型输出（随机action）
        raw_action = np.random.randn(16) * 0.1

        # 模拟当前状态
        current_state = np.random.randn(16) * 0.5

        # 后处理
        processed_action = postprocessor.process(raw_action, current_state)

        print(f"\nStep {step+1}:")
        print(f"  Raw action magnitude: {np.linalg.norm(raw_action[:14]):.4f}")
        print(f"  Processed action magnitude: {np.linalg.norm(processed_action[:14]):.4f}")

    print("\n" + "="*60)


if __name__ == "__main__":
    example_usage()
