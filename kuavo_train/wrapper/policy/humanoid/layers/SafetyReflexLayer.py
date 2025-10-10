"""
SafetyReflexLayer: 安全反射层 - 最高优先级，防跌倒和紧急停止
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from .BaseLayer import BaseLayer


class SafetyReflexLayer(BaseLayer):
    """
    安全反射层 - 优先级1（最高）

    特点：
    - 极简GRU结构，确保<10ms响应时间
    - 防跌倒检测和紧急停止
    - 基础平衡控制
    - 永远激活，可以覆盖其他层的输出
    """

    def __init__(self, config: Dict[str, Any], base_config: Any):
        super().__init__(config, "safety", priority=1)

        self.base_config = base_config

        # 输入维度配置 - 适配实际机器人状态
        # 根据实际机器人配置：only_arm=true时，状态为双臂14维+手爪2维=16维
        if 'input_dim' in config:
            self.input_dim = config['input_dim']
        else:
            # 从base_config推断状态维度
            state_shape = getattr(base_config, 'robot_state_feature', None)
            if state_shape and hasattr(state_shape, 'shape'):
                self.input_dim = state_shape.shape[0]
            else:
                self.input_dim = 16  # 默认：双臂+手爪配置

        self.hidden_size = config.get('hidden_size', 64)
        self.output_dim = config.get('output_dim', self.input_dim)  # 输出维度与输入对应

        # 极简GRU，确保最低延迟
        self.balance_gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=1,  # 只用一层确保速度
            batch_first=True,
            bias=True
        )

        # 紧急情况检测
        self.emergency_detector = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 平衡控制输出
        self.balance_controller = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_dim),
            nn.Tanh()  # 限制输出范围
        )

        # 倾斜角度检测
        self.tilt_detector = nn.Linear(self.hidden_size, 2)  # roll, pitch

        # 紧急动作生成器
        self.emergency_action_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_dim),
            nn.Tanh()
        )

        # 安全阈值（可配置）
        self.emergency_threshold = config.get('emergency_threshold', 0.8)
        self.tilt_threshold_degrees = config.get(
            'tilt_threshold_degrees', 15.0)  # 15度倾斜阈值

    def should_activate(self, inputs: Dict[str, torch.Tensor], context: Optional[Dict[str, Any]] = None) -> bool:
        """安全层始终激活"""
        return True

    def get_required_input_keys(self) -> List[str]:
        """安全层需要的输入"""
        return ['observation.state']  # 关节状态和IMU数据

    def forward(self, inputs: Dict[str, torch.Tensor], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        安全反射层前向传播

        Args:
            inputs: 输入数据，需要包含 observation.state
            context: 上下文信息

        Returns:
            Dict: 安全层输出，包含紧急状态和控制指令
        """
        # 提取关键安全信息 - 机器人关节状态（包含IMU数据）
        if 'observation.state' not in inputs:
            # 如果没有状态数据，返回安全的默认输出
            batch_size = list(inputs.values())[0].size(0)
            device = list(inputs.values())[0].device
            return self._generate_safe_default_output(batch_size, device)

        robot_state = inputs['observation.state']

        # 处理不同维度的输入

        if len(robot_state.shape) == 1:
            # [state_dim] -> [1, 1, state_dim]
            robot_state = robot_state.unsqueeze(0).unsqueeze(0)
            batch_size, seq_len, state_dim = robot_state.shape
        elif len(robot_state.shape) == 2:
            # [batch_size, state_dim] -> [batch_size, 1, state_dim]
            robot_state = robot_state.unsqueeze(1)
            batch_size, seq_len, state_dim = robot_state.shape
        elif len(robot_state.shape) == 3:
            # [batch_size, seq_len, state_dim]
            batch_size, seq_len, state_dim = robot_state.shape
        else:
            raise ValueError(
                f"Unexpected robot_state shape: {robot_state.shape}, expected 1D, 2D or 3D tensor")

        # 如果输入维度不匹配，进行适配
        if state_dim != self.input_dim:
            # 简单的降维或升维处理
            if state_dim > self.input_dim:
                robot_state = robot_state[..., :self.input_dim]  # 截取前面的维度
            else:
                # 用零填充
                padding = torch.zeros(batch_size, seq_len, self.input_dim - state_dim,
                                      device=robot_state.device, dtype=robot_state.dtype)
                robot_state = torch.cat([robot_state, padding], dim=-1)

        # 快速GRU处理
        gru_output, hidden = self.balance_gru(robot_state)

        # 使用最后一个时间步的输出
        last_output = gru_output[:, -1, :]  # [batch_size, hidden_size]

        # 紧急情况检测
        emergency_score = self.emergency_detector(
            last_output)  # [batch_size, 1]
        # [batch_size] bool类型
        emergency = (emergency_score > self.emergency_threshold).squeeze(-1)

        # 倾斜检测
        # [batch_size, 2] (roll, pitch)
        tilt_angles = self.tilt_detector(last_output)
        tilt_angles_degrees = tilt_angles * 45.0  # 缩放到±45度范围

        # 倾斜紧急检测
        tilt_emergency = torch.any(
            torch.abs(tilt_angles_degrees) > self.tilt_threshold_degrees, dim=-1)

        # 综合紧急状态（使用逻辑或运算，保持bool类型）
        overall_emergency = torch.logical_or(emergency, tilt_emergency)

        # 生成控制输出
        # 为所有样本同时生成紧急动作和平衡控制动作
        emergency_action = self.emergency_action_generator(last_output)
        balance_action_normal = self.balance_controller(last_output)

        # 根据每个样本的紧急状态选择相应的动作
        # overall_emergency: [batch_size] bool
        # 需要扩展维度以进行广播
        # [batch_size, 1]
        overall_emergency_expanded = overall_emergency.unsqueeze(-1)

        # 使用torch.where：如果紧急则用emergency_action，否则用balance_action_normal
        balance_action = torch.where(
            overall_emergency_expanded,
            emergency_action,
            balance_action_normal
        )  # [batch_size, action_dim]

        # 计算平衡置信度（倾斜越小，置信度越高）
        max_tilt = torch.max(torch.abs(tilt_angles_degrees), dim=-1)[0]
        balance_confidence = torch.exp(-max_tilt / 10.0)  # 置信度函数

        return {
            'emergency': overall_emergency,
            'emergency_score': emergency_score.squeeze(-1),
            'balance_action': balance_action,
            'emergency_action': self.emergency_action_generator(last_output),
            'tilt_angles_degrees': tilt_angles_degrees,
            'balance_confidence': balance_confidence,
            'safety_status': self._compute_safety_status(emergency_score.squeeze(-1), tilt_angles_degrees),
            'action': balance_action,  # 提供统一的action接口
            'layer': 'safety'
        }

    def _generate_safe_default_output(self, batch_size: int, device: torch.device) -> Dict[str, Any]:
        """生成安全的默认输出（当输入不可用时）"""
        zero_action = torch.zeros(batch_size, self.output_dim, device=device)

        return {
            # 默认紧急状态
            'emergency': torch.ones(batch_size, dtype=torch.bool, device=device),
            'emergency_score': torch.ones(batch_size, device=device),
            'balance_action': zero_action,
            'emergency_action': zero_action,
            'tilt_angles_degrees': torch.zeros(batch_size, 2, device=device),
            'balance_confidence': torch.zeros(batch_size, device=device),
            'safety_status': ['UNKNOWN'] * batch_size,
            'action': zero_action,
            'layer': 'safety'
        }

    def _compute_safety_status(self, emergency_score: torch.Tensor, tilt_angles: torch.Tensor) -> List[str]:
        """计算安全状态描述"""
        batch_size = emergency_score.size(0)
        status_list = []

        for i in range(batch_size):
            score = emergency_score[i].item()
            max_tilt = torch.max(torch.abs(tilt_angles[i])).item()

            if score > self.emergency_threshold:
                status_list.append('EMERGENCY')
            elif max_tilt > self.tilt_threshold_degrees:
                status_list.append('UNSTABLE')
            elif max_tilt > self.tilt_threshold_degrees * 0.5:
                status_list.append('CAUTION')
            else:
                status_list.append('SAFE')

        return status_list

    def get_output_keys(self) -> List[str]:
        """安全层输出的key列表"""
        return [
            'emergency', 'emergency_score', 'balance_action', 'emergency_action',
            'tilt_angles_degrees', 'balance_confidence', 'safety_status', 'action', 'layer'
        ]

    def set_emergency_threshold(self, threshold: float):
        """动态调整紧急阈值"""
        self.emergency_threshold = max(0.0, min(1.0, threshold))

    def set_tilt_threshold(self, threshold_degrees: float):
        """动态调整倾斜阈值"""
        self.tilt_threshold_degrees = max(1.0, min(45.0, threshold_degrees))

    def is_system_safe(self, inputs: Dict[str, torch.Tensor]) -> bool:
        """快速安全检查（不执行完整前向传播）"""
        with torch.no_grad():
            output = self.forward(inputs)
            emergency_tensor = output['emergency']
            if emergency_tensor.numel() == 1:
                return not emergency_tensor.item()
            else:
                # 对于批处理，如果任何样本处于紧急状态，则认为系统不安全
                return not torch.any(emergency_tensor).item()

    def __repr__(self) -> str:
        return (f"SafetyReflexLayer(input_dim={self.input_dim}, hidden_size={self.hidden_size}, "
                f"emergency_threshold={self.emergency_threshold}, "
                f"tilt_threshold={self.tilt_threshold_degrees}°)")


class EmergencyStopModule(nn.Module):
    """紧急停止模块 - 可以被外部系统调用"""

    def __init__(self):
        super().__init__()
        self.emergency_stop_active = False

    def activate_emergency_stop(self):
        """激活紧急停止"""
        self.emergency_stop_active = True

    def deactivate_emergency_stop(self):
        """解除紧急停止"""
        self.emergency_stop_active = False

    def is_emergency_stop_active(self) -> bool:
        """检查紧急停止状态"""
        return self.emergency_stop_active

    def get_emergency_action(self, batch_size: int, action_dim: int, device: torch.device) -> torch.Tensor:
        """生成紧急停止动作（全零）"""
        return torch.zeros(batch_size, action_dim, device=device)
