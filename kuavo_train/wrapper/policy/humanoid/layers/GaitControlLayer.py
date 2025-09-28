"""
GaitControlLayer: 步态控制层 - 混合GRU+Transformer架构
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from .BaseLayer import BaseLayer


class GaitControlLayer(BaseLayer):
    """
    步态控制层 - 优先级2

    特点：
    - 混合GRU + 轻量Transformer架构
    - 负载适应和地形适应
    - 双足步态规划和控制
    - 约20ms响应时间
    """

    def __init__(self, config: Dict[str, Any], base_config: Any):
        super().__init__(config, "gait", priority=2)

        self.base_config = base_config

        # 配置参数
        self.input_dim = getattr(base_config, 'robot_state_feature', type('obj', (object,), {'shape': [64]})).shape[0]
        self.gru_hidden = config.get('gru_hidden', 128)
        self.gru_layers = config.get('gru_layers', 2)
        self.tf_layers = config.get('tf_layers', 2)
        self.tf_heads = config.get('tf_heads', 4)
        self.tf_dim = config.get('tf_dim', 128)

        # GRU用于步态状态跟踪
        self.gait_state_gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            batch_first=True,
            dropout=0.1 if self.gru_layers > 1 else 0
        )

        # 轻量Transformer用于步态规划
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.gru_hidden,
            nhead=self.tf_heads,
            dim_feedforward=self.gru_hidden * 2,
            dropout=0.1,
            batch_first=True
        )
        self.gait_planner = nn.TransformerEncoder(encoder_layer, num_layers=self.tf_layers)

        # 负载适应模块
        self.load_adapter = LoadAdaptationModule(self.gru_hidden)

        # 输出投影
        self.output_projection = nn.Linear(self.gru_hidden, 32)  # 假设32维关节输出

    def should_activate(self, inputs: Dict[str, torch.Tensor], context: Optional[Dict[str, Any]] = None) -> bool:
        """当机器人需要移动时激活"""
        if context is None:
            return True
        return context.get('requires_locomotion', True)

    def get_required_input_keys(self) -> List[str]:
        return ['observation.state']

    def forward(self, inputs: Dict[str, torch.Tensor], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """步态控制前向传播"""
        # 获取关节状态
        robot_state = inputs.get('observation.state')
        if robot_state is None:
            batch_size = list(inputs.values())[0].size(0)
            device = list(inputs.values())[0].device
            return self._generate_default_output(batch_size, device)

        batch_size, seq_len, state_dim = robot_state.shape

        # 维度适配
        if state_dim != self.input_dim:
            if state_dim > self.input_dim:
                robot_state = robot_state[..., :self.input_dim]
            else:
                padding = torch.zeros(batch_size, seq_len, self.input_dim - state_dim,
                                    device=robot_state.device, dtype=robot_state.dtype)
                robot_state = torch.cat([robot_state, padding], dim=-1)

        # GRU处理步态状态
        gru_output, gru_hidden = self.gait_state_gru(robot_state)

        # Transformer步态规划（如果序列足够长）
        if seq_len >= 10:  # 至少200ms历史
            planned_gait = self.gait_planner(gru_output)
        else:
            planned_gait = gru_output

        # 负载适应
        adapted_gait = self.load_adapter(planned_gait, context)

        # 最终输出
        final_output = self.output_projection(adapted_gait[:, -1, :])  # 使用最后时间步

        return {
            'gait_features': gru_output,
            'planned_gait': planned_gait,
            'adapted_gait': adapted_gait,
            'action': final_output,
            'layer': 'gait'
        }

    def _generate_default_output(self, batch_size: int, device: torch.device) -> Dict[str, Any]:
        """生成默认输出"""
        zero_features = torch.zeros(batch_size, 10, self.gru_hidden, device=device)
        zero_action = torch.zeros(batch_size, 32, device=device)

        return {
            'gait_features': zero_features,
            'planned_gait': zero_features,
            'adapted_gait': zero_features,
            'action': zero_action,
            'layer': 'gait'
        }


class LoadAdaptationModule(nn.Module):
    """负载适应模块"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.adaptation_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )

    def forward(self, gait_features: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """负载适应处理"""
        # 简化实现：直接返回原特征加上适应调整
        adaptation = self.adaptation_net(gait_features)
        return gait_features + 0.1 * adaptation  # 小幅调整