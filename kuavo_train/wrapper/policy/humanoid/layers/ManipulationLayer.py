"""
ManipulationLayer: 操作控制层 - Transformer主导
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from .BaseLayer import BaseLayer


class ManipulationLayer(BaseLayer):
    """
    操作控制层 - 优先级3

    特点：
    - Transformer主导架构
    - 处理抓取、摆放等精细操作
    - 约束满足和双臂协调
    - 约100ms响应时间
    """

    def __init__(self, config: Dict[str, Any], base_config: Any):
        super().__init__(config, "manipulation", priority=3)

        self.base_config = base_config

        # 配置参数
        self.hidden_size = config.get('hidden_size', 512)
        self.num_layers = config.get('layers', 3)
        self.num_heads = config.get('heads', 8)
        self.dim_feedforward = config.get('dim_feedforward', 2048)

        # 特征维度计算（视觉+状态）- 适配实际机器人配置
        self.visual_dim = 1280  # EfficientNet-B0输出
        state_shape = getattr(base_config, 'robot_state_feature', None)
        if state_shape and hasattr(state_shape, 'shape'):
            self.state_dim = state_shape.shape[0]
        else:
            # 默认配置：only_arm=true时的双臂+手爪配置
            self.state_dim = 16

        self.input_projection = nn.Linear(self.visual_dim + self.state_dim, self.hidden_size)

        # 主要的Transformer网络
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.manipulation_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # 约束满足模块
        self.constraint_solver = ConstraintSatisfactionModule(self.hidden_size)

        # 双臂协调模块
        self.bimanual_coordinator = BimanualCoordinationModule(self.hidden_size)

        # 输出投影
        self.action_head = nn.Linear(self.hidden_size, 32)  # 动作输出

    def should_activate(self, inputs: Dict[str, torch.Tensor], context: Optional[Dict[str, Any]] = None) -> bool:
        """当需要精细操作时激活"""
        if context is None:
            return True
        return context.get('requires_manipulation', True)

    def get_required_input_keys(self) -> List[str]:
        return ['observation.state']  # 至少需要状态，视觉特征可选

    def forward(self, inputs: Dict[str, torch.Tensor], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """操作控制前向传播"""
        # 提取特征
        features = self._extract_features(inputs)
        if features is None:
            return self._generate_default_output(inputs)

        batch_size, seq_len, _ = features.shape

        # Transformer处理
        manipulation_features = self.manipulation_transformer(features)

        # 约束满足
        constraint_solution = self.constraint_solver(manipulation_features, context)

        # 双臂协调
        coordinated_actions = self.bimanual_coordinator(manipulation_features, context)

        # 最终动作
        final_action = self.action_head(manipulation_features[:, -1, :])

        return {
            'manipulation_features': manipulation_features,
            'constraint_solution': constraint_solution,
            'coordinated_actions': coordinated_actions,
            'action': final_action,
            'layer': 'manipulation'
        }

    def _extract_features(self, inputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """提取并融合多模态特征"""
        features_list = []

        print(f"🔍 ManipulationLayer: Available input keys: {list(inputs.keys())}")

        # 状态特征
        if 'observation.state' in inputs:
            state_features = inputs['observation.state']
            print(f"🔍 ManipulationLayer: state_features.shape = {state_features.shape}")
            # 处理维度：确保是3D tensor [batch_size, seq_len, state_dim]
            if len(state_features.shape) == 1:
                state_features = state_features.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
            elif len(state_features.shape) == 2:
                state_features = state_features.unsqueeze(1)  # [batch_size, 1, state_dim]
            features_list.append(state_features)
            print(f"🔍 ManipulationLayer: Processed state_features.shape = {state_features.shape}")

        # 视觉特征（如果可用）
        if 'observation.images' in inputs:
            # 简化：直接使用均值池化
            visual_features = inputs['observation.images']
            if len(visual_features.shape) > 3:
                visual_features = visual_features.mean(dim=(-2, -1))  # 全局平均池化
            # 确保是3D tensor
            if len(visual_features.shape) == 2:
                visual_features = visual_features.unsqueeze(1)
            features_list.append(visual_features)
        else:
            # 如果没有视觉特征，需要用零填充以匹配预训练模型的输入维度
            if features_list:
                batch_size, seq_len = features_list[0].shape[:2]
                device = features_list[0].device
                # 创建1280维的零视觉特征
                zero_visual = torch.zeros(batch_size, seq_len, self.visual_dim, device=device)
                features_list.append(zero_visual)

        if not features_list:
            return None

        # 特征拼接和投影
        combined_features = torch.cat(features_list, dim=-1)
        print(f"🔍 ManipulationLayer: combined_features.shape = {combined_features.shape}")
        print(f"🔍 ManipulationLayer: input_projection expects: {self.input_projection.in_features}")

        projected_features = self.input_projection(combined_features)
        print(f"🔍 ManipulationLayer: projected_features.shape = {projected_features.shape}")

        return projected_features

    def _generate_default_output(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """生成默认输出"""
        batch_size = list(inputs.values())[0].size(0)
        device = list(inputs.values())[0].device

        zero_features = torch.zeros(batch_size, 10, self.hidden_size, device=device)
        zero_action = torch.zeros(batch_size, 32, device=device)

        return {
            'manipulation_features': zero_features,
            'constraint_solution': {},
            'coordinated_actions': zero_action,
            'action': zero_action,
            'layer': 'manipulation'
        }


class ConstraintSatisfactionModule(nn.Module):
    """约束满足模块"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.constraint_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """约束满足处理"""
        constraint_satisfaction = self.constraint_net(features)
        return {
            'constraint_satisfaction_score': constraint_satisfaction,
            'constraints_met': constraint_satisfaction > 0.5
        }


class BimanualCoordinationModule(nn.Module):
    """双臂协调模块"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.coordination_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 32)  # 输出协调动作
        )

    def forward(self, features: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """双臂协调处理"""
        coordinated_action = self.coordination_net(features[:, -1, :])
        return coordinated_action