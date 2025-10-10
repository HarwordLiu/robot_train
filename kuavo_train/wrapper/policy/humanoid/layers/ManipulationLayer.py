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

        # 计算实际的视觉输入维度（3个RGB相机 + 3个深度相机）
        # head_cam_h: 3, depth_h: 1, wrist_cam_l: 3, depth_l: 1, wrist_cam_r: 3, depth_r: 1
        # 总共: 3+1+3+1+3+1 = 12 通道
        actual_visual_dim = 12  # 默认值，可以根据配置调整

        # 视觉投影层：将实际的视觉维度投影到标准的visual_dim
        self.visual_projection = nn.Linear(actual_visual_dim, self.visual_dim)

        # 总输入投影层
        self.input_projection = nn.Linear(
            self.visual_dim + self.state_dim, self.hidden_size)

        # 主要的Transformer网络
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.manipulation_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers)

        # 约束满足模块
        self.constraint_solver = ConstraintSatisfactionModule(self.hidden_size)

        # 双臂协调模块
        self.bimanual_coordinator = BimanualCoordinationModule(
            self.hidden_size, self.state_dim)

        # 输出投影 - 动作维度应该与状态维度一致
        self.action_head = nn.Linear(self.hidden_size, self.state_dim)  # 动作输出

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
        constraint_solution = self.constraint_solver(
            manipulation_features, context)

        # 双臂协调
        coordinated_actions = self.bimanual_coordinator(
            manipulation_features, context)

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

        # 状态特征
        if 'observation.state' in inputs:
            state_features = inputs['observation.state']
            # 处理维度：确保是3D tensor [batch_size, seq_len, state_dim]
            if len(state_features.shape) == 1:
                state_features = state_features.unsqueeze(
                    0).unsqueeze(0)  # [1, 1, state_dim]
            elif len(state_features.shape) == 2:
                state_features = state_features.unsqueeze(
                    1)  # [batch_size, 1, state_dim]
            features_list.append(state_features)

        # 视觉特征（如果可用）- 处理多相机输入
        visual_features_list = []

        # 查找所有图像和深度图的key
        image_keys = [k for k in inputs.keys() if k.startswith(
            'observation.images.') or k.startswith('observation.depth')]

        if image_keys:
            for key in image_keys:
                img_feature = inputs[key]
                # 对图像进行全局平均池化 [batch_size, channels, H, W] -> [batch_size, channels]
                if len(img_feature.shape) == 4:
                    img_feature = img_feature.mean(
                        dim=(-2, -1))  # [batch_size, channels]
                # 展平通道维度
                if len(img_feature.shape) == 2:
                    visual_features_list.append(img_feature)

        # 如果有视觉特征，拼接所有相机的特征
        if visual_features_list:
            # 拼接所有相机特征 [batch_size, sum_of_channels]
            combined_visual = torch.cat(visual_features_list, dim=-1)

            # 投影到标准视觉维度，使用固定的投影层
            # 注意：如果实际输入维度与预期不符，会在训练/推理时报错，这是期望的行为
            combined_visual = self.visual_projection(combined_visual)

            # 确保是3D tensor [batch_size, seq_len, visual_dim]
            if len(combined_visual.shape) == 2:
                combined_visual = combined_visual.unsqueeze(1)
            features_list.append(combined_visual)
        else:
            # 如果没有视觉特征，使用零填充
            if features_list:
                batch_size, seq_len = features_list[0].shape[:2]
                device = features_list[0].device
                zero_visual = torch.zeros(
                    batch_size, seq_len, self.visual_dim, device=device)
                features_list.append(zero_visual)

        if not features_list:
            return None

        # 特征拼接和投影
        combined_features = torch.cat(features_list, dim=-1)
        projected_features = self.input_projection(combined_features)
        return projected_features

    def _generate_default_output(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """生成默认输出"""
        batch_size = list(inputs.values())[0].size(0)
        device = list(inputs.values())[0].device

        zero_features = torch.zeros(
            batch_size, 10, self.hidden_size, device=device)
        zero_action = torch.zeros(batch_size, self.state_dim, device=device)

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

    def __init__(self, feature_dim: int, action_dim: int = 16):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.coordination_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_dim)  # 输出协调动作
        )

    def forward(self, features: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """双臂协调处理"""
        coordinated_action = self.coordination_net(features[:, -1, :])
        return coordinated_action
