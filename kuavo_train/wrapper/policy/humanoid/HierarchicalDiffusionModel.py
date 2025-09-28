"""
HierarchicalDiffusionModel: 分层架构的Diffusion模型
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from kuavo_train.wrapper.policy.diffusion.DiffusionModelWrapper import CustomDiffusionModelWrapper


class HierarchicalDiffusionModel(CustomDiffusionModelWrapper):
    """
    分层架构的Diffusion模型

    继承自CustomDiffusionModelWrapper，增加对分层输出的处理能力
    """

    def __init__(self, config):
        super().__init__(config)

        # 分层特征融合网络
        self.hierarchical_fusion = HierarchicalFeatureFusion(config)

    def compute_loss(self, batch: Dict[str, torch.Tensor], layer_outputs: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        计算分层Diffusion损失

        Args:
            batch: 输入批次
            layer_outputs: 分层输出结果

        Returns:
            torch.Tensor: Diffusion损失
        """
        if layer_outputs is None:
            # 如果没有分层输出，使用原有逻辑
            return super().compute_loss(batch)

        # 融合分层特征
        enhanced_batch = self.hierarchical_fusion(batch, layer_outputs)

        # 使用增强后的批次计算损失
        return super().compute_loss(enhanced_batch)


class HierarchicalFeatureFusion(nn.Module):
    """分层特征融合模块"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 特征融合网络
        self.safety_fusion = nn.Linear(64, 32)   # 安全层特征融合
        self.gait_fusion = nn.Linear(32, 64)     # 步态层特征融合 (修正维度)
        self.manipulation_fusion = nn.Linear(512, 128)  # 操作层特征融合
        self.planning_fusion = nn.Linear(1024, 256)     # 规划层特征融合

    def forward(self, batch: Dict[str, torch.Tensor], layer_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        融合分层特征到原始批次中

        Args:
            batch: 原始批次
            layer_outputs: 分层输出

        Returns:
            Dict: 增强后的批次
        """
        enhanced_batch = batch.copy()

        # 提取并融合各层特征
        fused_features = []

        if 'safety' in layer_outputs:
            safety_features = layer_outputs['safety'].get('balance_action')
            if safety_features is not None:
                fused_features.append(self.safety_fusion(safety_features))

        if 'gait' in layer_outputs:
            gait_features = layer_outputs['gait'].get('action')
            if gait_features is not None:
                fused_features.append(self.gait_fusion(gait_features))

        if 'manipulation' in layer_outputs:
            manip_features = layer_outputs['manipulation'].get('action')
            if manip_features is not None:
                fused_features.append(self.manipulation_fusion(manip_features))

        if 'planning' in layer_outputs:
            planning_features = layer_outputs['planning'].get('action')
            if planning_features is not None:
                fused_features.append(self.planning_fusion(planning_features))

        # 如果有融合特征，添加到批次中
        if fused_features:
            # 简单的特征融合策略：平均
            fused_feature = torch.stack(fused_features, dim=0).mean(dim=0)
            enhanced_batch['hierarchical_features'] = fused_feature

        return enhanced_batch