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

    继承自CustomDiffusionModelWrapper，支持分层架构训练
    注意：分层架构的价值在于课程学习和层间协调，而不是特征融合
    """

    def __init__(self, config):
        super().__init__(config)

    def compute_loss(self, batch: Dict[str, torch.Tensor], layer_outputs: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        计算分层Diffusion损失

        Args:
            batch: 输入批次
            layer_outputs: 分层输出结果（用于课程学习，不直接融合到Diffusion模型）

        Returns:
            torch.Tensor: Diffusion损失
        """
        # 直接使用原始批次计算损失
        # 分层架构的价值在于课程学习和层间协调，而不是特征融合
        return super().compute_loss(batch)


# 分层架构的价值在于课程学习和层间协调，不需要特征融合
