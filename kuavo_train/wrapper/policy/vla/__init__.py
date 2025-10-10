"""
VLA (Vision-Language-Action) Transformer Policy

现代化的Token化策略架构，支持灵活的输入维度配置
"""
from .VLAPolicyWrapper import VLAPolicyWrapper
from .VLAConfigWrapper import VLAConfigWrapper

__all__ = ['VLAPolicyWrapper', 'VLAConfigWrapper']
