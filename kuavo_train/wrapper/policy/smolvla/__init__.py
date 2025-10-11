"""
SmolVLA Policy Wrapper for Kuavo Project

SmolVLA是HuggingFace发布的轻量级Vision-Language-Action模型，
使用顺序fine-tuning策略支持多任务学习。
"""

from .SmolVLAPolicyWrapper import SmolVLAPolicyWrapper
from .SmolVLAConfigWrapper import SmolVLAConfigWrapper

__all__ = ['SmolVLAPolicyWrapper', 'SmolVLAConfigWrapper']
