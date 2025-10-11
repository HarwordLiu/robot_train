"""
SmolVLA Configuration Wrapper for Kuavo Project

扩展lerobot的SmolVLAConfig以支持Kuavo项目的特定需求
"""

from dataclasses import dataclass
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("smolvla_kuavo")
@dataclass
class SmolVLAConfigWrapper(SmolVLAConfig):
    """
    Kuavo项目的SmolVLA配置扩展类

    继承SmolVLAConfig的所有功能，可以添加Kuavo特定的配置项

    当前直接继承SmolVLAConfig，未来可以添加：
    - Kuavo特定的相机配置
    - 双臂机器人特定参数
    - 自定义的训练策略
    """

    def __post_init__(self):
        """
        后初始化处理

        可以在这里添加Kuavo特定的配置验证和处理逻辑
        """
        super().__post_init__()

        # 验证Kuavo双臂机器人的动作维度
        if self.max_action_dim != 16:
            print(f"⚠️  Warning: max_action_dim is {self.max_action_dim}, expected 16 for Kuavo dual-arm robot")

        # 打印SmolVLA配置摘要
        print(f"📋 SmolVLA Config Summary (Kuavo):")
        print(f"   - VLM Model: {self.vlm_model_name}")
        print(f"   - Max Action Dim: {self.max_action_dim}")
        print(f"   - Chunk Size: {self.chunk_size}")
        print(f"   - Action Steps: {self.n_action_steps}")
        print(f"   - Freeze Vision: {self.freeze_vision_encoder}")
        print(f"   - Train Expert Only: {self.train_expert_only}")
