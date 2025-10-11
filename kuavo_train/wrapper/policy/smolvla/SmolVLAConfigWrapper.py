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

        # 注意：为了使用SmolVLA预训练权重，max_action_dim和max_state_dim应该为32（与预训练模型一致）
        # Kuavo实际是16维，数据会在加载时自动填充到32维
        if self.max_action_dim == 32 and self.max_state_dim == 32:
            print("✅ Using SmolVLA pretrained dimensions (32D). Kuavo 16D data will be auto-padded.")
        elif self.max_action_dim != 32 or self.max_state_dim != 32:
            print(f"⚠️  Warning: max_action_dim={self.max_action_dim}, max_state_dim={self.max_state_dim}")
            print(f"   For pretrained SmolVLA, both should be 32. Current config may not load pretrained weights.")

        # 打印SmolVLA配置摘要
        print(f"📋 SmolVLA Config Summary (Kuavo):")
        print(f"   - VLM Model: {self.vlm_model_name}")
        print(f"   - Max Action Dim: {self.max_action_dim} (Kuavo actual: 16, auto-padded)")
        print(f"   - Max State Dim: {self.max_state_dim} (Kuavo actual: 16, auto-padded)")
        print(f"   - Chunk Size: {self.chunk_size}")
        print(f"   - Action Steps: {self.n_action_steps}")
        print(f"   - Freeze Vision: {self.freeze_vision_encoder}")
        print(f"   - Train Expert Only: {self.train_expert_only}")
