"""
SmolVLA Configuration Wrapper for Kuavo Project

扩展lerobot的SmolVLAConfig以支持Kuavo项目的特定需求
"""

from dataclasses import dataclass, fields
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

    重要：自动将所有 OmegaConf 对象转换为原生 Python 对象，
    确保可以使用 lerobot 的标准保存方式，无需依赖 omegaconf。
    """

    def _convert_omegaconf_to_native(self):
        """
        将配置中所有 OmegaConf 对象转换为原生 Python 对象

        这确保了配置可以被 JSON 序列化，支持 lerobot 的标准保存/加载方式。
        只在需要时导入 omegaconf，避免不必要的依赖。
        """
        try:
            from omegaconf import DictConfig, ListConfig, OmegaConf
        except ImportError:
            # 如果没有安装 omegaconf，说明配置已经是原生对象，无需转换
            return

        # 遍历所有 dataclass 字段
        for field in fields(self):
            value = getattr(self, field.name)

            # 转换 OmegaConf 对象为原生 Python 对象
            if isinstance(value, (DictConfig, ListConfig)):
                # OmegaConf.to_container 会递归转换所有嵌套的 DictConfig/ListConfig
                native_value = OmegaConf.to_container(value, resolve=True)
                setattr(self, field.name, native_value)

    def __post_init__(self):
        """
        后初始化处理

        1. 首先转换所有 OmegaConf 对象为原生 Python 对象
        2. 然后执行父类的验证逻辑
        3. 最后执行 Kuavo 特定的配置验证
        """
        # 第一步：转换 OmegaConf 对象（必须在父类 __post_init__ 之前）
        self._convert_omegaconf_to_native()

        # 第二步：调用父类的后初始化
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
