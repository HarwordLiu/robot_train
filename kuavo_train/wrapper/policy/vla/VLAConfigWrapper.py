"""
VLAConfigWrapper: VLA策略的配置类
"""
from typing import Any, Dict, Union, Tuple
from dataclasses import dataclass, field
from kuavo_train.wrapper.policy.diffusion.DiffusionConfigWrapper import CustomDiffusionConfigWrapper
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("vla_transformer")
@dataclass
class VLAConfigWrapper(CustomDiffusionConfigWrapper):
    """
    VLA Transformer策略配置

    继承自CustomDiffusionConfigWrapper以复用diffusion相关配置
    """

    # Token化配置（优化后的默认值）
    patch_size: int = 32  # Vision patch大小（从16增大到32）
    token_embed_dim: int = 256  # 统一token维度（从512降到256）
    image_size: Union[int, Tuple[int, int]] = (192, 256)  # 输入图像尺寸（保持3:4宽高比）
    use_pretrained_patches: bool = True  # 是否使用预训练patch embeddings

    # Transformer Encoder配置
    transformer_depth: int = 6  # Encoder层数（从8降到6）
    transformer_heads: int = 8  # 注意力头数
    transformer_dim_feedforward: int = 1024  # FFN维度（从2048降到1024）
    transformer_dropout: float = 0.25  # Dropout率（从0.1增加到0.25）

    # Diffusion Decoder配置
    num_denoiser_layers: int = 3  # Decoder层数（从4降到3）
    denoiser_heads: int = 8  # Decoder注意力头数
    denoiser_dim_feedforward: int = 1024  # Decoder FFN维度（从2048降到1024）

    # State配置（核心）
    state_config: Dict[str, Any] = field(default_factory=dict)

    # 图像和训练配置
    use_depth: bool = True  # 是否使用深度图像
    use_amp: bool = True  # 是否使用混合精度训练（训练脚本使用）

    # 架构配置（用于优化器选择）
    use_unet: bool = False  # VLA使用Transformer，不使用UNet
    use_transformer: bool = True  # VLA使用Transformer架构

    def __post_init__(self):
        """后初始化处理"""
        super().__post_init__()

        # 验证state_config
        if 'joints' not in self.state_config:
            print(
                "⚠️  Warning: state_config missing 'joints' key. Using empty joint list.")
            self.state_config['joints'] = []

        # 打印配置摘要
        num_joints = len(self.state_config.get('joints', []))
        print(f"📋 VLA Config Summary:")
        print(f"   - Token dimension: {self.token_embed_dim}")
        print(f"   - Transformer depth: {self.transformer_depth}")
        print(f"   - Number of joints: {num_joints}")
        print(f"   - Action horizon: {self.horizon}")
        print(f"   - Diffusion steps: {self.num_train_timesteps}")
