"""
SmolVLA Configuration Wrapper for Kuavo Project

扩展lerobot的SmolVLAConfig以支持Kuavo项目的特定需求
"""

from dataclasses import dataclass, fields
from pathlib import Path
from copy import deepcopy
from typing import TypeVar, List, Tuple
import torch
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.policies import PreTrainedConfig, PolicyFeature

T = TypeVar("T", bound="SmolVLAConfigWrapper")


@PreTrainedConfig.register_subclass("smolvla_kuavo")
@dataclass
class SmolVLAConfigWrapper(SmolVLAConfig):
    """
    Kuavo项目的SmolVLA配置扩展类 - 支持深度相机

    继承SmolVLAConfig的所有功能，添加深度相机支持：
    - 深度相机配置
    - 多相机融合参数
    - 深度图像预处理设置

    重要：自动将所有 OmegaConf 对象转换为原生 Python 对象，
    确保可以使用 lerobot 的标准保存方式，无需依赖 omegaconf。
    """

    # 深度相机支持
    use_depth: bool = True
    depth_features: List[str] = None

    # 深度图像预处理
    depth_resize_with_padding: List[int] = None
    depth_normalization_range: List[float] = None
    use_depth_padding: bool = True  # 深度图是否使用padding方式保持长宽比

    def __post_init__(self):
        """
        后初始化处理

        1. 设置默认值
        2. 转换 OmegaConf 对象为原生 Python 对象
        3. 重新将 input_features 和 output_features 转换为 PolicyFeature 对象
        4. 执行父类的验证逻辑
        5. 执行 Kuavo 特定的配置验证
        """
        # 设置默认值
        if self.depth_features is None:
            self.depth_features = [
                "observation.depth_h", "observation.depth_l", "observation.depth_r"
            ]

        if self.depth_resize_with_padding is None:
            self.depth_resize_with_padding = [512, 512]

        if self.depth_normalization_range is None:
            self.depth_normalization_range = [0.0, 1000.0]

        # 第一步：转换 OmegaConf 对象（必须在父类 __post_init__ 之前）
        self._convert_omegaconf_to_native()

        # 第二步：重新将 features 转换为 PolicyFeature 对象
        # 这是必要的，因为 _convert_omegaconf_to_native 会将它们转换为字典
        if hasattr(self, 'input_features') and self.input_features is not None:
            self.input_features = self._normalize_feature_dict(
                self.input_features)
        if hasattr(self, 'output_features') and self.output_features is not None:
            self.output_features = self._normalize_feature_dict(
                self.output_features)

        # 第三步：调用父类的后初始化
        super().__post_init__()

        # 验证深度配置
        if self.use_depth and not self.depth_features:
            raise ValueError("use_depth=True but no depth_features specified")

        # 注意：为了使用SmolVLA预训练权重，max_action_dim和max_state_dim应该为32（与预训练模型一致）
        # Kuavo实际是16维，数据会在加载时自动填充到32维
        if self.max_action_dim == 32 and self.max_state_dim == 32:
            print(
                "✅ Using SmolVLA pretrained dimensions (32D). Kuavo 16D data will be auto-padded.")
        elif self.max_action_dim != 32 or self.max_state_dim != 32:
            print(
                f"⚠️  Warning: max_action_dim={self.max_action_dim}, max_state_dim={self.max_state_dim}")
            print(
                f"   For pretrained SmolVLA, both should be 32. Current config may not load pretrained weights.")

        # 打印SmolVLA配置摘要
        print(f"📋 SmolVLA Config Summary (Kuavo with Depth):")
        print(f"   - VLM Model: {self.vlm_model_name}")
        print(
            f"   - Max Action Dim: {self.max_action_dim} (Kuavo actual: 16, auto-padded)")
        print(
            f"   - Max State Dim: {self.max_state_dim} (Kuavo actual: 16, auto-padded)")
        print(f"   - Chunk Size: {self.chunk_size}")
        print(f"   - Action Steps: {self.n_action_steps}")
        print(f"   - Freeze Vision: {self.freeze_vision_encoder}")
        print(f"   - Train Expert Only: {self.train_expert_only}")
        print(f"   - Use Depth: {self.use_depth}")
        print(f"   - Depth Features: {self.depth_features}")

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

    def _normalize_feature_dict(self, d):
        """
        将字典格式的 features 转换为 PolicyFeature 对象

        当 OmegaConf 配置被转换为原生 Python 对象后，input_features 和 output_features
        会变成字典，需要重新转换为 PolicyFeature 对象以供策略模型使用。

        Args:
            d: 字典或包含字典的字典

        Returns:
            包含 PolicyFeature 对象的字典
        """
        if not isinstance(d, dict):
            return d

        return {
            k: PolicyFeature(**v) if isinstance(v,
                                                dict) and not isinstance(v, PolicyFeature) else v
            for k, v in d.items()
        }

    def __post_init__(self):
        """
        后初始化处理

        1. 首先转换所有 OmegaConf 对象为原生 Python 对象
        2. 重新将 input_features 和 output_features 转换为 PolicyFeature 对象
        3. 然后执行父类的验证逻辑
        4. 最后执行 Kuavo 特定的配置验证
        """
        # 第一步：转换 OmegaConf 对象（必须在父类 __post_init__ 之前）
        self._convert_omegaconf_to_native()

        # 第二步：重新将 features 转换为 PolicyFeature 对象
        # 这是必要的，因为 _convert_omegaconf_to_native 会将它们转换为字典
        if hasattr(self, 'input_features') and self.input_features is not None:
            self.input_features = self._normalize_feature_dict(
                self.input_features)
        if hasattr(self, 'output_features') and self.output_features is not None:
            self.output_features = self._normalize_feature_dict(
                self.output_features)

        # 第三步：调用父类的后初始化
        super().__post_init__()

        # 注意：为了使用SmolVLA预训练权重，max_action_dim和max_state_dim应该为32（与预训练模型一致）
        # Kuavo实际是16维，数据会在加载时自动填充到32维
        if self.max_action_dim == 32 and self.max_state_dim == 32:
            print(
                "✅ Using SmolVLA pretrained dimensions (32D). Kuavo 16D data will be auto-padded.")
        elif self.max_action_dim != 32 or self.max_state_dim != 32:
            print(
                f"⚠️  Warning: max_action_dim={self.max_action_dim}, max_state_dim={self.max_state_dim}")
            print(
                f"   For pretrained SmolVLA, both should be 32. Current config may not load pretrained weights.")

        # 打印SmolVLA配置摘要
        print(f"📋 SmolVLA Config Summary (Kuavo):")
        print(f"   - VLM Model: {self.vlm_model_name}")
        print(
            f"   - Max Action Dim: {self.max_action_dim} (Kuavo actual: 16, auto-padded)")
        print(
            f"   - Max State Dim: {self.max_state_dim} (Kuavo actual: 16, auto-padded)")
        print(f"   - Chunk Size: {self.chunk_size}")
        print(f"   - Action Steps: {self.n_action_steps}")
        print(f"   - Freeze Vision: {self.freeze_vision_encoder}")
        print(f"   - Train Expert Only: {self.train_expert_only}")

    def _save_pretrained(self, save_directory: Path) -> None:
        """
        保存配置到指定目录

        在保存前，将不能被 JSON 序列化的对象（如 torch.device）转换为可序列化格式。

        Args:
            save_directory: 保存目录路径
        """
        import draccus
        from lerobot.configs.policies import CONFIG_NAME

        # 创建深拷贝以避免修改原始配置
        cfg_copy = deepcopy(self)

        # 将 torch.device 转换为字符串
        if hasattr(cfg_copy, 'device') and isinstance(cfg_copy.device, torch.device):
            cfg_copy.device = str(cfg_copy.device)

        # 使用 draccus 保存配置
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(cfg_copy, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs,
    ) -> T:
        """
        从预训练路径加载配置

        这个方法调用父类的 from_pretrained，确保正确处理配置文件中的 type 字段。

        Args:
            pretrained_name_or_path: 预训练模型路径或 HuggingFace 模型 ID
            其他参数同 PreTrainedConfig.from_pretrained

        Returns:
            加载的配置对象
        """
        # 调用父类 PreTrainedConfig 的 from_pretrained，触发 Choice 机制识别子类
        parent_cls = PreTrainedConfig

        return parent_cls.from_pretrained(
            pretrained_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            **policy_kwargs,
        )
