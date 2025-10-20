"""
SmolVLA Diffusion Configuration Wrapper for Kuavo Project

基于 SmolVLA 但使用 Diffusion 而非 Flow Matching 进行动作生成
"""

from dataclasses import dataclass, fields
from pathlib import Path
from copy import deepcopy
from typing import TypeVar, List, Tuple, Optional
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.policies import PreTrainedConfig, PolicyFeature

T = TypeVar("T", bound="SmolVLADiffusionConfigWrapper")


@PreTrainedConfig.register_subclass("smolvla_diffusion_kuavo")
@dataclass
class SmolVLADiffusionConfigWrapper(SmolVLAConfig):
    """
    Kuavo项目的 SmolVLA Diffusion 配置扩展类

    主要变化：
    1. 使用 Diffusion 替代 Flow Matching
    2. 完全冻结视觉层
    3. 添加 Diffusion 特定配置
    4. 保持与原始 SmolVLA 的兼容性
    """

    # ==================== Diffusion 特定配置 ====================
    use_diffusion: bool = True  # 启用 Diffusion
    num_inference_steps: int = 50  # 推理步数
    num_train_timesteps: int = 1000  # 训练时间步数
    noise_schedule: str = "linear"  # 噪声调度类型
    prediction_type: str = "epsilon"  # 预测类型

    # 噪声调度参数
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # DDIM 采样配置
    use_ddim_sampling: bool = True
    ddim_eta: float = 0.0

    # 其他 Diffusion 参数
    clip_sample: bool = False
    clip_sample_range: float = 1.0
    variance_type: str = "fixed_small"  # fixed_small, fixed_large, learned, etc.

    # ==================== 视觉层配置（完全冻结）====================
    # 重写父类配置，确保视觉层完全冻结
    unfreeze_vision_layers: Optional[List[int]] = None
    freeze_vision_layers: Optional[List[int]] = None
    freeze_vision_ratio: Optional[float] = None

    # ==================== 学习率配置（简化版）====================
    use_layerwise_lr: bool = False  # 简化策略，不使用分层学习率
    vision_encoder_lr: Optional[float] = None  # 视觉层冻结，不需要
    expert_lr: Optional[float] = None  # 使用统一学习率

    # ==================== 其他优化配置 ====================
    # Diffusion 通常需要不同的优化策略
    optimizer_lr: float = 2.0e-5  # 默认学习率（比 Flow Matching 稍低）

    def __post_init__(self):
        """
        后初始化处理
        """
        # 设置默认值
        if self.unfreeze_vision_layers is None:
            self.unfreeze_vision_layers = None

        if self.freeze_vision_layers is None:
            self.freeze_vision_layers = None

        if self.freeze_vision_ratio is None:
            self.freeze_vision_ratio = None

        # 确保 Diffusion 配置正确
        assert self.use_diffusion, "SmolVLADiffusionConfig 必须设置 use_diffusion=True"

        # 验证配置
        if self.noise_schedule not in ["linear", "cosine", "sqrt_linear"]:
            raise ValueError(f"不支持的 noise_schedule: {self.noise_schedule}")

        if self.prediction_type not in ["epsilon", "v_prediction", "sample"]:
            raise ValueError(f"不支持的 prediction_type: {self.prediction_type}")

        # 转换 OmegaConf 对象（如果存在）
        self._convert_omegaconf_to_native()

        # 重新转换 features 为 PolicyFeature 对象
        if hasattr(self, 'input_features') and self.input_features is not None:
            self.input_features = self._normalize_feature_dict(self.input_features)
        if hasattr(self, 'output_features') and self.output_features is not None:
            self.output_features = self._normalize_feature_dict(self.output_features)

        # 调用父类初始化
        super().__post_init__()

        # Diffusion 特定验证
        if self.use_diffusion and self.num_inference_steps > self.num_train_timesteps:
            print(f"⚠️ 警告: num_inference_steps ({self.num_inference_steps}) > num_train_timesteps ({self.num_train_timesteps})")

        # 打印配置摘要
        self._print_config_summary()

    def _convert_omegaconf_to_native(self):
        """
        将配置中所有 OmegaConf 对象转换为原生 Python 对象
        """
        try:
            from omegaconf import DictConfig, ListConfig, OmegaConf
        except ImportError:
            return

        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (DictConfig, ListConfig)):
                native_value = OmegaConf.to_container(value, resolve=True)
                setattr(self, field.name, native_value)

    def _normalize_feature_dict(self, d):
        """将字典格式的 features 转换为 PolicyFeature 对象"""
        if not isinstance(d, dict):
            return d

        return {
            k: PolicyFeature(**v) if isinstance(v, dict) and not isinstance(v, PolicyFeature) else v
            for k, v in d.items()
        }

    def _print_config_summary(self):
        """打印配置摘要"""
        print("\n" + "="*70)
        print("🚀 SmolVLA Diffusion Config Initialized for Kuavo Project")
        print("="*70)
        print(f"📋 Configuration Summary:")
        print(f"   - VLM Model: {self.vlm_model_name}")
        print(f"   - Action Generation: Diffusion (not Flow Matching)")
        print(f"   - Vision Encoder: FROZEN (all layers)")
        print(f"   - Train Expert Only: {self.train_expert_only}")
        print(f"   - Max Action Dim: {self.max_action_dim}")
        print(f"   - Max State Dim: {self.max_state_dim}")
        print(f"   - Chunk Size: {self.chunk_size}")
        print(f"   - Action Steps: {self.n_action_steps}")

        print(f"\n🎭 Diffusion Parameters:")
        print(f"   - Train Timesteps: {self.num_train_timesteps}")
        print(f"   - Inference Steps: {self.num_inference_steps}")
        print(f"   - Noise Schedule: {self.noise_schedule}")
        print(f"   - Prediction Type: {self.prediction_type}")
        print(f"   - Beta Range: [{self.beta_start}, {self.beta_end}]")
        print(f"   - Use DDIM: {self.use_ddim_sampling}")
        if self.use_ddim_sampling:
            print(f"   - DDIM Eta: {self.ddim_eta}")

        print(f"\n🧠 Learning Rate:")
        print(f"   - Optimizer LR: {self.optimizer_lr:.2e}")
        print(f"   - Use Layerwise LR: {self.use_layerwise_lr}")

        print(f"\n👁️  Vision Config:")
        print(f"   - Use Depth: {self.use_depth}")
        if self.use_depth and self.depth_features:
            print(f"   - Depth Features: {self.depth_features}")
        print(f"   - Image Size: {self.resize_imgs_with_padding}")

        print("="*70 + "\n")

    def _save_pretrained(self, save_directory: Path) -> None:
        """
        保存配置到指定目录
        """
        import draccus
        from lerobot.configs.policies import CONFIG_NAME

        # 创建深拷贝
        cfg_copy = deepcopy(self)

        # 转换 torch.device 为字符串
        if hasattr(cfg_copy, 'device') and isinstance(cfg_copy.device, torch.device):
            cfg_copy.device = str(cfg_copy.device)

        # 保存配置
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

        支持从 SmolVLA 预训练模型加载，并自动转换为 Diffusion 配置
        """
        print(f"\n{'='*70}")
        print(f"📂 Loading SmolVLA Diffusion Config from: {pretrained_name_or_path}")
        print(f"{'='*70}")

        # 如果提供了 Diffusion 特定参数，更新 policy_kwargs
        diffusion_defaults = {
            'use_diffusion': True,
            'num_inference_steps': 50,
            'num_train_timesteps': 1000,
            'noise_schedule': 'linear',
            'prediction_type': 'epsilon',
            'use_ddim_sampling': True,
            'ddim_eta': 0.0,
            'freeze_vision_encoder': True,  # 确保视觉层冻结
        }

        # 合并参数（用户参数优先）
        for key, value in diffusion_defaults.items():
            if key not in policy_kwargs:
                policy_kwargs[key] = value

        # 尝试从预训练路径加载
        pretrained_path = Path(pretrained_name_or_path)
        if pretrained_path.exists():
            # 本地配置文件
            config_file = pretrained_path / "config.json"
            if config_file.exists():
                import json
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)

                # 如果是原始 SmolVLA 配置，转换类型
                if config_dict.get("_type_name") == "smolvla_kuavo":
                    config_dict["_type_name"] = "smolvla_diffusion_kuavo"
                    print("✅ 转换 SmolVLA 配置为 SmolVLA Diffusion 配置")

                # 应用 Diffusion 默认值
                config_dict.update(policy_kwargs)

                # 创建配置实例
                from draccus import decode
                config = decode(SmolVLADiffusionConfigWrapper, config_dict)
                print("✅ 从本地配置文件加载成功")
                return config

        # 如果不是本地文件，调用父类方法
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

    def create_scheduler(self, device: Optional[str] = None):
        """
        创建噪声调度器

        Args:
            device: 设备

        Returns:
            噪声调度器实例
        """
        from .diffusion_scheduler import DDIMScheduler, DDPMScheduler

        if self.use_ddim_sampling:
            scheduler = DDIMScheduler(
                num_train_timesteps=self.num_train_timesteps,
                num_inference_steps=self.num_inference_steps,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                beta_schedule=self.noise_schedule,
                prediction_type=self.prediction_type,
                clip_sample=self.clip_sample,
                clip_sample_range=self.clip_sample_range,
            )
        else:
            scheduler = DDPMScheduler(
                num_train_timesteps=self.num_train_timesteps,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                beta_schedule=self.noise_schedule,
                prediction_type=self.prediction_type,
                clip_sample=self.clip_sample,
                clip_sample_range=self.clip_sample_range,
            )

        if device is not None:
            scheduler = scheduler.to(device)

        return scheduler