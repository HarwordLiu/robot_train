"""
VLAPolicyWrapper: VLA Transformer策略主类
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from collections import deque

from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from .VLAConfigWrapper import VLAConfigWrapper
from .tokenizers.VisionTokenizer import VisionTokenizer
from .tokenizers.StateTokenizer import StateTokenizer
from .decoders.DiffusionDecoder import DiffusionDecoder

# 导入归一化工具
from lerobot.policies.normalize import Normalize, Unnormalize

# 导入图像处理工具
from kuavo_train.utils.augmenter import crop_image, resize_image


class VLAPolicyWrapper(CustomDiffusionPolicyWrapper):
    """
    VLA Transformer策略

    架构流程：
    输入 → Token化 → Transformer Encoder → Token空间Diffusion → 动作输出

    核心特点：
    - 所有输入维度通过配置文件定义
    - 完整token化架构
    - 在token空间做diffusion
    - 对标OpenVLA/RT-2等SOTA方法
    """

    def __init__(
        self,
        config: VLAConfigWrapper,
        dataset_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    ):
        """
        初始化VLA策略

        Args:
            config: VLA配置对象
            dataset_stats: 数据集统计信息（用于归一化）
        """
        # 不调用CustomDiffusionPolicyWrapper的__init__，因为架构完全不同
        # 直接初始化nn.Module
        nn.Module.__init__(self)

        self.config = config

        # 构建归一化器（使用lerobot的Normalize类）
        self.normalize_inputs = Normalize(
            config.input_features,
            config.normalization_mapping,
            dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_features,
            config.normalization_mapping,
            dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features,
            config.normalization_mapping,
            dataset_stats
        )

        # 从配置获取关节配置
        self.joint_configs = config.state_config.get('joints', [])
        if not self.joint_configs:
            raise ValueError("state_config must contain 'joints' list")

        # 图像预处理配置
        self.crop_shape = getattr(config, 'crop_shape', None)
        self.crop_is_random = getattr(config, 'crop_is_random', True)

        # 获取动作维度
        if hasattr(config, 'output_features') and 'action' in config.output_features:
            self.action_dim = config.output_features["action"].shape[0]
        else:
            # 从joint配置推断
            self.action_dim = len(self.joint_configs)

        print(f"🤖 Initializing VLA Transformer Policy")
        print(f"   Action dim: {self.action_dim}")
        print(f"   Horizon: {config.horizon}")
        print(f"   Token dim: {config.token_embed_dim}")

        # ==================== Tokenizers ====================
        self.vision_tokenizer = VisionTokenizer(
            patch_size=config.patch_size,
            embed_dim=config.token_embed_dim,
            image_size=config.image_size
        )

        self.state_tokenizer = StateTokenizer(
            embed_dim=config.token_embed_dim,
            max_joints=50
        )

        # ==================== Transformer Encoder ====================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.token_embed_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_dim_feedforward,
            dropout=config.transformer_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN架构
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_depth
        )

        # ==================== Diffusion Decoder ====================
        self.diffusion_decoder = DiffusionDecoder(
            action_dim=self.action_dim,
            horizon=config.horizon,
            context_dim=config.token_embed_dim,
            num_train_timesteps=config.num_train_timesteps,
            num_denoiser_layers=config.num_denoiser_layers,
            num_heads=config.denoiser_heads,
            dim_feedforward=config.denoiser_dim_feedforward,
            noise_scheduler_type=config.noise_scheduler_type,
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            prediction_type=config.prediction_type,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range
        )

        # 观测和动作队列（用于推理）
        self._queues = None
        self.reset()

        print(f"✅ VLA Transformer Policy initialized successfully")

    def _preprocess_images(self, images: torch.Tensor, random_crop: bool = True, image_type: str = "rgb") -> torch.Tensor:
        """
        预处理图像：crop + resize到VisionTokenizer期望的尺寸

        Args:
            images: [B, C, H, W] 输入图像
            random_crop: 是否随机裁剪（训练时True，推理时False）
            image_type: 图像类型 ("rgb" 或 "depth")

        Returns:
            processed_images: [B, C, image_size, image_size] 处理后的图像
        """
        if self.crop_shape is not None:
            # 1. Crop到crop_shape
            images, _ = crop_image(
                images,
                target_range=self.crop_shape,
                random_crop=(random_crop and self.crop_is_random)
            )

        # 2. Resize到VisionTokenizer期望的image_size
        target_size = [self.config.image_size, self.config.image_size]
        images = resize_image(images, target_size=target_size, image_type=image_type)

        return images

    def reset(self):
        """重置观测和动作队列"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(
                maxlen=self.config.n_obs_steps)
        if self.config.use_depth and self.config.depth_features:
            self._queues["observation.depth"] = deque(
                maxlen=self.config.n_obs_steps)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        训练forward

        Args:
            batch: 输入批次，包含:
                - observation.image.{camera_name}: [B, T, C, H, W] 各相机RGB图像
                - observation.depth.{camera_name}: [B, T, C, H, W] 各相机深度图（可选）
                - observation.state: [B, T, state_dim] 状态
                - action: [B, horizon, action_dim] 目标动作

        Returns:
            loss: 标量tensor
            info: 额外信息字典（可为None）
        """
        # 1. 预处理图像：crop + resize（训练时随机crop）
        if self.config.image_features:
            batch = dict(batch)  # shallow copy
            for key in self.config.image_features:
                images = batch[key]
                # 处理时间维度：取最后一帧
                if len(images.shape) == 5:  # [B, T, C, H, W]
                    images = images[:, -1]  # [B, C, H, W]
                images = self._preprocess_images(images, random_crop=True, image_type="rgb")
                batch[key] = images  # [B, C, 224, 224]

        if self.config.use_depth and self.config.depth_features:
            batch = dict(batch)
            for key in self.config.depth_features:
                depth = batch[key]
                if len(depth.shape) == 5:
                    depth = depth[:, -1]
                depth = self._preprocess_images(depth, random_crop=True, image_type="depth")
                batch[key] = depth

        # 2. 归一化输入
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # 3. Stack各相机图像
        if self.config.image_features:
            # Stack所有相机: [B, num_cameras, C, H, W]
            rgb_images = torch.stack([batch[key] for key in self.config.image_features], dim=1)
        else:
            rgb_images = None

        if self.config.use_depth and self.config.depth_features:
            depth_images = torch.stack([batch[key] for key in self.config.depth_features], dim=1)
        else:
            depth_images = None

        # 4. Token化所有输入
        if rgb_images is not None:
            # rgb_images: [B, num_cameras, C, H, W]
            # VisionTokenizer会处理多相机输入
            vision_tokens = self.vision_tokenizer(rgb_images, depth_images)
        else:
            # 如果没有视觉输入，创建零tokens
            batch_size = batch['observation.state'].shape[0]
            device = batch['observation.state'].device
            num_patches = self.vision_tokenizer.num_patches
            vision_tokens = torch.zeros(
                batch_size, num_patches, self.config.token_embed_dim,
                device=device
            )

        # State tokens
        state = batch['observation.state']
        if len(state.shape) == 3:  # [B, T, state_dim]
            state = state[:, -1]  # 取最后一帧 [B, state_dim]

        state_tokens = self.state_tokenizer(state, self.joint_configs)

        # 3. 拼接所有tokens
        all_tokens = torch.cat([vision_tokens, state_tokens], dim=1)
        # [B, num_vision_tokens + num_state_tokens, token_embed_dim]

        # 4. Transformer编码
        context_tokens = self.transformer_encoder(all_tokens)

        # 5. Diffusion loss
        target_actions = batch['action']
        loss = self.diffusion_decoder.compute_loss(
            target_actions, context_tokens)

        # 返回loss和空的info字典
        return loss, None

    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        推理forward：生成动作

        Args:
            batch: 观测数据，包含各相机的observation keys

        Returns:
            action: [B, action_dim] 下一步动作
        """
        # 1. 预处理图像：crop + resize（推理时中心crop）
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                images = batch[key]
                if len(images.shape) == 5:  # [B, T, C, H, W]
                    images = images[:, -1]  # [B, C, H, W]
                images = self._preprocess_images(images, random_crop=False, image_type="rgb")
                batch[key] = images

        if self.config.use_depth and self.config.depth_features:
            batch = dict(batch)
            for key in self.config.depth_features:
                depth = batch[key]
                if len(depth.shape) == 5:
                    depth = depth[:, -1]
                depth = self._preprocess_images(depth, random_crop=False, image_type="depth")
                batch[key] = depth

        # 2. 归一化输入
        batch = self.normalize_inputs(batch)

        # 3. Stack各相机图像
        if self.config.image_features:
            rgb_images = torch.stack([batch[key] for key in self.config.image_features], dim=1)
        else:
            rgb_images = None

        if self.config.use_depth and self.config.depth_features:
            depth_images = torch.stack([batch[key] for key in self.config.depth_features], dim=1)
        else:
            depth_images = None

        # 4. Token化输入
        if rgb_images is not None:
            vision_tokens = self.vision_tokenizer(rgb_images, depth_images)
        else:
            batch_size = batch['observation.state'].shape[0]
            device = batch['observation.state'].device
            num_patches = self.vision_tokenizer.num_patches
            vision_tokens = torch.zeros(
                batch_size, num_patches, self.config.token_embed_dim,
                device=device
            )

        state = batch['observation.state']
        if len(state.shape) == 3:
            state = state[:, -1]

        state_tokens = self.state_tokenizer(state, self.joint_configs)

        # 3. 拼接并编码
        all_tokens = torch.cat([vision_tokens, state_tokens], dim=1)
        context_tokens = self.transformer_encoder(all_tokens)

        # 4. Diffusion采样
        num_inference_steps = self.config.num_inference_steps or 50
        actions = self.diffusion_decoder.sample(
            context_tokens,
            num_inference_steps=num_inference_steps
        )

        # 5. Unnormalize输出（重要！采样出来的动作是归一化后的值）
        unnormalized_actions = self.unnormalize_outputs({'action': actions})['action']

        # 6. 返回第一步动作
        return unnormalized_actions[:, 0]

    def _save_pretrained(self, save_directory: Path) -> None:
        """保存模型"""
        print(f"💾 Saving VLA model to {save_directory}")

        save_directory.mkdir(parents=True, exist_ok=True)

        # 保存配置
        self.config._save_pretrained(save_directory)

        # 保存模型权重
        state_dict = self.state_dict()

        # 使用safetensors格式
        from safetensors.torch import save_file
        model_file = save_directory / "model.safetensors"
        save_file(state_dict, model_file)

        print(f"✅ VLA model saved successfully")

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        """从预训练模型加载"""
        print(f"📂 Loading VLA model from {pretrained_name_or_path}")

        # 加载配置
        config = VLAConfigWrapper.from_pretrained(pretrained_name_or_path)

        # 创建模型
        model = cls(config)

        # 加载权重
        from safetensors.torch import load_file
        model_file = Path(pretrained_name_or_path) / "model.safetensors"
        if model_file.exists():
            state_dict = load_file(model_file)
            model.load_state_dict(state_dict)
            print(f"✅ VLA model loaded successfully")
        else:
            print(f"⚠️  Model weights not found, using random initialization")

        return model
