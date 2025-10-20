from typing import Optional
from tkinter import NO
from sympy import N
import torch.nn as nn
from kuavo_train.wrapper.policy.diffusion.DiffusionConfigWrapper import CustomDiffusionConfigWrapper
import math
from collections import deque
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
)

from lerobot.policies.diffusion.modeling_diffusion import (_make_noise_scheduler,
                                                           _replace_submodules,
                                                           DiffusionConditionalUnet1d,
                                                           SpatialSoftmax,
                                                           DiffusionModel
                                                           )
from kuavo_train.wrapper.policy.diffusion.transformer_diffusion import TransformerForDiffusion
from kuavo_train.wrapper.policy.diffusion.flow_matching_scheduler import (
    FlowMatchingScheduler,
    create_flow_matching_scheduler,
)

OBS_DEPTH = "observation.depth"


class CustomDiffusionModelWrapper(DiffusionModel):
    def __init__(self, config: CustomDiffusionConfigWrapper):
        super().__init__(config)

        # self.config = config

        # Build observation encoders (depending on which observations are provided).
        # global_cond_dim = self.config.robot_state_feature.shape[0]
        global_cond_dim = 0

        if self.config.robot_state_feature:
            if self.config.use_state_encoder:
                self.state_encoder = FeatureEncoder(
                    in_dim=self.config.robot_state_feature.shape[0], out_dim=self.config.state_feature_dim)
                global_cond_dim = self.config.state_feature_dim
            else:
                global_cond_dim = self.config.robot_state_feature.shape[0]

        if self.config.image_features:

            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config)
                            for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
                self.rgb_attn_layer = nn.MultiheadAttention(
                    embed_dim=encoders[0].feature_dim, num_heads=8, batch_first=True)
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
                self.rgb_attn_layer = nn.MultiheadAttention(
                    embed_dim=self.rgb_encoder.feature_dim, num_heads=8, batch_first=True)
        if self.config.use_depth and self.config.depth_features:
            num_depth = len(self.config.depth_features)
            if self.config.use_separate_depth_encoder_per_camera:
                encoders = [DiffusionDepthEncoder(
                    config) for _ in range(num_depth)]
                self.depth_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_depth
                self.depth_attn_layer = nn.MultiheadAttention(
                    embed_dim=encoders[0].feature_dim, num_heads=8, batch_first=True)
            else:
                self.depth_encoder = DiffusionDepthEncoder(config)
                global_cond_dim += self.depth_encoder.feature_dim * num_depth
                self.depth_attn_layer = nn.MultiheadAttention(
                    embed_dim=self.depth_encoder.feature_dim, num_heads=8, batch_first=True)
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        # global_cond_dim *= self.config.n_obs_steps

        if config.use_unet:
            self.unet = DiffusionConditionalUnet1d(
                config, global_cond_dim=global_cond_dim * self.config.n_obs_steps)
        elif config.use_transformer:
            # self.unet = DiffusionTransformer(config)
            # from kuavo_train.wrapper.policy.diffusion.DiT_1D_model import DiT_S
            # self.unet = DiT_S(input_length=config.horizon, input_dim=config.action_feature.shape[0], cond_dim=global_cond_dim)
            # print("hello!!!", self.config.transformer_n_emb)
            self.unet = TransformerForDiffusion(
                input_dim=config.output_features["action"].shape[0],
                output_dim=config.output_features["action"].shape[0],
                horizon=config.horizon,
                n_obs_steps=config.n_obs_steps,
                cond_dim=global_cond_dim,
                n_layer=self.config.transformer_n_layer,
                n_head=self.config.transformer_n_head,
                n_emb=self.config.transformer_n_emb,
                p_drop_emb=self.config.transformer_dropout,
                p_drop_attn=self.config.transformer_dropout,
                causal_attn=False,
                time_as_cond=True,
                obs_as_cond=True,
                n_cond_layers=0
            )
        else:
            raise ValueError(
                "Either `use_unet` or `use_transformer` must be True in the configuration.")

        if self.config.use_depth and self.config.depth_features:
            feat_dim = self.depth_attn_layer.embed_dim
            self.multimodalfuse = nn.ModuleDict({
                "depth_q": nn.MultiheadAttention(embed_dim=feat_dim, num_heads=8, batch_first=True),
                "rgb_q": nn.MultiheadAttention(embed_dim=feat_dim, num_heads=8, batch_first=True)
            })

        # 根据配置选择调度器类型
        self.use_flow_matching = getattr(config, 'use_flow_matching', False)

        if self.use_flow_matching:
            # 使用 Flow Matching 调度器
            print("\n" + "="*70)
            print("🌊 使用 Flow Matching 调度器")
            print("="*70)
            self.noise_scheduler = create_flow_matching_scheduler(
                scheduler_type=getattr(
                    config, 'flow_matching_type', 'conditional'),
                num_inference_steps=getattr(config, 'num_inference_steps', 10),
                sigma=getattr(config, 'flow_sigma', 0.0),
                use_ode_solver=getattr(config, 'ode_solver', 'euler'),
                device=config.device if hasattr(config, 'device') else 'cpu',
            )
            print(f"✅ Flow Matching 配置:")
            print(
                f"   - 类型: {getattr(config, 'flow_matching_type', 'conditional')}")
            print(f"   - 推理步数: {getattr(config, 'num_inference_steps', 10)}")
            print(f"   - ODE求解器: {getattr(config, 'ode_solver', 'euler')}")
            print("="*70 + "\n")
        else:
            # 使用传统 Diffusion 调度器
            print("\n📊 使用传统 Diffusion (DDPM/DDIM) 调度器")
            self.noise_scheduler = _make_noise_scheduler(
                config.noise_scheduler_type,
                num_train_timesteps=config.num_train_timesteps,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                beta_schedule=config.beta_schedule,
                clip_sample=config.clip_sample,
                clip_sample_range=config.clip_sample_range,
                prediction_type=config.prediction_type,
            )

        if config.num_inference_steps is None:
            if hasattr(self.noise_scheduler, 'config'):
                self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
            else:
                self.num_inference_steps = getattr(
                    config, 'num_train_timesteps', 100)
        else:
            self.num_inference_steps = config.num_inference_steps

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        计算损失函数

        根据配置自动选择:
        - Diffusion: 预测噪声 ε
        - Flow Matching: 预测速度场 v_t

        Args:
            batch: 批次数据

        Returns:
            loss: 损失值
        """
        from lerobot.constants import ACTION

        # 准备条件特征
        global_cond = self._prepare_global_conditioning(batch)

        # 获取目标动作
        trajectory = batch[ACTION]  # [B, horizon, action_dim]
        batch_size = trajectory.shape[0]

        if self.use_flow_matching:
            # ========== Flow Matching 训练 ==========
            # 1. 采样初始噪声 x_0
            noise = torch.randn_like(trajectory)

            # 2. 采样时间步 t ∈ [0, 1]
            timesteps = torch.rand(batch_size, device=trajectory.device)

            # 3. 线性插值得到 x_t = (1-t)·x_0 + t·x_1
            noisy_trajectory = self.noise_scheduler.add_noise(
                original_samples=trajectory,
                noise=noise,
                timesteps=timesteps
            )

            # 4. 模型预测速度场 v_t
            if self.config.use_transformer:
                # Transformer 需要整数时间步
                timesteps_for_model = (timesteps * 1000).long()
            else:
                timesteps_for_model = (
                    timesteps * self.config.num_train_timesteps).long()

            pred_velocity = self.unet(
                noisy_trajectory,
                timesteps_for_model,
                global_cond=global_cond
            )

            # 5. 计算目标速度场: v_t = x_1 - x_0
            target_velocity = self.noise_scheduler.get_velocity(
                trajectory, noise, timesteps
            )

            # 6. 计算损失
            loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")

            # 可选：对 padding 进行 mask
            if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
                in_episode_bound = ~batch["action_is_pad"]
                loss = loss * in_episode_bound.unsqueeze(-1)

            return loss.mean()

        else:
            # ========== 传统 Diffusion 训练 ==========
            # 1. 采样随机噪声
            eps = torch.randn_like(trajectory)

            # 2. 采样时间步 t ∈ [0, T]
            timesteps = torch.randint(
                low=0,
                high=self.noise_scheduler.config.num_train_timesteps,
                size=(batch_size,),
                device=trajectory.device,
            ).long()

            # 3. 添加噪声: x_t = √(α_t)·x_0 + √(1-α_t)·ε
            noisy_trajectory = self.noise_scheduler.add_noise(
                trajectory, eps, timesteps)

            # 4. 模型预测噪声
            pred = self.unet(noisy_trajectory, timesteps,
                             global_cond=global_cond)

            # 5. 计算损失
            if self.config.prediction_type == "epsilon":
                target = eps  # 预测噪声
            elif self.config.prediction_type == "sample":
                target = trajectory  # 预测原始样本
            else:
                raise ValueError(
                    f"Unsupported prediction type {self.config.prediction_type}")

            loss = F.mse_loss(pred, target, reduction="none")

            # 可选：对 padding 进行 mask
            if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
                in_episode_bound = ~batch["action_is_pad"]
                loss = loss * in_episode_bound.unsqueeze(-1)

            return loss.mean()

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps, n_camera = batch[OBS_STATE].shape[:3]

        global_cond_feats = []
        # global_cond_feats = [batch[OBS_STATE]]

        # Extract image features.
        img_features = None
        depth_features = None
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                images_per_camera = einops.rearrange(
                    batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s n ...", b=batch_size, s=n_obs_steps
                )
            else:
                # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES],
                                     "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s n ...", b=batch_size, s=n_obs_steps
                )
            img_features = einops.rearrange(
                img_features, "b s n ... -> (b s) n ...", b=batch_size, s=n_obs_steps)
            img_features = self.rgb_attn_layer(
                query=img_features, key=img_features, value=img_features)[0]
            # img_features = einops.rearrange(
            #         img_features, "(b s) n ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            #     )
            # global_cond_feats.append(img_features)
            # print("global_cond_feats.shape",np.array(global_cond_feats[0].cpu().detach().numpy()).shape)
        if self.config.use_depth and self.config.depth_features:
            if self.config.use_separate_depth_encoder_per_camera:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                depth_per_camera = einops.rearrange(
                    batch[OBS_DEPTH], "b s n ... -> n (b s) ...")
                depth_features_list = torch.cat(
                    [
                        encoder(depth)
                        for encoder, depth in zip(self.depth_encoder, depth_per_camera, strict=True)
                    ]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                depth_features = einops.rearrange(
                    depth_features_list, "(n b s) ... -> b s n ...", b=batch_size, s=n_obs_steps
                )
            else:
                # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
                depth_features = self.depth_encoder(
                    einops.rearrange(batch[OBS_DEPTH],
                                     "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                depth_features = einops.rearrange(
                    depth_features, "(b s n) ... -> b s n ...", b=batch_size, s=n_obs_steps
                )
            depth_features = einops.rearrange(
                depth_features, "b s n ... -> (b s) n ...", b=batch_size, s=n_obs_steps)
            depth_features = self.depth_attn_layer(
                query=depth_features, key=depth_features, value=depth_features)[0]
            # depth_features = einops.rearrange(
            #         depth_features, "(b s) n ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            #     )
            # global_cond_feats.append(depth_features)
            # print("global_cond_feats.shape",np.array(global_cond_feats[0].cpu().detach().numpy()).shape)
        if (img_features is not None) and (depth_features is not None):
            # img_features = einops.rearrange(img_features, "(b s) n ... -> n (b s) ...")
            # depth_features = einops.rearrange(depth_features, "(b s) n ... -> n (b s) ...")
            rgb_q_fuse = self.multimodalfuse["rgb_q"](
                query=img_features, key=depth_features, value=depth_features)[0]
            depth_q_fuse = self.multimodalfuse["depth_q"](
                query=depth_features, key=img_features, value=img_features)[0]
            rgb_q_fuse = einops.rearrange(
                rgb_q_fuse, "(b s) n ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
            depth_q_fuse = einops.rearrange(
                depth_q_fuse, "(b s) n ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
            global_cond_feats.extend([rgb_q_fuse, depth_q_fuse])
        elif img_features is not None:
            img_features = einops.rearrange(
                img_features, "(b s) n ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
            global_cond_feats.append(img_features)

        if self.config.robot_state_feature:
            if self.config.use_state_encoder:
                state_features = self.state_encoder(batch[OBS_STATE])
                global_cond_feats.append(state_features)
            else:
                global_cond_feats.append(batch[OBS_STATE])

        if self.config.env_state_feature:
            # print(f"Using environment state feature: {OBS_ENV_STATE}")
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # Concatenate features then flatten to (B, global_cond_dim).
        if self.config.use_transformer:
            # Concatenate features to (B, To, cond_dim).
            return torch.cat(global_cond_feats, dim=-1)
        else:
            return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)


class DiffusionRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: CustomDiffusionConfigWrapper):
        super().__init__()
        # # Set up optional preprocessing.
        # if config.crop_shape is not None:
        #     self.do_crop = True
        #     # Always use center crop for eval
        #     self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
        #     if config.crop_is_random:
        #         self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
        #     else:
        #         self.maybe_random_crop = self.center_crop
        # else:
        #     self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.image_features` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.image_features`.

        # Note: we have a check in the config class to make sure all images have the same shape.
        images_shape = next(iter(config.image_features.values())).shape

        if config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        elif config.crop_shape is not None:
            if isinstance(list(config.crop_shape)[0], (list, tuple)):
                (x_start, x_end), (y_start, y_end) = config.crop_shape
                dummy_shape_h_w = (x_end-x_start, y_end-y_start)
            else:
                dummy_shape_h_w = config.crop_shape
        else:
            dummy_shape_h_w = images_shape[1:]

        # dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(
            feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(
            config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        # if self.do_crop:
        #     if self.training:  # noqa: SIM108
        #         x = self.maybe_random_crop(x)
        #     else:
        #         # Always use center crop for eval.
        #         x = self.center_crop(x)
        # 确保输入数据类型正确
        if x.dtype != torch.float32:
            x = x.float()

        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


class DiffusionDepthEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: CustomDiffusionConfigWrapper):
        super().__init__()
        # # Set up optional preprocessing.
        # if config.crop_shape is not None:
        #     self.do_crop = True
        #     # Always use center crop for eval
        #     self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
        #     if config.crop_is_random:
        #         self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
        #     else:
        #         self.maybe_random_crop = self.center_crop
        # else:
        #     self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.depth_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        # self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        # change the first conv layer
        modules = list(backbone_model.children())[:-2]
        if isinstance(modules[0], nn.Conv2d):
            old_conv = modules[0]
            modules[0] = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            with torch.no_grad():
                modules[0].weight = nn.Parameter(
                    old_conv.weight.mean(dim=1, keepdim=True))

        self.backbone = nn.Sequential(*modules)

        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.image_features` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.image_features`.

        # Note: we have a check in the config class to make sure all images have the same shape.
        images_shape = next(iter(config.depth_features.values())).shape

        if config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        elif config.crop_shape is not None:
            if isinstance(list(config.crop_shape)[0], (list, tuple)):
                (x_start, x_end), (y_start, y_end) = config.crop_shape
                dummy_shape_h_w = (x_end-x_start, y_end-y_start)
            else:
                dummy_shape_h_w = config.crop_shape
        else:
            dummy_shape_h_w = images_shape[1:]

        # dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(
            feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(
            config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        # if self.do_crop:
        #     if self.training:  # noqa: SIM108
        #         x = self.maybe_random_crop(x)
        #     else:
        #         # Always use center crop for eval.
        #         x = self.center_crop(x)
        # 确保输入数据类型正确
        if x.dtype != torch.float32:
            x = x.float()

        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


"""
    state encoder
"""


class FeatureEncoder(nn.Module):
    """
    通用特征编码器
    将输入特征编码为指定维度的输出特征
    """

    def __init__(self, in_dim: int, out_dim: int = 128):
        """
        初始化特征编码器

        Args:
            in_dim: 输入特征维度
            out_dim: 输出特征维度
        """
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, in_dim]

        Returns:
            编码后的特征 [batch_size, out_dim]
        """
        if x.dim() == 2:
            return self.encoder(x)
        elif x.dim() == 3:
            B, T, D = x.shape            # x.shape = [64, 2, 14]
            x = x.view(B * T, D)         # => [128, 14]
            x = self.encoder(x)          # Linear + BatchNorm1d + ReLU
            x = x.view(B, T, -1)         # => [64, 2, out_dim]
            return x


"""
Transformer模块
包含用于扩散策略的Transformer网络实现
"""


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = query.shape[:2]

        # 线性变换
        Q = self.w_q(query).view(batch_size, seq_len,
                                 self.n_head, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1,
                               self.n_head, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1,
                                 self.n_head, self.d_k).transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # scores = scores.masked_fill(mask == 0, -1e9)
            # Fix
            # 更改为mask='-inf'
            scores = scores.masked_fill(mask == float('-inf'), float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 输出投影和残差连接
        output = self.w_out(context)
        return self.layer_norm(output + query)


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

    # def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    #     x = self.attention(x, x, x, mask)
    #     x = self.feed_forward(x)
    #     return x

    # Fix
    # 之前的代码没有mask参数，且attention和feed_forward都使用了x作为query、key和value
    # 修改为：兼容交叉注意力机制，attention使用action_emb作为query，key和value使用cond_emb
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention(query, key, value, mask)
        x = self.feed_forward(x)
        return x


class ConditionalTransformer(nn.Module):
    """条件Transformer网络"""

    def __init__(self,
                 action_dim: int,
                 cond_dim: int,
                 horizon: int = 64,
                 n_obs_steps: int = 4,
                 n_emb: int = 256,
                 n_head: int = 8,
                 n_layer: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.n_emb = n_emb

        # 时间步嵌入
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.time_mlp = nn.Sequential(
            nn.Linear(n_emb, n_emb * 2),
            nn.GELU(),
            nn.Linear(n_emb * 2, n_emb)
        )

        # 动作嵌入
        self.action_emb = nn.Linear(action_dim, n_emb)

        # 条件嵌入
        self.cond_emb = nn.Linear(cond_dim, n_emb)
        print(f"cond_dim: {cond_dim}, n_emb: {n_emb}")

        # 位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, 1000, n_emb) * 0.02)

        # Transformer层
        self.encoder = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.Mish(),
            nn.Linear(4 * n_emb, n_emb)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )

        # 输出层
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, action_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Mask
        T = horizon
        S = n_obs_steps+1

        mask = (torch.triu(torch.ones(T, T)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        self.register_buffer("mask", mask)

        t, s = torch.meshgrid(
            torch.arange(T),
            torch.arange(S),
            indexing='ij'
        )
        # add one dimension since time is the first token in cond
        mask = t >= (s-1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        self.register_buffer('memory_mask', mask)

    # Fix
    # 之前的代码用的是自注意力机制，且没有加mask，将action_emb， cond_emb， time_emb_expanded全部加起来
    # 修改为：交叉注意力机制，使用action_emb+time_emb_expanded作为query，cond_emb+time_emb_expanded作为key和value
    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor,
                global_cond: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            actions: 噪声动作序列 [B, T, action_dim]
            timesteps: 时间步 [B]
            global_cond: 条件特征 [B, cond_dim] 或 [B, T, cond_dim]

        Returns:
            预测的噪声 [B, T, action_dim]
        """
        B, T = actions.shape[:2]

        # 时间步嵌入
        time_emb = self.time_emb(timesteps)  # [B, n_emb]
        time_emb = self.time_mlp(time_emb).unsqueeze(1)   # [B, 1, n_emb]

        # 动作嵌入
        action_emb = self.action_emb(actions)  # [B, T, n_emb]

        # 条件嵌入
        if global_cond.dim() == 2:  # [B, To * cond_dim]
            if global_cond.shape[1] % self.cond_dim != 0:
                raise ValueError(
                    f"输入条件维度（To * cond_dim） {global_cond.shape[1]} 不能被cond_dim {self.cond_dim} 整除")
            To = global_cond.shape[1]//self.cond_dim
            cond_emb = self.cond_emb(
                global_cond.view(B, To, -1))  # [B, To, n_emb]
        else:  # [B, To, cond_dim]
            To = global_cond.shape[1]
            cond_emb = self.cond_emb(global_cond)  # [B, To, n_emb]

        # 添加时间步信息到每个时间步
        cond_emb = torch.cat([time_emb, cond_emb], dim=1)
        tc = cond_emb.shape[1]

        # 添加位置编码
        cond_emb = cond_emb + self.pos_emb[:, :tc, :]
        cond_emb = self.dropout(cond_emb)

        cond_emb = self.encoder(cond_emb)  # [B, To+1, n_emb]

        # 添加位置编码
        action_emb = action_emb + self.pos_emb[:, :T, :]
        action_emb = self.dropout(action_emb)

        x = self.decoder(
            tgt=action_emb,
            memory=cond_emb,
            tgt_mask=self.mask,
            memory_mask=self.memory_mask
        )

        # 输出
        x = self.ln_f(x)
        noise_pred = self.head(x)  # [B, T, action_dim]

        return noise_pred


class DiffusionTransformer(nn.Module):
    """扩散策略专用的Transformer"""

    def __init__(self, config: CustomDiffusionConfigWrapper):
        super().__init__()

        # 从配置获取参数
        action_dim = config.action_feature.shape[0]
        self.pred_horizon = config.horizon

        # 动态计算条件维度
        vision_dim = config.spatial_softmax_num_keypoints * \
            2 * len(config.rgb_image_features)
        if config.use_state_encoder:
            state_dim = config.state_feature_dim
        else:
            state_dim = config.robot_state_feature.shape[0]
        cond_dim = vision_dim + state_dim

        # Transformer参数
        n_emb = config.transformer_n_emb
        n_head = config.transformer_n_head
        n_layer = config.transformer_n_layer
        dropout = config.transformer_dropout

        # 观测步长
        n_obs_steps = config.n_obs_steps

        self.transformer = ConditionalTransformer(
            action_dim=action_dim,
            cond_dim=cond_dim,
            n_obs_steps=n_obs_steps,
            horizon=self.pred_horizon,
            n_emb=n_emb,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor,
                global_cond: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            sample: 噪声样本 [B*T, action_dim] 或 [B, T, action_dim]
            timestep: 时间步 [B*T] 或 [B]
            cond: 条件特征 [B*T, cond_dim] 或 [B, T, cond_dim]

        Returns:
            预测的噪声
        """

        # 通过Transformer
        print(sample.shape, timestep.shape, global_cond.shape)
        raise ValueError("stop!")
        noise_pred = self.transformer(sample, timestep, global_cond)

        return noise_pred
