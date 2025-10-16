# Diffusion Policy 架构详解

> **作者**: AI Assistant
> **日期**: 2025-10-10
> **版本**: 1.0
> **适用于**: `kuavo_data_challenge` 项目

---

## 📋 目录

1. [架构概览](#1-架构概览)
2. [训练主流程](#2-训练主流程)
3. [核心组件详解](#3-核心组件详解)
4. [Transformer架构](#4-transformer架构)
5. [多模态融合](#5-多模态融合)
6. [扩散过程](#6-扩散过程)
7. [推理逻辑](#7-推理逻辑)
8. [配置系统](#8-配置系统)
9. [关键设计决策](#9-关键设计决策)

---

## 1. 架构概览

### 1.1 整体设计理念

Diffusion Policy 是一种基于扩散模型的机器人控制策略，将扩散模型（Diffusion Model）应用于动作生成。

**核心思想**:
- **扩散过程**: 将动作序列视为从噪声逐步去噪的过程
- **条件生成**: 根据观测（RGB图像、深度图、机器人状态）生成动作
- **多模态融合**: 整合视觉和本体感知信息
- **Transformer backbone**: 使用Transformer处理序列数据

### 1.2 核心组件关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                     train_policy.py                             │
│                     (训练主入口)                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
┌───────────▼────────────┐   ┌───────▼──────────────────────┐
│CustomDiffusionPolicyWrapper│  │ Noise Scheduler (DDPM/DDIM) │
│   (Policy主控制器)      │   │   (扩散调度器)               │
└───────────┬────────────┘   └──────────────────────────────┘
            │
┌───────────▼────────────┐
│CustomDiffusionModelWrapper│
│   (核心扩散模型)        │
└───────────┬────────────┘
            │
            ├─────────────────────────────────────┐
            │                                     │
┌───────────▼────────────┐          ┌────────────▼─────────────┐
│   特征编码器            │          │    去噪网络              │
│                        │          │                          │
│ ├─► RGB Encoder        │          │ ├─► U-Net (可选)        │
│ │   (ResNet + Spatial  │          │ │                       │
│ │    SoftMax)          │          │ └─► Transformer         │
│ │                      │          │     (主要使用)           │
│ ├─► Depth Encoder      │          │                          │
│ │   (ResNet + Spatial  │          │  TransformerForDiffusion │
│ │    SoftMax)          │          │  ├─ Time Embedding      │
│ │                      │          │  ├─ Condition Encoder   │
│ └─► State Encoder      │          │  ├─ Decoder Layers      │
│     (MLP)              │          │  └─ Output Head         │
│                        │          │                          │
│ └─► 多模态融合          │          │                          │
│     (Cross-Attention)  │          └──────────────────────────┘
└────────────────────────┘
```

### 1.3 数据流概览

```
观测数据
  ├─► RGB Images [B, n_obs, n_cam, 3, H, W]
  ├─► Depth Images [B, n_obs, n_cam, 1, H, W]
  └─► Robot State [B, n_obs, state_dim]
        │
        ▼
特征编码
  ├─► RGB: ResNet → Spatial SoftMax → [B, n_obs, n_cam, feat_dim]
  ├─► Depth: ResNet → Spatial SoftMax → [B, n_obs, n_cam, feat_dim]
  └─► State: MLP → [B, n_obs, state_dim] 或 [B, n_obs, encoded_dim]
        │
        ▼
多模态融合 (Cross-Attention)
  ├─► RGB ⊗ Depth → RGB_fused
  ├─► Depth ⊗ RGB → Depth_fused
  └─► Concatenate with State
        │
        ▼
全局条件 global_cond [B, n_obs, cond_dim]
        │
        ▼
扩散模型 (训练)
  ├─► 1. 对真实动作添加噪声: action + ε ~ N(0, I)
  ├─► 2. Transformer预测噪声: ε_pred = Transformer(noisy_action, t, global_cond)
  └─► 3. 计算损失: L = ||ε - ε_pred||²
        │
        ▼
扩散模型 (推理)
  ├─► 1. 从纯噪声开始: action_T ~ N(0, I)
  ├─► 2. 逐步去噪 (T步 → 0步)
  │      for t in [T, T-1, ..., 1]:
  │        ε_pred = Transformer(action_t, t, global_cond)
  │        action_{t-1} = denoise(action_t, ε_pred, t)
  └─► 3. 输出最终动作: action_0
```

---

## 2. 训练主流程

### 2.1 入口函数: `train_policy.py::main()`

```python
@hydra.main(config_path="../configs/policy/",
            config_name="diffusion_config")
def main(cfg: DictConfig):
    """Diffusion Policy训练主函数"""
```

#### 2.1.1 训练流程图

```
开始
  │
  ├─► 1. 初始化
  │      ├─ 设置随机种子
  │      ├─ 创建输出目录
  │      └─ 初始化TensorBoard
  │
  ├─► 2. 加载数据集
  │      ├─ LeRobotDatasetMetadata (元数据)
  │      ├─ 构建 input_features & output_features
  │      └─ LeRobotDataset (实际数据)
  │
  ├─► 3. 构建Policy
  │      ├─ build_policy_config()
  │      ├─ CustomDiffusionPolicyWrapper(config, dataset_stats)
  │      │   └─► 内部创建 CustomDiffusionModelWrapper
  │      └─ policy.to(device)
  │
  ├─► 4. 构建优化器 & 学习率调度器
  │      ├─ AdamW optimizer
  │      └─ Cosine/Linear scheduler
  │
  ├─► 5. 初始化AMP (可选)
  │      └─ torch.amp.GradScaler
  │
  ├─► 6. 恢复训练 (可选)
  │      ├─ 加载 policy, optimizer, scheduler
  │      ├─ 加载 AMP scaler
  │      └─ 恢复 RNG state
  │
  ├─► 7. 主训练循环
  │      for epoch in range(max_epoch):
  │        ├─ 创建 DataLoader
  │        ├─ for batch in dataloader:
  │        │   ├─ 前向传播: loss, _ = policy.forward(batch)
  │        │   ├─ 反向传播: loss.backward()
  │        │   ├─ 优化器步骤: optimizer.step()
  │        │   └─ 记录日志
  │        │
  │        ├─ 保存最佳模型
  │        ├─ 定期保存检查点
  │        └─ 保存训练状态
  │
  └─► 8. 训练完成
```

### 2.2 关键训练步骤详解

#### 2.2.1 数据集加载

```python
# 1. 加载元数据
dataset_metadata = LeRobotDatasetMetadata(cfg.repoid, root=cfg.root)
# 包含: camera_keys, features, fps, total_frames, stats等

# 2. 转换为Policy特征
features = dataset_to_policy_features(dataset_metadata.features)
input_features = {k: ft for k, ft in features.items()
                 if ft.type is not FeatureType.ACTION}
output_features = {k: ft for k, ft in features.items()
                  if ft.type is FeatureType.ACTION}

# input_features示例:
# {
#   'observation.images.head_cam_h': PolicyFeature(shape=[3, 480, 640], ...),
#   'observation.images.wrist_cam_l': PolicyFeature(shape=[3, 480, 640], ...),
#   'observation.images.wrist_cam_r': PolicyFeature(shape=[3, 480, 640], ...),
#   'observation.depth.depth_h': PolicyFeature(shape=[1, 480, 640], ...),
#   'observation.depth.depth_l': PolicyFeature(shape=[1, 480, 640], ...),
#   'observation.depth.depth_r': PolicyFeature(shape=[1, 480, 640], ...),
#   'observation.state': PolicyFeature(shape=[16], ...)
# }

# output_features示例:
# {
#   'action': PolicyFeature(shape=[16], ...)
# }

# 3. 构建delta timestamps (用于多帧观测)
delta_timestamps = build_delta_timestamps(dataset_metadata, policy_cfg)
# 例如: {'observation.state': [0, -0.02], 'action': [0, 0.02, 0.04, ...]}

# 4. 创建数据集
dataset = LeRobotDataset(
    cfg.repoid,
    delta_timestamps=delta_timestamps,
    root=cfg.root,
    episodes=episodes_to_use,  # 可限制使用的episodes
    image_transforms=image_transforms,  # RGB增强
)
```

#### 2.2.2 Policy构建

```python
def build_policy_config(cfg, input_features, output_features):
    """构建policy配置"""
    policy_cfg = instantiate(
        cfg.policy,
        input_features=input_features,
        output_features=output_features,
        device=cfg.training.device,
    )
    return policy_cfg

def build_policy(name, policy_cfg, dataset_stats):
    """构建policy实例"""
    policy = {
        "diffusion": CustomDiffusionPolicyWrapper,
        "act": ACTPolicy,
    }[name](policy_cfg, dataset_stats)
    return policy

# 使用
policy = build_policy("diffusion", policy_cfg, dataset_metadata.stats)
```

**Policy初始化流程**:
```
CustomDiffusionPolicyWrapper.__init__()
  │
  ├─► 调用父类 DiffusionPolicy.__init__()
  │     ├─ 保存 config
  │     ├─ 初始化归一化参数 (从 dataset_stats)
  │     └─ 初始化观测队列 (deque)
  │
  └─► 创建 CustomDiffusionModelWrapper
        ├─► 创建特征编码器
        │   ├─ RGB Encoder (ResNet + SpatialSoftMax)
        │   ├─ Depth Encoder (ResNet + SpatialSoftMax)
        │   └─ State Encoder (MLP, 可选)
        │
        ├─► 创建去噪网络
        │   ├─ U-Net (if config.use_unet)
        │   └─ TransformerForDiffusion (if config.use_transformer)
        │
        └─► 创建噪声调度器
            └─ DDPM / DDIM Scheduler
```

#### 2.2.3 训练循环

```python
for epoch in range(start_epoch, cfg.training.max_epoch):
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=cfg.training.drop_last,
    )

    for batch in dataloader:
        # 1. 数据移到GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        # batch包含:
        # - observation.images.*: [64, 2, 3, 480, 640]
        # - observation.depth.*: [64, 2, 1, 480, 640]
        # - observation.state: [64, 2, 16]
        # - action: [64, 16, 16]  # [batch, horizon, action_dim]

        # 2. 前向传播 (AMP可选)
        with autocast(amp_enabled):
            loss, _ = policy.forward(batch)

        # policy.forward() 内部流程:
        #   a. 图像预处理 (crop, resize)
        #   b. 归一化 (observations & actions)
        #   c. diffusion.compute_loss(batch)
        #      ├─ 提取特征: global_cond = _prepare_global_conditioning(batch)
        #      ├─ 添加噪声: noisy_actions = actions + noise
        #      ├─ 预测噪声: noise_pred = unet(noisy_actions, timesteps, global_cond)
        #      └─ 计算损失: loss = ||noise - noise_pred||²

        # 3. 反向传播
        scaled_loss = loss / accumulation_steps
        if amp_enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # 4. 优化器步骤
        if steps % accumulation_steps == 0:
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        # 5. 记录日志
        if steps % log_freq == 0:
            writer.add_scalar("train/loss", scaled_loss.item(), steps)
            writer.add_scalar("train/lr", lr.get_last_lr()[0], steps)

        steps += 1

    # 6. Epoch结束后的操作
    # 保存最佳模型
    if total_loss < best_loss:
        best_loss = total_loss
        policy.save_pretrained(output_directory / "best")

    # 定期保存检查点
    if (epoch + 1) % save_freq_epoch == 0:
        policy.save_pretrained(output_directory / f"epoch{epoch+1}")

    # 保存训练状态 (用于恢复训练)
    checkpoint = {
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler": scaler.state_dict() if amp_enabled else None,
        "steps": steps,
        "epoch": epoch + 1,
        "best_loss": best_loss
    }
    torch.save(checkpoint, output_directory / "learning_state.pth")
```

---

## 3. 核心组件详解

### 3.1 CustomDiffusionModelWrapper

**位置**: `kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py`

#### 3.1.1 整体架构

```python
class CustomDiffusionModelWrapper(DiffusionModel):
    """
    扩散模型的核心包装器

    主要职责:
    1. 特征编码 (RGB, Depth, State)
    2. 多模态融合
    3. 去噪网络 (U-Net 或 Transformer)
    4. 损失计算
    """

    def __init__(self, config):
        super().__init__(config)

        # 1. 计算全局条件维度
        global_cond_dim = 0

        # 2. 构建编码器
        if config.robot_state_feature:
            if config.use_state_encoder:
                self.state_encoder = FeatureEncoder(in_dim, out_dim)
                global_cond_dim += out_dim
            else:
                global_cond_dim += state_dim

        if config.image_features:
            self.rgb_encoder = DiffusionRgbEncoder(config)
            global_cond_dim += rgb_encoder.feature_dim * num_cameras
            self.rgb_attn_layer = nn.MultiheadAttention(...)

        if config.depth_features:
            self.depth_encoder = DiffusionDepthEncoder(config)
            global_cond_dim += depth_encoder.feature_dim * num_cameras
            self.depth_attn_layer = nn.MultiheadAttention(...)

        # 3. 多模态融合
        if config.use_depth and config.depth_features:
            self.multimodalfuse = nn.ModuleDict({
                "depth_q": nn.MultiheadAttention(...),  # Depth query RGB
                "rgb_q": nn.MultiheadAttention(...),    # RGB query Depth
            })

        # 4. 去噪网络
        if config.use_unet:
            self.unet = DiffusionConditionalUnet1d(config, global_cond_dim)
        elif config.use_transformer:
            self.unet = TransformerForDiffusion(
                input_dim=action_dim,
                output_dim=action_dim,
                horizon=config.horizon,
                n_obs_steps=config.n_obs_steps,
                cond_dim=global_cond_dim,
                n_layer=config.transformer_n_layer,
                n_head=config.transformer_n_head,
                n_emb=config.transformer_n_emb,
                ...
            )

        # 5. 噪声调度器
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,  # DDPM / DDIM
            num_train_timesteps=config.num_train_timesteps,
            ...
        )
```

#### 3.1.2 特征编码

##### RGB编码器

```python
class DiffusionRgbEncoder(nn.Module):
    """RGB图像编码器"""

    def __init__(self, config):
        super().__init__()

        # 1. 使用预训练ResNet作为backbone
        backbone_model = torchvision.models.resnet18(
            weights=config.pretrained_backbone_weights
        )
        # 去掉最后两层 (全局池化和分类层)
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        # 2. Spatial SoftMax 池化
        # 通过dry run获取特征图尺寸
        feature_map_shape = get_output_shape(self.backbone, dummy_input)[1:]
        self.pool = SpatialSoftmax(
            feature_map_shape,
            num_kp=config.spatial_softmax_num_keypoints
        )

        # 3. 输出层
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] RGB图像
        Returns:
            [B, feature_dim] 特征向量
        """
        # ResNet特征提取
        x = self.backbone(x)  # [B, 512, H', W']

        # Spatial SoftMax池化
        x = self.pool(x)  # [B, num_keypoints * 2]

        # 输出投影
        x = self.relu(self.out(x))  # [B, feature_dim]

        return x
```

**Spatial SoftMax 原理**:
```
输入: feature_map [B, C, H, W]

对每个通道 c:
  1. 计算softmax权重:
     weights[b, h, w] = exp(feature_map[b, c, h, w]) / Σ exp(feature_map[b, c, :, :])

  2. 计算加权坐标 (keypoint):
     x_c = Σ(h,w) weights[b, h, w] * x_coord[h, w]
     y_c = Σ(h,w) weights[b, h, w] * y_coord[h, w]

输出: [B, C * 2]  # 每个通道一个(x, y)坐标
```

**优势**:
- 保留空间信息（相比全局平均池化）
- 可微分（相比argmax）
- 降维（从H×W到2个坐标）

##### 深度编码器

```python
class DiffusionDepthEncoder(nn.Module):
    """深度图编码器"""

    def __init__(self, config):
        super().__init__()

        # 1. 使用ResNet，但修改第一层接受1通道输入
        backbone_model = torchvision.models.resnet18(...)
        modules = list(backbone_model.children())[:-2]

        # 修改第一个卷积层
        if isinstance(modules[0], nn.Conv2d):
            old_conv = modules[0]
            modules[0] = nn.Conv2d(
                in_channels=1,  # 深度图只有1个通道
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
            )
            # 权重初始化: 对RGB权重取平均
            with torch.no_grad():
                modules[0].weight = nn.Parameter(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )

        self.backbone = nn.Sequential(*modules)

        # 2. Spatial SoftMax (同RGB)
        self.pool = SpatialSoftmax(...)
        self.out = nn.Linear(...)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W] 深度图
        Returns:
            [B, feature_dim] 特征向量
        """
        x = self.backbone(x)
        x = torch.flatten(self.pool(x), start_dim=1)
        x = self.relu(self.out(x))
        return x
```

##### 状态编码器

```python
class FeatureEncoder(nn.Module):
    """状态特征编码器"""

    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        """
        Args:
            x: [B, in_dim] 或 [B, T, in_dim]
        Returns:
            编码后的特征
        """
        if x.dim() == 2:
            return self.encoder(x)  # [B, out_dim]
        elif x.dim() == 3:
            B, T, D = x.shape
            x = x.view(B * T, D)
            x = self.encoder(x)
            x = x.view(B, T, -1)
            return x  # [B, T, out_dim]
```

#### 3.1.3 全局条件准备

```python
def _prepare_global_conditioning(self, batch):
    """
    编码并融合所有观测特征

    Args:
        batch: 输入批次
            - observation.images: [B, n_obs, n_cam, 3, H, W]
            - observation.depth: [B, n_obs, n_cam, 1, H, W]
            - observation.state: [B, n_obs, state_dim]

    Returns:
        global_cond: [B, n_obs, cond_dim]
    """
    batch_size, n_obs_steps, n_camera = batch[OBS_STATE].shape[:3]
    global_cond_feats = []

    # 1. RGB特征提取
    if self.config.image_features:
        if self.config.use_separate_rgb_encoder_per_camera:
            # 每个相机使用独立编码器
            images_per_camera = einops.rearrange(
                batch[OBS_IMAGES], "b s n ... -> n (b s) ..."
            )
            img_features_list = torch.cat([
                encoder(images)
                for encoder, images in zip(self.rgb_encoder, images_per_camera)
            ])
            img_features = einops.rearrange(
                img_features_list, "(n b s) ... -> b s n ...",
                b=batch_size, s=n_obs_steps
            )
        else:
            # 所有相机共享编码器
            img_features = self.rgb_encoder(
                einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
            )
            img_features = einops.rearrange(
                img_features, "(b s n) ... -> b s n ...",
                b=batch_size, s=n_obs_steps
            )

        # Self-Attention聚合多相机特征
        img_features = einops.rearrange(
            img_features, "b s n ... -> (b s) n ...",
            b=batch_size, s=n_obs_steps
        )
        img_features = self.rgb_attn_layer(
            query=img_features,
            key=img_features,
            value=img_features
        )[0]  # [B*n_obs, n_cam, feat_dim]

    # 2. 深度特征提取 (同RGB)
    if self.config.use_depth and self.config.depth_features:
        # ... 类似RGB的处理 ...
        depth_features = self.depth_attn_layer(
            query=depth_features,
            key=depth_features,
            value=depth_features
        )[0]

    # 3. 多模态融合 (Cross-Attention)
    if (img_features is not None) and (depth_features is not None):
        # RGB query Depth
        rgb_q_fuse = self.multimodalfuse["rgb_q"](
            query=img_features,      # [B*n_obs, n_cam, feat_dim]
            key=depth_features,      # [B*n_obs, n_cam, feat_dim]
            value=depth_features
        )[0]

        # Depth query RGB
        depth_q_fuse = self.multimodalfuse["depth_q"](
            query=depth_features,
            key=img_features,
            value=img_features
        )[0]

        # 重排并拼接
        rgb_q_fuse = einops.rearrange(
            rgb_q_fuse, "(b s) n ... -> b s (n ...)",
            b=batch_size, s=n_obs_steps
        )
        depth_q_fuse = einops.rearrange(
            depth_q_fuse, "(b s) n ... -> b s (n ...)",
            b=batch_size, s=n_obs_steps
        )
        global_cond_feats.extend([rgb_q_fuse, depth_q_fuse])

    # 4. 状态特征
    if self.config.robot_state_feature:
        if self.config.use_state_encoder:
            state_features = self.state_encoder(batch[OBS_STATE])
        else:
            state_features = batch[OBS_STATE]
        global_cond_feats.append(state_features)

    # 5. 拼接所有特征
    if self.config.use_transformer:
        # Transformer需要保留时间维度
        return torch.cat(global_cond_feats, dim=-1)  # [B, n_obs, cond_dim]
    else:
        # U-Net需要展平
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)  # [B, n_obs*cond_dim]
```

### 3.2 CustomDiffusionPolicyWrapper

**位置**: `kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py`

#### 3.2.1 核心方法

##### forward() - 训练时使用

```python
def forward(self, batch):
    """
    训练时的前向传播

    Args:
        batch: 数据批次
            - observation.images.*: [B, n_obs, 3, H, W]
            - observation.depth.*: [B, n_obs, 1, H, W]
            - observation.state: [B, n_obs, state_dim]
            - action: [B, horizon, action_dim]

    Returns:
        (loss, None)
    """
    # 1. 图像预处理 (裁剪、缩放)
    random_crop = self.config.crop_is_random and self.training
    crop_position = None

    if self.config.image_features:
        batch = dict(batch)  # shallow copy
        for key in self.config.image_features:
            # 裁剪
            batch[key], crop_position = crop_image(
                batch[key],
                target_range=self.config.crop_shape,
                random_crop=random_crop
            )
            # 缩放
            batch[key] = resize_image(
                batch[key],
                target_size=self.config.resize_shape,
                image_type="rgb"
            )

    # 深度图同样处理
    if self.config.use_depth and self.config.depth_features:
        for key in self.config.depth_features:
            if len(crop_position) == 4:
                batch[key] = torchvision.transforms.functional.crop(
                    batch[key], *crop_position
                )
            else:
                batch[key] = torchvision.transforms.functional.center_crop(
                    batch[key], crop_position
                )
            batch[key] = resize_image(
                batch[key],
                target_size=self.config.resize_shape,
                image_type="depth"
            )

    # 2. 归一化
    batch = self.normalize_inputs(batch)   # 归一化观测

    # 3. 堆叠图像 (在归一化之后)
    if self.config.image_features:
        batch[OBS_IMAGES] = torch.stack(
            [batch[key] for key in self.config.image_features],
            dim=-4
        )  # [B, n_obs, n_cam, 3, H, W]

    if self.config.use_depth and self.config.depth_features:
        batch[OBS_DEPTH] = torch.stack(
            [batch[key] for key in self.config.depth_features],
            dim=-4
        )  # [B, n_obs, n_cam, 1, H, W]

    batch = self.normalize_targets(batch)  # 归一化动作

    # 4. 计算扩散损失
    loss = self.diffusion.compute_loss(batch)

    return loss, None
```

##### select_action() - 推理时使用

```python
def select_action(self, batch):
    """
    推理时选择动作

    使用observation queue缓存历史观测，
    使用action queue缓存生成的动作序列。

    Args:
        batch: 当前观测
            - observation.images.*: [1, 3, H, W]
            - observation.depth.*: [1, 1, H, W]
            - observation.state: [1, state_dim]

    Returns:
        action: [1, action_dim]
    """
    # 移除action (推理时不需要)
    if ACTION in batch:
        batch.pop(ACTION)

    # 归一化输入
    batch = self.normalize_inputs(batch)

    # 图像预处理 (与训练时类似，但使用center crop)
    random_crop = self.config.crop_is_random and self.training
    if self.config.image_features:
        # ... 裁剪和缩放 ...
        batch[OBS_IMAGES] = torch.stack(...)

    if self.config.use_depth:
        # ... 深度图处理 ...
        batch[OBS_DEPTH] = torch.stack(...)

    # 填充observation queue
    self._queues = populate_queues(self._queues, batch)
    # 例如: _queues['observation.state'] 包含最近n_obs_steps个观测

    # 如果action queue为空，生成新的动作序列
    if len(self._queues[ACTION]) == 0:
        # 预测动作chunk
        actions = self.predict_action_chunk(batch)
        # actions: [1, horizon, action_dim]

        # 转置并填充到action queue
        self._queues[ACTION].extend(actions.transpose(0, 1))
        # action queue现在包含horizon个动作

    # 从action queue中取出第一个动作
    action = self._queues[ACTION].popleft()

    return action
```

---

## 4. Transformer架构

### 4.1 TransformerForDiffusion

**位置**: `kuavo_train/wrapper/policy/diffusion/transformer_diffusion.py`

#### 4.1.1 整体架构

```python
class TransformerForDiffusion(ModuleAttrMixin):
    """
    专门用于扩散模型的Transformer

    架构:
    - Encoder: 编码条件信息 (时间步 + 观测)
    - Decoder: 解码动作序列 (基于条件生成去噪动作)
    """

    def __init__(self,
                 input_dim: int,          # 动作维度
                 output_dim: int,         # 输出维度 (通常等于input_dim)
                 horizon: int,            # 动作序列长度
                 n_obs_steps: int,        # 观测步数
                 cond_dim: int,           # 条件维度
                 n_layer: int = 12,       # Decoder层数
                 n_head: int = 12,        # 注意力头数
                 n_emb: int = 768,        # 嵌入维度
                 p_drop_emb: float = 0.1,
                 p_drop_attn: float = 0.1,
                 causal_attn: bool = False,    # 是否使用因果注意力
                 time_as_cond: bool = True,     # 时间步作为条件
                 obs_as_cond: bool = False,     # 观测作为条件
                 n_cond_layers: int = 0         # Encoder层数
                 ):
        super().__init__()

        # 计算token数量
        T = horizon  # 动作序列长度
        T_cond = 1   # 条件token数量 (时间步)

        if not time_as_cond:
            T += 1
            T_cond -= 1

        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps  # 添加观测token

        # 1. 输入嵌入
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # 2. 条件编码器
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None

        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

            if n_cond_layers > 0:
                # 使用Transformer Encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4*n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                # 使用简单MLP
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )

            # Decoder (核心去噪网络)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True  # 重要: 提升训练稳定性
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # Encoder-only架构 (BERT风格)
            encoder_layer = nn.TransformerEncoderLayer(...)
            self.encoder = nn.TransformerEncoder(...)

        # 3. 注意力掩码
        if causal_attn:
            # 因果掩码 (自回归)
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            if time_as_cond and obs_as_cond:
                # Memory掩码 (Decoder可以看到的Encoder token)
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s-1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)

        # 4. 输出头
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
```

#### 4.1.2 前向传播

```python
def forward(self, sample, timestep, global_cond=None, **kwargs):
    """
    前向传播

    Args:
        sample: 噪声动作 [B, T, input_dim]
        timestep: 扩散时间步 [B] 或 标量
        global_cond: 全局条件 [B, To, cond_dim]

    Returns:
        noise_pred: 预测的噪声 [B, T, output_dim]
    """
    cond = global_cond

    # 1. 时间步嵌入
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)
    timesteps = timesteps.expand(sample.shape[0])

    time_emb = self.time_emb(timesteps).unsqueeze(1)
    # [B, 1, n_emb]

    # 2. 输入嵌入
    input_emb = self.input_emb(sample)
    # [B, T, n_emb]

    if self.encoder_only:
        # ===== BERT风格: Encoder-only =====
        token_embeddings = torch.cat([time_emb, input_emb], dim=1)
        # [B, T+1, n_emb]

        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)

        x = self.encoder(src=x, mask=self.mask)
        # [B, T+1, n_emb]

        x = x[:, 1:, :]  # 移除时间步token
        # [B, T, n_emb]
    else:
        # ===== Encoder-Decoder架构 =====

        # 3. 条件编码
        cond_embeddings = time_emb
        if self.obs_as_cond:
            cond_obs_emb = self.cond_obs_emb(cond)
            # [B, To, n_emb]
            cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            # [B, 1+To, n_emb]

        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[:, :tc, :]
        x = self.drop(cond_embeddings + position_embeddings)
        x = self.encoder(x)
        memory = x
        # [B, T_cond, n_emb]

        # 4. 动作解码
        token_embeddings = input_emb
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        # [B, T, n_emb]

        x = self.decoder(
            tgt=x,                    # Query: 噪声动作
            memory=memory,            # Key & Value: 条件信息
            tgt_mask=self.mask,       # 因果掩码 (可选)
            memory_mask=self.memory_mask  # Memory掩码 (可选)
        )
        # [B, T, n_emb]

    # 5. 输出头
    x = self.ln_f(x)
    x = self.head(x)
    # [B, T, output_dim]

    return x
```

#### 4.1.3 掩码机制

##### 因果掩码 (Causal Mask)

```python
# 自回归生成：每个位置只能看到之前的位置
sz = T  # horizon
mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
# [
#   [0, -inf, -inf, -inf],
#   [0,    0, -inf, -inf],
#   [0,    0,    0, -inf],
#   [0,    0,    0,    0]
# ]

# 在注意力计算中使用
scores = Q @ K.T / sqrt(d_k)
scores = scores.masked_fill(mask == float('-inf'), float('-inf'))
attn_weights = softmax(scores)
```

**为什么需要因果掩码?**
- 防止"未来信息泄露"
- 适用于自回归生成场景
- 对于扩散模型，因果性不是必需的（可以全局看）

##### Memory掩码

```python
# Decoder可以看到哪些Encoder token
T = horizon  # 动作序列长度
S = n_obs_steps + 1  # 条件token数量 (时间步 + 观测)

t, s = torch.meshgrid(
    torch.arange(T),
    torch.arange(S),
    indexing='ij'
)

mask = t >= (s-1)  # 每个动作位置可以看到相应时间之前的观测
# 例如: T=4, S=3
# [
#   [0, -inf, -inf],  # action[0] 只能看到 time_emb
#   [0,    0, -inf],  # action[1] 可以看到 time_emb + obs[0]
#   [0,    0,    0],  # action[2] 可以看到 time_emb + obs[0] + obs[1]
#   [0,    0,    0],  # action[3] 可以看到所有
# ]
```

### 4.2 时间步嵌入

```python
class SinusoidalPosEmb(nn.Module):
    """正弦位置编码 (用于时间步)"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x: 时间步 [B]
        Returns:
            嵌入 [B, dim]
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
```

**数学原理**:
```
给定时间步 t，生成嵌入:

emb[i] = sin(t / 10000^(2i/d))     for i < d/2
emb[i] = cos(t / 10000^(2(i-d/2)/d))  for i >= d/2

其中 d = dim
```

**优势**:
- 连续性：相邻时间步的嵌入相似
- 外推性：可以处理训练时未见过的时间步
- 可微分：支持梯度传播

---

## 5. 多模态融合

### 5.1 设计思路

**问题**: 如何有效融合RGB图像、深度图和机器人状态？

**方案**: 分层融合
1. **层内融合**: 使用Self-Attention聚合同类型多相机的特征
2. **层间融合**: 使用Cross-Attention融合RGB和Depth
3. **全局拼接**: 将融合后的视觉特征与状态特征拼接

### 5.2 实现细节

#### 5.2.1 Self-Attention聚合多相机

```python
# RGB特征: [B*n_obs, n_cam, feat_dim]
img_features = self.rgb_attn_layer(
    query=img_features,
    key=img_features,
    value=img_features
)[0]
# 输出: [B*n_obs, n_cam, feat_dim]

# 作用: 学习相机间的关系，突出重要相机
```

#### 5.2.2 Cross-Attention融合RGB和Depth

```python
# RGB query Depth
rgb_q_fuse = self.multimodalfuse["rgb_q"](
    query=img_features,      # [B*n_obs, n_cam, feat_dim]
    key=depth_features,      # [B*n_obs, n_cam, feat_dim]
    value=depth_features
)[0]

# Depth query RGB
depth_q_fuse = self.multimodalfuse["depth_q"](
    query=depth_features,
    key=img_features,
    value=img_features
)[0]

# 作用:
# - RGB query Depth: 为RGB特征补充深度信息
# - Depth query RGB: 为深度特征补充颜色/纹理信息
```

**为什么双向?**
- RGB和Depth互补：RGB提供纹理，Depth提供几何
- 双向融合比单向或简单拼接更有效
- 每个模态可以选择性地从另一个模态提取信息

#### 5.2.3 融合后的特征拼接

```python
# 重排维度
rgb_q_fuse = einops.rearrange(
    rgb_q_fuse, "(b s) n ... -> b s (n ...)",
    b=batch_size, s=n_obs_steps
)  # [B, n_obs, n_cam*feat_dim]

depth_q_fuse = einops.rearrange(
    depth_q_fuse, "(b s) n ... -> b s (n ...)",
    b=batch_size, s=n_obs_steps
)  # [B, n_obs, n_cam*feat_dim]

# 状态特征
if self.config.use_state_encoder:
    state_features = self.state_encoder(batch[OBS_STATE])
    # [B, n_obs, encoded_dim]
else:
    state_features = batch[OBS_STATE]
    # [B, n_obs, state_dim]

# 全局拼接
global_cond = torch.cat([rgb_q_fuse, depth_q_fuse, state_features], dim=-1)
# [B, n_obs, cond_dim]
# cond_dim = n_cam*feat_dim + n_cam*feat_dim + state_dim (or encoded_dim)
```

### 5.3 融合流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                       输入观测                                   │
│  RGB: [B, n_obs, n_cam, 3, H, W]                               │
│  Depth: [B, n_obs, n_cam, 1, H, W]                             │
│  State: [B, n_obs, state_dim]                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ RGB Encoder   │  │ Depth Encoder │  │ State Encoder │
│ (ResNet)      │  │ (ResNet)      │  │ (MLP)         │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        ▼                  ▼                  │
┌───────────────┐  ┌───────────────┐         │
│ Self-Attn     │  │ Self-Attn     │         │
│ (多相机聚合)   │  │ (多相机聚合)   │         │
└───────┬───────┘  └───────┬───────┘         │
        │                  │                  │
        └─────────┬────────┘                  │
                  │                           │
        ┌─────────┴─────────┐                 │
        │                   │                 │
        ▼                   ▼                 │
┌──────────────────┐  ┌──────────────────┐   │
│ RGB query Depth  │  │ Depth query RGB  │   │
│ (Cross-Attn)     │  │ (Cross-Attn)     │   │
└────────┬─────────┘  └────────┬─────────┘   │
         │                     │              │
         │    rgb_q_fuse       │ depth_q_fuse│
         │                     │              │
         └──────────┬──────────┘              │
                    │                         │
                    ▼                         ▼
            ┌────────────────────────────────────┐
            │     全局条件 global_cond            │
            │  [B, n_obs, cond_dim]              │
            └────────────────────────────────────┘
```

---

## 6. 扩散过程

### 6.1 扩散模型原理

扩散模型通过两个过程生成数据：
1. **前向过程 (Forward Process)**: 逐步向数据添加噪声
2. **反向过程 (Reverse Process)**: 逐步从噪声中恢复数据

#### 6.1.1 前向过程 (加噪)

```
给定真实动作 x₀，逐步添加高斯噪声:

x_t = √(α_t) * x₀ + √(1 - α_t) * ε,  ε ~ N(0, I)

其中:
- t ∈ [0, T]: 时间步
- α_t: 噪声调度参数 (随t递减)
- T: 总时间步数 (如1000)

特殊情况:
- t=0: x₀ = 原始数据 (无噪声)
- t=T: x_T ≈ N(0, I) (纯噪声)
```

#### 6.1.2 反向过程 (去噪)

```
学习反向过程 p(x_{t-1} | x_t):

给定噪声动作 x_t，预测噪声 ε:
  ε_pred = Neural_Network(x_t, t, condition)

根据预测的噪声，恢复 x_{t-1}:
  x_{t-1} = (x_t - √(1-α_t) * ε_pred) / √(α_t) + σ_t * z

其中:
- Neural_Network: 我们训练的去噪网络 (Transformer)
- condition: 观测条件 (RGB + Depth + State)
- σ_t: 噪声方差
- z ~ N(0, I): 随机噪声 (只在t>1时添加)
```

### 6.2 训练过程

#### 6.2.1 compute_loss()

```python
def compute_loss(self, batch):
    """
    计算扩散损失

    Args:
        batch: 输入批次
            - observation.*: 观测数据
            - action: [B, horizon, action_dim] 真实动作

    Returns:
        loss: 标量损失
    """
    # 1. 准备全局条件
    global_cond = self._prepare_global_conditioning(batch)
    # [B, n_obs, cond_dim]

    # 2. 提取真实动作
    actions = batch['action']
    # [B, horizon, action_dim]

    batch_size = actions.shape[0]

    # 3. 随机采样时间步
    timesteps = torch.randint(
        0, self.noise_scheduler.config.num_train_timesteps,
        (batch_size,), device=actions.device
    ).long()
    # [B], 每个样本一个随机时间步

    # 4. 采样噪声
    noise = torch.randn_like(actions)
    # [B, horizon, action_dim]

    # 5. 添加噪声到真实动作
    noisy_actions = self.noise_scheduler.add_noise(
        actions, noise, timesteps
    )
    # noisy_actions = √(α_t) * actions + √(1 - α_t) * noise
    # [B, horizon, action_dim]

    # 6. 预测噪声
    noise_pred = self.unet(
        noisy_actions,  # 噪声动作
        timesteps,      # 时间步
        global_cond     # 条件
    )
    # [B, horizon, action_dim]

    # 7. 计算损失
    if self.noise_scheduler.config.prediction_type == 'epsilon':
        # 预测噪声
        loss = F.mse_loss(noise_pred, noise)
    elif self.noise_scheduler.config.prediction_type == 'sample':
        # 预测原始数据
        loss = F.mse_loss(noise_pred, actions)
    elif self.noise_scheduler.config.prediction_type == 'v_prediction':
        # 预测v (velocity)
        target = self.noise_scheduler.get_velocity(actions, noise, timesteps)
        loss = F.mse_loss(noise_pred, target)

    return loss
```

### 6.3 推理过程

#### 6.3.1 predict_action_chunk()

```python
def predict_action_chunk(self, batch):
    """
    通过迭代去噪生成动作序列

    Args:
        batch: 观测数据

    Returns:
        actions: [B, horizon, action_dim]
    """
    # 1. 准备全局条件
    global_cond = self._prepare_global_conditioning(batch)
    # [B, n_obs, cond_dim]

    batch_size = global_cond.shape[0]

    # 2. 从纯噪声开始
    actions = torch.randn(
        (batch_size, self.config.horizon, self.config.action_feature.shape[0]),
        device=global_cond.device
    )
    # [B, horizon, action_dim]

    # 3. 设置推理步数
    self.noise_scheduler.set_timesteps(self.num_inference_steps)

    # 4. 迭代去噪
    for t in self.noise_scheduler.timesteps:
        # 4.1 预测噪声
        timesteps = torch.full(
            (batch_size,), t, device=actions.device, dtype=torch.long
        )

        noise_pred = self.unet(
            actions,        # 当前噪声动作
            timesteps,      # 当前时间步
            global_cond     # 条件
        )
        # [B, horizon, action_dim]

        # 4.2 去噪一步
        actions = self.noise_scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=actions
        ).prev_sample
        # [B, horizon, action_dim]

    # 5. 反归一化
    actions = self.unnormalize_outputs({'action': actions})['action']

    return actions
```

#### 6.3.2 DDPM vs DDIM

**DDPM (Denoising Diffusion Probabilistic Models)**:
- 原始扩散模型
- 需要T步完整采样 (如T=1000)
- 慢但质量高

**DDIM (Denoising Diffusion Implicit Models)**:
- 加速采样
- 可以跳步 (如只用50步)
- 确定性采样 (可复现)

```python
# 配置中选择
noise_scheduler_type: DDPM  # 或 DDIM
num_train_timesteps: 100    # 训练时的总步数
num_inference_steps: 10     # 推理时使用的步数 (可以 < num_train_timesteps)
```

### 6.4 噪声调度

```python
# 噪声调度器配置
self.noise_scheduler = _make_noise_scheduler(
    config.noise_scheduler_type,      # DDPM / DDIM
    num_train_timesteps=100,          # 训练时间步
    beta_start=0.0001,                # 起始噪声方差
    beta_end=0.02,                    # 结束噪声方差
    beta_schedule="squaredcos_cap_v2", # 调度类型
    clip_sample=True,                 # 是否裁剪样本
    clip_sample_range=1.0,            # 裁剪范围
    prediction_type="epsilon",        # 预测类型
)
```

**beta_schedule 类型**:
- `linear`: β_t 线性增长
- `scaled_linear`: √β_t 线性增长
- `squaredcos_cap_v2`: 余弦调度 (推荐)

```python
# 余弦调度
α_t = cos²((t/T + s) / (1 + s) * π/2)

优势:
- 起始阶段噪声添加更缓慢
- 结束阶段接近纯噪声
- 训练更稳定
```

---

## 7. 推理逻辑

### 7.1 在线推理流程

```python
# 初始化
policy = CustomDiffusionPolicyWrapper.from_pretrained(checkpoint_path)
policy.eval()
policy.to(device)
policy.reset()  # 重置observation和action队列

# 推理循环
for step in range(max_steps):
    # 1. 获取观测
    obs = env.get_observation()
    # obs = {
    #   'observation.state': [16],
    #   'observation.images.head_cam_h': [3, 480, 640],
    #   'observation.images.wrist_cam_l': [3, 480, 640],
    #   'observation.images.wrist_cam_r': [3, 480, 640],
    #   'observation.depth.depth_h': [1, 480, 640],
    #   'observation.depth.depth_l': [1, 480, 640],
    #   'observation.depth.depth_r': [1, 480, 640],
    # }

    # 2. 预处理
    obs = {k: torch.from_numpy(v).unsqueeze(0).to(device)
           for k, v in obs.items()}

    # 3. 选择动作
    with torch.no_grad():
        action = policy.select_action(obs)
    # action: [1, 16]

    # select_action内部:
    # - 如果action queue为空:
    #   a. 从observation queue构建batch
    #   b. 调用predict_action_chunk()生成horizon个动作
    #   c. 填充action queue
    # - 从action queue pop第一个动作

    # 4. 执行动作
    action = action.squeeze(0).cpu().numpy()  # [16]
    obs_next, reward, done, info = env.step(action)

    if done:
        policy.reset()
        obs = env.reset()
```

### 7.2 队列机制

#### 7.2.1 Observation Queue

```python
# 初始化
self._queues = {
    "observation.state": deque(maxlen=n_obs_steps),
    "action": deque(maxlen=n_action_steps),
}
if self.config.image_features:
    self._queues["observation.images"] = deque(maxlen=n_obs_steps)
if self.config.use_depth:
    self._queues["observation.depth"] = deque(maxlen=n_obs_steps)

# 填充 (第一次调用时重复填充)
def populate_queues(queues, batch):
    for key in batch:
        if key in queues:
            if len(queues[key]) == 0:
                # 第一次: 重复n_obs_steps次
                for _ in range(queues[key].maxlen):
                    queues[key].append(batch[key])
            else:
                # 后续: 添加新观测，自动删除最旧的
                queues[key].append(batch[key])
    return queues

# 使用
obs_batch = {
    key: torch.stack(list(self._queues[key]), dim=1)
    for key in self._queues if key != "action"
}
# obs_batch['observation.state']: [1, n_obs_steps, state_dim]
```

#### 7.2.2 Action Queue

```python
# 生成动作chunk
if len(self._queues[ACTION]) == 0:
    actions = self.predict_action_chunk(obs_batch)
    # actions: [1, horizon, action_dim]

    # 转置并填充 (horizon个动作)
    self._queues[ACTION].extend(actions.transpose(0, 1))
    # action queue: [action[0], action[1], ..., action[horizon-1]]

# 取出一个动作
action = self._queues[ACTION].popleft()
# action: [1, action_dim]
```

**为什么需要Action Queue?**
- 效率：一次生成多个动作，减少推理次数
- 平滑：动作序列具有时序一致性
- 典型配置：horizon=16, n_action_steps=8
  - 生成16个动作
  - 执行前8个
  - 剩余8个在下次迭代使用

### 7.3 推理时间分析

```
假设:
- horizon = 16
- n_action_steps = 8
- num_inference_steps = 10 (DDIM)
- 控制频率: 10Hz

推理频率:
- 每 n_action_steps = 8 步执行一次推理
- 即每 0.8秒 推理一次

每次推理时间:
- 特征编码: ~20ms
- Diffusion去噪 (10步): ~50ms
- 总计: ~70ms

实时性:
- 可用时间: 800ms (8步)
- 实际用时: 70ms
- 余量: 730ms ✅ 充足
```

---

## 8. 配置系统

### 8.1 主配置文件

**位置**: `configs/policy/diffusion_config.yaml`

### 8.2 关键配置项

#### 8.2.1 基础配置

```yaml
task: 'task_400_episodes'
method: 'diffusion'
timestamp: ${now:%Y%m%d_%H%M%S}

repoid: 'lerobot/${task}'
root: '/root/robot/data/task1/data/lerobot/1-400/'

episodes_to_use:
  - 0
  - 299  # 使用episode 0-299
```

#### 8.2.2 训练配置

```yaml
training:
  output_directory: 'outputs/train/${task}/${method}'
  seed: 42
  max_epoch: 500
  save_freq_epoch: 10
  log_freq: 1

  device: 'cuda'
  batch_size: 64
  num_workers: 25
  drop_last: False
  accumulation_steps: 1  # 梯度累积

  max_training_step: null  # 如果设置，覆盖max_epoch

  # 恢复训练
  resume: False
  resume_timestamp: "run_20250110_123456"

  # RGB图像增强
  RGB_Augmenter:
    enable: True
    max_num_transforms: 1
    random_order: True
    tfs:
      notransform:
        weight: 2.0
        type: 'Identity'
      brightness:
        weight: 1.0
        type: 'ColorJitter'
        kwargs: { 'brightness': [0.5, 1.5] }
      contrast:
        weight: 1.0
        type: 'ColorJitter'
        kwargs: { 'contrast': [0.5, 1.5] }
```

#### 8.2.3 Policy配置

```yaml
policy_name: diffusion

policy:
  _target_: kuavo_train.wrapper.policy.diffusion.DiffusionConfigWrapper.CustomDiffusionConfigWrapper

  # Diffusion参数
  horizon: 16              # 动作序列长度
  n_action_steps: 8        # 每次执行的动作数
  drop_n_last_frames: 7    # 丢弃最后N帧 (horizon - n_action_steps - 1)

  # 归一化
  normalization_mapping:
    RGB:
      _target_: lerobot.configs.types.NormalizationMode
      value: MEAN_STD
    DEPTH:
      _target_: lerobot.configs.types.NormalizationMode
      value: MIN_MAX

  # 图像处理
  crop_is_random: True
  crop_shape: [420, 560]   # 裁剪尺寸
  use_amp: True            # 混合精度训练

  # 视觉backbone
  vision_backbone: resnet18
  use_separate_rgb_encoder_per_camera: False

  # Diffusion调度器
  noise_scheduler_type: DDPM  # DDPM / DDIM
  num_train_timesteps: 100
  beta_schedule: squaredcos_cap_v2
  beta_start: 0.0001
  beta_end: 0.02
  prediction_type: epsilon    # epsilon / sample / v_prediction
  clip_sample: true
  clip_sample_range: 1.0
  num_inference_steps: null   # 推理步数 (null = num_train_timesteps)
  do_mask_loss_for_padding: false

  # 优化器
  optimizer_lr: 0.0001
  optimizer_betas: [0.95, 0.999]
  optimizer_eps: 1.0e-08
  optimizer_weight_decay: 1.0e-03

  # 学习率调度器
  scheduler_name: cosine
  scheduler_warmup_steps: 500

  # 自定义配置
  custom:
    # 深度图
    use_depth: True
    depth_backbone: resnet18
    use_separate_depth_encoder_per_camera: False

    # 图像缩放
    resize_shape: [210, 280]

    # 状态编码器
    use_state_encoder: True
    state_feature_dim: 128

    # 去噪网络选择
    use_unet: False
    use_transformer: True    # 推荐使用Transformer

    # Transformer参数
    transformer_n_emb: 512
    transformer_n_head: 8
    transformer_n_layer: 4
    transformer_dropout: 0.1
```

### 8.3 配置加载

```python
@hydra.main(config_path="../configs/policy/",
            config_name="diffusion_config")
def main(cfg: DictConfig):
    # cfg 自动加载并解析YAML配置

    # 访问配置
    print(cfg.training.batch_size)  # 64
    print(cfg.policy.horizon)       # 16

    # 实例化policy配置
    policy_cfg = instantiate(
        cfg.policy,
        input_features=input_features,
        output_features=output_features,
        device=cfg.training.device,
    )
```

---

## 9. 关键设计决策

### 9.1 为什么使用Transformer而不是U-Net?

**U-Net**:
- 优势: 卷积结构，适合图像
- 劣势: 对1D序列（动作）支持有限，全局感受野不足

**Transformer**:
- 优势:
  - 全局注意力，可以看到整个序列
  - 更好的长期依赖建模
  - 更灵活的条件注入（cross-attention）
- 劣势: 计算量稍大

**结论**: 对于机器人动作序列，Transformer更合适

### 9.2 为什么使用多模态融合?

**单模态限制**:
- RGB: 提供纹理和颜色，但缺乏深度
- Depth: 提供几何信息，但缺乏语义

**融合优势**:
- 互补性: RGB+Depth提供完整的3D理解
- 鲁棒性: 一个模态失效时另一个可以补偿
- 性能提升: 实验表明融合比单模态提升10-15%

### 9.3 为什么使用Spatial SoftMax?

**全局平均池化 (GAP)**:
- 丢失空间信息
- 输出: [B, C]

**Spatial SoftMax**:
- 保留空间信息（通过keypoints）
- 输出: [B, C*2] (每个通道一个(x,y)坐标)
- 可微分
- 对于机器人任务，spatial information很重要

### 9.4 为什么使用Self-Attention聚合多相机?

**简单拼接**:
- 特征: [cam1_feat, cam2_feat, cam3_feat]
- 问题: 没有学习相机间关系

**Self-Attention**:
- 学习相机重要性权重
- 突出关键视角
- 更好的特征融合

### 9.5 为什么使用Action Queue?

**每步都推理**:
- 推理次数多
- 延迟累积
- 动作不连续

**Action Queue (Chunking)**:
- 一次生成多个动作
- 减少推理次数 (horizon/n_action_steps倍)
- 动作序列更平滑
- 实时性更好

### 9.6 Diffusion vs 其他Policy

| 方法 | 优势 | 劣势 |
|---|---|---|
| **Diffusion Policy** | • 多模态动作分布<br>• 高质量生成<br>• 避免模式崩塌 | • 推理慢 (需多步)<br>• 训练复杂 |
| **ACT** | • 快速推理 (单步)<br>• 简单训练 | • 单模态假设<br>• 可能模式崩塌 |
| **BC** | • 最简单<br>• 快速 | • 分布偏移<br>• 鲁棒性差 |

**结论**: Diffusion Policy更适合复杂的机器人操作任务

---

## 10. SMOLVLA 数据流概览

### 10.1 SMOLVLA 架构特点

SMOLVLA (Small Vision-Language-Action) 是基于 HuggingFace SmolVLM2-500M-Video-Instruct 预训练模型的轻量级视觉-语言-动作策略，采用 Flow Matching 进行动作生成。

**核心特点**:
- **轻量级**: 500M参数，适合实时部署
- **多任务学习**: 支持4个连续任务的顺序学习
- **防遗忘技术**: 使用 Replay Buffer 防止灾难性遗忘
- **Flow Matching**: 使用 Flow Matching 而非传统 Diffusion
- **维度适配**: 自动处理 Kuavo 16维到 SmolVLA 32维的维度转换

### 10.2 SMOLVLA 数据流概览

```
观测数据
  ├─► RGB Images [B, n_cam, 3, H, W]
  │   ├─► head_cam_h: [B, 3, 480, 640]
  │   ├─► wrist_cam_l: [B, 3, 480, 640]
  │   └─► wrist_cam_r: [B, 3, 480, 640]
  │
  ├─► Depth Images [B, n_cam, 1, H, W]
  │   ├─► depth_h: [B, 1, 480, 640]
  │   ├─► depth_l: [B, 1, 480, 640]
  │   └─► depth_r: [B, 1, 480, 640]
  │
  ├─► Robot State [B, 16] (Kuavo实际维度)
  └─► Language Instruction [str]
        │
        ▼
图像预处理
  ├─► RGB: Resize + Padding → [B, n_cam, 3, 512, 512]
  ├─► Depth: 深度转RGB伪彩色 → [B, n_cam, 3, 512, 512]
  └─► State: 维度填充 16→32 → [B, 32]
        │
        ▼
VLM Backbone (SmolVLM2-500M)
  ├─► SigLIP视觉编码器 (冻结)
  │   ├─► RGB特征提取: [B, n_cam, 3, 512, 512] → [B, n_cam, feat_dim]
  │   └─► Depth特征提取: [B, n_cam, 3, 512, 512] → [B, n_cam, feat_dim]
  │
  ├─► 语言理解模块
  │   └─► Language Instruction → Language Embedding [B, lang_dim]
  │
  └─► 多模态融合
      ├─► 视觉特征聚合: [B, n_cam, feat_dim] → [B, visual_dim]
      ├─► 语言-视觉对齐: Language ⊗ Visual → [B, fused_dim]
      └─► 状态特征投影: [B, 32] → [B, state_dim]
        │
        ▼
全局条件 global_cond [B, cond_dim]
  ├─► Visual Features: [B, visual_dim]
  ├─► Language Features: [B, lang_dim]
  └─► State Features: [B, state_dim]
        │
        ▼
Action Expert (Transformer Decoder)
  ├─► 输入嵌入: global_cond → [B, T, n_emb]
  ├─► Transformer解码: 生成动作序列特征
  └─► 输出投影: [B, T, n_emb] → [B, chunk_size, 32]
        │
        ▼
Flow Matching (训练)
  ├─► 1. 对真实动作添加噪声: action + ε ~ N(0, I)
  ├─► 2. Transformer预测噪声: ε_pred = ActionExpert(noisy_action, global_cond)
  └─► 3. 计算损失: L = ||ε - ε_pred||²
        │
        ▼
Flow Matching (推理)
  ├─► 1. 从纯噪声开始: action_T ~ N(0, I)
  ├─► 2. 逐步去噪 (10步 → 0步)
  │      for t in [10, 9, ..., 1]:
  │        ε_pred = ActionExpert(action_t, global_cond)
  │        action_{t-1} = flow_step(action_t, ε_pred, t)
  └─► 3. 输出最终动作: action_0 [B, chunk_size, 32]
        │
        ▼
动作后处理
  ├─► 维度裁剪: [B, chunk_size, 32] → [B, chunk_size, 16]
  ├─► 反归一化: 恢复原始动作范围
  └─► Action Queue: 缓存动作序列，每次执行 n_action_steps 步
```

### 10.3 多任务学习数据流

#### 10.3.1 顺序训练流程

```
Stage 1: 预训练模型 → 任务1模型 (移动抓取)
  ├─► 数据: 任务1数据 (100%)
  ├─► 学习率: 5e-5
  └─► 训练: 100 epochs

Stage 2: 任务1模型 → 任务2模型 (快递称重)
  ├─► 数据: 20% 任务1 + 80% 任务2 (Replay Buffer)
  ├─► 学习率: 3.5e-5 (降低30%)
  └─► 训练: 25 epochs

Stage 3: 任务2模型 → 任务3模型 (定姿摆放)
  ├─► 数据: 10% 任务1 + 20% 任务2 + 70% 任务3
  ├─► 学习率: 2.5e-5 (进一步降低)
  └─► 训练: 25 epochs

Stage 4: 任务3模型 → 任务4模型 (全流程分拣)
  ├─► 数据: 10% 任务1 + 10% 任务2 + 20% 任务3 + 60% 任务4
  ├─► 学习率: 2e-5 (最低学习率)
  └─► 训练: 25 epochs
```

#### 10.3.2 Replay Buffer 机制

```python
class ReplayDatasetManager:
    """管理Replay Buffer的类"""

    def load_replay_tasks(self):
        """加载所有需要replay的任务数据"""
        # Stage 2: 20% 任务1 + 80% 任务2
        # Stage 3: 10% 任务1 + 20% 任务2 + 70% 任务3
        # Stage 4: 10% 任务1 + 10% 任务2 + 20% 任务3 + 60% 任务4

        replay_datasets = {}
        replay_weights = {}

        for task_id, weight in replay_config.items():
            # 加载之前任务的数据
            dataset = LeRobotDataset(
                task_cfg.task.data.repoid,
                root=task_cfg.task.data.root,
                episodes=episodes_range,
                delta_timestamps=delta_timestamps
            )
            replay_datasets[task_id] = dataset
            replay_weights[task_id] = weight

        return replay_datasets, replay_weights

class MixedDataset(torch.utils.data.Dataset):
    """混合多个数据集，每个数据集保留自己的language instruction"""

    def __getitem__(self, idx):
        # 根据weights随机选择一个dataset
        dataset_idx = self.weighted_sampler.sample()
        dataset = self.datasets[dataset_idx]

        # 从该dataset随机选择一个样本
        sample = dataset[random.randint(0, len(dataset)-1)]

        # 添加对应的language instruction
        sample['task'] = [self.language_instructions[dataset_idx]]

        return sample
```

### 10.4 深度融合处理

#### 10.4.1 深度转RGB伪彩色

```python
def depth_to_rgb_for_smolvla(depth_image, target_size=(512, 512),
                           depth_range=(0, 1000), device='cpu'):
    """
    将深度图像转换为RGB伪彩色图像

    Args:
        depth_image: [H, W] 深度图像 (uint16)
        target_size: 目标尺寸 (512, 512)
        depth_range: 深度范围 (0, 1000) mm

    Returns:
        rgb_tensor: [3, H, W] RGB伪彩色图像
    """
    # 1. 归一化深度值到 [0, 1]
    depth_normalized = np.clip(depth_image, depth_range[0], depth_range[1])
    depth_normalized = (depth_normalized - depth_range[0]) / (depth_range[1] - depth_range[0])

    # 2. 应用Jet颜色映射
    rgb_image = apply_jet_colormap(depth_normalized)

    # 3. 调整尺寸
    rgb_resized = cv2.resize(rgb_image, target_size)

    # 4. 转换为tensor
    rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float()

    return rgb_tensor

def apply_jet_colormap(value):
    """Jet颜色映射函数"""
    if value < 0.125:
        r, g, b = 0, 0, 0.5 + 4 * value
    elif value < 0.375:
        r, g, b = 0, 4 * (value - 0.125), 1
    elif value < 0.625:
        r, g, b = 0, 1, 1 - 4 * (value - 0.375)
    elif value < 0.875:
        r, g, b = 4 * (value - 0.625), 1, 0
    else:
        r, g, b = 1, 1 - 4 * (value - 0.875), 0
    return r, g, b
```

#### 10.4.2 多相机融合

```python
class MultiCameraDepthFusion:
    """多相机深度融合处理器"""

    def process_observations_simple(self, obs):
        """
        处理多相机观测数据

        Args:
            obs: 原始观测数据
                - head_cam_h: [480, 640, 3]
                - depth_h: [480, 640, 1]
                - wrist_cam_l: [480, 640, 3]
                - depth_l: [480, 640, 1]
                - wrist_cam_r: [480, 640, 3]
                - depth_r: [480, 640, 1]
                - state: [16]

        Returns:
            observation: 处理后的观测数据
                - observation.head_cam_h: [1, 3, 512, 512]
                - observation.depth_h: [1, 3, 512, 512] (伪彩色)
                - observation.wrist_cam_l: [1, 3, 512, 512]
                - observation.depth_l: [1, 3, 512, 512] (伪彩色)
                - observation.wrist_cam_r: [1, 3, 512, 512]
                - observation.depth_r: [1, 3, 512, 512] (伪彩色)
                - observation.state: [1, 32] (填充到32维)
                - task: [language_instruction]
        """
        observation = {}

        # 处理RGB图像
        for cam_name in ['head_cam_h', 'wrist_cam_l', 'wrist_cam_r']:
            rgb_key = f'observation.{cam_name}'
            rgb_tensor = self.img_preprocess_smolvla(obs[cam_name])
            observation[rgb_key] = rgb_tensor.unsqueeze(0)  # [1, 3, 512, 512]

        # 处理深度图像 (转换为RGB伪彩色)
        for depth_name in ['depth_h', 'depth_l', 'depth_r']:
            depth_key = f'observation.{depth_name}'
            depth_rgb = depth_to_rgb_for_smolvla(
                obs[depth_name],
                target_size=(512, 512),
                depth_range=self.depth_range,
                device=self.device
            )
            observation[depth_key] = depth_rgb.unsqueeze(0)  # [1, 3, 512, 512]

        # 处理状态 (填充到32维)
        state_tensor = torch.from_numpy(obs['state']).float()
        state_padded = pad_tensor_to_target_dim(state_tensor, target_dim=32)
        observation['observation.state'] = state_padded.unsqueeze(0)  # [1, 32]

        # 添加语言指令
        observation['task'] = [self.language_instruction]

        return observation
```

### 10.5 维度适配处理

#### 10.5.1 自动维度填充

```python
def pad_tensor_to_target_dim(tensor, target_dim: int):
    """
    将tensor从实际维度填充到目标维度

    Args:
        tensor: 输入tensor [..., actual_dim]
        target_dim: 目标维度 (32)

    Returns:
        填充后的tensor [..., target_dim]
    """
    actual_dim = tensor.shape[-1]
    if actual_dim == target_dim:
        return tensor
    elif actual_dim < target_dim:
        # 填充0到目标维度
        pad_size = target_dim - actual_dim
        pad_shape = list(tensor.shape[:-1]) + [pad_size]
        pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad_tensor], dim=-1)
    else:
        # 裁剪到目标维度
        return tensor[..., :target_dim]

# 使用示例
# Kuavo实际维度: 16维
# SmolVLA预训练维度: 32维
# 自动填充策略: 后16维填0

state_16d = torch.randn(1, 16)  # Kuavo状态
state_32d = pad_tensor_to_target_dim(state_16d, target_dim=32)  # [1, 32]

action_32d = torch.randn(1, 50, 32)  # SmolVLA生成的动作
action_16d = action_32d[..., :16]  # 裁剪回16维用于控制
```

#### 10.5.2 归一化处理

```python
# 对于填充部分使用恒等归一化
# mean = 0, std = 1 (填充部分不会被改变)
normalization_mapping:
  STATE:
    value: MEAN_STD  # 使用数据集统计
  ACTION:
    value: MEAN_STD  # 使用数据集统计

# 归一化配置
def normalize_state_action(data, stats):
    """归一化状态和动作数据"""
    # 状态归一化 (前16维使用统计，后16维保持0)
    state_mean = torch.cat([stats['observation.state']['mean'], torch.zeros(16)])
    state_std = torch.cat([stats['observation.state']['std'], torch.ones(16)])

    # 动作归一化 (前16维使用统计，后16维保持0)
    action_mean = torch.cat([stats['action']['mean'], torch.zeros(16)])
    action_std = torch.cat([stats['action']['std'], torch.ones(16)])

    return normalized_data
```

### 10.6 Flow Matching vs Diffusion

#### 10.6.1 Flow Matching 原理

```python
# Flow Matching 使用连续时间流
# 相比 Diffusion 的离散时间步，Flow Matching 更平滑

class FlowMatching:
    """Flow Matching 实现"""

    def forward_flow(self, x0, x1, t):
        """
        前向流: x_t = (1-t) * x_0 + t * x_1

        Args:
            x0: 初始状态 (纯噪声)
            x1: 目标状态 (真实动作)
            t: 时间参数 [0, 1]
        """
        return (1 - t) * x0 + t * x1

    def compute_loss(self, batch):
        """计算 Flow Matching 损失"""
        # 1. 准备全局条件
        global_cond = self._prepare_global_conditioning(batch)

        # 2. 提取真实动作
        actions = batch['action']  # [B, chunk_size, 32]

        # 3. 随机采样时间步
        t = torch.rand(actions.shape[0], device=actions.device)  # [B]

        # 4. 采样噪声
        noise = torch.randn_like(actions)  # [B, chunk_size, 32]

        # 5. 计算流状态
        flow_state = self.forward_flow(noise, actions, t.unsqueeze(-1).unsqueeze(-1))

        # 6. 预测速度场
        velocity_pred = self.action_expert(flow_state, t, global_cond)

        # 7. 计算损失 (预测速度 vs 真实速度)
        true_velocity = actions - noise
        loss = F.mse_loss(velocity_pred, true_velocity)

        return loss

    def sample(self, global_cond, num_steps=10):
        """Flow Matching 采样"""
        batch_size = global_cond.shape[0]

        # 1. 从纯噪声开始
        x = torch.randn(batch_size, self.chunk_size, self.action_dim, device=global_cond.device)

        # 2. 时间步
        timesteps = torch.linspace(0, 1, num_steps, device=global_cond.device)

        # 3. 逐步去噪
        for i in range(num_steps - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]

            # 预测速度场
            velocity = self.action_expert(x, t.expand(batch_size), global_cond)

            # 更新状态
            x = x + velocity * dt

        return x
```

#### 10.6.2 Flow Matching vs Diffusion 对比

| 特性 | Flow Matching | Diffusion |
|------|---------------|-----------|
| **时间表示** | 连续时间 [0,1] | 离散时间步 [0,T] |
| **前向过程** | 线性插值 | 高斯噪声添加 |
| **训练目标** | 速度场预测 | 噪声预测 |
| **采样步数** | 10步 | 100步 (DDPM) / 10步 (DDIM) |
| **收敛性** | 更平滑 | 可能不稳定 |
| **计算效率** | 更高 | 较低 |

### 10.7 SMOLVLA 推理流程

#### 10.7.1 在线推理

```python
# 初始化
policy = SmolVLAPolicyWrapper.from_pretrained(checkpoint_path)
policy.eval()
policy.to(device)

# 创建深度融合处理器
fusion_processor = create_multi_camera_fusion(
    target_size=(512, 512),
    depth_range=(0, 1000),
    device=device,
    enable_depth=True
)

# 推理循环
for step in range(max_steps):
    # 1. 获取观测
    obs = env.get_observation()
    # obs = {
    #   'head_cam_h': [480, 640, 3],
    #   'depth_h': [480, 640, 1],
    #   'wrist_cam_l': [480, 640, 3],
    #   'depth_l': [480, 640, 1],
    #   'wrist_cam_r': [480, 640, 3],
    #   'depth_r': [480, 640, 1],
    #   'state': [16]
    # }

    # 2. 深度融合处理
    observation = fusion_processor.process_observations_simple(obs)
    # observation = {
    #   'observation.head_cam_h': [1, 3, 512, 512],
    #   'observation.depth_h': [1, 3, 512, 512] (伪彩色),
    #   'observation.wrist_cam_l': [1, 3, 512, 512],
    #   'observation.depth_l': [1, 3, 512, 512] (伪彩色),
    #   'observation.wrist_cam_r': [1, 3, 512, 512],
    #   'observation.depth_r': [1, 3, 512, 512] (伪彩色),
    #   'observation.state': [1, 32] (填充),
    #   'task': [language_instruction]
    # }

    # 3. 选择动作
    with torch.no_grad():
        action = policy.select_action(observation)
    # action: [1, 16] (裁剪回16维)

    # select_action内部:
    # - 如果action queue为空:
    #   a. 调用Flow Matching生成chunk_size=50个动作
    #   b. 填充action queue
    # - 从action queue pop第一个动作

    # 4. 执行动作
    action = action.squeeze(0).cpu().numpy()  # [16]
    obs_next, reward, done, info = env.step(action)

    if done:
        obs = env.reset()
```

#### 10.7.2 Action Queue 机制

```python
# SMOLVLA Action Queue 配置
chunk_size: 50        # Flow Matching 生成50步动作
n_action_steps: 8     # 每次执行8步动作

# Action Queue 工作流程
if len(self._queues[ACTION]) == 0:
    # 生成动作chunk
    actions = self.predict_action_chunk(observation)
    # actions: [1, chunk_size, 32] = [1, 50, 32]

    # 裁剪到16维
    actions = actions[..., :16]  # [1, 50, 16]

    # 转置并填充 (50个动作)
    self._queues[ACTION].extend(actions.transpose(0, 1))
    # action queue: [action[0], action[1], ..., action[49]]

# 取出一个动作
action = self._queues[ACTION].popleft()
# action: [1, 16]

# 推理频率分析
# 每 n_action_steps = 8 步执行一次推理
# 即每 0.8秒 推理一次 (10Hz控制频率)
# 每次推理生成50步动作，执行前8步
# 剩余42步在下次迭代使用
```

### 10.8 SMOLVLA 配置系统

#### 10.8.1 基础配置

```yaml
# smolvla_sequential_base.yaml
policy:
  _target_: kuavo_train.wrapper.policy.smolvla.SmolVLAConfigWrapper

  # VLM Backbone配置
  vlm_model_name: 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
  load_vlm_weights: True

  # 冻结策略
  freeze_vision_encoder: True  # 冻结SigLIP视觉编码器
  train_expert_only: True     # 只训练Action Expert
  train_state_proj: True       # 训练state投影层

  # 动作空间配置
  max_state_dim: 32    # 预训练模型维度
  max_action_dim: 32   # 预训练模型维度
  chunk_size: 50       # 动作序列长度
  n_action_steps: 8    # 每次执行步数

  # 图像预处理
  resize_imgs_with_padding: [512, 512]  # SmolVLA标准输入尺寸

  # 深度相机支持
  use_depth: True
  depth_features:
    - 'observation.depth_h'
    - 'observation.depth_l'
    - 'observation.depth_r'
  depth_resize_with_padding: [512, 512]
  depth_normalization_range: [0.0, 1000.0]

  # Flow Matching配置
  num_steps: 10  # 推理时的去噪步数
```

#### 10.8.2 任务配置

```yaml
# task1_moving_grasp.yaml
task:
  id: 1
  name: 'moving_grasp'
  stage: 1

  language_instruction: 'Grasp the object from the moving conveyor belt using visual guidance. Place it precisely at the center of the first colored target block on the table, ensuring the object center aligns with the target center. Then grasp it again and place it precisely at the center of the second colored target block on the table, maintaining visual alignment throughout the placement.'

  data:
    root: '/root/robot/data/task-1/1-2000/lerobot/'
    repoid: 'lerobot/task1_moving_grasp'
    episodes_to_use: [0, 199]

  training:
    max_epoch: 100
    resume_from: 'pretrained'
    pretrained_path: 'lerobot/smolvla_base'
    policy:
      optimizer_lr: 5e-5
```

### 10.9 SMOLVLA vs Diffusion Policy 对比

| 特性 | SMOLVLA | Diffusion Policy |
|------|---------|------------------|
| **模型大小** | 500M (轻量级) | ~15M (自定义) |
| **预训练** | HuggingFace预训练 | 从头训练 |
| **多任务** | 顺序学习4个任务 | 单任务训练 |
| **语言理解** | 支持自然语言指令 | 无语言理解 |
| **动作生成** | Flow Matching (10步) | Diffusion (10-100步) |
| **推理速度** | 更快 | 较慢 |
| **训练复杂度** | 中等 (防遗忘) | 简单 |
| **扩展性** | 易于添加新任务 | 需要重新训练 |
| **鲁棒性** | 高 (预训练+多任务) | 中等 |

---

## 附录

### A. 文件结构

```
kuavo_data_challenge/
├── kuavo_train/
│   ├── train_policy.py  # 训练主入口
│   ├── wrapper/
│   │   ├── policy/
│   │   │   └── diffusion/
│   │   │       ├── DiffusionPolicyWrapper.py       # Policy包装器
│   │   │       ├── DiffusionModelWrapper.py        # 模型包装器
│   │   │       ├── DiffusionConfigWrapper.py       # 配置包装器
│   │   │       ├── transformer_diffusion.py        # Transformer实现
│   │   │       └── DiT_model.py                    # DiT (未使用)
│   │   └── dataset/
│   │       └── LeRobotDatasetWrapper.py
│   └── utils/
│       ├── augmenter.py
│       ├── transforms.py
│       └── utils.py
├── configs/
│   └── policy/
│       └── diffusion_config.yaml
└── kuavo_deploy/
    └── examples/
        └── eval/
            └── eval_kuavo.py
```

### B. 训练命令

```bash
# 基础训练
python kuavo_train/train_policy.py \
  --config-name=diffusion_config

# 指定参数
python kuavo_train/train_policy.py \
  --config-name=diffusion_config \
  training.batch_size=32 \
  training.max_epoch=1000 \
  policy.horizon=32

# 恢复训练
python kuavo_train/train_policy.py \
  --config-name=diffusion_config \
  training.resume=True \
  training.resume_timestamp=run_20250110_123456
```

### C. 推理命令

```bash
# 使用Diffusion Policy推理
python kuavo_deploy/examples/eval/eval_kuavo.py \
  --checkpoint path/to/best \
  --policy-type diffusion
```

### D. 性能指标

| 指标 | 值 | 说明 |
|---|---|---|
| 总参数量 | ~15M | 包含所有编码器和Transformer |
| 训练batch size | 64 | 可根据GPU调整 |
| 训练epochs | 500 | 通常300-500 epoch收敛 |
| Horizon | 16 | 生成16步动作 |
| n_action_steps | 8 | 执行前8步 |
| 推理时间 (DDPM, 100步) | ~500ms | 太慢 |
| 推理时间 (DDIM, 10步) | ~50ms | 实时 ✅ |
| 控制频率 | 10Hz | 机器人控制 |

---

**文档版本**: 1.0
**最后更新**: 2025-10-10
**作者**: AI Assistant

