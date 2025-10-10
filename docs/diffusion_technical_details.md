# Diffusion Policy 技术细节与实战

> 深入技术实现和实际使用指南

---

## 目录

1. [扩散数学原理](#1-扩散数学原理)
2. [Transformer详细实现](#2-transformer详细实现)
3. [数据流与维度](#3-数据流与维度)
4. [实战训练指南](#4-实战训练指南)
5. [调试与优化](#5-调试与优化)
6. [常见问题](#6-常见问题)

---

## 1. 扩散数学原理

### 1.1 前向扩散过程

**定义**: 给定数据分布 q(x₀)，逐步添加高斯噪声直到变为标准正态分布。

```
q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1 - ᾱ_t) I)

其中:
- x_0: 原始数据 (真实动作)
- x_t: t时刻的噪声数据
- ᾱ_t = ∏(1 - β_i) for i=1 to t
- β_t: 噪声调度参数
```

**直接采样**:
```python
def add_noise(self, original_samples, noise, timesteps):
    """
    直接采样 x_t

    x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
    """
    alphas_cumprod = self.alphas_cumprod.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = (
        sqrt_alpha_prod * original_samples
        + sqrt_one_minus_alpha_prod * noise
    )
    return noisy_samples
```

### 1.2 反向去噪过程

**目标**: 学习反向分布 p_θ(x_{t-1} | x_t)

```
训练目标 (epsilon prediction):
L = E_{x_0, ε, t} [ ||ε - ε_θ(x_t, t, c)||² ]

其中:
- ε ~ N(0, I): 真实噪声
- ε_θ: 神经网络预测的噪声
- x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε
- c: 条件信息 (观测)
```

**DDPM采样**:
```python
def ddpm_step(self, model_output, timestep, sample):
    """
    DDPM单步去噪

    x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ) + σ_t * z
    """
    t = timestep

    # 获取参数
    alpha_t = self.alphas[t]
    alpha_bar_t = self.alphas_cumprod[t]
    beta_t = 1 - alpha_t

    # 预测 x_0
    if self.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_t ** 0.5 * model_output
        ) / alpha_t ** 0.5
    elif self.prediction_type == "sample":
        pred_original_sample = model_output

    # 裁剪
    if self.clip_sample:
        pred_original_sample = torch.clamp(
            pred_original_sample,
            -self.clip_sample_range,
            self.clip_sample_range
        )

    # 计算 x_{t-1} 的均值
    pred_original_sample_coeff = (
        alpha_bar_prev ** 0.5 * beta_t / (1 - alpha_bar_t)
    )
    current_sample_coeff = alpha_t ** 0.5 * (1 - alpha_bar_prev) / (1 - alpha_bar_t)

    pred_prev_sample = (
        pred_original_sample_coeff * pred_original_sample
        + current_sample_coeff * sample
    )

    # 添加噪声 (除了最后一步)
    if t > 0:
        noise = torch.randn_like(sample)
        variance = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
        pred_prev_sample = pred_prev_sample + (variance ** 0.5) * noise

    return pred_prev_sample
```

**DDIM采样** (确定性, 更快):
```python
def ddim_step(self, model_output, timestep, sample, eta=0.0):
    """
    DDIM单步去噪 (可跳步)

    x_{t-1} = √(ᾱ_{t-1}) * pred_x_0
             + √(1 - ᾱ_{t-1} - σ_t²) * ε_θ
             + σ_t * z

    当 eta=0 时, σ_t=0, 完全确定性
    """
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

    alpha_bar_t = self.alphas_cumprod[timestep]
    alpha_bar_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else 1.0

    beta_t = 1 - alpha_bar_t

    # 预测 x_0
    if self.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_t ** 0.5 * model_output
        ) / alpha_bar_t ** 0.5

    # DDIM公式
    pred_sample_direction = (1 - alpha_bar_prev - eta**2 * variance) ** 0.5 * model_output

    pred_prev_sample = (
        alpha_bar_prev ** 0.5 * pred_original_sample
        + pred_sample_direction
    )

    if eta > 0:
        noise = torch.randn_like(sample)
        variance = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * (1 - alpha_bar_t / alpha_bar_prev)
        pred_prev_sample = pred_prev_sample + eta * (variance ** 0.5) * noise

    return pred_prev_sample
```

### 1.3 噪声调度

**线性调度**:
```python
betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
```

**余弦调度** (推荐):
```python
def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦调度

    ᾱ_t = f(t) / f(0)
    f(t) = cos²((t/T + s) / (1 + s) * π/2)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

**对比**:
| 调度类型 | 优势 | 劣势 | 适用场景 |
|---|---|---|---|
| Linear | 简单 | 起始噪声过大 | 简单任务 |
| Cosine | 稳定，起始缓慢 | 略复杂 | 推荐默认 |
| Quadratic | 非线性 | 需调参 | 特殊需求 |

---

## 2. Transformer详细实现

### 2.1 多头注意力机制

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
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

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [B, T_q, d_model]
            key: [B, T_k, d_model]
            value: [B, T_v, d_model]  (T_k == T_v)
            mask: [T_q, T_k] 或 [B, T_q, T_k]

        Returns:
            output: [B, T_q, d_model]
        """
        batch_size, seq_len = query.shape[:2]

        # 1. 线性变换并分头
        Q = self.w_q(query).view(batch_size, seq_len, self.n_head, self.d_k)
        K = self.w_k(key).view(batch_size, -1, self.n_head, self.d_k)
        V = self.w_v(value).view(batch_size, -1, self.n_head, self.d_k)

        # 转置: [B, n_head, seq_len, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 2. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # [B, n_head, T_q, T_k]

        # 3. 应用mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T_q, T_k]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [B, 1, T_q, T_k]
            scores = scores.masked_fill(mask == float('-inf'), float('-inf'))

        # 4. Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. 加权求和
        context = torch.matmul(attn_weights, V)
        # [B, n_head, T_q, d_k]

        # 6. 合并多头
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        # 7. 输出投影和残差连接
        output = self.w_out(context)
        output = self.layer_norm(output + query)

        return output
```

**注意力机制可视化**:
```
Query: "我想知道第3步动作应该是什么"
Keys:  ["观测t=0", "观测t=1", "观测t=2"]
Values: [obs_emb_0, obs_emb_1, obs_emb_2]

注意力权重: [0.1, 0.2, 0.7]  # 第3步主要看第2步的观测

输出: 0.1*obs_emb_0 + 0.2*obs_emb_1 + 0.7*obs_emb_2
```

### 2.2 Position Encoding

```python
class PositionalEncoding(nn.Module):
    """位置编码 (可学习)"""

    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        seq_len = x.size(1)
        return x + self.pos_emb[:, :seq_len, :]
```

**为什么需要位置编码?**
- Transformer本身没有位置信息
- Self-Attention是置换不变的
- 位置编码告诉模型"哪个token在哪个位置"

**可学习 vs 固定**:
| 类型 | 优势 | 劣势 |
|---|---|---|
| 可学习 (Learnable) | 灵活，适应任务 | 不能外推 |
| 固定 (Sinusoidal) | 可外推到更长序列 | 不灵活 |

### 2.3 Feed-Forward Network

```python
class FeedForward(nn.Module):
    """Position-wise Feed-Forward网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        residual = x

        x = self.linear1(x)      # [B, T, d_ff]
        x = self.activation(x)
        x = self.dropout(x)

        x = self.linear2(x)      # [B, T, d_model]
        x = self.dropout(x)

        x = self.layer_norm(x + residual)

        return x
```

**FFN作用**:
- 增加模型表达能力
- 每个位置独立处理 (position-wise)
- 通常 d_ff = 4 * d_model

---

## 3. 数据流与维度

### 3.1 训练时的完整数据流

```
Step 1: Dataset.__getitem__()
  输出:
    observation.images.head_cam_h: [2, 3, 480, 640]     # [n_obs, C, H, W]
    observation.images.wrist_cam_l: [2, 3, 480, 640]
    observation.images.wrist_cam_r: [2, 3, 480, 640]
    observation.depth.depth_h: [2, 1, 480, 640]
    observation.depth.depth_l: [2, 1, 480, 640]
    observation.depth.depth_r: [2, 1, 480, 640]
    observation.state: [2, 16]                          # [n_obs, state_dim]
    action: [16, 16]                                    # [horizon, action_dim]

Step 2: DataLoader (batch_size=64)
  输出:
    observation.images.*: [64, 2, 3, 480, 640]
    observation.depth.*: [64, 2, 1, 480, 640]
    observation.state: [64, 2, 16]
    action: [64, 16, 16]

Step 3: Policy.forward() - 图像预处理
  裁剪: crop_shape=[420, 560]
    observation.images.*: [64, 2, 3, 420, 560]
  缩放: resize_shape=[210, 280]
    observation.images.*: [64, 2, 3, 210, 280]
    observation.depth.*: [64, 2, 1, 210, 280]

Step 4: Policy.forward() - 归一化
  observation.images.*: (x - mean) / std
  observation.depth.*: (x - min) / (max - min)
  observation.state: (x - mean) / std
  action: (x - mean) / std

Step 5: Policy.forward() - 堆叠图像
  observation.images:
    torch.stack([head_cam_h, wrist_cam_l, wrist_cam_r], dim=-4)
    -> [64, 2, 3, 3, 210, 280]  # [B, n_obs, n_cam, C, H, W]

  observation.depth:
    torch.stack([depth_h, depth_l, depth_r], dim=-4)
    -> [64, 2, 3, 1, 210, 280]

Step 6: DiffusionModel._prepare_global_conditioning()

  6.1 RGB编码
    输入: [64, 2, 3, 3, 210, 280]
    重排: einops.rearrange("b s n c h w -> (b s n) c h w")
    -> [64*2*3=384, 3, 210, 280]

    RGB Encoder (ResNet18):
    -> [384, 512, 7, 9]  # 特征图

    Spatial SoftMax:
    -> [384, 64*2=128]   # 64个keypoints，每个2个坐标

    输出层:
    -> [384, 128]

    重排回: einops.rearrange("(b s n) d -> b s n d", b=64, s=2)
    -> [64, 2, 3, 128]

  6.2 Depth编码 (类似RGB)
    -> [64, 2, 3, 128]

  6.3 Self-Attention聚合多相机
    重排: "(b s) n d", b=64, s=2
    -> [128, 3, 128]

    Self-Attention (RGB):
    -> [128, 3, 128]

    Self-Attention (Depth):
    -> [128, 3, 128]

  6.4 Cross-Attention融合
    RGB query Depth:
    -> [128, 3, 128]

    Depth query RGB:
    -> [128, 3, 128]

    重排并拼接:
    rgb_fuse: [128, 3, 128] -> [64, 2, 3*128=384]
    depth_fuse: [128, 3, 128] -> [64, 2, 384]

  6.5 状态编码
    输入: [64, 2, 16]
    State Encoder (if enabled):
    -> [64, 2, 128]

  6.6 全局拼接
    global_cond = cat([rgb_fuse, depth_fuse, state_features], dim=-1)
    -> [64, 2, 384+384+128=896]

Step 7: DiffusionModel.compute_loss()

  7.1 提取动作
    actions: [64, 16, 16]

  7.2 采样时间步
    timesteps: [64]  # 随机整数 0-99

  7.3 采样噪声
    noise: [64, 16, 16]  # ~ N(0, I)

  7.4 添加噪声
    noisy_actions = √(ᾱ_t) * actions + √(1 - ᾱ_t) * noise
    -> [64, 16, 16]

  7.5 Transformer前向

    7.5.1 时间步嵌入
      timesteps: [64]
      SinusoidalPosEmb:
      -> [64, 512]
      MLP:
      -> [64, 512]
      unsqueeze(1):
      -> [64, 1, 512]

    7.5.2 动作嵌入
      noisy_actions: [64, 16, 16]
      input_emb:
      -> [64, 16, 512]

    7.5.3 条件嵌入
      global_cond: [64, 2, 896]
      cond_obs_emb:
      -> [64, 2, 512]

      cat([time_emb, cond_obs_emb], dim=1):
      -> [64, 3, 512]  # 1 time + 2 obs

    7.5.4 位置编码
      cond_embeddings + cond_pos_emb:
      -> [64, 3, 512]

      input_emb + pos_emb:
      -> [64, 16, 512]

    7.5.5 Encoder (简单MLP)
      -> [64, 3, 512]

    7.5.6 Decoder (4层Transformer)
      tgt: [64, 16, 512]        # Query
      memory: [64, 3, 512]      # Key & Value

      Decoder Layer 1:
        Self-Attention(tgt, tgt, tgt):
        -> [64, 16, 512]

        Cross-Attention(tgt, memory, memory):
        -> [64, 16, 512]

        FFN:
        -> [64, 16, 512]

      ... (Layer 2, 3, 4)

      输出: [64, 16, 512]

    7.5.7 输出头
      LayerNorm:
      -> [64, 16, 512]

      Linear:
      -> [64, 16, 16]  # 预测的噪声

  7.6 计算损失
    noise_pred: [64, 16, 16]
    noise: [64, 16, 16]

    loss = MSE(noise_pred, noise)
    -> 标量

Step 8: 反向传播
  loss.backward()
  optimizer.step()
```

### 3.2 推理时的数据流

```
Step 1: Env.get_observation()
  输出:
    observation.images.*: [3, 480, 640]
    observation.depth.*: [1, 480, 640]
    observation.state: [16]

Step 2: 添加batch维度
  -> [1, 3, 480, 640], [1, 1, 480, 640], [1, 16]

Step 3: Policy.select_action() - 预处理
  同训练时的Step 3-5
  -> observation.images: [1, 1, 3, 3, 210, 280]  # n_obs=1 (首次)
     observation.depth: [1, 1, 3, 1, 210, 280]
     observation.state: [1, 1, 16]

Step 4: 填充observation queue
  首次调用: 重复n_obs_steps=2次
  -> observation.state queue: [[1, 16], [1, 16]]

  后续调用: 添加新观测，删除最旧的
  -> observation.state queue: [[1, 16]_old, [1, 16]_new]

Step 5: 如果action queue为空，生成动作

  5.1 从queue构建batch
    observation.state: stack(queue) -> [1, 2, 16]
    observation.images: stack(queue) -> [1, 2, 3, 3, 210, 280]

  5.2 predict_action_chunk()

    5.2.1 准备条件
      global_cond = _prepare_global_conditioning(batch)
      -> [1, 2, 896]

    5.2.2 初始化纯噪声
      actions = randn([1, 16, 16])

    5.2.3 迭代去噪 (DDIM, 10步)
      for t in [99, 90, 81, 72, 63, 54, 45, 36, 27, 18, 9]:
        # 预测噪声
        timesteps = [t]
        noise_pred = transformer(actions, timesteps, global_cond)
        # [1, 16, 16]

        # 去噪一步
        actions = ddim_step(noise_pred, t, actions)
        # [1, 16, 16]

    5.2.4 反归一化
      actions = actions * std + mean
      -> [1, 16, 16]

  5.3 填充action queue
    actions.transpose(0, 1):
    -> [[1, 16], [1, 16], ..., [1, 16]]  # 16个动作

    extend到queue

Step 6: 从action queue取动作
  action = queue.popleft()
  -> [1, 16]

Step 7: 执行动作
  action = action.squeeze(0).cpu().numpy()
  -> [16]

  env.step(action)
```

---

## 4. 实战训练指南

### 4.1 训练前准备

#### 4.1.1 数据检查

```bash
# 检查数据集结构
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
metadata = LeRobotDatasetMetadata('lerobot/task_400_episodes', root='/path/to/data')
print('Camera keys:', metadata.camera_keys)
print('Total episodes:', metadata.info['total_episodes'])
print('Total frames:', metadata.info['total_frames'])
print('FPS:', metadata.fps)
print('Features:', metadata.features.keys())
"

# 预期输出:
# Camera keys: ['head_cam_h', 'wrist_cam_l', 'wrist_cam_r']
# Total episodes: 400
# Total frames: 50000
# FPS: 10
# Features: dict_keys([...])
```

#### 4.1.2 配置检查

```python
# 检查配置一致性
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs/policy/", config_name="diffusion_config")
def check_config(cfg):
    print(f"Horizon: {cfg.policy.horizon}")
    print(f"n_action_steps: {cfg.policy.n_action_steps}")
    print(f"drop_n_last_frames: {cfg.policy.drop_n_last_frames}")

    # 验证: drop_n_last_frames == horizon - n_action_steps - 1
    assert cfg.policy.drop_n_last_frames == cfg.policy.horizon - cfg.policy.n_action_steps - 1
    print("✅ Configuration is valid!")

if __name__ == "__main__":
    check_config()
```

### 4.2 训练策略

#### 4.2.1 学习率调优

```yaml
# 配置文件
policy:
  optimizer_lr: 0.0001  # 基础学习率
  scheduler_name: cosine
  scheduler_warmup_steps: 500

# 学习率曲线
# Warmup (0-500 steps): lr从0线性增长到optimizer_lr
# Cosine decay (500-end): lr从optimizer_lr余弦衰减到0
```

**调优建议**:
| 情况 | 建议 |
|---|---|
| 训练不稳定 | 降低lr (0.00005) |
| 收敛太慢 | 增加lr (0.0002) |
| Loss震荡 | 增加warmup_steps (1000) |
| Overfitting | 增加weight_decay |

#### 4.2.2 Batch Size选择

```python
# 计算显存占用 (粗略估计)
def estimate_memory(batch_size, horizon, n_obs_steps, n_cameras=3):
    # 图像: B * n_obs * n_cam * 3 * H * W * 4 bytes
    image_mem = batch_size * n_obs_steps * n_cameras * 3 * 210 * 280 * 4

    # 动作: B * horizon * action_dim * 4 bytes
    action_mem = batch_size * horizon * 16 * 4

    # 模型参数: ~15M * 4 bytes
    model_mem = 15e6 * 4

    # 激活值: ~2倍模型参数
    activation_mem = 2 * model_mem

    # 梯度: ~1倍模型参数
    gradient_mem = model_mem

    total_mem = image_mem + action_mem + model_mem + activation_mem + gradient_mem
    return total_mem / 1e9  # GB

# 示例
print(f"Batch=32: {estimate_memory(32, 16, 2):.2f} GB")
print(f"Batch=64: {estimate_memory(64, 16, 2):.2f} GB")
print(f"Batch=128: {estimate_memory(128, 16, 2):.2f} GB")

# 输出:
# Batch=32: ~8 GB
# Batch=64: ~12 GB
# Batch=128: ~20 GB
```

**推荐配置**:
| GPU | VRAM | 推荐Batch Size | 备注 |
|---|---|---|---|
| RTX 3090 | 24GB | 64 | 最佳 |
| RTX 3080 | 10GB | 32 | 需要梯度累积 |
| V100 | 32GB | 96 | 可以更大 |
| A100 | 80GB | 128 | 超大batch |

#### 4.2.3 梯度累积

```yaml
training:
  batch_size: 32
  accumulation_steps: 2  # 等效batch_size=64
```

```python
# 训练代码会自动处理
for step, batch in enumerate(dataloader):
    loss, _ = policy.forward(batch)

    # 缩放损失
    scaled_loss = loss / accumulation_steps
    scaled_loss.backward()

    # 每accumulation_steps步更新一次
    if step % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4.3 监控训练

#### 4.3.1 TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir outputs/train/task_400_episodes/diffusion/

# 访问 http://localhost:6006
```

**关键指标**:
1. **train/loss**: 应该稳定下降
   - 正常: 从~1.0下降到0.05-0.1
   - 异常: 震荡或不下降 → 检查lr
2. **train/lr**: 学习率曲线
   - 应该先warmup后decay
3. **GPU利用率**: 应该>80%
   - 低于80% → 增加num_workers或batch_size

#### 4.3.2 训练日志

```python
# 添加自定义日志
# 在train_policy.py中
if steps % log_freq == 0:
    writer.add_scalar("train/loss", scaled_loss.item(), steps)
    writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], steps)

    # 额外监控
    writer.add_scalar("train/grad_norm", torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0), steps)
    writer.add_scalar("train/param_norm", torch.norm(torch.cat([p.flatten() for p in policy.parameters()])), steps)
```

### 4.4 常见训练问题

#### 4.4.1 Loss不下降

**可能原因**:
1. 学习率过高/过低
2. 数据归一化问题
3. 模型架构问题

**诊断**:
```python
# 检查数据归一化
dataset_stats = dataset_metadata.stats
print("Action mean:", dataset_stats['action']['mean'])
print("Action std:", dataset_stats['action']['std'])

# 应该接近0和1
# 如果差距很大，重新计算stats
```

#### 4.4.2 Nan/Inf Loss

**可能原因**:
1. 学习率爆炸
2. 梯度爆炸
3. 数值不稳定

**解决方案**:
```python
# 1. 梯度裁剪
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

# 2. 使用AMP
policy:
  use_amp: True

# 3. 降低学习率
policy:
  optimizer_lr: 0.00005
```

---

## 5. 调试与优化

### 5.1 快速验证

```python
# test_diffusion_policy.py
import torch
from kuavo_train.wrapper.policy.diffusion import CustomDiffusionPolicyWrapper

# 1. 创建虚拟配置
config = ...  # 从yaml加载

# 2. 创建policy
policy = CustomDiffusionPolicyWrapper(config, dataset_stats=None)
policy.eval()

# 3. 测试前向传播
batch = {
    'observation.images.head_cam_h': torch.randn(2, 2, 3, 210, 280),
    'observation.state': torch.randn(2, 2, 16),
    'action': torch.randn(2, 16, 16),
}

with torch.no_grad():
    # 训练模式
    loss, _ = policy.forward(batch)
    print(f"Loss: {loss.item()}")

    # 推理模式
    obs = {k: v[:1, :1] for k, v in batch.items() if 'observation' in k}
    action = policy.select_action(obs)
    print(f"Action shape: {action.shape}")  # [1, 16]
```

### 5.2 性能分析

```python
# profile_diffusion.py
import torch
from torch.profiler import profile, ProfilerActivity

policy = ...  # 加载policy
batch = ...   # 准备batch

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    for _ in range(10):
        loss, _ = policy.forward(batch)
        loss.backward()

# 打印统计
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))

# 导出Chrome trace
prof.export_chrome_trace("diffusion_trace.json")
# 在 chrome://tracing 中打开
```

### 5.3 内存优化

#### 5.3.1 梯度检查点

```python
# 在Transformer中使用梯度检查点
from torch.utils.checkpoint import checkpoint

class TransformerForDiffusion(nn.Module):
    def __init__(self, ..., use_checkpoint=False):
        ...
        self.use_checkpoint = use_checkpoint

    def forward(self, sample, timestep, global_cond):
        ...
        if self.use_checkpoint and self.training:
            x = checkpoint(self.decoder, x, memory, ...)
        else:
            x = self.decoder(x, memory, ...)
        ...
```

#### 5.3.2 混合精度

```yaml
policy:
  use_amp: True

# 自动使用float16进行前向和反向，权重保持float32
```

---

## 6. 常见问题

### 6.1 维度不匹配

**问题**: `RuntimeError: size mismatch`

**诊断**:
```python
def check_dimensions(batch):
    print("=== Batch Dimensions ===")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:40s}: {list(v.shape)}")

check_dimensions(batch)
```

### 6.2 OOM (Out of Memory)

**解决方案**:
1. 减小batch_size
2. 使用梯度累积
3. 启用AMP
4. 减小图像分辨率
5. 减少相机数量

### 6.3 推理速度慢

**优化方法**:
1. 使用DDIM代替DDPM
2. 减少推理步数 (100→10)
3. 使用TorchScript
4. 使用ONNX Runtime

```python
# 使用DDIM加速
config:
  noise_scheduler_type: DDIM
  num_train_timesteps: 100
  num_inference_steps: 10  # 只用10步

# 加速10倍!
```

### 6.4 训练不稳定

**常见原因**:
1. 学习率过高
2. BatchNorm不稳定
3. 初始化不当

**解决**:
```yaml
policy:
  optimizer_lr: 0.00005  # 降低
  scheduler_warmup_steps: 1000  # 增加warmup
  optimizer_weight_decay: 0.001  # 增加正则化

  custom:
    use_group_norm: True  # 使用GroupNorm代替BatchNorm
```

---

**相关文档**:
- [主架构文档](diffusion_policy_architecture.md)
- [训练脚本](../kuavo_train/train_policy.py)

