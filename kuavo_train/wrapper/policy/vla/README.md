# VLA (Vision-Language-Action) Transformer Policy

现代化的Token化策略架构，对标OpenVLA/RT-2等SOTA方法。

## 核心特点

### 1. 完整Token化架构

```
输入 → 全部token化 → 统一512维空间 → Transformer处理 → Token空间Diffusion → 输出
```

- **Vision Tokens**: RGB和Depth图像转为patch tokens
- **State Tokens**: 每个关节独立token化
- **Action Tokens**: 动作序列token化（训练）/反token化（推理）

### 2. 灵活的维度配置

**所有输入维度通过配置文件定义**，无硬编码：

- 支持16维（只有手臂）
- 支持36维（手臂+腿部）
- 支持任意维度扩展

### 3. Token空间Diffusion

在512维token空间进行diffusion，而非动作空间：

- 更平滑的潜在空间
- 更好的学习效果
- 更容易扩展维度

## 架构组件

### Tokenizers

#### VisionTokenizer
```python
# 将RGB+Depth图像转为tokens
vision_tokens = vision_tokenizer(rgb_images, depth_images)
# 输出: [B, num_patches, 512]
```

#### StateTokenizer
```python
# 每个关节独立token化
state_tokens = state_tokenizer(state, joint_configs)
# 输出: [B, num_joints, 512]
```

#### ActionTokenizer
```python
# 训练: 动作 → tokens
action_tokens = action_tokenizer.tokenize(actions)

# 推理: tokens → 动作
actions = action_tokenizer.detokenize(tokens)
```

### DiffusionDecoder

```python
# 训练: 计算diffusion loss
loss = diffusion_decoder.compute_loss(target_actions, context_tokens)

# 推理: DDPM采样
actions = diffusion_decoder.sample(context_tokens)
```

### VLAPolicyWrapper

主策略类，集成所有组件。

## 使用方法

### 1. 训练（16维配置）

```bash
python kuavo_train/train_vla_policy.py --config-name=vla_config
```

### 2. 训练（36维配置）

```bash
python kuavo_train/train_vla_policy.py --config-name=vla_config_36dim
```

### 3. 推理

```python
from kuavo_train.wrapper.policy.vla import VLAPolicyWrapper

# 加载模型
policy = VLAPolicyWrapper.from_pretrained("path/to/checkpoint")

# 推理
action = policy.select_action(observation)
```

## 配置说明

### 关节配置（核心）

在`vla_config.yaml`中详细定义每个关节：

```yaml
state_config:
  joints:
    - idx: 0              # 在state向量中的索引
      type: 'shoulder'    # 关节类型（共享embedding）
      side: 0             # 0=left, 1=right
      id: 0               # 唯一ID
      name: 'left_shoulder_pitch'  # 名称（可选）
```

### 关节类型

支持的关节类型：
- `shoulder`: 肩部关节
- `elbow`: 肘部关节
- `wrist`: 腕部关节
- `gripper`: 手爪
- `hip`: 髋部关节
- `knee`: 膝部关节
- `ankle`: 踝部关节

### Token化配置

```yaml
policy:
  patch_size: 16              # Vision patch大小
  token_embed_dim: 512        # 统一token维度
  image_size: 224             # 输入图像尺寸
```

### Transformer配置

```yaml
policy:
  transformer_depth: 8        # Encoder层数
  transformer_heads: 8        # 注意力头数
  transformer_dim_feedforward: 2048  # FFN维度
```

### Diffusion配置

```yaml
policy:
  horizon: 16                 # 动作序列长度
  num_train_timesteps: 100    # 训练时的diffusion步数
  num_inference_steps: 50     # 推理时的采样步数
  noise_scheduler_type: DDPM  # 噪声调度器类型
```

## 维度扩展示例

### 从16维扩展到36维

只需修改配置文件中的`state_config`，添加新关节定义：

```yaml
# 原16维：只有手臂
state_config:
  joints:
    - {idx: 0, type: 'shoulder', side: 0, id: 0, name: 'left_shoulder_pitch'}
    # ... 共16个关节

# 新36维：手臂+腿部
state_config:
  joints:
    - {idx: 0, type: 'shoulder', side: 0, id: 0, name: 'left_shoulder_pitch'}
    # ... 前16个关节
    - {idx: 16, type: 'hip', side: 0, id: 16, name: 'left_hip_yaw'}
    # ... 新增20个腿部关节
```

**无需修改代码！** 所有维度自动从配置推断。

## 优势对比

### vs 传统Diffusion Policy

| 特性 | VLA Transformer | 传统Diffusion |
|------|----------------|---------------|
| 输入表示 | Token化 | 特征拼接 |
| Diffusion空间 | 512维token | 16维动作 |
| 维度扩展 | 修改配置 | 修改代码 |
| 架构风格 | 现代化 | 传统 |

### vs 分层架构

| 特性 | VLA Transformer | 分层架构 |
|------|----------------|---------|
| 训练方式 | 端到端 | 课程学习 |
| 架构复杂度 | 统一简洁 | 多层复杂 |
| 推理效率 | 单次前向 | 多层串行 |
| 适用场景 | 通用任务 | 特定任务 |

## 技术细节

### Token空间Diffusion的优势

1. **更高维的潜在空间**: 512维 vs 16维
2. **更平滑的流形**: Token空间经过Transformer学习，更适合diffusion
3. **更好的泛化**: 高维空间表达能力更强

### 关节Token化的设计

```python
# 每个关节token = type_embedding + side_embedding + id_embedding
token = joint_type_embeddings[type](value)
      + side_embedding(side)
      + joint_id_embedding(id)
```

- **Type embedding**: 同类型关节共享（如所有shoulder关节）
- **Side embedding**: 区分左右
- **ID embedding**: 唯一标识每个关节

这种设计使得：
- 新关节可以复用类型embedding
- 参数共享，训练更高效
- 自然支持维度扩展

## 开发者指南

### 添加新的关节类型

1. 在`StateTokenizer`中添加新类型：

```python
self.joint_type_embeddings = nn.ModuleDict({
    'shoulder': nn.Linear(1, embed_dim),
    # ... 现有类型
    'new_type': nn.Linear(1, embed_dim),  # 新类型
})
```

2. 在配置中使用新类型：

```yaml
joints:
  - {idx: 36, type: 'new_type', side: 2, id: 36, name: 'new_joint'}
```

### 自定义Tokenizer

继承基类并实现自己的tokenizer：

```python
class CustomTokenizer(nn.Module):
    def forward(self, inputs):
        # 自定义token化逻辑
        return tokens
```

## 性能优化

### 训练技巧

1. **使用AMP**: `use_amp: True`
2. **调整batch size**: 根据显存调整
3. **Gradient Accumulation**: 模拟更大batch size

### 推理优化

1. **减少采样步数**: `num_inference_steps: 20`（牺牲质量换速度）
2. **使用DDIM**: 更快的采样算法
3. **模型量化**: FP16/INT8量化

## 常见问题

### Q: 如何选择token_embed_dim?

**A**: 512是标准选择，平衡性能和效率。更大（如768、1024）可能提升性能但需要更多资源。

### Q: 如何处理变长序列?

**A**: 使用padding和mask：

```python
loss = diffusion_decoder.compute_loss(actions, context_tokens, mask=mask)
```

### Q: 如何迁移预训练模型?

**A**: 使用`expand_action_dim`方法：

```python
action_tokenizer.expand_action_dim(new_action_dim=36, freeze_old_weights=True)
```

## 引用

如果使用本模块，请引用：

```
VLA Transformer Policy for Humanoid Robot Control
Inspired by OpenVLA and RT-2
```

## License

与主项目相同。

