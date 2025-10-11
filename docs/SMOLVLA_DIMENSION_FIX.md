# SmolVLA维度适配修复说明

## 问题描述

训练SmolVLA时遇到两个关键错误：

### 错误1：权重加载失败
```
size mismatch for model.state_proj.weight: copying a param with shape torch.Size([960, 32]) from checkpoint, the shape in current model is torch.Size([960, 16]).
```

### 错误2：训练时维度不匹配
```
RuntimeError: The size of tensor a (267) must match the size of tensor b (233) at non-singleton dimension 2
```

## 根本原因

- **Kuavo机器人**: 16维动作空间（7+1 左臂 + 7+1 右臂）
- **SmolVLA预训练模型**: 32维动作和状态空间
- **问题**: 直接修改配置为16维会导致无法加载预训练权重

## 解决方案

### 1. 配置文件修复

**文件**: `configs/policy/smolvla_sequential_base.yaml`

```yaml
# 动作空间配置（Kuavo双臂16关节机器人）
# 注意：为了使用SmolVLA预训练权重，state/action维度必须与预训练模型一致（32维）
# Kuavo的16维数据会在数据加载时自动填充到32维（后16维填0）
max_state_dim: 32   # 使用预训练模型的32维状态空间
max_action_dim: 32  # 使用预训练模型的32维动作空间
chunk_size: 50
n_action_steps: 8
```

**关键变更**:
- ✅ `max_state_dim`: 16 → 32
- ✅ `max_action_dim`: 保持32
- ✅ 添加详细注释说明填充策略

### 2. 训练脚本修复

**文件**: `kuavo_train/train_smolvla_sequential.py`

#### 添加填充函数

```python
def pad_tensor_to_target_dim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    将tensor从实际维度填充到目标维度

    Kuavo实际: 16维 → SmolVLA需要: 32维
    填充策略: 后16维填充0
    """
    actual_dim = tensor.shape[-1]
    if actual_dim < target_dim:
        pad_size = target_dim - actual_dim
        pad_shape = list(tensor.shape[:-1]) + [pad_size]
        pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad_tensor], dim=-1)
    return tensor
```

#### 修改DataLoader的collate_fn

在`create_dataloader_with_language`和`create_mixed_dataloader`中：

```python
def collate_fn_with_language(batch):
    """为batch添加language instruction并填充action/state维度"""
    # ... 默认collate处理 ...

    # 填充action和state维度（从Kuavo的16维到SmolVLA的32维）
    for key in batch_dict.keys():
        if isinstance(batch_dict[key], torch.Tensor):
            if 'action' in key.lower():
                # 填充action: 16维 → 32维
                batch_dict[key] = pad_tensor_to_target_dim(batch_dict[key], 32)
            elif 'state' in key.lower() or 'observation.state' in key:
                # 填充state: 16维 → 32维
                batch_dict[key] = pad_tensor_to_target_dim(batch_dict[key], 32)

    return batch_dict
```

### 3. ConfigWrapper修复

**文件**: `kuavo_train/wrapper/policy/smolvla/SmolVLAConfigWrapper.py`

```python
def __post_init__(self):
    super().__post_init__()

    # 验证维度配置
    if self.max_action_dim == 32 and self.max_state_dim == 32:
        print("✅ Using SmolVLA pretrained dimensions (32D). Kuavo 16D data will be auto-padded.")

    print(f"📋 SmolVLA Config Summary (Kuavo):")
    print(f"   - Max Action Dim: {self.max_action_dim} (Kuavo actual: 16, auto-padded)")
    print(f"   - Max State Dim: {self.max_state_dim} (Kuavo actual: 16, auto-padded)")
```

## 工作原理

### 数据流程

```
Kuavo Robot (16D)
    ↓
LeRobotDataset 加载 (16D)
    ↓
collate_fn 自动填充 (16D → 32D)
    ↓
SmolVLA模型输入 (32D)
    ↓
预训练权重正确加载 ✅
```

### 填充策略

- **原始数据**: `[x1, x2, ..., x16]`
- **填充后**: `[x1, x2, ..., x16, 0, 0, ..., 0]`  (后16维填0)
- **推理时**: SmolVLA输出32维，只使用前16维作为Kuavo控制命令

## 验证清单

训练前请确认：

- [x] `max_state_dim = 32`
- [x] `max_action_dim = 32`
- [x] collate_fn包含填充逻辑
- [x] ConfigWrapper显示正确的维度信息

## 预期输出

训练启动时应该看到：

```
✅ Using SmolVLA pretrained dimensions (32D). Kuavo 16D data will be auto-padded.
📋 SmolVLA Config Summary (Kuavo):
   - Max Action Dim: 32 (Kuavo actual: 16, auto-padded)
   - Max State Dim: 32 (Kuavo actual: 16, auto-padded)

======================================================================
📂 Loading SmolVLA from: lerobot/smolvla_base
======================================================================
✅ Loaded weights from HuggingFace: lerobot/smolvla_base
```

**关键**: 不应再看到 `size mismatch` 错误！

## 常见问题

### Q1: 为什么不直接改模型为16维？

**A**: SmolVLA预训练模型的所有层都是按32维设计的。改为16维会：
- 无法加载预训练权重
- 失去预训练的VLM能力
- 需要从头训练（效果差、时间长）

### Q2: 填充0会影响性能吗？

**A**: 不会，因为：
- SmolVLA的Action Expert会学习忽略填充维度
- 训练时所有数据都一致填充
- 实际控制只使用前16维

### Q3: 如何确认填充是否生效？

**A**: 检查训练日志：
```python
# 在训练循环中添加debug（可选）
print(f"Action shape: {batch['action'].shape}")  # 应该是 [B, 50, 32]
```

## 相关文件

- `configs/policy/smolvla_sequential_base.yaml`: 主配置文件
- `kuavo_train/train_smolvla_sequential.py`: 训练脚本（包含填充逻辑）
- `kuavo_train/wrapper/policy/smolvla/SmolVLAConfigWrapper.py`: 配置包装器

## 总结

通过**配置维度对齐（32D）+ 数据自动填充**的策略，成功解决了Kuavo 16维机器人与SmolVLA 32维预训练模型的兼容性问题，既能使用预训练权重，又能适配Kuavo的实际动作空间。
