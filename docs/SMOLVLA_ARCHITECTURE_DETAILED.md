# SmolVLA 策略架构与技术实现详解

## 1. 概述

SmolVLA (Small Vision-Language-Action) 是基于 HuggingFace 发布的轻量级视觉-语言-动作模型，专门为 Kuavo 双臂机器人项目设计的多任务学习策略。该策略采用顺序微调（Sequential Fine-tuning）方法，通过防遗忘技术实现多任务学习。

### 1.1 核心特点

- **轻量级架构**: 基于 SmolVLM2-500M-Video-Instruct 预训练模型
- **多任务学习**: 支持4个连续任务的顺序学习
- **防遗忘技术**: 使用 Replay Buffer 和冻结策略防止灾难性遗忘
- **Flow Matching**: 使用 Flow Matching 而非传统 Diffusion 进行动作生成
- **维度适配**: 自动处理 Kuavo 16维到 SmolVLA 32维的维度转换

## 2. 架构设计

### 2.1 整体架构

```
输入层 → VLM Backbone → Action Expert → Flow Matching → 动作输出
  ↓           ↓              ↓              ↓
图像+状态    SigLIP视觉     Transformer    动作序列
语言指令     编码器         解码器         生成
```

### 2.2 核心组件

#### 2.2.1 VLM Backbone
- **模型**: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- **视觉编码器**: SigLIP (冻结)
- **语言理解**: 支持自然语言指令
- **参数量**: 约500M参数

#### 2.2.2 Action Expert
- **架构**: Transformer Decoder
- **功能**: 将视觉-语言特征转换为动作序列
- **训练策略**: 只训练 Action Expert，冻结视觉编码器

#### 2.2.3 Flow Matching
- **方法**: Flow Matching (非传统 Diffusion)
- **优势**: 更平滑的生成过程，更好的收敛性
- **步数**: 推理时10步去噪

## 3. 多任务学习策略

### 3.1 顺序训练流程

```
Stage 1: 预训练模型 → 任务1模型 (移动抓取)
Stage 2: 任务1模型 → 任务2模型 (快递称重)
Stage 3: 任务2模型 → 任务3模型 (定姿摆放)
Stage 4: 任务3模型 → 任务4模型 (全流程分拣)
```

### 3.2 防遗忘技术

#### 3.2.1 Replay Buffer
- **策略**: 比例混合策略
- **Stage 2**: 20% 任务1 + 80% 任务2
- **Stage 3**: 10% 任务1 + 20% 任务2 + 70% 任务3
- **Stage 4**: 10% 任务1 + 10% 任务2 + 20% 任务3 + 60% 任务4

#### 3.2.2 冻结策略
- **冻结视觉编码器**: `freeze_vision_encoder: True`
- **只训练 Action Expert**: `train_expert_only: True`
- **训练状态投影层**: `train_state_proj: True`

#### 3.2.3 学习率衰减
- **任务1**: 5e-5 (从预训练开始)
- **任务2**: 3.5e-5 (降低30%)
- **任务3**: 2.5e-5 (进一步降低)
- **任务4**: 2e-5 (最低学习率)

## 4. 技术实现细节

### 4.1 维度处理

#### 4.1.1 维度适配
```python
# Kuavo 实际维度: 16维
# SmolVLA 预训练维度: 32维
# 自动填充策略: 后16维填0

def pad_tensor_to_target_dim(tensor, target_dim: int):
    actual_dim = tensor.shape[-1]
    if actual_dim < target_dim:
        pad_size = target_dim - actual_dim
        pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype)
        return torch.cat([tensor, pad_tensor], dim=-1)
```

#### 4.1.2 归一化处理
```python
# 对于填充部分使用恒等归一化
# mean = 0, std = 1 (填充部分不会被改变)
normalization_mapping:
  STATE:
    value: MEAN_STD  # 使用数据集统计
  ACTION:
    value: MEAN_STD  # 使用数据集统计
```

### 4.2 数据流处理

#### 4.2.1 Action Chunks
```python
# 配置动作序列长度
chunk_size: 50  # Flow Matching 生成50步
n_action_steps: 8  # 每次执行8步

# Delta timestamps 配置
delta_timestamps = {
    "observation.state": [0],  # 当前帧
    "action": [i / dataset_fps for i in range(chunk_size)]  # 未来50帧
}
```

#### 4.2.2 图像预处理
```python
# SmolVLA 标准输入尺寸
resize_imgs_with_padding: [512, 512]
empty_cameras: 0  # 不使用空相机占位
```

### 4.3 训练优化

#### 4.3.1 优化器配置
```python
# 针对 batch_size=64 优化
optimizer_betas: [0.9, 0.999]  # beta2=0.999 对大batch更稳定
optimizer_eps: 1.0e-08
optimizer_weight_decay: 5.0e-7  # 适度正则化
optimizer_grad_clip_norm: 1.0  # VLM embedding空间大，需要严格梯度控制
```

#### 4.3.2 学习率调度
```python
# VLM+Action Expert 异构架构需要更长warmup
scheduler_warmup_steps: 1500
scheduler_decay_steps: 25000  # 充分的cosine decay
scheduler_decay_lr: 1e-6  # 最终学习率衰减到很小
```

## 5. 配置文件结构

### 5.1 基础配置 (smolvla_sequential_base.yaml)

```yaml
# SmolVLA策略配置
policy:
  _target_: kuavo_train.wrapper.policy.smolvla.SmolVLAConfigWrapper

  # VLM Backbone配置
  vlm_model_name: 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
  load_vlm_weights: True

  # 冻结策略
  freeze_vision_encoder: True
  train_expert_only: True
  train_state_proj: True

  # 动作空间配置
  max_state_dim: 32  # 预训练模型维度
  max_action_dim: 32  # 预训练模型维度
  chunk_size: 50
  n_action_steps: 8

  # Flow Matching配置
  num_steps: 10
```

### 5.2 任务配置示例

#### 任务1: 移动抓取
```yaml
task:
  id: 1
  name: 'moving_grasp'
  language_instruction: 'Grasp the object from the conveyor belt using visual guidance. Place it precisely at the first marked target location on the table. Then grasp it again and place it precisely at the second marked target location on the table.'

  training:
    max_epoch: 100
    resume_from: 'pretrained'
    pretrained_path: 'lerobot/smolvla_base'
    policy:
      optimizer_lr: 5e-5
```

#### 任务2: 快递称重
```yaml
task:
  id: 2
  name: 'weighing'
  language_instruction: 'Pick up the package from the conveyor belt, weigh it on the electronic scale, then pick it up again and place it in the designated storage container'

  training:
    max_epoch: 25
    resume_from: 'task'
    resume_task_id: 1
    policy:
      optimizer_lr: 3.5e-5  # 降低学习率
```

## 6. 训练流程

### 6.1 训练脚本 (train_smolvla_sequential.py)

#### 6.1.1 主要功能
- **Replay Buffer 管理**: 自动混合之前任务的数据
- **多任务验证**: 验证所有之前任务的性能
- **维度填充**: 自动处理16维到32维的转换
- **检查点管理**: 支持训练状态保存和恢复

#### 6.1.2 关键类

**ReplayDatasetManager**
```python
class ReplayDatasetManager:
    """管理Replay Buffer的类"""

    def load_replay_tasks(self):
        """加载所有需要replay的任务数据"""
        # 根据stage配置加载之前任务的数据
        # 使用比例混合策略
```

**MixedDataset**
```python
class MixedDataset(torch.utils.data.Dataset):
    """混合多个数据集，每个数据集保留自己的language instruction"""

    def __getitem__(self, idx):
        # 根据weights随机选择一个dataset
        # 从该dataset随机选择一个样本
        # 添加language instruction
```

### 6.2 训练循环

```python
for epoch in range(max_epoch):
    # 训练阶段
    for batch in dataloader:
        loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()

    # 多任务验证
    if epoch % validation_freq_epoch == 0:
        validation_results = validate_all_tasks(policy, cfg, task_id)

    # 保存最佳模型
    if avg_loss < best_loss:
        policy.save_pretrained(best_path)
```

## 7. 推理与部署

### 7.1 推理流程

```python
# 加载模型
policy = SmolVLAPolicyWrapper.from_pretrained(checkpoint_path)

# 准备输入
batch = {
    'observation.images': images,  # [B, C, H, W]
    'observation.state': state,   # [B, 32] (16维填充到32维)
    'task': [language_instruction]  # 任务指令
}

# 生成动作
action = policy.select_action(batch)  # [B, 16] (裁剪回16维)
```

### 7.2 部署配置

```yaml
# kuavo_smolvla_sim_env.yaml
policy:
  _target_: kuavo_train.wrapper.policy.smolvla.SmolVLAPolicyWrapper
  pretrained_name_or_path: "path/to/checkpoint"

# 推理配置
inference:
  num_steps: 10  # Flow Matching步数
  n_action_steps: 8  # 每次执行步数
```

## 8. 性能特点

### 8.1 优势
- **轻量级**: 500M参数，适合实时部署
- **多任务**: 一个模型支持4个不同任务
- **防遗忘**: 有效防止灾难性遗忘
- **语言理解**: 支持自然语言指令

### 8.2 技术挑战
- **维度适配**: 需要处理16维到32维的转换
- **顺序训练**: 需要精心设计的学习率衰减
- **数据混合**: Replay Buffer的比例需要仔细调优

## 9. 使用示例

### 9.1 训练任务1
```bash
python kuavo_train/train_smolvla_sequential.py \
    --config-path=../configs/policy \
    --config-name=smolvla_sequential_base \
    task=tasks/task1_moving_grasp
```

### 9.2 训练任务2
```bash
python kuavo_train/train_smolvla_sequential.py \
    --config-path=../configs/policy \
    --config-name=smolvla_sequential_base \
    task=tasks/task2_weighing
```

### 9.3 推理使用
```python
from kuavo_train.wrapper.policy.smolvla import SmolVLAPolicyWrapper

# 加载模型
policy = SmolVLAPolicyWrapper.from_pretrained("path/to/checkpoint")

# 执行任务
action = policy.select_action({
    'observation.images': images,
    'observation.state': state,
    'task': ['Pick up the package from the conveyor belt']
})
```

## 10. 总结

SmolVLA 策略为 Kuavo 双臂机器人提供了一个高效的多任务学习解决方案。通过顺序微调和防遗忘技术，该策略能够在保持轻量级的同时实现多任务学习，为机器人操作提供了强大的视觉-语言-动作理解能力。

关键技术点包括：
- Flow Matching 动作生成
- Replay Buffer 防遗忘
- 维度自适应处理
- 语言指令理解
- 顺序训练策略

该实现为机器人多任务学习提供了一个完整的解决方案，具有良好的扩展性和实用性。
