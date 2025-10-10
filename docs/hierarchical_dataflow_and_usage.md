# 分层架构 - 数据流与实战指南

> 详细说明数据如何在系统中流动，以及如何实际使用该架构

---

## 目录

1. [完整数据流详解](#1-完整数据流详解)
2. [训练数据流](#2-训练数据流)
3. [推理数据流](#3-推理数据流)
4. [实战示例](#4-实战示例)
5. [调试技巧](#5-调试技巧)
6. [性能优化](#6-性能优化)
7. [常见问题](#7-常见问题)

---

## 1. 完整数据流详解

### 1.1 数据流概览图

```
┌─────────────────────────────────────────────────────────────────┐
│                         原始数据源                               │
│  LeRobot Dataset: /robot/data/task1/lerobot/                   │
│    ├─ episode_0000.parquet                                     │
│    ├─ episode_0001.parquet                                     │
│    └─ ...                                                      │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Dataset 加载                                │
│  LeRobotDataset.__getitem__() 返回:                             │
│    {                                                            │
│      'observation.state': [2, 16],          # [seq_len, dim]   │
│      'observation.images.head_cam_h': [2, 3, 480, 640],        │
│      'observation.images.wrist_cam_l': [2, 3, 480, 640],       │
│      'observation.images.wrist_cam_r': [2, 3, 480, 640],       │
│      'observation.depth.depth_h': [2, 1, 480, 640],            │
│      'observation.depth.depth_l': [2, 1, 480, 640],            │
│      'observation.depth.depth_r': [2, 1, 480, 640],            │
│      'action': [16, 16]                     # [horizon, dim]   │
│    }                                                            │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DataLoader 批处理                           │
│  Batch (batch_size=64):                                         │
│    {                                                            │
│      'observation.state': [64, 2, 16],                         │
│      'observation.images.head_cam_h': [64, 2, 3, 480, 640],    │
│      ... (其他相机)                                              │
│      'action': [64, 16, 16]                                    │
│    }                                                            │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                  图像预处理 (Policy._preprocess_batch)           │
│  1. 裁剪: crop_image(target_range=[420, 560])                  │
│  2. 缩放: resize_image(target_size=[210, 280])                 │
│  3. 堆叠: torch.stack([...], dim=-4)                           │
│                                                                 │
│  输出:                                                          │
│    batch[OBS_IMAGES]: [64, 2, 6, 210, 280]  # 6相机 (3RGB+3D)  │
│    batch[OBS_DEPTH]: [64, 2, 3, 210, 280]                      │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    归一化 (normalize_inputs/targets)             │
│  observation.state: (x - mean) / std                           │
│  observation.images: (x - mean) / std                          │
│  action: (x - mean) / std                                      │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              HierarchicalScheduler 分层处理                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ Layer 1: SafetyReflexLayer                          │       │
│  │   输入: observation.state [64, 2, 16]               │       │
│  │   输出: {                                           │       │
│  │     'emergency': [64] bool,                         │       │
│  │     'balance_action': [64, 16],                     │       │
│  │     'action': [64, 16]                              │       │
│  │   }                                                 │       │
│  └─────────────────────────────────────────────────────┘       │
│                        │                                        │
│                        ▼ (更新context)                          │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ Layer 2: GaitControlLayer                           │       │
│  │   输入: observation.state [64, 2, 16] + context     │       │
│  │   输出: {                                           │       │
│  │     'gait_features': [64, 2, 128],                  │       │
│  │     'action': [64, 16]                              │       │
│  │   }                                                 │       │
│  └─────────────────────────────────────────────────────┘       │
│                        │                                        │
│                        ▼ (更新context)                          │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ Layer 3: ManipulationLayer                          │       │
│  │   输入: observation.* [64, ...] + context           │       │
│  │   处理: 多模态特征融合 + Transformer                 │       │
│  │   输出: {                                           │       │
│  │     'manipulation_features': [64, 1, 512],          │       │
│  │     'action': [64, 16]                              │       │
│  │   }                                                 │       │
│  └─────────────────────────────────────────────────────┘       │
│                        │                                        │
│                        ▼ (更新context)                          │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ Layer 4: GlobalPlanningLayer (可选激活)             │       │
│  │   输入: observation.* [64, ...] + context           │       │
│  │   处理: 长期记忆 + 任务分解                         │       │
│  │   输出: {                                           │       │
│  │     'global_plan': [64, 64],                        │       │
│  │     'action': [64, 16]                              │       │
│  │   }                                                 │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
│  汇总输出: layer_outputs = {                                    │
│    'safety': {...},                                             │
│    'gait': {...},                                               │
│    'manipulation': {...},                                       │
│    'planning': {...}                                            │
│  }                                                              │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│            HierarchicalDiffusionModel 损失计算                   │
│  diffusion_loss = compute_loss(batch)                          │
│  # Diffusion模型的标准损失（不融合layer_outputs）               │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              分层损失聚合 (_aggregate_hierarchical_loss)         │
│                                                                 │
│  total_loss = diffusion_loss                                   │
│                                                                 │
│  for layer_name in enabled_layers:                             │
│    if 'loss' in layer_outputs[layer_name]:                     │
│      layer_weight = task_layer_weights[layer_name]             │
│      layer_loss = layer_outputs[layer_name]['loss']            │
│      total_loss += layer_weight * layer_loss                   │
│                                                                 │
│  例如:                                                          │
│    total_loss = diffusion_loss                                 │
│                + 2.0 * safety_loss                             │
│                + 1.0 * gait_loss                               │
│                + 2.0 * manipulation_loss                       │
│                + 0.5 * planning_loss                           │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    反向传播 & 优化器步骤                         │
│  scaled_loss = total_loss / accumulation_steps                 │
│  scaler.scale(scaled_loss).backward()                          │
│  scaler.step(optimizer)                                        │
│  lr_scheduler.step()                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 张量维度追踪

#### 训练时 (batch_size=64)

| 阶段 | Key | Shape | 说明 |
|---|---|---|---|
| Dataset | observation.state | [2, 16] | 2个时间步，16维状态 |
| DataLoader | observation.state | [64, 2, 16] | 批处理 |
| Dataset | observation.images.head_cam_h | [2, 3, 480, 640] | 2步，RGB |
| DataLoader | observation.images.head_cam_h | [64, 2, 3, 480, 640] | 批处理 |
| Preprocess | observation.images.head_cam_h | [64, 2, 3, 210, 280] | 裁剪缩放 |
| Preprocess | OBS_IMAGES | [64, 2, 6, 210, 280] | 堆叠6相机 |
| SafetyLayer | robot_state (input) | [64, 2, 16] | 输入 |
| SafetyLayer | gru_output | [64, 2, 64] | GRU输出 |
| SafetyLayer | action (output) | [64, 16] | 动作输出 |
| ManipulationLayer | state_features | [64, 1, 16] | 状态 |
| ManipulationLayer | combined_visual | [64, 1, 1280] | 视觉 |
| ManipulationLayer | projected_features | [64, 1, 512] | 投影 |
| ManipulationLayer | manipulation_features | [64, 1, 512] | Transformer输出 |
| ManipulationLayer | action (output) | [64, 16] | 动作输出 |

#### 推理时 (batch_size=1)

| 阶段 | Key | Shape | 说明 |
|---|---|---|---|
| Env | observation.state | [16] | 单个时间步 |
| Policy | observation.state | [1, 1, 16] | 添加batch和seq维度 |
| SafetyLayer | action | [1, 16] | 单样本输出 |
| Final | action | [16] | 移除batch维度 |

---

## 2. 训练数据流

### 2.1 单个训练步骤

```python
# 伪代码，展示完整流程
for epoch in range(max_epoch):
    for batch in dataloader:
        # 1. 数据移到GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        # batch['observation.state']: [64, 2, 16]
        # batch['action']: [64, 16, 16]

        # 2. 前向传播
        with autocast(amp_enabled):
            loss, hierarchical_info = policy.forward(batch, curriculum_info)

        # 内部流程:
        #   a. _preprocess_batch(batch)
        #      - crop_image, resize_image
        #      - 堆叠图像 -> OBS_IMAGES

        #   b. normalize_inputs(batch)
        #      - 归一化观测

        #   c. normalize_targets(batch)
        #      - 归一化动作

        #   d. _identify_task(batch, curriculum_info)
        #      - 推断任务类型

        #   e. scheduler(batch, task_info)
        #      - 按优先级调用各层
        #      - 返回 layer_outputs

        #   f. diffusion.compute_loss(batch, layer_outputs)
        #      - 计算Diffusion损失

        #   g. _aggregate_hierarchical_loss(diffusion_loss, layer_outputs)
        #      - 聚合所有层的损失

        # 3. 反向传播
        scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()

        # 4. 优化器步骤
        if steps % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()

        # 5. 记录日志
        if steps % log_freq == 0:
            writer.add_scalar("train/loss", scaled_loss.item(), steps)
            # 记录分层信息
            for key, value in hierarchical_info.items():
                writer.add_scalar(f"hierarchical/{key}", value, steps)
```

### 2.2 课程学习数据流

```python
# 阶段1: 只训练 manipulation 层
curriculum_info = {
    'stage': 'manipulation_first',
    'enabled_layers': ['manipulation']
}

# Policy内部:
self.enabled_layers = ['manipulation']

# 损失聚合:
total_loss = diffusion_loss + 2.0 * manipulation_loss
# safety_loss, gait_loss, planning_loss 不计算

# --------------------

# 阶段2: 训练 manipulation + safety
curriculum_info = {
    'stage': 'manipulation_safety',
    'enabled_layers': ['manipulation', 'safety']
}

self.enabled_layers = ['manipulation', 'safety']

total_loss = diffusion_loss + 2.0 * manipulation_loss + 2.0 * safety_loss
# gait_loss, planning_loss 不计算
```

### 2.3 任务特定训练数据流

```python
# 多任务DataLoader
datasets = {
    1: task1_dataset,  # 300 episodes
    2: task2_dataset,  # 200 episodes
}

# 加权采样
task_weights = {1: 0.6, 2: 0.4}  # 基于复杂度和数据量
sample_weights = [0.6] * 300 + [0.4] * 200
sampler = WeightedRandomSampler(sample_weights, ...)

dataloader = DataLoader(combined_dataset, sampler=sampler, ...)

# 训练循环
for batch in dataloader:
    # batch中可能包含来自不同任务的样本
    # 根据task_id获取特定权重
    task_loss_weights = task_manager.get_task_specific_loss_weights(batch)

    loss, info = policy.forward(batch, task_weights=task_loss_weights)
```

---

## 3. 推理数据流

### 3.1 在线推理 (Real-time)

```python
# 初始化
policy = HumanoidDiffusionPolicy.from_pretrained(checkpoint_path)
policy.eval()
policy.to(device)
policy.reset()  # 清空observation queue

# 推理循环
for step in range(max_steps):
    # 1. 获取观测
    obs = env.get_observation()
    # obs = {
    #   'observation.state': [16],  # 注意：没有batch维度
    #   'observation.images.head_cam_h': [3, 480, 640],
    #   ...
    # }

    # 2. 移到GPU并添加batch维度
    obs = {k: torch.from_numpy(v).unsqueeze(0).to(device)
           for k, v in obs.items()}
    # obs = {
    #   'observation.state': [1, 16],
    #   'observation.images.head_cam_h': [1, 3, 480, 640],
    #   ...
    # }

    # 3. 选择动作
    with torch.no_grad():
        action = policy.select_action(obs)
    # action: [1, 16]

    # 4. 移除batch维度，移到CPU
    action = action.squeeze(0).cpu().numpy()
    # action: [16]

    # 5. 反归一化
    action = denormalize_action(action, dataset_stats)

    # 6. 执行动作
    obs_next, reward, done, info = env.step(action)
```

### 3.2 推理模式 (Inference Mode with Latency Budget)

```python
policy = HumanoidDiffusionPolicy.from_pretrained(checkpoint_path)
policy.eval()

for step in range(max_steps):
    obs = env.get_observation()
    obs = preprocess_obs(obs)

    # 设置延迟预算
    task_info = {
        'latency_budget_ms': 50.0,  # 50ms预算
        'requires_locomotion': True,
        'requires_manipulation': True,
        'task_complexity': 'medium'
    }

    with torch.no_grad():
        # HierarchicalScheduler会根据预算自适应激活层
        layer_outputs = policy.scheduler.inference_mode(
            obs, task_info, latency_budget_ms=50.0
        )

    # 从层输出提取动作
    action = policy._extract_action_from_layers(layer_outputs, obs)

    # 检查是否在预算内
    inference_stats = layer_outputs['_inference_stats']
    if not inference_stats['within_budget']:
        print(f"⚠️  超出预算: {inference_stats['total_time_ms']:.1f}ms")

    env.step(action)
```

### 3.3 离线评估

```python
# 离线评估：在保存的episode上评估
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repoid, root=data_path)
policy = HumanoidDiffusionPolicy.from_pretrained(checkpoint_path)
policy.eval()

total_mse = 0.0
num_samples = 0

for i in range(len(dataset)):
    sample = dataset[i]

    # 提取观测和真实动作
    obs = {k: v for k, v in sample.items() if k.startswith('observation.')}
    true_action = sample['action'][0]  # 第一个动作

    # 预测动作
    with torch.no_grad():
        pred_action = policy.select_action(obs)

    # 计算误差
    mse = torch.mean((pred_action - true_action) ** 2)
    total_mse += mse.item()
    num_samples += 1

avg_mse = total_mse / num_samples
print(f"平均MSE: {avg_mse:.4f}")
```

---

## 4. 实战示例

### 4.1 从零开始训练

```bash
# 1. 准备数据
# 确保数据在正确位置: /robot/data/task1/lerobot/

# 2. 检查配置
# configs/policy/humanoid_diffusion_config.yaml
# - root: 数据路径
# - use_hierarchical: True
# - curriculum_learning.enable: True

# 3. 开始训练
python kuavo_train/train_hierarchical_policy.py \
  --config-name=humanoid_diffusion_config

# 4. 监控训练
tensorboard --logdir outputs/train/task_400_episodes/humanoid_hierarchical/

# 5. 检查日志
tail -f hierarchical_training.log
```

### 4.2 课程学习训练

```python
# 配置文件中定义课程学习阶段
curriculum_learning:
  enable: True
  universal_stages:
    stage1:
      name: 'manipulation_first'
      layers: ['manipulation']
      epochs: 8
    stage2:
      name: 'manipulation_safety'
      layers: ['manipulation', 'safety']
      epochs: 7
    # ...

# 训练时自动执行课程学习
# 输出日志:
# 🎓 开始课程学习阶段: stage1_manipulation_first
#    激活层: ['manipulation']
#    训练轮次: 8
# ...
# ✅ 课程阶段 stage1_manipulation_first 完成，最佳损失: 0.0523
```

### 4.3 恢复训练

```bash
# 1. 找到之前的检查点
ls outputs/train/task_400_episodes/humanoid_hierarchical/

# 2. 修改配置
training:
  resume: True
  resume_timestamp: "run_20250110_123456"

# 3. 恢复训练
python kuavo_train/train_hierarchical_policy.py \
  --config-name=humanoid_diffusion_config

# 输出日志:
# 从检查点恢复训练: outputs/.../run_20250110_123456
# ✅ Policy 保存成功
# 已恢复训练从epoch 150, step 12000
```

### 4.4 部署到真实机器人

```python
# eval_kuavo.py
from kuavo_train.wrapper.policy.humanoid.HumanoidDiffusionPolicy import HumanoidDiffusionPolicy

# 1. 加载模型
checkpoint_path = "outputs/train/.../best"
policy = HumanoidDiffusionPolicy.from_pretrained(
    checkpoint_path,
    use_hierarchical=True,
    hierarchical=hierarchical_config  # 需要提供hierarchical配置
)
policy.eval()
policy.to('cuda')

# 2. 初始化环境
from kuavo_deploy.kuavo_env.kuavo_real_env import KuavoRealEnv
env = KuavoRealEnv(config)

# 3. 推理循环
obs, info = env.reset()
policy.reset()

for step in range(1000):
    # 选择动作
    action = policy.select_action(obs)

    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action.numpy())

    # 记录层输出（可选）
    layer_outputs = policy.get_last_layer_outputs()
    if layer_outputs and 'safety' in layer_outputs:
        if layer_outputs['safety']['emergency']:
            print("🚨 检测到紧急情况！")
            break

    if terminated or truncated:
        break

env.close()
```

---

## 5. 调试技巧

### 5.1 查看层激活情况

```python
# 在训练循环中
loss, hierarchical_info = policy.forward(batch, curriculum_info)

# hierarchical_info 包含:
# {
#   'curriculum_stage': 'stage2_manipulation_safety',
#   'enabled_layers': ['manipulation', 'safety'],
#   'task_weights': {'safety': 2.0, 'manipulation': 2.0, ...},
#   'layer_performance': {
#     'safety': {'loss': 0.05, 'execution_time': 8.2, 'active': True},
#     'manipulation': {'loss': 0.08, 'execution_time': 95.3, 'active': True},
#     'gait': {'active': False},
#     'planning': {'active': False}
#   }
# }

# 打印激活层
active_layers = [k for k, v in hierarchical_info['layer_performance'].items()
                 if v.get('active', False)]
print(f"激活层: {active_layers}")
```

### 5.2 可视化损失分解

```python
# 在TensorBoard中查看
writer.add_scalar("hierarchical/total_loss", total_loss.item(), steps)
writer.add_scalar("hierarchical/diffusion_loss", diffusion_loss.item(), steps)

for layer_name, layer_output in layer_outputs.items():
    if 'loss' in layer_output:
        writer.add_scalar(f"hierarchical/{layer_name}_loss",
                         layer_output['loss'].item(), steps)
```

### 5.3 检查数据维度

```python
def check_batch_shapes(batch, stage=""):
    """检查batch中所有张量的维度"""
    print(f"\n{'='*60}")
    print(f"Batch shapes at stage: {stage}")
    print(f"{'='*60}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:40s}: {list(value.shape)}")
    print(f"{'='*60}\n")

# 在训练循环中使用
check_batch_shapes(batch, "After DataLoader")
batch = policy._preprocess_batch(batch)
check_batch_shapes(batch, "After Preprocess")
```

### 5.4 单步调试

```python
# 测试单个层
layer = policy.scheduler.layers['safety']
layer.eval()

# 构造测试输入
test_input = {
    'observation.state': torch.randn(1, 1, 16).to(device)
}

# 前向传播
with torch.no_grad():
    output = layer.forward(test_input)

print(f"Emergency: {output['emergency']}")
print(f"Balance action: {output['balance_action'].shape}")
```

### 5.5 性能分析

```python
# 使用PyTorch Profiler
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    for i, batch in enumerate(dataloader):
        if i >= 10:  # 只分析10个batch
            break
        loss, info = policy.forward(batch)
        loss.backward()

# 打印统计信息
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 导出Chrome trace
prof.export_chrome_trace("trace.json")
# 在 chrome://tracing 中查看
```

---

## 6. 性能优化

### 6.1 内存优化

#### 梯度累积

```python
# 在配置中设置
training:
  batch_size: 32            # 减小实际batch size
  accumulation_steps: 2     # 累积2步，等效batch_size=64

# 训练循环会自动处理
scaled_loss = loss / accumulation_steps
scaled_loss.backward()

if steps % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

#### 混合精度训练

```python
# 在配置中启用
policy:
  use_amp: True

# 自动使用AMP
scaler = torch.amp.GradScaler(enabled=True)

with autocast(amp_enabled=True):
    loss, info = policy.forward(batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 内存释放

```python
# 在训练循环中定期清理
if steps % 100 == 0:
    torch.cuda.empty_cache()
```

### 6.2 速度优化

#### DataLoader优化

```python
dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=25,           # 多进程加载
    pin_memory=True,          # 固定内存，加速传输
    prefetch_factor=2,        # 预取2个batch
    persistent_workers=True   # 保持worker进程
)
```

#### 编译模型 (PyTorch 2.0+)

```python
# 编译Policy模型
policy = torch.compile(policy, mode="reduce-overhead")

# 或者只编译特定层
policy.scheduler.layers['manipulation'] = torch.compile(
    policy.scheduler.layers['manipulation']
)
```

#### 减少不必要的计算

```python
# 推理时关闭不必要的层
policy.scheduler.set_layer_enabled('planning', False)

# 或者使用inference_mode
layer_outputs = policy.scheduler.inference_mode(
    batch, task_info, latency_budget_ms=50.0
)
```

### 6.3 分布式训练

```python
# 使用PyTorch DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化分布式
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# 包装模型
policy = DDP(policy.to(local_rank), device_ids=[local_rank])

# 训练循环
for batch in dataloader:
    batch = {k: v.to(local_rank) for k, v in batch.items()}
    loss, info = policy(batch)
    loss.backward()
    optimizer.step()
```

---

## 7. 常见问题

### 7.1 维度不匹配

**问题**: `RuntimeError: size mismatch`

**原因**: 输入维度与层期望的不一致

**解决**:
```python
# 检查配置
# humanoid_diffusion_config.yaml
hierarchical:
  layers:
    safety:
      input_dim: 16   # 确保与实际状态维度一致
      output_dim: 16

# 如果机器人配置改变（如启用双腿），需要更新input_dim
```

### 7.2 OOM (Out of Memory)

**问题**: `CUDA out of memory`

**解决**:
1. 减小batch_size
2. 启用梯度累积
3. 启用AMP
4. 减少图像分辨率
5. 禁用不必要的层

```python
# 配置调整
training:
  batch_size: 32             # 从64减小到32
  accumulation_steps: 2      # 等效batch_size=64

policy:
  use_amp: True              # 启用混合精度

  custom:
    resize_shape: [180, 240]  # 从[210, 280]减小
```

### 7.3 训练不收敛

**问题**: Loss不下降或发散

**可能原因及解决**:

1. **学习率过高**
```python
policy:
  optimizer_lr: 0.00005  # 从0.0001减小
```

2. **层权重不平衡**
```python
hierarchical:
  layer_weights:
    safety: 1.0      # 从2.0减小
    manipulation: 1.0
    # 避免某层权重过大导致训练不稳定
```

3. **课程学习阶段epochs不足**
```python
universal_stages:
  stage1:
    epochs: 15  # 从8增加到15
```

4. **数据归一化问题**
```python
# 检查dataset_stats是否正确
print(dataset_stats['observation.state']['mean'])
print(dataset_stats['observation.state']['std'])
```

### 7.4 推理速度慢

**问题**: 推理延迟超过预算

**解决**:

1. **使用推理模式**
```python
layer_outputs = policy.scheduler.inference_mode(
    batch, task_info, latency_budget_ms=50.0
)
```

2. **禁用复杂层**
```python
policy.scheduler.set_layer_enabled('planning', False)
```

3. **减小模型大小**
```python
# 使用更小的配置
hierarchical:
  layers:
    manipulation:
      hidden_size: 256  # 从512减小
      layers: 2         # 从3减小
```

4. **使用TensorRT优化**
```python
import torch_tensorrt

# 编译为TensorRT
trt_model = torch_tensorrt.compile(
    policy,
    inputs=[torch_tensorrt.Input((1, 1, 16))],
    enabled_precisions={torch.float16}
)
```

### 7.5 层不激活

**问题**: 某些层从不激活

**检查**:
```python
# 查看should_activate逻辑
layer = policy.scheduler.layers['planning']
should_activate = layer.should_activate(batch, context)
print(f"Planning layer should activate: {should_activate}")

# 检查context
print(f"Task complexity: {context.get('task_complexity')}")
# planning层只在task_complexity='high'或'very_high'时激活

# 解决：在task_info中设置正确的复杂度
task_info = {
    'task_complexity': 'high'  # 改为'high'
}
```

### 7.6 课程学习跳过

**问题**: 课程学习阶段被跳过

**检查**:
```python
# 1. 确认配置启用
curriculum_learning:
  enable: True  # 必须为True

# 2. 检查stages配置
universal_stages:
  stage1:  # 必须有定义
    name: '...'
    layers: [...]
    epochs: 10

# 3. 检查训练日志
# 应该看到: 🎓 启动课程学习，共X个阶段
```

---

## 总结

本文详细说明了分层架构中的数据流动：

1. **训练数据流**: Dataset → DataLoader → Preprocess → Normalize → HierarchicalScheduler → Loss → Backward
2. **推理数据流**: Env → Observation → Policy → HierarchicalScheduler → Action → Env
3. **实战示例**: 从训练到部署的完整流程
4. **调试技巧**: 如何定位和解决问题
5. **性能优化**: 内存、速度、分布式训练
6. **常见问题**: 典型错误及解决方案

---

**相关文档**:
- [主架构文档](hierarchical_policy_architecture.md)
- [层详细设计](hierarchical_layers_detailed.md)
- [训练脚本](../kuavo_train/train_hierarchical_policy.py)

