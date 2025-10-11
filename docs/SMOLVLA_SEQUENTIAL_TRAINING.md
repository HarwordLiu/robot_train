# SmolVLA顺序多任务训练指南

## 📋 概述

本文档介绍如何使用SmolVLA进行顺序多任务训练（Sequential Multi-Task Fine-tuning）。

**目标**：训练一个能够执行4个不同机器人任务的多任务模型。

**策略**：顺序Fine-tuning（策略B）
- Stage 1: HuggingFace预训练 → 任务1模型
- Stage 2: 任务1模型 → 任务2模型
- Stage 3: 任务2模型 → 任务3模型
- Stage 4: 任务3模型 → 任务4模型（最终多任务模型）

**防遗忘技术**：
1. ✅ Replay Buffer - 混合之前任务数据
2. ✅ Lower Learning Rate - 逐步降低学习率
3. ✅ Freeze Layers - 冻结VLM底层
4. ✅ Multi-Task Validation - 定期验证所有任务

---

## 🎯 四个机器人任务

### 任务1：移动目标抓取
- **描述**：机器人从移动的传送带上抓取物体，放置于桌面后，再将其推送至指定区域内
- **Language**: "Pick up the moving object from the conveyor belt, place it on the table, and push it to the designated area"
- **数据路径**: `/root/robot/data/task1/data/lerobot/1-400/`
- **Episodes**: 200个

### 任务2：快递袋称重
- **描述**：机器人从移动的传送带上拾取快递袋，先放置在电子秤上完成称重，随后再次拾起并放入指定收纳筐中
- **Language**: "Pick up the package from the conveyor belt, weigh it on the electronic scale, then pick it up again and place it in the designated storage container"
- **数据路径**: `/root/robot/data/task2/data/lerobot/1-400/`
- **Episodes**: 200个

### 任务3：日化产品定姿摆放
- **描述**：机器人从杂乱摆放的日化瓶中随机拾取一瓶，传递至另一只手后，再按照指定姿态将其摆放在目标位置
- **Language**: "Pick up a bottle from the cluttered daily chemical bottles, transfer it to the other hand, and place it in the specified pose with the label facing up..."
- **数据路径**: `/root/robot/data/task3/data/lerobot/1-400/`
- **Episodes**: 200个

### 任务4：全流程分拣
- **描述**：机器人从指定起始点出发，移动至流利架前拾取工件，随后转身移动至放置架，将工件放置在物料筐内指定位置
- **Language**: "Move from the starting point to the rack, pick up the workpiece, turn around, move to the placement rack, and place it in the designated position..."
- **数据路径**: `/root/robot/data/task4/data/lerobot/1-400/`
- **Episodes**: 200个

---

## 🚀 快速开始

### 前置要求

1. **Python环境**：Python 3.10+
2. **GPU**：NVIDIA GPU with ≥16GB VRAM（推荐24GB+）
3. **依赖安装**：
```bash
pip install -r requirements_total.txt
```

4. **数据准备**：确保4个任务的数据已转换为LeRobot格式

---

## 📝 训练流程

### Stage 1: 训练任务1（移动目标抓取）

```bash
# 从HuggingFace预训练SmolVLA开始
python kuavo_train/train_smolvla_sequential.py \
  --config-path=../configs/policy \
  --config-name=smolvla_sequential_base \
  task=tasks/task1_moving_grasp
```

**预期输出**：
- 训练时间：~2-3小时（20 epochs）
- 最终loss：< 0.5
- 模型保存：`outputs/train/smolvla_sequential/task1_moving_grasp/best/`

**检查点**：
- [ ] Loss曲线下降正常
- [ ] 最终loss < 0.5
- [ ] 模型能成功抓取物体

---

### Stage 2: 训练任务2（快递袋称重）

```bash
# 自动从任务1的checkpoint继续训练
python kuavo_train/train_smolvla_sequential.py \
  --config-path=../configs/policy \
  --config-name=smolvla_sequential_base \
  task=tasks/task2_weighing
```

**关键特性**：
- ✅ 自动加载任务1模型
- ✅ 混合20%任务1数据（Replay Buffer）
- ✅ 学习率降至0.00005
- ✅ 每2个epoch验证任务1和2

**预期输出**：
- 训练时间：~3-4小时（25 epochs）
- 任务2 loss：< 0.5
- **任务1 loss：< 0.7**（防遗忘验证）

**检查点**：
- [ ] 任务2 loss下降
- [ ] 任务1 loss保持稳定（< 0.7）
- [ ] 模型能执行称重任务

⚠️ **如果任务1 loss > 0.8，说明有遗忘，需要调整replay ratio**

---

### Stage 3: 训练任务3（日化产品摆放）

```bash
python kuavo_train/train_smolvla_sequential.py \
  --config-path=../configs/policy \
  --config-name=smolvla_sequential_base \
  task=tasks/task3_placement
```

**Replay策略**：
- 10% 任务1
- 20% 任务2
- 70% 任务3

**预期输出**：
- 训练时间：~4-5小时（30 epochs）
- 任务3 loss：< 0.5
- 任务1/2 loss：< 0.8

**检查点**：
- [ ] 任务3 loss下降
- [ ] 任务1/2 loss保持稳定
- [ ] 模型能按姿态摆放物体

---

### Stage 4: 训练任务4（全流程分拣）

```bash
python kuavo_train/train_smolvla_sequential.py \
  --config-path=../configs/policy \
  --config-name=smolvla_sequential_base \
  task=tasks/task4_sorting
```

**Replay策略**（最平衡）：
- 10% 任务1
- 10% 任务2
- 20% 任务3
- 60% 任务4

**预期输出**：
- 训练时间：~5-6小时（35 epochs）
- 所有任务loss：< 0.7
- **这是最终多任务模型！**

**最终检查点**：
- [ ] 任务4 loss < 0.5
- [ ] 任务1/2/3 loss < 1.0
- [ ] **模型能通过language切换4个任务**

---

## 📊 监控训练

### TensorBoard

```bash
tensorboard --logdir=outputs/train/smolvla_sequential/
```

**关键曲线**：
1. `train/loss` - 当前任务训练loss
2. `train/lr` - 学习率变化
3. `validation/task1_loss` - 任务1验证loss（监控遗忘）
4. `validation/task2_loss` - 任务2验证loss
5. `validation/task3_loss` - 任务3验证loss
6. `validation/task4_loss` - 任务4验证loss

### 验证日志

训练过程中会定期输出多任务验证结果：

```
🔍 Multi-Task Validation (Tasks 1-2)
======================================================================

📊 Validating Task 1...
  Task 1 Validation Loss: 0.58

📊 Validating Task 2...
  Task 2 Validation Loss: 0.45

⚠️  Forgetting Analysis:
  Task 1: ✅ Well Retained (loss=0.58)
======================================================================
```

---

## 🎯 成功标准

### Stage 1 成功标准
- ✅ 任务1 loss < 0.5
- ✅ 模型能抓取移动物体

### Stage 2 成功标准
- ✅ 任务2 loss < 0.5
- ✅ **任务1 loss < 0.7**（允许最多40%退化）
- ✅ 模型能称重并放置

### Stage 3 成功标准
- ✅ 任务3 loss < 0.5
- ✅ 任务1/2 loss < 0.8
- ✅ 模型能按姿态摆放

### Stage 4 成功标准（最终）
- ✅ 任务4 loss < 0.5
- ✅ **所有任务loss < 1.0**
- ✅ **模型能用language切换任务**

---

## 🔧 故障排除

### 问题1：GPU内存不足

**症状**：CUDA Out of Memory

**解决方案**：
```yaml
# 修改 smolvla_sequential_base.yaml
training:
  batch_size: 8  # 从16降至8
```

### 问题2：任务1严重遗忘

**症状**：训练任务2后，任务1 loss > 1.0

**解决方案**：
```yaml
# 修改 smolvla_sequential_base.yaml
sequential:
  stage2_replay:
    task1: 0.3  # 从0.2增加到0.3
    task2: 0.7
```

### 问题3：训练速度太慢

**解决方案**：
```yaml
training:
  num_workers: 4  # 减少workers
  batch_size: 24  # 增加batch size（如果GPU允许）
```

### 问题4：Loss不下降

**可能原因**：
1. 学习率太小
2. 数据质量问题
3. Replay比例不当

**诊断方法**：
```bash
# 查看训练日志
tail -f smolvla_sequential_training.log

# 检查TensorBoard
tensorboard --logdir=outputs/train/smolvla_sequential/
```

---

## 📁 输出文件结构

```
outputs/
└── train/
    └── smolvla_sequential/
        ├── task1_moving_grasp/
        │   ├── best/
        │   │   ├── model.safetensors
        │   │   └── config.json
        │   ├── epoch5/
        │   ├── epoch10/
        │   └── training_results.json
        │
        ├── task2_weighing/
        │   ├── best/  ← 会任务1+2
        │   └── training_results.json
        │
        ├── task3_placement/
        │   ├── best/  ← 会任务1+2+3
        │   └── training_results.json
        │
        └── task4_sorting/
            ├── best/  ← 最终多任务模型
            └── training_results.json
```

### training_results.json 示例

```json
{
  "task_id": 4,
  "task_name": "sorting",
  "description": "全流程分拣任务...",
  "language_instruction": "Move from the starting point...",
  "best_loss": 0.52,
  "final_validation": {
    "1": 0.65,  // 任务1仍然work
    "2": 0.58,  // 任务2仍然work
    "3": 0.50,  // 任务3仍然work
    "4": 0.52   // 任务4新学会
  },
  "training_epochs": 35,
  "learning_rate": 0.00002
}
```

---

## 🧪 测试最终模型

### 单任务测试

```python
from kuavo_train.wrapper.policy.smolvla.SmolVLAPolicyWrapper import SmolVLAPolicyWrapper

# 加载最终模型
policy = SmolVLAPolicyWrapper.from_pretrained(
    'outputs/train/smolvla_sequential/task4_sorting/best'
)

# 测试任务1
batch['task'] = "Pick up the moving object from the conveyor belt..."
action = policy.select_action(batch)

# 测试任务2
batch['task'] = "Pick up the package from the conveyor belt, weigh it..."
action = policy.select_action(batch)
```

### 多任务切换测试

```python
# 验证模型能根据language instruction切换任务
tasks = [
    ("任务1", "Pick up the moving object..."),
    ("任务2", "Pick up the package..."),
    ("任务3", "Pick up a bottle..."),
    ("任务4", "Move from the starting point..."),
]

for task_name, instruction in tasks:
    batch['task'] = instruction
    action = policy.select_action(batch)
    print(f"{task_name}: {action.shape}")
```

---

## ⏱️ 预计时间

- **实现时间**：已完成 ✅
- **Stage 1训练**：2-3小时
- **Stage 2训练**：3-4小时
- **Stage 3训练**：4-5小时
- **Stage 4训练**：5-6小时
- **总训练时间**：~15-18小时

---

## 📚 相关文档

- [SmolVLA Paper](https://arxiv.org/abs/2506.01844)
- [HuggingFace SmolVLA](https://huggingface.co/lerobot/smolvla_base)
- [Lerobot Documentation](https://github.com/huggingface/lerobot)

---

## 🎉 预期最终效果

完成所有4个stage后，你将拥有：

1. ✅ **4个独立任务模型**（task1/best, task2/best, task3/best）
2. ✅ **1个多任务模型**（task4/best），能通过language执行所有任务
3. ✅ **完整训练历史**，展示顺序学习过程
4. ✅ **防遗忘验证数据**，证明模型保留了之前任务的知识

**这将是Kuavo项目的第一个多任务机器人学习系统！** 🚀
