# SmolVLA 灵活视觉层冻结功能 - 快速开始

## 🎉 新功能概述

现在你可以**灵活配置 SmolVLA 视觉编码器的冻结策略**，精确控制哪些层参与训练，哪些层保持冻结！

### ✨ 核心优势

1. **提高推理成功率**：只解冻任务相关的高层，保护通用视觉特征
2. **防止灾难性遗忘**：在顺序多任务学习中，逐步增加冻结层数
3. **减少过拟合**：冻结大部分层，提供强正则化效果
4. **节省训练资源**：冻结层不计算梯度，降低显存和计算需求

---

## 🚀 快速使用

### 方式 1: 使用负数索引（最推荐）

```yaml
# 在配置文件中添加：
policy:
  unfreeze_vision_layers: [-1, -2, -3, -4, -5]  # 只解冻最后5层
```

**优点**：直观、灵活，不需要知道总层数

### 方式 2: 使用比例（最简单）

```yaml
policy:
  freeze_vision_ratio: 0.75  # 冻结前75%的层
```

**优点**：一个数字搞定，适合快速实验

### 方式 3: 指定要冻结的层

```yaml
policy:
  freeze_vision_layers: [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 冻结前9层
```

**优点**：精确控制，适合高级用户

---

## 📋 推荐配置速查表

| 任务 | 解冻层数 | 配置 | 适用场景 |
|-----|---------|------|---------|
| **任务1** | 9层 | `unfreeze_vision_layers: [-1, -2, -3, -4, -5, -6, -7, -8, -9]` | 从预训练开始，快速收敛 |
| **任务2** | 7层 | `unfreeze_vision_layers: [-1, -2, -3, -4, -5, -6, -7]` | 防止遗忘任务1 |
| **任务3** | 5层 | `unfreeze_vision_layers: [-1, -2, -3, -4, -5]` | 最保守，保护任务1+2 |
| **任务4** | 3层 | `unfreeze_vision_layers: [-1, -2, -3]` | 极度保守，整合所有任务 |

---

## 🎯 使用示例

### 示例 1: 训练任务1（默认配置已添加）

```bash
python kuavo_train/train_smolvla_sequential.py \
    --config-path=../configs/policy \
    --config-name=smolvla_sequential_base \
    task=tasks/task1_moving_grasp
```

**效果**：解冻 Layer 18-26（最后9层），冻结 Layer 0-17

### 示例 2: 自定义冻结策略

编辑 `configs/policy/tasks/task1_moving_grasp.yaml`：

```yaml
policy:
  # 实验不同策略
  unfreeze_vision_layers: [-1, -2, -3]  # 只解冻最后3层（更保守）
```

### 示例 3: 使用比例策略

```yaml
policy:
  freeze_vision_ratio: 0.67  # 冻结前67%的层（前18层）
```

---

## 📊 训练时会看到的日志

```
======================================================================
🔧 应用灵活视觉层冻结策略
======================================================================
Vision Encoder 总层数: 27

策略: 解冻指定层 [-1, -2, -3, -4, -5, -6, -7, -8, -9]

✅ 冻结策略应用完成:
   🔒 冻结层数: 18 / 27
   🔓 解冻层数: 9 / 27
   🔒 冻结层索引: [0...17]
   🔓 解冻层索引: [18...26]
======================================================================
```

---

## 🔍 层级功能速查

详细说明见 `docs/SMOLVLA_VISION_LAYERS_GUIDE.md`

### 快速参考：

- **Layer 0-8（底层）**：边缘、纹理、形状 → ✅ **建议冻结**
- **Layer 9-17（中层）**：物体识别、空间关系 → 🔄 **可选解冻**
- **Layer 18-26（高层）**：抓取点、动作关联 → ✅ **建议解冻**

---

## ⚙️ 高级技巧

### 技巧 1: 配合分层学习率使用

```yaml
policy:
  use_layerwise_lr: True
  vision_encoder_lr: 5.0e-6  # 解冻层的学习率（很低）
  expert_lr: 2.0e-5          # Expert的学习率（较高）
  unfreeze_vision_layers: [-1, -2, -3, -4, -5]
```

### 技巧 2: 根据数据量调整

```yaml
# 数据少（<500 episodes）- 更保守
unfreeze_vision_layers: [-1, -2, -3]

# 数据充足（>2000 episodes）- 更激进
unfreeze_vision_layers: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
```

### 技巧 3: 顺序学习中逐步增加冻结

```yaml
# 任务1: 解冻9层
unfreeze_vision_layers: [-1, -2, -3, -4, -5, -6, -7, -8, -9]

# 任务2: 解冻7层（更保守）
unfreeze_vision_layers: [-1, -2, -3, -4, -5, -6, -7]

# 任务3: 解冻5层（最保守）
unfreeze_vision_layers: [-1, -2, -3, -4, -5]

# 任务4: 解冻3层（极度保守）
unfreeze_vision_layers: [-1, -2, -3]
```

---

## 🐛 常见问题

### Q1: 三种配置方式的优先级？

**优先级（从高到低）：**
1. `unfreeze_vision_layers` （最高）
2. `freeze_vision_layers`
3. `freeze_vision_ratio`
4. `freeze_vision_encoder`（默认行为）

### Q2: 如何验证冻结策略是否生效？

查看训练日志中的 "🔧 应用灵活视觉层冻结策略" 部分，会显示：
- 冻结/解冻的层数
- 具体层索引

### Q3: 可以训练中途更改冻结策略吗？

不建议。冻结策略在模型初始化时应用，中途更改需要重新初始化模型。

### Q4: 负数索引如何工作？

```python
-1 → Layer 26 (最后一层)
-2 → Layer 25 (倒数第二层)
-9 → Layer 18
```

### Q5: 如果不配置这些参数会怎样？

使用默认的 `freeze_vision_encoder: False` 行为，即**全部解冻**。

---

## 📚 相关文档

1. **详细层级说明**：`docs/SMOLVLA_VISION_LAYERS_GUIDE.md`
2. **配置文件**：`configs/policy/smolvla_sequential_base.yaml`
3. **任务配置**：`configs/policy/tasks/task[1-4]_*.yaml`

---

## 💡 最佳实践

### ✅ 推荐做法

1. **任务1从保守策略开始**：解冻9层
2. **后续任务逐步增加冻结**：任务2→7层，任务3→5层，任务4→3层
3. **配合低学习率使用**：解冻层的学习率设为 1e-6 到 5e-6
4. **监控多任务验证**：每个epoch验证所有之前的任务

### ❌ 避免做法

1. **不要过度解冻**：超过50%的层容易过拟合
2. **不要频繁改变策略**：每个任务保持一致的冻结策略
3. **不要忽视学习率**：解冻更多层时必须降低学习率

---

## 🎓 实验建议

### 对比实验

建议进行以下对比实验，找到最优策略：

```bash
# 实验1: 保守策略（解冻5层）
unfreeze_vision_layers: [-1, -2, -3, -4, -5]

# 实验2: 平衡策略（解冻9层，默认）
unfreeze_vision_layers: [-1, -2, -3, -4, -5, -6, -7, -8, -9]

# 实验3: 激进策略（解冻18层）
freeze_vision_ratio: 0.33

# 实验4: 全部解冻（当前做法）
# 注释掉 unfreeze_vision_layers
```

**评估指标**：
- 训练loss收敛速度
- 验证loss
- 多任务验证结果（防遗忘能力）
- 实际推理成功率

---

## 📞 反馈与支持

如果你发现某个冻结策略特别有效，或者遇到问题，欢迎反馈！

**文件修改列表**：
- ✅ `kuavo_train/wrapper/policy/smolvla/SmolVLAConfigWrapper.py`
- ✅ `kuavo_train/wrapper/policy/smolvla/SmolVLAPolicyWrapper.py`
- ✅ `configs/policy/smolvla_sequential_base.yaml`
- ✅ `configs/policy/tasks/task1_moving_grasp.yaml`
- ✅ `configs/policy/tasks/task2_weighing.yaml`
- ✅ `configs/policy/tasks/task3_placement.yaml`
- ✅ `configs/policy/tasks/task4_sorting.yaml`
- ✅ `docs/SMOLVLA_VISION_LAYERS_GUIDE.md`

**更新日期**：2025-10-17
**版本**：v1.0

---

## 🚦 开始训练

现在你已经准备好了！运行训练命令：

```bash
# 训练任务1（已配置好默认策略）
python kuavo_train/train_smolvla_sequential.py \
    --config-path=../configs/policy \
    --config-name=smolvla_sequential_base \
    task=tasks/task1_moving_grasp

# 查看日志，确认冻结策略是否正确应用
```

**Good luck! 🎉**

