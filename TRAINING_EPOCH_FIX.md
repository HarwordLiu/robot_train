# 训练Epoch数量问题分析与解决方案

## 🔍 **问题分析**

你遇到的问题是：**指定了10个训练轮次，但实际执行了30次**

### 根本原因

1. **配置文件中的 `universal_stages` 设置**：
   ```yaml
   universal_stages:
     stage1:
       epochs: 30  # 这里设置了30个epoch
     stage2:
       epochs: 70  # 这里设置了70个epoch
     stage3:
       epochs: 100 # 这里设置了100个epoch
     stage4:
       epochs: 100 # 这里设置了100个epoch
   ```

2. **测试训练模式被禁用**：
   ```yaml
   test_training_mode: False  # 不使用测试模式
   test_training_epochs: 10    # 这个设置只在test_training_mode=True时生效
   ```

3. **代码逻辑**：
   - 当 `test_training_mode: False` 时，使用配置文件中的实际epoch数
   - 当 `test_training_mode: True` 时，才会使用 `test_training_epochs` 的值

## ✅ **解决方案**

### 方案1：修改配置文件中的epoch设置（推荐）

```yaml
universal_stages:
  stage1:
    name: 'safety_only'
    layers: ['safety']
    epochs: 10  # 改为10
  stage2:
    name: 'safety_gait'
    layers: ['safety', 'gait']
    epochs: 10  # 改为10
  stage3:
    name: 'safety_gait_manipulation'
    layers: ['safety', 'gait', 'manipulation']
    epochs: 10  # 改为10
  stage4:
    name: 'full_hierarchy'
    layers: ['safety', 'gait', 'manipulation', 'planning']
    epochs: 10  # 改为10
```

### 方案2：启用测试训练模式

```yaml
test_training_mode: True   # 启用测试模式
test_training_epochs: 10   # 每个阶段10个epoch
```

## 📊 **训练流程说明**

### 课程学习阶段
1. **Stage 1**: Safety层 - 10 epochs
2. **Stage 2**: Safety + Gait层 - 10 epochs
3. **Stage 3**: Safety + Gait + Manipulation层 - 10 epochs
4. **Stage 4**: 全分层架构 - 10 epochs

**总训练轮次**: 4 × 10 = 40 epochs（课程学习阶段）

### 主要训练循环
- 在课程学习完成后，还会进行主要训练循环
- 主要训练循环使用 `max_epoch: 500` 设置

## 🎯 **建议的训练策略**

### 快速验证模式（当前推荐）
```yaml
test_training_mode: True
test_training_epochs: 5  # 每个阶段5个epoch，总共20个epoch
```

### 正常训练模式
```yaml
test_training_mode: False
universal_stages:
  stage1: {epochs: 20}
  stage2: {epochs: 30}
  stage3: {epochs: 40}
  stage4: {epochs: 50}
```

## 🔧 **已实施的修复**

1. ✅ 将 `universal_stages` 中所有阶段的epochs改为10
2. ✅ 将 `min_epochs_per_phase` 改为10
3. ✅ 保持 `test_training_mode: False` 以使用配置文件设置

现在重新运行训练应该会看到每个阶段只训练10个epoch。
