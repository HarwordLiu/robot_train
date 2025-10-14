# 优化训练启动指南

## ✅ 已应用的优化

### 1. 语言指令优化
- 强调"visual guidance"、"precisely"、"marked target location"

### 2. 训练参数优化
- **max_epoch**: 100 → 150
- **optimizer_lr**: 0.0001 → 0.00009（配合batch_size=64）
- **scheduler_warmup_steps**: 2000 → 3000
- **scheduler_decay_steps**: 30000 → 40000

### 3. 模型配置优化
- **chunk_size**: 50 → 75（更长动作序列）
- **n_action_steps**: 8 → 10
- **num_steps**: 10 → 15（更精确的Flow Matching）
- **batch_size**: 32 → 64（更稳定梯度）
- **validation_freq_epoch**: 2 → 1（更频繁验证）

### 4. 数据增强
- ✅ 已集成到训练代码
- 50%概率应用增强（颜色抖动、噪声、遮挡等）

---

## 🚀 开始训练

### 方法1: 直接训练（推荐）

```bash
cd /Users/HarowrdLiu/learn/robot/kuavo_data_challenge

python kuavo_train/train_smolvla_sequential.py \
    --config-path=../configs/policy \
    --config-name=smolvla_sequential_base \
    task=tasks/task1_moving_grasp
```

### 方法2: 使用screen/tmux（避免断连）

```bash
# 创建screen会话
screen -S training

# 运行训练
cd /Users/HarowrdLiu/learn/robot/kuavo_data_challenge
python kuavo_train/train_smolvla_sequential.py \
    --config-path=../configs/policy \
    --config-name=smolvla_sequential_base \
    task=tasks/task1_moving_grasp

# 退出screen: Ctrl+A, D
# 重新连接: screen -r training
```

---

## 📊 监控训练

### TensorBoard

```bash
# 新开终端
cd /Users/HarowrdLiu/learn/robot/kuavo_data_challenge
tensorboard --logdir outputs/train/task1_moving_grasp/smolvla_sequential/
```

访问: http://localhost:6006

### 查看日志

```bash
tail -f smolvla_sequential_training.log
```

---

## 🎯 预期效果

| 指标 | 当前 | 优化后预期 |
|------|------|-----------|
| Loss | 0.0113 | < 0.01 |
| 放置准确性 | ❌ 桌子上 | ✅ 目标位置 |
| 抓取鲁棒性 | ❌ <50% | ✅ >80% |
| 训练时间 | 5-6h | 8-10h |

---

## 📝 关键参数总结

```yaml
# 任务配置: configs/policy/tasks/task1_moving_grasp.yaml
max_epoch: 150
optimizer_lr: 0.00009
scheduler_warmup_steps: 3000
scheduler_decay_steps: 40000

# 基础配置: configs/policy/smolvla_sequential_base.yaml
chunk_size: 75
n_action_steps: 10
num_steps: 15
batch_size: 64
validation_freq_epoch: 1

# 数据增强: 自动启用，50%概率
```

---

## ⚠️ 注意事项

1. **显存**: batch_size=64需要约16GB显存，如果不够请降低batch_size
2. **训练时间**: 预计8-10小时，请保持训练环境稳定
3. **验证频率**: 每个epoch都会验证，注意观察validation loss
4. **最佳模型**: 自动保存在`outputs/train/.../best/`

---

## 🔧 如果显存不够

修改 `configs/policy/smolvla_sequential_base.yaml`:

```yaml
# 降低batch_size
batch_size: 48  # 或 40、32

# 对应调整学习率
# 在 task1_moving_grasp.yaml 中:
optimizer_lr: 0.00008  # batch_size=48
# 或
optimizer_lr: 0.00007  # batch_size=40
```

---

## 📂 输出目录

```
outputs/train/task1_moving_grasp/smolvla_sequential/run_YYYYMMDD_HHMMSS/
├── best/                    # 最佳模型
├── epoch10/                 # 每10个epoch保存
├── epoch20/
├── ...
└── training_results.json    # 训练结果
```

---

祝训练成功！🎉

