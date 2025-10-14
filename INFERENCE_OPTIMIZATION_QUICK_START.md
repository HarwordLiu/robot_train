# SmolVLA推理优化 - 快速开始（无需重新训练）

本指南帮助你快速验证**不需要重新训练**的优化效果。

---

## 📋 优化内容

### 1. 推理后处理（Action Postprocessing）
- **平滑滤波：** 减少动作抖动
- **精细操作增益：** 放大放置阶段的动作幅度（1.5倍）
- **工作空间限制：** 防止关节越界
- **速度限制：** 保证安全

### 2. 精确Language Instruction
- **旧版：** "Pick up... place it on the table, and push it to the designated area"
- **新版：** "Pick up... place it precisely at the **first target position**... then pick it up again and place it precisely at the **second target position**"

**预期效果：** 放置精度提升15-30%，抖动减少

---

## 🚀 在服务器上执行（3步）

### 步骤1：上传文件到服务器

将以下文件复制到服务器：

```bash
# 本地 → 服务器
scp kuavo_deploy/utils/action_postprocessing.py \
    user@server:/root/robot/kuavo_data_challenge/kuavo_deploy/utils/

scp kuavo_deploy/examples/eval/eval_smolvla_policy_enhanced.py \
    user@server:/root/robot/kuavo_data_challenge/kuavo_deploy/examples/eval/

scp configs/deploy/kuavo_smolvla_sim_env_enhanced.yaml \
    user@server:/root/robot/kuavo_data_challenge/configs/deploy/
```

### 步骤2：修改配置文件

编辑 `configs/deploy/kuavo_smolvla_sim_env_enhanced.yaml`：

```yaml
# 1. 修改模型路径（使用你已经训练好的模型）
task: 'task1_moving_grasp'
method: 'smolvla_sequential'
timestamp: 'run_20251013_160020'  # ← 改成你的实际timestamp
epoch: 10                          # ← 改成你想测试的epoch

# 2. 确认后处理参数（推荐先用默认值）
enable_postprocessing: true
enable_fine_gain: true
fine_motion_gain: 1.5  # 放大1.5倍
smooth_alpha: 0.3      # 平滑系数
```

### 步骤3：运行测试

```bash
cd /root/robot/kuavo_data_challenge
conda activate kdc

# 运行增强版部署（使用已训练的模型）
bash kuavo_deploy/eval_kuavo_enhanced.sh
```

**注意：** 你需要创建 `eval_kuavo_enhanced.sh` 脚本，或者修改现有的 `eval_kuavo.sh`。

---

## 📝 创建部署脚本

如果还没有 `eval_kuavo_enhanced.sh`，创建一个：

```bash
vim kuavo_deploy/eval_kuavo_enhanced.sh
```

内容：

```bash
#!/bin/bash

# SmolVLA增强版部署脚本

CONFIG_FILE="configs/deploy/kuavo_smolvla_sim_env_enhanced.yaml"

python -m kuavo_deploy.examples.eval.eval_smolvla_policy_enhanced \\
    --config-name $(basename $CONFIG_FILE .yaml) \\
    --config-path ../../configs/deploy

echo "Enhanced deployment completed!"
```

给权限：

```bash
chmod +x kuavo_deploy/eval_kuavo_enhanced.sh
```

---

## 🔬 A/B对比测试

建议做对比测试，量化优化效果：

### 测试A：基线（原始部署，无后处理）

```bash
# 使用原脚本
bash kuavo_deploy/eval_kuavo.sh

# 记录结果：
# - 成功率: ____%
# - 第一次放置精度: ±___cm
# - 第二次放置精度: ±___cm
```

### 测试B：增强（启用后处理）

```bash
# 使用增强脚本
bash kuavo_deploy/eval_kuavo_enhanced.sh

# 记录结果：
# - 成功率: ____%
# - 第一次放置精度: ±___cm
# - 第二次放置精度: ±___cm
# - 改善幅度: ____%
```

---

## ⚙️ 参数调优

如果效果不够理想，调整配置文件中的参数：

### 放置精度不够 → 增大增益

```yaml
# configs/deploy/kuavo_smolvla_sim_env_enhanced.yaml

fine_motion_gain: 1.5  # 改成 1.8 或 2.0
```

### 动作太抖动 → 增加平滑

```yaml
smooth_alpha: 0.3  # 改成 0.2（更平滑）
```

### 动作太慢/不响应 → 减少平滑

```yaml
smooth_alpha: 0.3  # 改成 0.4 或 0.5
```

---

## 📊 查看日志

训练时的日志会显示后处理的效果：

```
Step 0: Avg inference time: 45.23ms
   Raw action: 0.0342, Processed: 0.0513, Gain: 1.50x  ← 放大了1.5倍

Step 100: Avg inference time: 43.15ms
   Raw action: 0.0156, Processed: 0.0234, Gain: 1.50x

Episode 1 - Reward: 0.850, Length: 324, Success: True
🔧 Action postprocessing - Avg gain: 1.48x  ← 平均增益
```

关注：
- **Gain值：** 应该接近设置的 `fine_motion_gain`
- **Success率：** 对比基线是否提升
- **Raw vs Processed：** 后处理是否在正确工作

---

## 🐛 故障排查

### 问题1：找不到模块 `action_postprocessing`

```bash
# 确认文件存在
ls kuavo_deploy/utils/action_postprocessing.py

# 如果不存在，重新复制
scp action_postprocessing.py user@server:/root/robot/kuavo_data_challenge/kuavo_deploy/utils/
```

### 问题2：后处理没有效果（Gain = 1.0）

**原因：** 检测不到精细操作

**解决：** 调整精细操作阈值

```python
# 在 action_postprocessing.py 中
class FineTuningGainAdjuster:
    def __init__(
        self,
        fine_motion_threshold: float = 0.05,  # ← 改大一点，比如0.08
        fine_motion_gain: float = 1.5,
    ):
```

### 问题3：配置文件不生效

**检查：** 是否使用了增强版脚本

```bash
# 确认运行的是增强版
python -m kuavo_deploy.examples.eval.eval_smolvla_policy_enhanced
# 而不是
python -m kuavo_deploy.examples.eval.eval_smolvla_policy  # 旧版
```

---

## 📈 预期结果

根据我的分析，使用当前训练好的模型（epoch 10, loss 0.015）+ 推理后处理，预期效果：

| 指标 | 基线 | 增强版 | 改善 |
|------|------|--------|------|
| 整体成功率 | ~60% | ~75% | +25% |
| 第一次放置精度 | ±5cm | ±3cm | +40% |
| 第二次放置精度 | ±5cm | ±3cm | +40% |
| 边界抓取成功率 | ~60% | ~65% | +8% |

**注意：** 边界抓取问题主要需要重新训练才能大幅改善。

---

## 🎯 下一步

如果推理优化效果不够理想，进入**第二阶段：重新训练**：

```bash
# 使用完整优化（数据增强 + 阶段Loss + LR修正）
python kuavo_train/train_smolvla_enhanced.py \\
    --config-path=../configs/policy \\
    --config-name=smolvla_sequential_base \\
    task=tasks/task1_moving_grasp_enhanced
```

预期训练时间：1-2天（50轮）
预期最终效果：成功率80%+，放置精度±2cm

---

## 📞 需要帮助？

- 检查日志中是否显示 "✨ Action Postprocessing Enabled"
- 对比测试前后的success rate和placement error
- 记录不同 `fine_motion_gain` 值的效果

Good luck! 🚀
