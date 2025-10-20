# Flow Matching 快速入门指南

## 🎯 一分钟快速启动

### 方法 1: 使用专用配置文件（最简单）

```bash
# 直接使用 Flow Matching 配置训练
python train_policy.py policy=flow_matching_config
```

### 方法 2: 修改现有配置

在 `configs/policy/diffusion_config.yaml` 中修改一行：

```yaml
policy:
  use_flow_matching: True  # 从 False 改为 True
  num_inference_steps: 10  # 可选：调整推理步数
```

然后正常训练：

```bash
python train_policy.py policy=diffusion_config
```

## ✅ 验证安装

运行测试脚本：

```bash
python test_flow_matching.py
```

预期输出：
```
✅ 所有测试通过！Flow Matching 已成功集成！
```

## 📊 性能对比

### 预期提升

| 指标 | Diffusion | Flow Matching | 提升 |
|------|-----------|---------------|------|
| 推理步数 | 50-100步 | 10步 | 5-10倍 ⚡ |
| 推理时间 | ~200ms | ~40ms | 5倍 🚀 |
| 训练时间 | 基准 | 相当或略快 | ≈ |
| 任务成功率 | 基准 | 相当或更好 | ✅ |

### 实际测试方法

1. **训练两个模型**

```bash
# Diffusion 版本
python train_policy.py policy=diffusion_config task=your_task

# Flow Matching 版本
python train_policy.py policy=flow_matching_config task=your_task
```

2. **对比推理时间**

```python
import time

# 加载两个模型
diffusion_policy = load_policy("diffusion_checkpoint")
flow_matching_policy = load_policy("flow_matching_checkpoint")

# 测试推理时间
obs = env.get_observation()

start = time.time()
action_diff = diffusion_policy.select_action(obs)
time_diff = time.time() - start

start = time.time()
action_flow = flow_matching_policy.select_action(obs)
time_flow = time.time() - start

print(f"Diffusion: {time_diff*1000:.2f}ms")
print(f"Flow Matching: {time_flow*1000:.2f}ms")
print(f"速度提升: {time_diff/time_flow:.1f}x")
```

## ⚙️ 参数调优指南

### 快速推理（实时控制）

```yaml
use_flow_matching: True
num_inference_steps: 5-10
ode_solver: "euler"
```

**适用场景**: 机器人实时控制、高频率任务

### 平衡性能

```yaml
use_flow_matching: True
num_inference_steps: 10-20
ode_solver: "euler"
```

**适用场景**: 大多数任务（推荐）

### 高质量生成

```yaml
use_flow_matching: True
num_inference_steps: 20-50
ode_solver: "rk4"
```

**适用场景**: 离线评估、演示视频

## 🔧 常见问题

### Q1: 训练损失不下降？

**A**: 检查学习率，Flow Matching 可能需要稍低的学习率：

```yaml
optimizer_lr: 0.00005  # 从 0.0001 降低
```

### Q2: 推理结果不稳定？

**A**: 增加推理步数或使用更精确的求解器：

```yaml
num_inference_steps: 20  # 增加到 20
ode_solver: "rk4"        # 使用 RK4
```

### Q3: 性能不如 Diffusion？

**A**: 确保训练充分，尝试以下调整：

```yaml
# 增加训练轮次
max_epoch: 600  # 从 500 增加

# 调整推理步数
num_inference_steps: 15  # 找到最优值
```

### Q4: 想切换回 Diffusion？

**A**: 只需修改一个参数：

```yaml
use_flow_matching: False
```

## 📝 检查清单

在正式使用前，确认：

- [ ] 测试脚本通过 (`python test_flow_matching.py`)
- [ ] 配置文件中 `use_flow_matching` 设置正确
- [ ] `num_inference_steps` 根据需求设置（10-20 推荐）
- [ ] 训练时监控损失曲线
- [ ] 对比测试 Diffusion 和 Flow Matching 性能

## 🚀 完整训练流程示例

```bash
# 1. 准备数据（与 Diffusion 相同）
python kuavo_data/CvtRosbag2Lerobot.py --config your_config

# 2. 训练 Flow Matching 模型
python train_policy.py \
  policy=flow_matching_config \
  task=your_task \
  training.batch_size=96 \
  training.max_epoch=500

# 3. 评估模型
python eval_policy.py \
  policy=flow_matching_config \
  checkpoint=path/to/checkpoint

# 4. 部署推理
python deploy_policy.py \
  --policy_path path/to/checkpoint \
  --env_config your_env_config
```

## 📚 更多信息

- 详细文档: `kuavo_train/wrapper/policy/diffusion/FLOW_MATCHING_README.md`
- 调度器代码: `kuavo_train/wrapper/policy/diffusion/flow_matching_scheduler.py`
- 配置文件: `configs/policy/flow_matching_config.yaml`
- 测试脚本: `test_flow_matching.py`

## 💡 最佳实践建议

1. **首次使用**: 从默认配置开始，不要过度调参
2. **训练监控**: 关注损失曲线，应该平稳下降
3. **A/B 测试**: 同时训练 Diffusion 和 Flow Matching 版本对比
4. **渐进调优**: 先确保模型收敛，再调整推理步数
5. **记录结果**: 记录不同配置下的性能指标

## 🎉 预期收益

采用 Flow Matching 后，你应该能看到：

✅ **推理速度提升 5-10倍**
✅ **实时控制频率提高**
✅ **相当或更好的任务成功率**
✅ **训练过程更稳定**
✅ **代码更简洁（无需复杂噪声调度）**

---

**开始你的 Flow Matching 之旅！** 🌊

如有问题，请查看详细文档或提交 Issue。

