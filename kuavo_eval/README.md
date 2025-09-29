# Kuavo 离线评估系统

## 📖 概述

Kuavo离线评估系统是一个专门设计的模型评估框架，用于在无需ROS环境的情况下评估人形机器人的模仿学习模型。系统支持两种模型类型：

- **分层架构模型** (`humanoid_diffusion`): 具有多层优先级的人形机器人控制策略
- **传统Diffusion模型** (`diffusion`): 标准的扩散策略模型

## 🚀 快速开始

### 1. 快速验证1-epoch模型

对于刚完成1-epoch训练的模型，使用快速验证检查模型是否正常工作：

```bash
python kuavo_eval/scripts/quick_validation.py \
  --config configs/eval/offline_hierarchical_eval.yaml \
  --checkpoint outputs/train/task_400_episodes/humanoid_hierarchical/run_xxx/epoch1
```

### 2. 完整模型评估

进行全面的模型性能评估：

```bash
python kuavo_eval/scripts/run_offline_eval.py \
  --config configs/eval/offline_hierarchical_eval.yaml \
  --episodes 10 \
  --output-dir outputs/evaluation/my_results
```

## 📁 目录结构

```
kuavo_eval/
├── core/                       # 核心评估器
│   ├── base_evaluator.py             # 基础评估器
│   ├── hierarchical_evaluator.py     # 分层架构评估器
│   └── diffusion_evaluator.py        # 传统diffusion评估器
├── utils/                      # 工具模块
│   ├── mock_observation.py           # Mock观测环境
│   ├── metrics_calculator.py         # 指标计算工具
│   └── report_generator.py           # 报告生成器
├── scripts/                    # 执行脚本
│   ├── run_offline_eval.py           # 主评估脚本
│   └── quick_validation.py           # 快速验证脚本
└── examples/                   # 使用示例
    └── eval_examples.py              # 使用示例代码

configs/ (项目根目录)            # 配置文件目录
├── eval/                           # 评估配置子目录
│   ├── base_eval_config.yaml          # 基础配置
│   ├── offline_hierarchical_eval.yaml # 分层架构配置
│   └── offline_diffusion_eval.yaml    # 传统diffusion配置
├── data/                           # 数据转换配置
├── deploy/                         # 部署配置
└── policy/                         # 训练策略配置
```

## ⚙️ 配置文件说明

### 基础配置 (`base_eval_config.yaml`)

包含所有评估器共用的配置参数：

```yaml
# 通用评估参数
common:
  device: 'cuda'
  seed: 42
  output_dir: 'outputs/evaluation'

# 测试数据配置
test_data:
  root: '/root/robot/data/data/task1/lerobot'
  episodes_range: [0, 50]
  max_episodes: 10
  max_steps_per_episode: 100

# 评估指标配置
evaluation:
  action_metrics: ['mse', 'mae', 'cosine_sim']
  save_predictions: True
  generate_plots: True
```

### 分层架构配置 (`offline_hierarchical_eval.yaml`)

继承基础配置并添加分层架构特有的评估项：

```yaml
defaults:
  - base_eval_config

model:
  type: 'humanoid_diffusion'
  checkpoint_path: 'outputs/train/task_400_episodes/humanoid_hierarchical/run_xxx/epoch1'

hierarchical_evaluation:
  enabled_layers: ['safety', 'manipulation']
  latency_budget_ms: 100.0
  layer_activation_analysis:
    enable: True
```

### 传统Diffusion配置 (`offline_diffusion_eval.yaml`)

```yaml
defaults:
  - base_eval_config

model:
  type: 'diffusion'
  checkpoint_path: 'outputs/train/task_400_episodes/diffusion_method/run_xxx/epoch1'

diffusion_evaluation:
  denoising_analysis:
    enable: True
  trajectory_smoothness:
    enable: True
```

## 📊 评估指标

### 动作精度指标
- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **Cosine Similarity**: 余弦相似度
- **L2 Norm**: L2范数差异

### 分层架构特有指标
- **层激活率**: 各层的激活频率
- **预算遵从率**: 推理延迟预算的遵从情况
- **层一致性**: 层间协调的一致性
- **安全覆盖率**: 安全层的覆盖频率

### 传统Diffusion特有指标
- **去噪质量**: 扩散过程的去噪效果
- **轨迹平滑度**: 动作轨迹的平滑程度
- **推理速度**: 推理步数和时间的关系
- **动作一致性**: 时序动作的一致性

## 🔧 使用指南

### 1. 修改配置文件

在开始评估前，需要修改配置文件中的关键路径：

```yaml
model:
  checkpoint_path: 'path/to/your/model/checkpoint'  # 修改为你的模型路径

test_data:
  root: 'path/to/your/lerobot/data'  # 修改为你的数据路径
```

### 2. 快速验证

对于1-epoch训练的模型，建议先进行快速验证：

```bash
# 验证分层架构模型
python kuavo_eval/scripts/quick_validation.py \
  --config configs/eval/offline_hierarchical_eval.yaml

# 验证传统diffusion模型
python kuavo_eval/scripts/quick_validation.py \
  --config configs/eval/offline_diffusion_eval.yaml
```

快速验证会：
- 只评估2个episodes，每个episode 10步
- 只计算核心指标（MSE, MAE）
- 进行模型健康检查
- 给出是否适合进一步评估的建议

### 3. 完整评估

通过快速验证后，进行完整的性能评估：

```bash
python kuavo_eval/scripts/run_offline_eval.py \
  --config configs/eval/offline_hierarchical_eval.yaml \
  --episodes 10 \
  --output-dir outputs/evaluation/detailed_results
```

### 4. 命令行参数

两个脚本都支持丰富的命令行参数：

```bash
# 基础参数
--config CONFIG_FILE       # 配置文件路径
--checkpoint CHECKPOINT     # 模型检查点路径（覆盖配置文件）
--device {cpu,cuda}         # 计算设备
--episodes NUM              # 最大评估episodes数
--output-dir DIR            # 输出目录

# run_offline_eval.py 特有参数
--verbose                   # 详细输出模式
--quick                     # 快速模式（减少episodes和steps）
--no-plots                  # 禁用图表生成
```

## 📈 输出结果

### 文件输出

评估完成后会生成以下文件：

1. **JSON报告**: `{model_type}_evaluation_report_{timestamp}.json`
   - 包含完整的评估数据
   - 适合程序化处理

2. **CSV摘要**: `{model_type}_evaluation_summary_{timestamp}.csv`
   - 表格格式的关键指标
   - 适合导入Excel等工具

3. **Markdown报告**: `{model_type}_evaluation_report_{timestamp}.md`
   - 人类可读的详细分析报告
   - 包含建议和结论

4. **可视化图表**: 各种PNG格式的分析图表
   - 动作精度对比图
   - Episode趋势图
   - 模型特有的性能图表

### 快速验证输出

快速验证会在控制台直接显示：

```
⚡ QUICK VALIDATION RESULTS
==================================================
✅ Status: SUCCESS
⏱️  Total Time: 15.32s
🤖 Model Type: humanoid_diffusion

📊 Key Metrics:
  mse: 0.0234
  mae: 0.1156
  avg_inference_time: 45.23
  budget_compliance: 0.95

🏥 Health Check:
  Overall Status: ✅ HEALTHY
  Pass Rate: 100.0% (4/4)

💡 Recommendations:
  🎉 Model appears to be working well!
  ✨ Ready for full evaluation or deployment testing
```

## ⚠️ 常见问题

### 1. 模型加载失败
**错误**: `FileNotFoundError: Model checkpoint not found`

**解决**:
- 检查配置文件中的 `checkpoint_path` 是否正确
- 确认模型训练已完成并保存了检查点

### 2. 数据路径错误
**错误**: `FileNotFoundError: Data root directory not found`

**解决**:
- 检查 `test_data.root` 路径是否正确
- 确认数据已转换为lerobot格式

### 3. GPU内存不足
**错误**: `CUDA out of memory`

**解决**:
- 使用 `--device cpu` 切换到CPU
- 减少配置文件中的 `batch_size`
- 减少 `max_episodes` 或 `max_steps_per_episode`

### 4. 权限问题
**错误**: `Permission denied`

**解决**:
```bash
chmod +x kuavo_eval/scripts/*.py
```

## 🛠️ 高级用法

### 1. 批量评估多个检查点

```bash
#!/bin/bash
for epoch in 1 5 10 20 50; do
    python kuavo_eval/scripts/run_offline_eval.py \
      --config configs/eval/offline_hierarchical_eval.yaml \
      --checkpoint outputs/train/task/method/run_xxx/epoch${epoch} \
      --output-dir outputs/evaluation/epoch_${epoch} \
      --quick
done
```

### 2. 对比不同模型类型

```bash
# 评估分层架构模型
python kuavo_eval/scripts/run_offline_eval.py \
  --config configs/eval/offline_hierarchical_eval.yaml \
  --output-dir outputs/evaluation/hierarchical

# 评估传统diffusion模型
python kuavo_eval/scripts/run_offline_eval.py \
  --config configs/eval/offline_diffusion_eval.yaml \
  --output-dir outputs/evaluation/diffusion
```

### 3. 自定义评估指标

修改配置文件添加新的指标：

```yaml
evaluation:
  action_metrics: ['mse', 'mae', 'rmse', 'max_error']

  # 添加自定义阈值
  custom_thresholds:
    mse_threshold: 0.01
    mae_threshold: 0.05
```

## 📞 支持

如果遇到问题或需要新功能：

1. 查看 `kuavo_eval/examples/eval_examples.py` 的使用示例
2. 检查配置文件的参数说明
3. 运行快速验证确认环境配置正确

## 🎯 最佳实践

1. **开发阶段**: 始终先运行快速验证
2. **训练监控**: 定期评估不同epoch的模型
3. **模型对比**: 使用相同配置评估不同模型以确保公平对比
4. **结果分析**: 重点关注健康检查和建议部分
5. **持续改进**: 根据评估结果调整训练策略

## 🚀 快速开始指南

### 步骤1: 修改配置文件
编辑 `configs/offline_hierarchical_eval.yaml` 或 `configs/offline_diffusion_eval.yaml`：

```yaml
model:
  checkpoint_path: 'outputs/train/task_400_episodes/humanoid_hierarchical/run_xxx/epoch1'

test_data:
  root: '/your/path/to/lerobot/data'
```

### 步骤2: 快速验证 (推荐1-epoch模型)
```bash
# 在项目根目录下执行
python kuavo_eval/scripts/quick_validation.py \
  --config configs/offline_hierarchical_eval.yaml
```

### 步骤3: 完整评估 (可选)
```bash
# 在项目根目录下执行
python kuavo_eval/scripts/run_offline_eval.py \
  --config configs/offline_hierarchical_eval.yaml \
  --episodes 5
```

📍 **重要提示**: 所有命令都应该在项目根目录下执行，配置文件现已移至 `configs/` 目录。