# 推理日志记录系统使用说明

## 概述

推理日志记录系统用于在仿真环境中记录模型推理过程的详细信息，包括：
- 每步的推理结果（动作、执行时间等）
- 分层架构的层激活情况
- 执行时间统计
- 任务相关信息

## 功能特点

### 1. 详细的步骤记录
- **动作信息**：记录每步输出的动作值、统计信息（均值、标准差、最小值、最大值）
- **观测形状**：记录输入观测数据的形状
- **推理时间**：精确记录每步的推理耗时
- **额外信息**：支持记录自定义的额外信息

### 2. 分层架构支持
对于 `hierarchical_diffusion` 策略类型，系统会自动记录：
- 每层的激活状态
- 每层的执行时间
- 层的激活计数
- 层的损失值（训练模式）
- 紧急状态（安全层）
- 层之间的交互信息

### 3. 回合总结
每个回合结束后自动生成：
- 成功/失败状态
- 总步数和总奖励
- 平均推理时间
- 层激活统计（分层架构）
- 层执行时间统计

### 4. 聚合报告
所有回合完成后自动生成聚合报告：
- 总体成功率
- 平均步数和推理时间
- 所有回合的汇总统计
- 层激活和执行时间的聚合统计

## 文件结构

运行推理后，会在以下位置生成日志文件：

```
outputs/eval/{task}/{method}/{timestamp}/epoch{epoch}/inference_logs/
├── inference_episode_0.jsonl                    # 第0回合的详细步骤记录
├── inference_episode_0_summary.json             # 第0回合的总结
├── inference_episode_1.jsonl                    # 第1回合的详细步骤记录
├── inference_episode_1_summary.json             # 第1回合的总结
├── ...
└── aggregated_inference_report.json             # 所有回合的聚合报告
```

## 日志文件说明

### 1. 步骤记录文件 (`inference_episode_N.jsonl`)

每行是一个 JSON 对象，记录一步的信息：

```json
{
  "timestamp": "2025-10-10T12:34:56.789",
  "episode": 0,
  "step": 5,
  "inference_time_ms": 45.2,
  "observation_shapes": {
    "observation.images.head_cam_h": [1, 3, 480, 640],
    "observation.state": [1, 16]
  },
  "action": {
    "shape": [16],
    "mean": 0.023,
    "std": 0.15,
    "min": -0.5,
    "max": 0.6,
    "values": [0.1, 0.2, ...]  // 每N步记录一次完整值
  },
  "hierarchical_layers": {  // 仅分层架构
    "safety": {
      "activated": true,
      "activation_count": 5,
      "execution_time_ms": 8.5,
      "emergency": false
    },
    "manipulation": {
      "activated": true,
      "activation_count": 5,
      "execution_time_ms": 35.2,
      "action_shape": [1, 16],
      "action_norm": 0.45
    }
  },
  "additional_info": {
    "reward_sum": 12.5
  }
}
```

### 2. 回合总结文件 (`inference_episode_N_summary.json`)

```json
{
  "episode_index": 0,
  "start_time": "2025-10-10T12:34:00.000",
  "end_time": "2025-10-10T12:35:30.000",
  "episode_duration_sec": 90.0,
  "success": true,
  "total_reward": 25.5,
  "total_steps": 100,
  "total_inference_time_sec": 4.52,
  "avg_inference_time_ms": 45.2,
  "hierarchical_stats": {  // 仅分层架构
    "layer_activation_counts": {
      "safety": 100,
      "manipulation": 98
    },
    "layer_avg_execution_times_ms": {
      "safety": 8.5,
      "manipulation": 35.2
    },
    "layer_total_execution_times_ms": {
      "safety": 850.0,
      "manipulation": 3449.6
    }
  },
  "additional_stats": {
    "episode_steps": 100,
    "marker1_position": [0.5, 0.3, 0.2],
    "marker2_position": [0.6, 0.4, 0.3]
  }
}
```

### 3. 聚合报告 (`aggregated_inference_report.json`)

```json
{
  "task_name": "task_400_episodes_humanoid_hierarchical",
  "generated_at": "2025-10-10T12:40:00.000",
  "total_episodes": 50,
  "success_count": 45,
  "success_rate": 0.9,
  "avg_steps_per_episode": 98.5,
  "avg_inference_time_ms": 46.3,
  "hierarchical_aggregated_stats": {  // 仅分层架构
    "safety": {
      "total_activations": 4925,
      "avg_execution_time_ms": 8.7
    },
    "manipulation": {
      "total_activations": 4850,
      "avg_execution_time_ms": 35.8
    }
  },
  "episodes": [...]  // 所有回合的总结
}
```

## 使用方法

### 1. 运行仿真评估

推理日志记录功能已自动集成到评估脚本中，无需额外配置。

使用 `eval_kuavo.sh` 脚本运行评估：

```bash
./kuavo_deploy/eval_kuavo.sh
# 选择选项 3，然后输入配置文件路径
# 选择选项 8 进行自动测试
```

或者直接使用 Python 脚本：

```bash
python kuavo_deploy/examples/scripts/script_auto_test.py \
  --task auto_test \
  --config configs/deploy/kuavo_hierarchical_sim_env.yaml
```

### 2. 查看日志

评估完成后，日志会自动保存到：
```
outputs/eval/{task}/{method}/{timestamp}/epoch{epoch}/inference_logs/
```

### 3. 配置选项

在配置文件中可以调整日志记录行为（可选）：

```yaml
# configs/deploy/kuavo_hierarchical_sim_env.yaml
hierarchical:
  enable_performance_logging: True  # 启用性能日志
  log_frequency: 1  # 每N步记录一次详细日志（默认为1，即每步都记录）
```

### 4. 日志记录的性能影响

- **JSONL 格式**：使用流式写入，每步立即写入磁盘，不会占用大量内存
- **最小开销**：日志记录的时间开销通常小于 1ms
- **可配置**：可以通过 `log_every_n_steps` 参数减少记录频率

## 数据分析示例

### Python 分析脚本示例

```python
import json
from pathlib import Path
import numpy as np

# 读取JSONL文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 分析推理时间
def analyze_inference_time(log_dir):
    log_files = Path(log_dir).glob("inference_episode_*.jsonl")

    all_times = []
    for log_file in log_files:
        steps = read_jsonl(log_file)
        times = [step['inference_time_ms'] for step in steps]
        all_times.extend(times)

    print(f"平均推理时间: {np.mean(all_times):.2f}ms")
    print(f"最小推理时间: {np.min(all_times):.2f}ms")
    print(f"最大推理时间: {np.max(all_times):.2f}ms")
    print(f"标准差: {np.std(all_times):.2f}ms")

# 分析层激活情况
def analyze_layer_activation(summary_file):
    with open(summary_file, 'r') as f:
        data = json.load(f)

    if 'hierarchical_aggregated_stats' in data:
        print("\n层激活统计:")
        for layer, stats in data['hierarchical_aggregated_stats'].items():
            print(f"{layer}:")
            print(f"  总激活次数: {stats['total_activations']}")
            print(f"  平均执行时间: {stats['avg_execution_time_ms']:.2f}ms")

# 使用示例
log_dir = "outputs/eval/task_400_episodes/humanoid_hierarchical/task_specific_run_20251009_033145/epoch30/inference_logs"
analyze_inference_time(log_dir)
analyze_layer_activation(Path(log_dir) / "aggregated_inference_report.json")
```

## 常见问题

### Q1: 日志文件太大怎么办？
A: 可以调整 `log_every_n_steps` 参数，例如设置为 10，则只记录每 10 步的详细动作值。

### Q2: 如何关闭日志记录？
A: 目前日志记录是自动启用的，如果需要关闭，可以在 `eval_kuavo.py` 中注释掉相关代码。

### Q3: 日志记录会影响性能吗？
A: 影响非常小（< 1ms），因为使用了流式写入和异步处理。

### Q4: 可以记录自定义信息吗？
A: 可以通过 `additional_info` 参数传递自定义信息到 `log_step` 方法。

### Q5: 如何查看层激活的详细情况？
A: 查看 `inference_episode_N.jsonl` 文件中的 `hierarchical_layers` 字段，可以看到每步每层的激活状态。

## 技术细节

### 日志记录流程

1. **初始化**：在 `eval_kuavo.py` 的 `main` 函数中创建 `InferenceLogger`
2. **步骤记录**：在推理循环中，每步调用 `logger.log_step()`
3. **回合总结**：回合结束时调用 `logger.save_episode_summary()`
4. **聚合报告**：所有回合完成后调用 `InferenceLogger.create_aggregated_report()`

### 层信息获取

- 在 `HierarchicalScheduler.forward()` 中添加 `_activation_summary` 到输出
- 在 `HumanoidDiffusionPolicy._hierarchical_select_action()` 中保存 `_last_layer_outputs`
- 通过 `policy.get_last_layer_outputs()` 获取层信息

## 后续改进建议

1. **可视化工具**：开发基于 Web 的可视化工具，实时展示推理过程
2. **性能分析**：添加更详细的性能瓶颈分析
3. **异常检测**：自动检测异常的推理模式
4. **对比分析**：支持多次评估结果的对比分析

## 相关文件

- `kuavo_deploy/utils/inference_logger.py` - 日志记录器实现
- `kuavo_train/wrapper/policy/humanoid/HierarchicalScheduler.py` - 层激活信息记录
- `kuavo_train/wrapper/policy/humanoid/HumanoidDiffusionPolicy.py` - 层输出信息获取
- `kuavo_deploy/examples/eval/auto_test/eval_kuavo.py` - 评估脚本（集成日志记录）
- `kuavo_deploy/examples/eval/auto_test/eval_kuavo_autotest.py` - 自动测试脚本（生成聚合报告）

## 联系方式

如有问题或建议，请联系开发团队。

