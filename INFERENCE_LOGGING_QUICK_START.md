# 推理日志记录功能 - 快速开始

## 概述

已为仿真环境的 `kuavo_hierarchical_sim_env` 模式添加了完整的推理日志记录功能，可以记录：
- ✅ 每步模型推理结果（动作、推理时间等）
- ✅ 层激活情况（分层架构）
- ✅ 执行时间统计
- ✅ 回合总结和聚合报告

## 快速使用

### 1. 运行仿真评估（自动记录日志）

```bash
# 使用交互式脚本
./kuavo_deploy/eval_kuavo.sh
# 选择: 3 -> 输入配置文件路径 -> 选择: 8 (自动测试)

# 或直接运行
python kuavo_deploy/examples/scripts/script_auto_test.py \
  --task auto_test \
  --config configs/deploy/kuavo_hierarchical_sim_env.yaml
```

### 2. 查看生成的日志

日志会自动保存在：
```
outputs/eval/{task}/{method}/{timestamp}/epoch{epoch}/inference_logs/
├── inference_episode_0.jsonl              # 详细步骤记录
├── inference_episode_0_summary.json       # 回合总结
├── inference_episode_1.jsonl
├── inference_episode_1_summary.json
└── aggregated_inference_report.json       # 聚合报告（所有回合完成后）
```

### 3. 分析日志

```bash
# 使用内置分析工具
python kuavo_deploy/utils/analyze_inference_logs.py \
  --log_dir outputs/eval/.../inference_logs

# 输出示例：
# 📊 推理时间分析
#   平均推理时间: 45.2ms
#   成功率: 90.0% (45/50)
# 🏗️ 分层架构激活分析
#   SAFETY层: 激活率 100%, 平均执行时间 8.5ms
#   MANIPULATION层: 激活率 98%, 平均执行时间 35.2ms
```

## 日志内容示例

### 步骤记录 (JSONL)
```json
{
  "timestamp": "2025-10-10T12:34:56",
  "episode": 0,
  "step": 5,
  "inference_time_ms": 45.2,
  "action": {"mean": 0.023, "std": 0.15, "values": [...]},
  "hierarchical_layers": {
    "safety": {"activated": true, "execution_time_ms": 8.5},
    "manipulation": {"activated": true, "execution_time_ms": 35.2}
  }
}
```

### 回合总结 (JSON)
```json
{
  "episode_index": 0,
  "success": true,
  "total_steps": 100,
  "avg_inference_time_ms": 45.2,
  "hierarchical_stats": {
    "layer_activation_counts": {"safety": 100, "manipulation": 98},
    "layer_avg_execution_times_ms": {"safety": 8.5, "manipulation": 35.2}
  }
}
```

## 主要特性

1. **自动记录** - 无需额外配置，运行评估即可自动记录
2. **详细信息** - 记录每步的推理结果、层激活、执行时间
3. **分层支持** - 自动识别分层架构，记录各层的激活情况
4. **聚合报告** - 自动生成所有回合的统计汇总
5. **性能优化** - 使用流式写入，对推理性能影响极小（< 1ms）

## 文件说明

### 新增文件
- `kuavo_deploy/utils/inference_logger.py` - 日志记录器实现
- `kuavo_deploy/utils/analyze_inference_logs.py` - 日志分析工具
- `kuavo_deploy/utils/INFERENCE_LOGGING_README.md` - 详细文档

### 修改文件
- `kuavo_train/wrapper/policy/humanoid/HierarchicalScheduler.py` - 添加层激活信息记录
- `kuavo_train/wrapper/policy/humanoid/HumanoidDiffusionPolicy.py` - 添加层输出信息获取
- `kuavo_deploy/examples/eval/auto_test/eval_kuavo.py` - 集成日志记录功能
- `kuavo_deploy/examples/eval/auto_test/eval_kuavo_autotest.py` - 添加聚合报告生成

## 配置选项（可选）

在配置文件中调整日志记录行为：

```yaml
# configs/deploy/kuavo_hierarchical_sim_env.yaml
hierarchical:
  enable_performance_logging: True  # 启用性能日志（默认True）
  log_frequency: 1  # 每N步记录一次（默认1）
```

## 常见问题

**Q: 日志记录会影响性能吗？**
A: 影响非常小（< 1ms），使用流式写入，不会占用大量内存。

**Q: 如何减少日志文件大小？**
A: 可以修改 `eval_kuavo.py` 中的 `log_every_n_steps` 参数，例如设置为 10。

**Q: 非分层架构也会记录吗？**
A: 是的，但不会记录层激活信息，只记录基本的推理结果。

**Q: 如何查看实时推理情况？**
A: 可以使用 `tail -f` 命令实时查看 JSONL 文件。

## 下一步

查看详细文档了解更多功能：
```bash
cat kuavo_deploy/utils/INFERENCE_LOGGING_README.md
```

## 联系方式

如有问题或建议，请联系开发团队。

