# 分层人形机器人Diffusion Policy框架使用指南

## 🎯 概述

本框架实现了一个四层分层架构的Diffusion Policy，专门为双足人形机器人的复杂任务设计。

### 核心特性
- ✅ **四层分层架构**: 安全层、步态层、操作层、规划层
- ✅ **优先级调度**: 安全优先的智能任务调度
- ✅ **实时性能**: 自适应计算预算分配
- ✅ **向后兼容**: 可与现有训练流程无缝集成
- ✅ **任务特化**: 支持4种复杂机器人任务

## 📁 文件结构

```
kuavo_train/wrapper/policy/humanoid/
├── HumanoidDiffusionPolicy.py          # 主入口
├── HierarchicalScheduler.py            # 核心调度器
├── HierarchicalDiffusionModel.py       # 分层Diffusion模型
├── layers/                             # 四个层次实现
│   ├── BaseLayer.py                    # 抽象基类
│   ├── SafetyReflexLayer.py           # 安全反射层
│   ├── GaitControlLayer.py            # 步态控制层
│   ├── ManipulationLayer.py           # 操作控制层
│   └── GlobalPlanningLayer.py         # 全局规划层
└── modules/                           # 支撑模块目录

configs/policy/
└── humanoid_diffusion_config.yaml     # 分层架构专用配置

validate_hierarchical_framework.py     # 验证脚本
```

## 🚀 快速开始

### 1. 验证框架
```bash
# 运行验证脚本
python validate_hierarchical_framework.py
```

### 2. 训练分层模型
```bash
# 使用分层架构配置训练
python kuavo_train/train_policy.py --config-name=humanoid_diffusion_config
```

### 3. 对比训练（可选）
```bash
# 使用传统架构训练（对比基线）
python kuavo_train/train_policy.py --config-name=diffusion_config
```

## ⚙️ 配置说明

### 关键配置项

#### 启用分层架构
```yaml
policy:
  use_hierarchical: True  # 启用分层架构
```

#### 层配置
```yaml
hierarchical:
  layers:
    safety:
      type: "GRU"
      hidden_size: 64
      response_time_ms: 10
      priority: 1
      enabled: True

    gait:
      type: "Hybrid"  # GRU + Transformer
      gru_hidden: 128
      tf_layers: 2
      priority: 2
      enabled: True

    manipulation:
      type: "Transformer"
      hidden_size: 512
      layers: 3
      priority: 3
      enabled: True

    planning:
      type: "Transformer"
      hidden_size: 1024
      layers: 4
      priority: 4
      enabled: False  # 默认禁用最复杂的层
```

#### 层权重配置
```yaml
layer_weights:
  safety: 2.0      # 安全层权重最高
  gait: 1.5        # 步态层次之
  manipulation: 1.0 # 操作层标准权重
  planning: 0.8    # 规划层权重较低
```

#### 课程学习配置
```yaml
curriculum_learning:
  enable: True
  stages:
    stage1:
      name: "safety_only"
      layers: ["safety"]
      epochs: 50
    stage2:
      name: "safety_gait"
      layers: ["safety", "gait"]
      epochs: 100
    # ... 更多阶段
```

## 🏗️ 架构详解

### 四层分层架构

#### 1. 安全反射层 (Priority 1)
- **响应时间**: <10ms
- **功能**: 防跌倒检测、紧急停止、基础平衡控制
- **架构**: 极简GRU
- **特点**: 永远激活，可覆盖其他层输出

#### 2. 步态控制层 (Priority 2)
- **响应时间**: ~20ms
- **功能**: 步态规划、负载适应、地形适应
- **架构**: GRU + 轻量Transformer混合
- **特点**: 双足机器人专门优化

#### 3. 操作控制层 (Priority 3)
- **响应时间**: ~100ms
- **功能**: 精细操作、约束满足、双臂协调
- **架构**: 中型Transformer
- **特点**: 处理抓取、摆放等复杂操作

#### 4. 全局规划层 (Priority 4)
- **响应时间**: ~500ms
- **功能**: 长期规划、任务分解、全局优化
- **架构**: 大型Transformer
- **特点**: 最复杂的推理，默认禁用

### 智能调度机制

#### 优先级调度
```python
# 按优先级顺序处理
for layer_name in ['safety', 'gait', 'manipulation', 'planning']:
    if layer.should_activate(batch, context):
        output = layer.forward_with_timing(batch, context)

        # 安全层可以立即返回
        if layer_name == 'safety' and output.get('emergency'):
            return emergency_action
```

#### 自适应激活
```python
# 根据任务复杂度自动选择激活的层
def should_activate(self, inputs, context):
    task_complexity = context.get('task_complexity', 'medium')
    if self.layer_name == 'planning':
        return task_complexity in ['high', 'very_high']
    return True
```

#### 实时预算分配
```python
# 推理时根据延迟预算分配计算资源
scheduler.inference_mode(
    batch,
    task_info,
    latency_budget_ms=50.0  # 50ms预算
)
```

## 📊 性能监控

### 获取性能统计
```python
# 获取各层性能统计
stats = policy.get_performance_stats()

# 检查层健康状态
health = scheduler.check_layer_health()

# 自动调优
scheduler.auto_tune_layers(target_latency_ms=50.0)
```

### 动态控制
```python
# 动态启用/禁用层
policy.set_layer_enabled('planning', False)

# 获取当前激活的层
active_layers = policy.get_active_layers()
```

## 🧪 测试与验证

### 运行完整验证
```bash
python validate_hierarchical_framework.py
```

### 预期测试结果
```
✅ SafetyReflexLayer: <10ms响应时间
✅ GaitControlLayer: <50ms响应时间
✅ ManipulationLayer: <200ms响应时间
✅ GlobalPlanningLayer: <1000ms响应时间
✅ HierarchicalScheduler: 正确的层激活和调度
```

### 性能基准
- **安全层**: 1-5ms (批次大小1-8)
- **步态层**: 5-20ms (依据序列长度)
- **操作层**: 20-100ms (依据复杂度)
- **规划层**: 100-500ms (最复杂任务)

## 🐛 故障排除

### 常见问题

#### 1. 导入错误
```bash
# 确保路径正确
export PYTHONPATH=$PYTHONPATH:/path/to/kuavo_data_challenge
```

#### 2. CUDA内存不足
```yaml
# 减少batch size
training:
  batch_size: 32  # 从64降到32

# 禁用最复杂的层
hierarchical:
  layers:
    planning:
      enabled: False
```

#### 3. 训练不收敛
```yaml
# 使用课程学习
curriculum_learning:
  enable: True

# 调整层权重
layer_weights:
  safety: 1.0
  gait: 1.0
  manipulation: 1.0
  planning: 0.5
```

#### 4. 推理延迟过高
```python
# 使用推理模式
scheduler.inference_mode(batch, task_info, latency_budget_ms=30.0)

# 自动调优
scheduler.auto_tune_layers(target_latency_ms=50.0)
```

### 调试技巧

#### 1. 单层测试
```python
# 单独测试某一层
layer = SafetyReflexLayer(config, base_config)
output = layer.forward(inputs)
```

#### 2. 性能分析
```python
# 获取详细性能统计
stats = scheduler.get_performance_stats()
print(stats)
```

#### 3. 逐步启用
```python
# 逐步启用层进行调试
scheduler.set_layer_enabled('planning', False)
scheduler.set_layer_enabled('manipulation', False)
# 只使用基础层进行调试
```

## 📈 优化建议

### 训练优化
1. **使用课程学习**: 从简单到复杂逐步训练
2. **调整层权重**: 根据任务重要性调整权重
3. **监控性能**: 实时监控各层性能并调优
4. **GPU内存管理**: 合理分配各层的计算资源

### 推理优化
1. **预算分配**: 根据实时性要求分配计算预算
2. **层选择**: 根据任务复杂度选择激活的层
3. **批处理**: 合理设置推理批次大小
4. **模型压缩**: 可考虑对大层进行模型压缩

## 🔄 版本兼容性

### 向后兼容
- 设置 `use_hierarchical: False` 即可使用传统架构
- 所有原有配置项保持兼容
- 训练脚本无需修改

### 迁移指南
1. **备份原配置**: 保留原 `diffusion_config.yaml`
2. **使用新配置**: 复制 `humanoid_diffusion_config.yaml`
3. **调整参数**: 根据需求调整层配置
4. **验证功能**: 运行验证脚本确认正常工作

## 📞 支持与反馈

遇到问题时请：
1. 首先运行验证脚本诊断问题
2. 查看生成的测试报告文件
3. 检查层的性能统计信息
4. 参考本指南的故障排除部分

---

**祝您使用愉快！🚀**