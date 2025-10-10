# Diffusion Policy 文档索引

> Diffusion Policy 的完整技术文档和使用指南

---

## 📚 文档目录

### 1. [Diffusion Policy 架构详解](diffusion_policy_architecture.md)

**适合人群**: 所有开发者和研究人员

**内容概览**:
- ✅ 架构概览与设计理念
- ✅ 训练主流程详解
- ✅ 核心组件实现 (编码器、Transformer、扩散模型)
- ✅ 多模态融合机制
- ✅ 扩散过程原理 (前向加噪、反向去噪)
- ✅ 推理逻辑与队列机制
- ✅ 配置系统说明
- ✅ 关键设计决策

**重点章节**:
1. **第2节: 训练主流程** - 理解整个训练pipeline
2. **第3节: 核心组件详解** - 深入每个模块的实现
3. **第4节: Transformer架构** - TransformerForDiffusion的详细解析
4. **第6节: 扩散过程** - DDPM和DDIM的数学原理与实现

---

### 2. [Diffusion Policy 技术细节与实战](diffusion_technical_details.md)

**适合人群**: 需要实际训练和部署的工程师

**内容概览**:
- ✅ 扩散数学原理 (前向扩散、反向去噪、噪声调度)
- ✅ Transformer详细实现 (多头注意力、位置编码、FFN)
- ✅ 完整数据流与维度变换
- ✅ 实战训练指南 (数据准备、配置调优、训练策略)
- ✅ 调试与优化技巧
- ✅ 常见问题解决方案

**重点章节**:
1. **第1节: 扩散数学原理** - 理解DDPM和DDIM的数学基础
2. **第3节: 数据流与维度** - 追踪每一步的张量形状变化
3. **第4节: 实战训练指南** - 从零开始训练一个Diffusion Policy
4. **第6节: 常见问题** - 快速定位和解决训练中的问题

---

### 3. [Diffusion vs Hierarchical 深度对比](diffusion_vs_hierarchical_comparison.md)

**适合人群**: 需要选择架构的研究者和工程师

**内容概览**:
- ✅ 架构对比 (可视化图表、组件差异)
- ✅ 核心设计差异 (设计理念、特征编码、去噪网络)
- ✅ 训练流程对比 (端到端 vs 课程学习)
- ✅ 推理流程对比 (正常 vs 紧急响应)
- ✅ 代码实现对比 (继承关系、forward函数)
- ✅ 适用场景分析 (何时用哪个)
- ✅ 性能与复杂度对比
- ✅ 如何选择 (决策树、渐进式策略)

**重点章节**:
1. **第1节: 架构对比概览** - 直观理解两种架构的区别
2. **第2节: 核心设计差异** - 深入理解设计理念
3. **第6节: 适用场景分析** - 帮助你选择合适的架构
4. **第8节: 如何选择** - 提供实用的决策指南

---

## 🚀 快速开始

### 第一次使用?

1. **从架构开始**: 阅读 [diffusion_policy_architecture.md](diffusion_policy_architecture.md) 的**第1节**，了解整体设计
2. **理解训练流程**: 阅读**第2节**，掌握训练的主要步骤
3. **深入核心模块**: 根据兴趣选择**第3-4节**的特定模块深入学习
4. **开始训练**: 参考 [diffusion_technical_details.md](diffusion_technical_details.md) 的**第4节**

### 已经熟悉扩散模型?

- 直接查看 [diffusion_policy_architecture.md](diffusion_policy_architecture.md) 的**第3-4节**，了解我们的具体实现
- 关注**多模态融合**部分（第5节），这是我们的特色

### 遇到问题?

1. 查看 [diffusion_technical_details.md](diffusion_technical_details.md) 的**第6节: 常见问题**
2. 使用**第3节: 数据流与维度**来追踪维度变化，定位问题
3. 参考**第5节: 调试与优化**的诊断工具

---

## 📋 对比: Diffusion vs Hierarchical

| 特性 | Diffusion Policy | Hierarchical Policy |
|---|---|---|
| **架构** | 单一扩散模型 | 四层分层架构 |
| **适用场景** | 通用机器人任务 | 人形机器人复杂任务 |
| **训练复杂度** | 中等 | 高 (需要课程学习) |
| **推理速度** | 慢 (多步去噪) | 较快 (可选择性激活层) |
| **动作分布** | 多模态 | 单模态 (每层独立) |
| **文档** | 本目录 | [hierarchical_*](README.md) |

📖 **详细对比**: 查看 [Diffusion vs Hierarchical 深度对比](diffusion_vs_hierarchical_comparison.md) 了解两种架构的详细差异、适用场景和选择建议

---

## 📖 文档组织

```
docs/
├── README_DIFFUSION.md                      # 本文件 (Diffusion文档索引)
├── diffusion_policy_architecture.md         # Diffusion架构详解 (主文档)
├── diffusion_technical_details.md           # Diffusion技术细节与实战
├── diffusion_vs_hierarchical_comparison.md  # 两种架构深度对比 ⭐新增
│
├── README.md                                # Hierarchical文档索引
├── hierarchical_policy_architecture.md      # Hierarchical架构详解
├── hierarchical_layers_detailed.md          # Hierarchical层详细说明
└── hierarchical_dataflow_and_usage.md       # Hierarchical数据流与使用
```

---

## 🔗 相关资源

### 代码文件

| 文件 | 说明 |
|---|---|
| `kuavo_train/train_policy.py` | 训练主脚本 |
| `kuavo_train/wrapper/policy/diffusion/DiffusionModelWrapper.py` | 核心扩散模型 |
| `kuavo_train/wrapper/policy/diffusion/DiffusionPolicyWrapper.py` | Policy包装器 |
| `kuavo_train/wrapper/policy/diffusion/transformer_diffusion.py` | Transformer实现 |
| `configs/policy/diffusion_config.yaml` | 配置文件 |
| `kuavo_deploy/examples/eval/eval_kuavo.py` | 推理脚本 |

### 配置文件

```yaml
# 主配置: configs/policy/diffusion_config.yaml
policy_name: diffusion
policy:
  horizon: 16
  n_action_steps: 8
  noise_scheduler_type: DDPM
  num_train_timesteps: 100
  use_amp: True
  custom:
    use_transformer: True
    transformer_n_emb: 512
    transformer_n_layer: 4
```

---

## 💡 使用建议

### 研究人员

- 重点阅读: 扩散数学原理、Transformer架构
- 关注: 多模态融合、噪声调度
- 实验: 尝试不同的调度策略、网络架构

### 工程师

- 重点阅读: 训练主流程、实战训练指南
- 关注: 配置系统、性能优化
- 实践: 快速验证、调试技巧

### 学生

- 建议路径:
  1. 架构概览 (了解整体)
  2. 扩散数学原理 (理解理论)
  3. 核心组件详解 (学习实现)
  4. 实战训练指南 (动手实践)

---

## 🎯 学习路径

### 初级 (1-2天)

**目标**: 能够运行训练和推理

- [ ] 阅读架构概览 (第1节)
- [ ] 理解训练主流程 (第2节)
- [ ] 跑通训练脚本
- [ ] 进行简单推理

**资源**:
- [架构详解](diffusion_policy_architecture.md) 第1-2节
- [实战指南](diffusion_technical_details.md) 第4节

### 中级 (3-5天)

**目标**: 理解核心实现，能够调试和优化

- [ ] 深入核心组件 (第3节)
- [ ] 理解Transformer架构 (第4节)
- [ ] 掌握多模态融合 (第5节)
- [ ] 学习调试技巧 (第5节)

**资源**:
- [架构详解](diffusion_policy_architecture.md) 第3-5节
- [技术细节](diffusion_technical_details.md) 第2-3节

### 高级 (5-7天)

**目标**: 完全掌握原理，能够修改和扩展

- [ ] 掌握扩散数学 (第6节 + 技术文档第1节)
- [ ] 追踪完整数据流 (技术文档第3节)
- [ ] 理解所有设计决策 (第9节)
- [ ] 能够修改网络架构
- [ ] 能够添加新功能

**资源**:
- 所有文档
- 源代码
- 论文: [Diffusion Policy](https://arxiv.org/abs/2303.04137)

---

## 📝 重要概念索引

### 扩散模型

- **前向过程**: 逐步添加噪声 → [架构文档 §6.1](diffusion_policy_architecture.md#61-扩散模型原理)
- **反向过程**: 逐步去噪生成动作 → [架构文档 §6.1](diffusion_policy_architecture.md#61-扩散模型原理)
- **DDPM vs DDIM**: 采样方法对比 → [架构文档 §6.3](diffusion_policy_architecture.md#632-ddpm-vs-ddim)
- **噪声调度**: 控制加噪程度 → [技术文档 §1.3](diffusion_technical_details.md#13-噪声调度)

### Transformer

- **多头注意力**: Self-Attention和Cross-Attention → [技术文档 §2.1](diffusion_technical_details.md#21-多头注意力机制)
- **位置编码**: 为序列添加位置信息 → [技术文档 §2.2](diffusion_technical_details.md#22-position-encoding)
- **Encoder-Decoder**: 条件生成架构 → [架构文档 §4.1](diffusion_policy_architecture.md#41-transformerfordiffusion)

### 多模态融合

- **Self-Attention聚合**: 多相机特征融合 → [架构文档 §5.2.1](diffusion_policy_architecture.md#521-self-attention聚合多相机)
- **Cross-Attention融合**: RGB与Depth融合 → [架构文档 §5.2.2](diffusion_policy_architecture.md#522-cross-attention融合rgb和depth)

### 实战技巧

- **学习率调优**: → [技术文档 §4.2.1](diffusion_technical_details.md#421-学习率调优)
- **Batch Size选择**: → [技术文档 §4.2.2](diffusion_technical_details.md#422-batch-size选择)
- **内存优化**: → [技术文档 §5.3](diffusion_technical_details.md#53-内存优化)

---

## 🔍 快速查找

### 我想了解...

| 问题 | 查看章节 |
|---|---|
| 整体架构是什么样的? | [架构文档 §1](diffusion_policy_architecture.md#1-架构概览) |
| 训练流程是怎样的? | [架构文档 §2](diffusion_policy_architecture.md#2-训练主流程) |
| 扩散模型如何工作? | [架构文档 §6](diffusion_policy_architecture.md#6-扩散过程) |
| Transformer的实现细节? | [技术文档 §2](diffusion_technical_details.md#2-transformer详细实现) |
| 如何开始训练? | [技术文档 §4](diffusion_technical_details.md#4-实战训练指南) |
| 遇到错误怎么办? | [技术文档 §6](diffusion_technical_details.md#6-常见问题) |
| 各步的tensor形状? | [技术文档 §3](diffusion_technical_details.md#3-数据流与维度) |
| 如何优化性能? | [技术文档 §5](diffusion_technical_details.md#5-调试与优化) |

---

## ⚡ 命令速查

```bash
# 训练
python kuavo_train/train_policy.py --config-name=diffusion_config

# 恢复训练
python kuavo_train/train_policy.py training.resume=True training.resume_timestamp=run_xxx

# 修改参数训练
python kuavo_train/train_policy.py training.batch_size=32 policy.horizon=32

# 推理
python kuavo_deploy/examples/eval/eval_kuavo.py --checkpoint path/to/best --policy-type diffusion

# TensorBoard监控
tensorboard --logdir outputs/train/
```

---

## 📞 获取帮助

### 文档中找不到答案?

1. 检查两个文档的目录
2. 使用Ctrl+F搜索关键词
3. 查看源代码注释
4. 参考配置文件示例

### 代码层面的问题?

- 查看文件头部的docstring
- 使用调试工具 (参见技术文档§5.1)
- 添加print语句追踪数据流

---

**文档版本**: 1.0
**最后更新**: 2025-10-10
**维护者**: AI Assistant

