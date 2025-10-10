# 分层人形机器人 Diffusion Policy 文档中心

> 欢迎！这里是分层架构的完整文档索引

---

## 📚 文档导航

### 🏗️ 核心架构文档

#### [1. 分层架构详解 (主文档)](./hierarchical_policy_architecture.md)
**推荐首先阅读**

涵盖内容:
- ✅ 整体架构设计理念
- ✅ 训练主流程详解
- ✅ 四层架构概述
- ✅ 课程学习机制
- ✅ 任务特定训练
- ✅ 推理逻辑
- ✅ 配置系统
- ✅ 关键设计决策

**适合人群**: 所有用户，特别是初次使用者

**阅读时间**: 约60分钟

---

#### [2. 各层详细设计](./hierarchical_layers_detailed.md)
**深入理解各层实现**

涵盖内容:
- ✅ BaseLayer 抽象基类
- ✅ SafetyReflexLayer 详细设计（GRU架构、紧急检测、倾斜检测）
- ✅ GaitControlLayer 详细设计（混合GRU+Transformer、步态周期建模）
- ✅ ManipulationLayer 详细设计（多模态融合、约束满足、双臂协调）
- ✅ GlobalPlanningLayer 详细设计（长期记忆、任务分解）
- ✅ 层间通信机制
- ✅ 损失函数设计

**适合人群**: 需要修改或扩展层功能的开发者

**阅读时间**: 约90分钟

---

#### [3. 数据流与实战指南](./hierarchical_dataflow_and_usage.md)
**实际使用和调试**

涵盖内容:
- ✅ 完整数据流详解（从数据集到损失）
- ✅ 训练数据流
- ✅ 推理数据流
- ✅ 实战示例（训练、恢复训练、部署）
- ✅ 调试技巧
- ✅ 性能优化
- ✅ 常见问题解答

**适合人群**: 实际使用该系统的研究员和工程师

**阅读时间**: 约75分钟

---

### 📊 对比与参考

#### [4. Diffusion vs Hierarchical 深度对比](./diffusion_vs_hierarchical_comparison.md) ⭐推荐
**帮助你理解两种架构的区别并做出选择**

涵盖内容:
- ✅ 架构对比 (可视化图表、组件差异)
- ✅ 核心设计差异 (设计理念对比)
- ✅ 训练/推理流程对比
- ✅ 适用场景分析
- ✅ 性能与复杂度对比
- ✅ 如何选择 (决策树、渐进式策略)

**适合人群**: 需要选择架构的研究者和工程师

---

#### [5. Diffusion训练分析](./diffusion_training_analysis.md)
Diffusion模型训练的详细分析

#### [6. Diffusion Policy 完整文档](./README_DIFFUSION.md)
查看普通 Diffusion Policy 的完整文档

---

## 🚀 快速开始

### 新用户推荐路径

```
1. 阅读 [架构详解] 第1-2节
   ├─ 了解整体设计
   └─ 理解四层架构的职责

2. 阅读 [数据流与实战] 第4节
   ├─ 尝试训练示例
   └─ 运行推理示例

3. 深入阅读 [各层详细设计]
   └─ 根据需要查阅特定层的实现

4. 遇到问题时参考 [数据流与实战] 第7节
   └─ 常见问题解答
```

### 开发者推荐路径

```
1. 快速浏览 [架构详解] 全文
   └─ 建立整体认知

2. 深入阅读 [各层详细设计]
   ├─ 理解每层的网络架构
   ├─ 理解激活条件
   └─ 理解损失函数

3. 参考 [数据流与实战] 第5节
   ├─ 学习调试技巧
   └─ 学习性能优化方法

4. 修改代码时查阅相关章节
```

---

## 📖 文档结构

```
docs/
├── README.md (本文件)
│   └─ 文档索引和导航
│
├── hierarchical_policy_architecture.md
│   ├─ 1. 架构概览
│   ├─ 2. 训练主流程
│   ├─ 3. 分层架构详解
│   ├─ 4. 课程学习机制
│   ├─ 5. 任务特定训练
│   ├─ 6. 推理逻辑
│   ├─ 7. 配置系统
│   └─ 8. 关键设计决策
│
├── hierarchical_layers_detailed.md
│   ├─ 1. BaseLayer 抽象基类
│   ├─ 2. SafetyReflexLayer 详解
│   ├─ 3. GaitControlLayer 详解
│   ├─ 4. ManipulationLayer 详解
│   ├─ 5. GlobalPlanningLayer 详解
│   ├─ 6. 层间通信机制
│   └─ 7. 损失函数设计
│
└── hierarchical_dataflow_and_usage.md
    ├─ 1. 完整数据流详解
    ├─ 2. 训练数据流
    ├─ 3. 推理数据流
    ├─ 4. 实战示例
    ├─ 5. 调试技巧
    ├─ 6. 性能优化
    └─ 7. 常见问题
```

---

## 🎯 按任务查找

### 我想要...

#### 训练一个新模型
1. 阅读 [架构详解 - 训练主流程](./hierarchical_policy_architecture.md#2-训练主流程)
2. 参考 [实战指南 - 从零开始训练](./hierarchical_dataflow_and_usage.md#41-从零开始训练)
3. 配置 [课程学习](./hierarchical_policy_architecture.md#4-课程学习机制)

#### 理解某个特定层的工作原理
1. 阅读 [架构详解 - 分层架构详解](./hierarchical_policy_architecture.md#3-分层架构详解)
2. 深入阅读 [各层详细设计](./hierarchical_layers_detailed.md) 的对应章节

#### 部署模型到机器人
1. 阅读 [实战指南 - 部署到真实机器人](./hierarchical_dataflow_and_usage.md#44-部署到真实机器人)
2. 了解 [推理逻辑](./hierarchical_policy_architecture.md#6-推理逻辑)
3. 学习 [推理模式](./hierarchical_layers_detailed.md#614-推理模式-inference-mode)

#### 调试训练问题
1. 参考 [调试技巧](./hierarchical_dataflow_and_usage.md#5-调试技巧)
2. 查看 [常见问题](./hierarchical_dataflow_and_usage.md#7-常见问题)

#### 优化训练速度或内存
1. 阅读 [性能优化](./hierarchical_dataflow_and_usage.md#6-性能优化)
2. 了解 [配置系统](./hierarchical_policy_architecture.md#7-配置系统)

#### 添加新的层或功能
1. 理解 [BaseLayer 抽象基类](./hierarchical_layers_detailed.md#1-baselayer-抽象基类)
2. 参考现有层的实现
3. 了解 [层间通信机制](./hierarchical_layers_detailed.md#6-层间通信机制)

#### 多任务训练
1. 阅读 [任务特定训练](./hierarchical_policy_architecture.md#5-任务特定训练)
2. 参考 [任务特定训练数据流](./hierarchical_dataflow_and_usage.md#23-任务特定训练数据流)

---

## 🔍 快速查找表

### 关键概念

| 概念 | 文档位置 | 页码 |
|---|---|---|
| 分层架构总览 | 架构详解 | 第1节 |
| 四层设计理念 | 架构详解 | 第3节 |
| SafetyReflexLayer | 各层详细设计 | 第2节 |
| GaitControlLayer | 各层详细设计 | 第3节 |
| ManipulationLayer | 各层详细设计 | 第4节 |
| GlobalPlanningLayer | 各层详细设计 | 第5节 |
| 课程学习 | 架构详解 | 第4节 |
| 任务特定训练 | 架构详解 | 第5节 |
| 数据流 | 数据流与实战 | 第1节 |
| 训练示例 | 数据流与实战 | 第4节 |
| 调试技巧 | 数据流与实战 | 第5节 |
| 性能优化 | 数据流与实战 | 第6节 |
| 常见问题 | 数据流与实战 | 第7节 |

### 代码位置

| 组件 | 文件路径 |
|---|---|
| 训练主脚本 | `kuavo_train/train_hierarchical_policy.py` |
| Policy主类 | `kuavo_train/wrapper/policy/humanoid/HumanoidDiffusionPolicy.py` |
| 调度器 | `kuavo_train/wrapper/policy/humanoid/HierarchicalScheduler.py` |
| SafetyLayer | `kuavo_train/wrapper/policy/humanoid/layers/SafetyReflexLayer.py` |
| GaitLayer | `kuavo_train/wrapper/policy/humanoid/layers/GaitControlLayer.py` |
| ManipulationLayer | `kuavo_train/wrapper/policy/humanoid/layers/ManipulationLayer.py` |
| PlanningLayer | `kuavo_train/wrapper/policy/humanoid/layers/GlobalPlanningLayer.py` |
| 任务管理器 | `kuavo_train/wrapper/policy/humanoid/TaskSpecificTrainingManager.py` |
| 配置文件 | `configs/policy/humanoid_diffusion_config.yaml` |

---

## 📝 更新日志

### 2025-10-10 - 文档v1.0发布
- ✅ 完成主架构文档
- ✅ 完成各层详细设计文档
- ✅ 完成数据流与实战指南
- ✅ 创建文档索引

---

## 🤝 贡献

如果您发现文档中的错误或有改进建议，请：
1. 在项目中提Issue
2. 或直接修改文档并提交PR

---

## 📞 联系方式

如有问题，请参考：
- 项目主README: [../README.md](../README.md)
- 训练日志: `hierarchical_training.log`
- TensorBoard: `tensorboard --logdir outputs/train/`

---

**祝您使用愉快！** 🎉

