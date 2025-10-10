# 分层人形机器人 Diffusion Policy 架构详解

> **作者**: AI Assistant
> **日期**: 2025-10-10
> **版本**: 1.0
> **适用于**: `kuavo_data_challenge` 项目

---

## 📋 目录

1. [架构概览](#1-架构概览)
2. [训练主流程](#2-训练主流程)
3. [分层架构详解](#3-分层架构详解)
4. [课程学习机制](#4-课程学习机制)
5. [任务特定训练](#5-任务特定训练)
6. [推理逻辑](#6-推理逻辑)
7. [配置系统](#7-配置系统)
8. [关键设计决策](#8-关键设计决策)

---

## 1. 架构概览

### 1.1 整体设计理念

分层人形机器人 Diffusion Policy（Hierarchical Humanoid Diffusion Policy）是一个**混合架构**，结合了：

- **Diffusion Policy**: 基础的扩散模型用于动作生成
- **分层控制**: 四层优先级结构，模拟人类认知层次
- **课程学习**: 渐进式训练，从简单到复杂
- **任务特定训练**: 支持多任务场景，防止灾难性遗忘

### 1.2 核心组件关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                     train_hierarchical_policy.py                │
│                          (训练主入口)                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
    ┌───────────▼────────────┐   ┌───────▼──────────────────────┐
    │ HumanoidDiffusionPolicy│   │ TaskSpecificTrainingManager  │
    │  (策略主控制器)         │   │   (任务管理器)                │
    └───────────┬────────────┘   └──────────────────────────────┘
                │
    ┌───────────┴────────────┐
    │                        │
┌───▼────────────┐  ┌────────▼──────────────┐
│HierarchicalScheduler│  │HierarchicalDiffusionModel│
│  (层调度器)     │  │   (Diffusion模型)      │
└───┬────────────┘  └───────────────────────┘
    │
    │ 管理四个层
    │
    ├─► SafetyReflexLayer (优先级1, ~10ms)
    ├─► GaitControlLayer (优先级2, ~20ms)
    ├─► ManipulationLayer (优先级3, ~100ms)
    └─► GlobalPlanningLayer (优先级4, ~500ms)
```

### 1.3 数据流概览

```
输入数据 (观测)
    │
    ├─► observation.state (关节状态)
    ├─► observation.images.* (RGB相机)
    └─► observation.depth.* (深度相机)
    │
    ▼
图像预处理 (裁剪、缩放、归一化)
    │
    ▼
分层处理 (HierarchicalScheduler)
    │
    ├─► Layer 1: SafetyReflexLayer
    │   └─► 输出: 紧急状态、平衡控制
    │
    ├─► Layer 2: GaitControlLayer
    │   └─► 输出: 步态特征、负载适应
    │
    ├─► Layer 3: ManipulationLayer
    │   └─► 输出: 操作特征、双臂协调
    │
    └─► Layer 4: GlobalPlanningLayer
        └─► 输出: 规划特征、任务分解
    │
    ▼
Diffusion Model 损失计算
    │
    ▼
层损失聚合 (加权求和)
    │
    ▼
最终损失 & 反向传播
```

---

## 2. 训练主流程

### 2.1 入口函数: `train_hierarchical_policy.py::main()`

```python
@hydra.main(config_path="../configs/policy/",
            config_name="humanoid_diffusion_config")
def main(cfg: DictConfig):
    """统一分层架构训练主函数"""
```

#### 训练流程图

```
开始
  │
  ├─► 1. 初始化日志系统
  │
  ├─► 2. 设置随机种子
  │
  ├─► 3. 检查训练模式
  │      ├─► 基础模式 (use_hierarchical=True, task_specific=False)
  │      └─► 任务特定模式 (task_specific=True)
  │
  ├─► 4. 加载数据集
  │      ├─► 基础模式: 直接加载 LeRobotDataset
  │      └─► 任务特定模式: 动态加载多任务数据
  │
  ├─► 5. 构建 Policy
  │      └─► HumanoidDiffusionPolicy (分层架构)
  │
  ├─► 6. 构建优化器 & 学习率调度器
  │
  ├─► 7. 课程学习 (可选)
  │      └─► 按阶段训练各层
  │
  ├─► 8. 主训练循环
  │      ├─► Epoch循环
  │      ├─► Batch循环
  │      ├─► 前向传播
  │      ├─► 损失计算
  │      ├─► 反向传播
  │      └─► 定期保存检查点
  │
  └─► 9. 训练完成，保存最终模型
```

### 2.2 关键训练步骤详解

#### 2.2.1 数据集加载

**基础模式:**
```python
# 从配置读取数据路径
dataset_metadata = LeRobotDatasetMetadata(cfg.repoid, root=cfg.root)

# 构建delta timestamps（观测和动作的时间偏移）
delta_timestamps = build_delta_timestamps(dataset_metadata, policy_cfg)

# 创建数据集
dataset = LeRobotDataset(
    cfg.repoid,
    delta_timestamps=delta_timestamps,
    root=cfg.root,
    episodes=episodes_to_use,  # 可限制使用的episodes
    image_transforms=image_transforms,  # 图像增强
)
```

**任务特定模式:**
```python
# 注册可用任务
task_manager.register_available_task(task_id, episode_count, data_path)

# 加载每个任务的数据集
for task_id in available_tasks:
    dataset, metadata = load_task_dataset(task_id, cfg, policy_cfg, transforms)
    datasets[task_id] = dataset

# 创建加权采样的DataLoader
dataloader = create_task_specific_dataloader(datasets, task_manager, cfg, device)
```

#### 2.2.2 Policy 构建

```python
def build_hierarchical_policy(policy_cfg, dataset_stats):
    """构建分层架构的policy"""
    return HumanoidDiffusionPolicy(policy_cfg, dataset_stats)
```

**Policy初始化流程:**
1. 检查 `use_hierarchical` 配置
2. 如果启用，创建 `HierarchicalScheduler` (管理四层)
3. 创建 `HierarchicalDiffusionModel` (替换标准Diffusion模型)
4. 初始化任务条件权重系统

#### 2.2.3 课程学习执行

```python
def run_curriculum_learning_stage(
    policy, stage_config, dataset, cfg, device, writer,
    current_step, optimizer, lr_scheduler, scaler,
    output_directory, amp_enabled, task_manager, dataloader
):
    """运行课程学习的单个阶段"""
```

**阶段执行流程:**
```
1. 解析阶段配置
   ├─► name: 阶段名称
   ├─► layers: 激活的层列表 ['safety', 'manipulation', ...]
   ├─► epochs: 训练轮次
   └─► target_task: 目标任务ID (任务特定模式)

2. 激活指定层
   └─► policy.set_curriculum_stage(enabled_layers)

3. 配置任务层权重 (任务特定模式)
   └─► policy.set_task_layer_weights(layer_weights)

4. Epoch循环
   ├─► Batch迭代
   ├─► 构建curriculum_info
   ├─► 前向传播: loss, hierarchical_info = policy.forward(batch, curriculum_info)
   ├─► 反向传播
   └─► 记录损失和层性能指标

5. 保存最佳模型
   └─► 基于epoch平均损失
```

#### 2.2.4 主训练循环

```python
for epoch in range(start_epoch, cfg.training.max_epoch):
    dataloader = create_dataloader(...)  # 创建或更新dataloader

    for batch in dataloader:
        # 数据移到设备
        batch = {k: v.to(device) for k, v in batch.items()}

        # 前向传播
        with autocast(amp_enabled):
            if use_task_specific:
                task_loss_weights = task_manager.get_task_specific_loss_weights(batch)
                loss, info = policy.forward(batch, task_weights=task_loss_weights)
            else:
                loss, info = policy.forward(batch)

        # 梯度累积
        scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()

        # 优化器步骤
        if steps % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()

        # 记录日志
        if steps % log_freq == 0:
            writer.add_scalar("train/loss", scaled_loss.item(), steps)
            # 记录分层架构的详细信息
            log_hierarchical_info(writer, info, steps)

        steps += 1

    # 保存检查点
    save_hierarchical_checkpoint(...)
```

---

## 3. 分层架构详解

### 3.1 HierarchicalScheduler (层调度器)

**位置**: `kuavo_train/wrapper/policy/humanoid/HierarchicalScheduler.py`

#### 3.1.1 核心职责

```python
class HierarchicalScheduler(nn.Module):
    """分层架构的核心调度器，负责管理四个层次的激活、调度和输出聚合"""
```

1. **层管理**: 创建和管理四个层
2. **优先级调度**: 按优先级顺序执行层
3. **上下文传递**: 在层之间传递信息
4. **紧急处理**: 安全层可立即中断
5. **性能监控**: 跟踪各层执行时间和激活次数

#### 3.1.2 初始化流程

```python
def __init__(self, hierarchical_config, base_config):
    # 1. 保存配置
    self.config = hierarchical_config
    self.base_config = base_config

    # 2. 构建四个层
    self.layers = self._build_layers()
    #    ├─► SafetyReflexLayer
    #    ├─► GaitControlLayer
    #    ├─► ManipulationLayer
    #    └─► GlobalPlanningLayer

    # 3. 设置优先级和权重
    self.layer_priorities = {name: layer.get_priority()
                            for name, layer in self.layers.items()}
    self.layer_weights = hierarchical_config.get('layer_weights', {})

    # 4. 初始化性能监控
    self.total_forward_calls = 0
    self.layer_activation_stats = {name: 0 for name in self.layers.keys()}
```

#### 3.1.3 前向传播流程

```python
def forward(self, batch, task_info=None):
    """分层处理前向传播"""

    # 1. 检查是否使用推理模式
    if task_info and 'latency_budget_ms' in task_info:
        return self.inference_mode(batch, task_info, latency_budget)

    # 2. 构建上下文
    context = self._build_context(batch, task_info)
    #    包含: batch_size, device, training, task_info

    # 3. 按优先级顺序处理各层
    outputs = {}
    for layer_name in self._get_processing_order():
        layer = self.layers[layer_name]

        # 3.1 检查是否应该激活
        if not layer.should_activate(batch, context):
            continue

        # 3.2 执行层的前向传播（带时间监控）
        layer_output = layer.forward_with_timing(batch, context)
        outputs[layer_name] = layer_output

        # 3.3 更新上下文（供后续层使用）
        context.update(layer_output)

        # 3.4 安全层紧急检查
        if layer_name == 'safety' and layer_output.get('emergency'):
            return {layer_name: layer_output}  # 立即返回

    # 4. 返回所有层的输出
    return outputs
```

#### 3.1.4 推理模式 (Inference Mode)

```python
def inference_mode(self, batch, task_info, latency_budget_ms=50.0):
    """根据延迟预算自适应激活层"""

    remaining_budget = latency_budget_ms
    outputs = {}

    for layer_name in self._get_processing_order():
        layer = self.layers[layer_name]

        # 1. 检查时间预算
        layer_budget = layer.get_latency_budget()
        if remaining_budget < layer_budget:
            print(f"⏰ Skipping {layer_name} due to time budget")
            continue

        # 2. 执行层推理
        start_time = time.time()
        layer_output = layer.forward_with_timing(batch, context)
        outputs[layer_name] = layer_output

        # 3. 更新剩余预算
        layer_time = (time.time() - start_time) * 1000
        remaining_budget -= layer_time

        # 4. 紧急情况立即返回
        if layer_name == 'safety' and layer_output.get('emergency'):
            break

    outputs['_inference_stats'] = {
        'total_time_ms': total_time,
        'remaining_budget_ms': remaining_budget,
        'within_budget': total_time <= latency_budget_ms
    }

    return outputs
```

### 3.2 Layer 1: SafetyReflexLayer (安全反射层)

**位置**: `kuavo_train/wrapper/policy/humanoid/layers/SafetyReflexLayer.py`

#### 3.2.1 设计理念

```
优先级: 1 (最高)
响应时间: <10ms
架构: 极简GRU
核心功能: 防跌倒、紧急停止、基础平衡
```

#### 3.2.2 网络架构

```python
class SafetyReflexLayer(BaseLayer):
    def __init__(self, config, base_config):
        super().__init__(config, "safety", priority=1)

        # 输入维度: 机器人关节状态
        # only_arm=True: 双臂14维 + 手爪2维 = 16维
        self.input_dim = 16
        self.hidden_size = 64
        self.output_dim = 16

        # 1. 极简GRU (只用一层确保速度)
        self.balance_gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # 2. 紧急情况检测器
        self.emergency_detector = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出0-1的紧急评分
        )

        # 3. 平衡控制器
        self.balance_controller = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_dim),
            nn.Tanh()  # 限制输出范围
        )

        # 4. 倾斜检测器
        self.tilt_detector = nn.Linear(self.hidden_size, 2)  # roll, pitch

        # 5. 紧急动作生成器
        self.emergency_action_generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_dim),
            nn.Tanh()
        )
```

#### 3.2.3 前向传播逻辑

```python
def forward(self, inputs, context=None):
    """安全反射层前向传播"""

    # 1. 提取关节状态
    robot_state = inputs['observation.state']
    # 处理维度: [batch, state_dim] -> [batch, 1, state_dim]
    if len(robot_state.shape) == 2:
        robot_state = robot_state.unsqueeze(1)

    # 2. GRU处理
    gru_output, hidden = self.balance_gru(robot_state)
    last_output = gru_output[:, -1, :]  # [batch, hidden_size]

    # 3. 紧急情况检测
    emergency_score = self.emergency_detector(last_output)  # [batch, 1]
    emergency = (emergency_score > self.emergency_threshold).squeeze(-1)  # [batch]

    # 4. 倾斜检测
    tilt_angles = self.tilt_detector(last_output)  # [batch, 2]
    tilt_angles_degrees = tilt_angles * 45.0  # 缩放到±45度
    tilt_emergency = torch.any(
        torch.abs(tilt_angles_degrees) > self.tilt_threshold_degrees,
        dim=-1
    )  # [batch]

    # 5. 综合紧急状态
    overall_emergency = torch.logical_or(emergency, tilt_emergency)

    # 6. 生成控制输出
    emergency_action = self.emergency_action_generator(last_output)
    balance_action_normal = self.balance_controller(last_output)

    # 7. 根据紧急状态选择动作
    overall_emergency_expanded = overall_emergency.unsqueeze(-1)
    balance_action = torch.where(
        overall_emergency_expanded,
        emergency_action,      # 紧急时使用紧急动作
        balance_action_normal  # 正常时使用平衡动作
    )

    # 8. 计算平衡置信度
    max_tilt = torch.max(torch.abs(tilt_angles_degrees), dim=-1)[0]
    balance_confidence = torch.exp(-max_tilt / 10.0)

    return {
        'emergency': overall_emergency,
        'emergency_score': emergency_score.squeeze(-1),
        'balance_action': balance_action,
        'emergency_action': emergency_action,
        'tilt_angles_degrees': tilt_angles_degrees,
        'balance_confidence': balance_confidence,
        'safety_status': self._compute_safety_status(...),
        'action': balance_action,
        'layer': 'safety'
    }
```

#### 3.2.4 激活条件

```python
def should_activate(self, inputs, context=None):
    """安全层始终激活"""
    return True
```

### 3.3 Layer 2: GaitControlLayer (步态控制层)

**位置**: `kuavo_train/wrapper/policy/humanoid/layers/GaitControlLayer.py`

#### 3.3.1 设计理念

```
优先级: 2
响应时间: ~20ms
架构: 混合GRU + 轻量Transformer
核心功能: 步态规划、负载适应、地形适应
```

#### 3.3.2 网络架构

```python
class GaitControlLayer(BaseLayer):
    def __init__(self, config, base_config):
        super().__init__(config, "gait", priority=2)

        self.input_dim = 16  # 双臂+手爪配置
        self.gru_hidden = 128
        self.gru_layers = 2
        self.tf_layers = 2
        self.tf_heads = 4

        # 1. GRU用于步态状态跟踪
        self.gait_state_gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            batch_first=True,
            dropout=0.1
        )

        # 2. 轻量Transformer用于步态规划
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.gru_hidden,
            nhead=self.tf_heads,
            dim_feedforward=self.gru_hidden * 2,
            dropout=0.1,
            batch_first=True
        )
        self.gait_planner = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.tf_layers
        )

        # 3. 负载适应模块
        self.load_adapter = LoadAdaptationModule(self.gru_hidden)

        # 4. 输出投影
        self.output_projection = nn.Linear(self.gru_hidden, self.input_dim)
```

#### 3.3.3 前向传播逻辑

```python
def forward(self, inputs, context=None):
    """步态控制前向传播"""

    # 1. 获取关节状态
    robot_state = inputs.get('observation.state')
    # 处理维度: [batch, state_dim] -> [batch, seq_len, state_dim]
    if len(robot_state.shape) == 2:
        robot_state = robot_state.unsqueeze(1)
    batch_size, seq_len, state_dim = robot_state.shape

    # 2. GRU处理步态状态
    gru_output, gru_hidden = self.gait_state_gru(robot_state)

    # 3. Transformer步态规划（如果序列足够长）
    if seq_len >= 10:  # 至少200ms历史
        planned_gait = self.gait_planner(gru_output)
    else:
        planned_gait = gru_output

    # 4. 负载适应
    adapted_gait = self.load_adapter(planned_gait, context)

    # 5. 最终输出
    final_output = self.output_projection(adapted_gait[:, -1, :])

    return {
        'gait_features': gru_output,
        'planned_gait': planned_gait,
        'adapted_gait': adapted_gait,
        'action': final_output,
        'layer': 'gait'
    }
```

#### 3.3.4 激活条件

```python
def should_activate(self, inputs, context=None):
    """当机器人需要移动时激活"""
    if context is None:
        return True
    return context.get('requires_locomotion', True)
```

### 3.4 Layer 3: ManipulationLayer (操作控制层)

**位置**: `kuavo_train/wrapper/policy/humanoid/layers/ManipulationLayer.py`

#### 3.4.1 设计理念

```
优先级: 3
响应时间: ~100ms
架构: Transformer主导
核心功能: 抓取、摆放、双臂协调、约束满足
```

#### 3.4.2 网络架构

```python
class ManipulationLayer(BaseLayer):
    def __init__(self, config, base_config):
        super().__init__(config, "manipulation", priority=3)

        self.hidden_size = 512
        self.num_layers = 3
        self.num_heads = 8
        self.dim_feedforward = 2048

        # 特征维度
        self.visual_dim = 1280  # EfficientNet-B0输出
        self.state_dim = 16
        actual_visual_dim = 12  # 3RGB + 3深度相机

        # 1. 视觉投影层
        self.visual_projection = nn.Linear(actual_visual_dim, self.visual_dim)

        # 2. 总输入投影
        self.input_projection = nn.Linear(
            self.visual_dim + self.state_dim,
            self.hidden_size
        )

        # 3. 主要的Transformer网络
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.manipulation_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # 4. 约束满足模块
        self.constraint_solver = ConstraintSatisfactionModule(self.hidden_size)

        # 5. 双臂协调模块
        self.bimanual_coordinator = BimanualCoordinationModule(
            self.hidden_size,
            self.state_dim
        )

        # 6. 输出投影
        self.action_head = nn.Linear(self.hidden_size, self.state_dim)
```

#### 3.4.3 前向传播逻辑

```python
def forward(self, inputs, context=None):
    """操作控制前向传播"""

    # 1. 提取和融合多模态特征
    features = self._extract_features(inputs)
    #    ├─► 状态特征: observation.state
    #    ├─► 视觉特征: observation.images.*
    #    └─► 深度特征: observation.depth.*
    batch_size, seq_len, _ = features.shape

    # 2. Transformer处理
    manipulation_features = self.manipulation_transformer(features)

    # 3. 约束满足
    constraint_solution = self.constraint_solver(manipulation_features, context)
    #    └─► 输出: constraint_satisfaction_score, constraints_met

    # 4. 双臂协调
    coordinated_actions = self.bimanual_coordinator(manipulation_features, context)

    # 5. 最终动作
    final_action = self.action_head(manipulation_features[:, -1, :])

    return {
        'manipulation_features': manipulation_features,
        'constraint_solution': constraint_solution,
        'coordinated_actions': coordinated_actions,
        'action': final_action,
        'layer': 'manipulation'
    }
```

#### 3.4.4 特征提取详解

```python
def _extract_features(self, inputs):
    """提取并融合多模态特征"""
    features_list = []

    # 1. 状态特征
    if 'observation.state' in inputs:
        state_features = inputs['observation.state']
        # 确保是3D: [batch, seq_len, state_dim]
        if len(state_features.shape) == 2:
            state_features = state_features.unsqueeze(1)
        features_list.append(state_features)

    # 2. 视觉特征（处理多相机输入）
    visual_features_list = []
    image_keys = [k for k in inputs.keys()
                  if k.startswith('observation.images.')
                  or k.startswith('observation.depth')]

    for key in image_keys:
        img_feature = inputs[key]
        # 全局平均池化: [batch, C, H, W] -> [batch, C]
        if len(img_feature.shape) == 4:
            img_feature = img_feature.mean(dim=(-2, -1))
        visual_features_list.append(img_feature)

    # 3. 拼接所有相机特征
    if visual_features_list:
        combined_visual = torch.cat(visual_features_list, dim=-1)
        # 投影到标准维度
        combined_visual = self.visual_projection(combined_visual)
        if len(combined_visual.shape) == 2:
            combined_visual = combined_visual.unsqueeze(1)
        features_list.append(combined_visual)
    else:
        # 零填充
        batch_size, seq_len = features_list[0].shape[:2]
        device = features_list[0].device
        zero_visual = torch.zeros(batch_size, seq_len, self.visual_dim, device=device)
        features_list.append(zero_visual)

    # 4. 特征拼接和投影
    combined_features = torch.cat(features_list, dim=-1)
    projected_features = self.input_projection(combined_features)

    return projected_features
```

#### 3.4.5 激活条件

```python
def should_activate(self, inputs, context=None):
    """当需要精细操作时激活"""
    if context is None:
        return True
    return context.get('requires_manipulation', True)
```

### 3.5 Layer 4: GlobalPlanningLayer (全局规划层)

**位置**: `kuavo_train/wrapper/policy/humanoid/layers/GlobalPlanningLayer.py`

#### 3.5.1 设计理念

```
优先级: 4 (最低，最复杂)
响应时间: ~500ms
架构: 大型Transformer
核心功能: 长期记忆、复杂任务规划、任务分解
```

#### 3.5.2 网络架构

```python
class GlobalPlanningLayer(BaseLayer):
    def __init__(self, config, base_config):
        super().__init__(config, "planning", priority=4)

        self.hidden_size = 1024
        self.num_layers = 4
        self.num_heads = 16
        self.dim_feedforward = 4096

        visual_dim = 1280
        self.state_dim = 16
        self.input_projection = nn.Linear(visual_dim + self.state_dim, self.hidden_size)

        # 1. 大型Transformer用于复杂推理
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.global_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # 2. 长期记忆模块
        self.memory_bank = LongTermMemoryModule(self.hidden_size)

        # 3. 任务分解模块
        self.task_decomposer = TaskDecompositionModule(self.hidden_size)

        # 4. 输出投影
        self.action_head = nn.Linear(self.hidden_size, self.state_dim)
        self.plan_head = nn.Linear(self.hidden_size, 64)
```

#### 3.5.3 前向传播逻辑

```python
def forward(self, inputs, context=None):
    """全局规划前向传播"""

    # 1. 编码全局状态
    global_state = self._encode_global_state(inputs, context)
    batch_size, seq_len, _ = global_state.shape

    # 2. 记忆检索
    relevant_memory = self.memory_bank.retrieve(global_state)
    #    └─► 使用注意力机制从记忆库检索相关信息

    # 3. 全局推理
    enhanced_state = torch.cat([global_state, relevant_memory], dim=-1)
    planning_output = self.global_transformer(enhanced_state)

    # 4. 任务分解
    task_plan = self.task_decomposer(planning_output, context)
    #    └─► 输出: task_scores, task_priorities, num_subtasks

    # 5. 输出动作和规划
    final_action = self.action_head(planning_output[:, -1, :])
    global_plan = self.plan_head(planning_output[:, -1, :])

    return {
        'global_features': planning_output,
        'task_plan': task_plan,
        'relevant_memory': relevant_memory,
        'global_plan': global_plan,
        'action': final_action,
        'layer': 'planning'
    }
```

#### 3.5.4 激活条件

```python
def should_activate(self, inputs, context=None):
    """当需要复杂规划时激活"""
    if context is None:
        return False  # 默认不激活最复杂的层

    task_complexity = context.get('task_complexity', 'medium')
    return task_complexity in ['high', 'very_high']
```

---

## 4. 课程学习机制

### 4.1 设计理念

课程学习（Curriculum Learning）模拟人类学习过程，从简单到复杂，逐步激活更多层。

### 4.2 课程学习配置

```yaml
# 在 humanoid_diffusion_config.yaml
hierarchical:
  curriculum_learning:
    enable: True
    universal_stages:
      stage1:
        name: 'manipulation_first'
        layers: ['manipulation']  # 先学抓取行为
        epochs: 8
        loss_threshold: 0.1

      stage2:
        name: 'manipulation_safety'
        layers: ['manipulation', 'safety']  # 加入安全约束
        epochs: 7
        loss_threshold: 0.08

      stage3:
        name: 'manipulation_safety_gait'
        layers: ['manipulation', 'safety', 'gait']  # 添加步态
        epochs: 5
        loss_threshold: 0.06

      stage4:
        name: 'full_hierarchy'
        layers: ['safety', 'gait', 'manipulation', 'planning']  # 全开
        epochs: 10
        loss_threshold: 0.05
```

### 4.3 课程学习执行流程

```python
# 在 train_hierarchical_policy.py
curriculum_stages = task_manager.get_current_curriculum_stages()

current_step = steps
for stage_name, stage_config in curriculum_stages.items():
    print(f"🎓 开始课程学习阶段: {stage_name}")
    print(f"   激活层: {stage_config['layers']}")
    print(f"   训练轮次: {stage_config['epochs']}")

    # 执行该阶段
    current_step = run_curriculum_learning_stage(
        policy, stage_config, dataset, cfg, device, writer,
        current_step, optimizer, lr_scheduler, scaler,
        output_directory, amp_enabled, task_manager, dataloader
    )
```

### 4.4 阶段内部执行

```python
def run_curriculum_learning_stage(...):
    # 1. 激活指定的层
    policy.set_curriculum_stage(enabled_layers)
    #    └─► 更新 policy.enabled_layers

    # 2. Epoch循环
    for epoch in range(stage_epochs):
        for batch in dataloader:
            # 3. 构建课程学习信息
            curriculum_info = {
                'stage': stage_name,
                'enabled_layers': enabled_layers
            }

            # 4. 前向传播（只计算激活层的损失）
            loss, hierarchical_info = policy.forward(batch, curriculum_info)

            # 5. 反向传播
            scaled_loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # 6. 保存最佳模型
        if avg_epoch_loss < best_stage_loss:
            policy.save_pretrained(best_save_path)
```

### 4.5 Policy中的课程学习处理

```python
# 在 HumanoidDiffusionPolicy.py
def _hierarchical_forward(self, batch, curriculum_info, task_weights):
    # 1. 更新课程学习状态
    self._update_curriculum_state(curriculum_info)
    #    └─► 更新 self.enabled_layers

    # 2. 分层处理
    layer_outputs = self.scheduler(batch, task_info)

    # 3. Diffusion损失计算
    diffusion_loss = self.diffusion.compute_loss(batch, layer_outputs)

    # 4. 分层损失聚合（只计算激活层的损失）
    total_loss = diffusion_loss
    for layer_name, layer_output in layer_outputs.items():
        if layer_name in self.enabled_layers:  # 只计算激活层
            layer_weight = self.task_layer_weights.get(layer_name, 1.0)
            layer_loss = layer_output['loss']
            total_loss = total_loss + layer_weight * layer_loss

    return total_loss, hierarchical_info
```

---

## 5. 任务特定训练

### 5.1 TaskSpecificTrainingManager

**位置**: `kuavo_train/wrapper/policy/humanoid/TaskSpecificTrainingManager.py`

#### 5.1.1 核心职责

```python
class TaskSpecificTrainingManager:
    """任务特定训练管理器，管理分层架构在多任务场景下的训练"""
```

1. **任务管理**: 注册和管理可用任务数据
2. **课程学习生成**: 为不同任务生成特定的课程学习阶段
3. **权重调整**: 任务特定的层权重配置
4. **防遗忘**: 数据采样策略和重放机制
5. **状态保存**: 保存和恢复任务训练状态

#### 5.1.2 任务定义

```python
# 预定义任务配置
self.task_definitions = {
    1: TaskInfo(
        task_id=1,
        name="dynamic_grasping",  # 动态抓取
        complexity_level=2,
        required_layers=["safety", "manipulation"],
        primary_capabilities=["object_detection", "trajectory_tracking", "grasp_control"]
    ),
    2: TaskInfo(
        task_id=2,
        name="package_weighing",  # 称重
        complexity_level=3,
        required_layers=["safety", "gait", "manipulation"],
        primary_capabilities=["dual_arm_coordination", "weight_estimation", "balance_control"]
    ),
    3: TaskInfo(
        task_id=3,
        name="precise_placement",  # 摆放
        complexity_level=3,
        required_layers=["safety", "manipulation", "planning"],
        primary_capabilities=["spatial_reasoning", "orientation_control", "precision_placement"]
    ),
    4: TaskInfo(
        task_id=4,
        name="full_process_sorting",  # 分拣
        complexity_level=4,
        required_layers=["safety", "gait", "manipulation", "planning"],
        primary_capabilities=["whole_body_coordination", "sequence_planning", "multi_modal_control"]
    )
}
```

#### 5.1.3 任务特定层权重

```python
self.task_layer_weights = {
    1: {  # 动态抓取：强调操作和安全
        "safety": 2.0,
        "gait": 0.5,        # 低权重
        "manipulation": 2.0,  # 高权重
        "planning": 0.8
    },
    2: {  # 称重：强调步态和平衡
        "safety": 2.0,
        "gait": 1.8,        # 高权重
        "manipulation": 1.5,
        "planning": 1.0
    },
    3: {  # 摆放：强调操作和规划
        "safety": 2.0,
        "gait": 0.8,
        "manipulation": 1.8,  # 高权重
        "planning": 2.0      # 高权重
    },
    4: {  # 分拣：平衡所有层
        "safety": 2.0,
        "gait": 1.5,
        "manipulation": 1.5,
        "planning": 1.5
    }
}
```

#### 5.1.4 任务特定课程学习

```python
self.task_curriculum_stages = {
    1: {  # 动态抓取 - 快速反应导向
        "stage1": {
            "name": "safety_reflex",
            "layers": ["safety"],
            "epochs": 30
        },
        "stage2": {
            "name": "basic_manipulation",
            "layers": ["safety", "manipulation"],
            "epochs": 70
        },
        "stage3": {
            "name": "full_grasping",
            "layers": ["safety", "gait", "manipulation"],
            "epochs": 100
        }
    },
    2: {  # 称重 - 平衡协调导向
        "stage1": {"name": "safety_base", "layers": ["safety"], "epochs": 25},
        "stage2": {"name": "gait_control", "layers": ["safety", "gait"], "epochs": 50},
        "stage3": {"name": "dual_arm_coord", "layers": ["safety", "gait", "manipulation"], "epochs": 75},
        "stage4": {"name": "full_weighing", "layers": ["safety", "gait", "manipulation", "planning"], "epochs": 100}
    },
    # ... 任务3、4的配置
}
```

### 5.2 渐进式多任务训练

```python
def _build_progressive_curriculum(self):
    """构建渐进式多任务课程学习"""
    stages = {}
    stage_counter = 1

    # 按任务复杂度排序
    sorted_tasks = sorted(self.available_tasks,
                         key=lambda x: self.task_definitions[x].complexity_level)

    # 为每个任务创建专门的适应阶段
    for task_id in sorted_tasks:
        task_info = self.task_definitions[task_id]
        task_stages = self.task_curriculum_stages[task_id]

        for stage_name, stage_config in task_stages.items():
            adapted_stage_name = f"stage{stage_counter}_{task_info.name}"
            stages[adapted_stage_name] = {
                **stage_config,
                "name": adapted_stage_name,
                "target_task": task_id,
                "task_weight": 1.0 / len(self.available_tasks),
                "epochs": max(10, stage_config["epochs"] // len(self.available_tasks))
            }
            stage_counter += 1

    # 添加最终融合阶段
    stages[f"stage{stage_counter}_integration"] = {
        "name": "multi_task_integration",
        "layers": ["safety", "gait", "manipulation", "planning"],
        "epochs": 50,
        "target_task": "all",
        "task_weight": "balanced"
    }

    return stages
```

### 5.3 多任务数据采样

```python
def get_task_data_sampling_strategy(self):
    """获取任务数据采样策略"""
    sampling_weights = {}
    total_episodes = sum(self.task_definitions[tid].episode_count
                        for tid in self.available_tasks)

    for task_id in self.available_tasks:
        task_info = self.task_definitions[task_id]

        # 基于复杂度和数据量的权重计算
        complexity_factor = task_info.complexity_level / 4.0
        data_factor = task_info.episode_count / total_episodes

        # 平衡复杂度和数据可用性
        sampling_weights[task_id] = 0.6 * complexity_factor + 0.4 * data_factor

    # 归一化权重
    total_weight = sum(sampling_weights.values())
    sampling_weights = {k: v/total_weight for k, v in sampling_weights.items()}

    return {
        "strategy": "weighted_sampling",
        "task_weights": sampling_weights,
        "anti_forgetting": True,
        "rehearsal_ratio": 0.2  # 20%的数据用于防遗忘
    }
```

### 5.4 创建任务特定DataLoader

```python
def create_task_specific_dataloader(datasets, task_manager, cfg, device):
    """创建任务特定的数据加载器（多任务加权采样）"""

    # 多任务情况 - 使用加权采样
    sampling_strategy = task_manager.get_task_data_sampling_strategy()
    task_weights = sampling_strategy.get("task_weights", {})

    combined_datasets = []
    sample_weights = []

    for task_id, dataset in datasets.items():
        task_weight = task_weights.get(task_id, 1.0 / len(datasets))
        dataset_size = len(dataset)

        combined_datasets.append(dataset)
        sample_weights.extend([task_weight] * dataset_size)

    combined_dataset = ConcatDataset(combined_datasets)

    # 加权随机采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(combined_dataset),
        replacement=True
    )

    return DataLoader(
        combined_dataset,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type != "cpu"),
        drop_last=cfg.training.drop_last
    )
```

---

## 6. 推理逻辑

### 6.1 动作选择流程 (Inference)

```python
# 在 HumanoidDiffusionPolicy.py
def select_action(self, batch):
    """选择动作（推理时使用）"""
    if self.use_hierarchical:
        return self._hierarchical_select_action(batch)
    else:
        return super().select_action(batch)
```

### 6.2 分层推理流程

```python
def _hierarchical_select_action(self, batch):
    """分层架构的动作选择"""

    # 1. 预处理
    batch = self._preprocess_batch(batch)
    batch = self.normalize_inputs(batch)

    # 2. 任务识别
    task_info = self._identify_task(batch)
    #    └─► 根据观测推断任务类型

    # 3. 分层推理
    with torch.no_grad():
        layer_outputs = self.scheduler(batch, task_info)

    # 4. 保存layer_outputs供日志记录
    self._last_layer_outputs = layer_outputs

    # 5. 从分层输出中提取最终动作
    return self._extract_action_from_layers(layer_outputs, batch)
```

### 6.3 动作提取逻辑

```python
def _extract_action_from_layers(self, layer_outputs, batch):
    """从分层输出中提取最终动作"""

    # 1. 优先级处理：安全层可以覆盖其他层的输出
    if 'safety' in layer_outputs:
        is_emergency = layer_outputs['safety'].get('emergency', False)
        if is_emergency:
            # 返回紧急动作
            return layer_outputs['safety'].get('emergency_action', torch.zeros_like(...))

    # 2. 正常情况下，使用最高级别可用层的输出
    for layer_name in ['planning', 'manipulation', 'gait', 'safety']:
        if layer_name in layer_outputs and 'action' in layer_outputs[layer_name]:
            return layer_outputs[layer_name]['action']

    # 3. 回退：使用传统diffusion输出
    return super().select_action(batch)
```

### 6.4 推理模式（Inference Mode）

```python
# 在 HierarchicalScheduler.py
def inference_mode(self, batch, task_info, latency_budget_ms=50.0):
    """根据延迟预算自适应激活层"""

    remaining_budget = latency_budget_ms
    outputs = {}

    # 按优先级顺序处理，在预算内尽可能多地激活层
    for layer_name in self._get_processing_order():
        layer = self.layers[layer_name]

        # 检查时间预算
        layer_budget = layer.get_latency_budget()
        if remaining_budget < layer_budget:
            print(f"⏰ Skipping {layer_name} due to time budget")
            continue

        # 执行层推理
        start_time = time.time()
        layer_output = layer.forward_with_timing(batch, context)
        outputs[layer_name] = layer_output

        # 更新剩余预算
        layer_time = (time.time() - start_time) * 1000
        remaining_budget -= layer_time

        # 安全层紧急情况立即返回
        if layer_name == 'safety' and layer_output.get('emergency'):
            break

    return outputs
```

### 6.5 实时部署示例

```python
# 在 kuavo_deploy/examples/eval/eval_kuavo.py
policy = HumanoidDiffusionPolicy.from_pretrained(
    checkpoint_path,
    use_hierarchical=True,
    hierarchical=hierarchical_config
)

# 推理循环
for step in range(max_steps):
    # 1. 获取观测
    obs = env.get_observation()

    # 2. 添加延迟预算到task_info（如果需要）
    task_info = {
        'latency_budget_ms': 50.0,  # 50ms预算
        'requires_locomotion': True,
        'requires_manipulation': True
    }

    # 3. 选择动作
    action = policy.select_action(obs)

    # 4. 执行动作
    env.step(action)

    # 5. 记录层输出（可选）
    layer_outputs = policy.get_last_layer_outputs()
    if layer_outputs:
        log_layer_performance(layer_outputs)
```

---

## 7. 配置系统

### 7.1 主配置文件

**位置**: `configs/policy/humanoid_diffusion_config.yaml`

### 7.2 关键配置项

#### 7.2.1 基础训练配置

```yaml
training:
  output_directory: 'outputs/train/${task}/${method}'
  seed: 42
  max_epoch: 500
  batch_size: 64
  num_workers: 25
  accumulation_steps: 1

  # 测试模式
  test_training_mode: False
  test_training_epochs: 10
```

#### 7.2.2 分层架构配置

```yaml
policy:
  use_hierarchical: True

  hierarchical:
    # 四层配置
    layers:
      safety:
        type: 'GRU'
        hidden_size: 64
        num_layers: 1
        input_dim: 16
        output_dim: 16
        response_time_ms: 10
        priority: 1

      gait:
        type: 'Hybrid'
        gru_hidden: 128
        tf_layers: 2
        priority: 2

      manipulation:
        type: 'Transformer'
        layers: 3
        hidden_size: 512
        priority: 3

      planning:
        type: 'Transformer'
        layers: 4
        hidden_size: 1024
        priority: 4

    # 层权重
    layer_weights:
      safety: 1.0
      gait: 1.0
      manipulation: 2.0  # 抓取任务高权重
      planning: 0.5

    # 课程学习
    curriculum_learning:
      enable: True
      universal_stages:
        stage1:
          name: 'manipulation_first'
          layers: ['manipulation']
          epochs: 8
        stage2:
          name: 'manipulation_safety'
          layers: ['manipulation', 'safety']
          epochs: 7
        # ...
```

#### 7.2.3 任务特定训练配置

```yaml
task_specific_training:
  enable: True

  data_config:
    base_path: '/robot/data'
    task_directories:
      1: 'task-1/1-2000/lerobot'
      2: 'task-2/lerobot'
      3: 'task-3/lerobot'
      4: 'task-4/lerobot'

  current_phase: 1

  training_strategy:
    progressive_learning: True
    min_epochs_per_phase: 10
    phase_transition_loss_threshold: 0.08

  memory_management:
    max_concurrent_tasks: 2
    max_episodes_per_task: 300
```

### 7.3 配置加载流程

```python
# 在 train_hierarchical_policy.py
@hydra.main(config_path="../configs/policy/",
            config_name="humanoid_diffusion_config")
def main(cfg: DictConfig):
    # 1. Hydra自动加载配置

    # 2. 检查训练模式
    use_task_specific = cfg.get('task_specific_training', {}).get('enable', False)

    # 3. 构建policy配置
    policy_cfg = build_policy_config(cfg, input_features, output_features)

    # 4. 实例化policy
    policy = HumanoidDiffusionPolicy(policy_cfg, dataset_stats)
    #    └─► 内部读取 hierarchical 配置
```

---

## 8. 关键设计决策

### 8.1 为什么不进行特征融合？

**设计决策**: 分层架构的价值在于**课程学习**和**层间协调**，而不是特征融合。

**原因:**
1. **Diffusion模型的独立性**: Diffusion模型本身已经很强大，直接融合层特征可能破坏其内部结构
2. **损失聚合更安全**: 通过加权损失聚合，各层可以独立学习，避免梯度冲突
3. **课程学习的核心**: 渐进式激活层才是关键，而不是特征拼接

**代码体现:**
```python
# 在 HierarchicalDiffusionModel.py
def compute_loss(self, batch, layer_outputs=None):
    """直接使用原始批次计算损失，不融合layer_outputs"""
    return super().compute_loss(batch)
```

### 8.2 为什么安全层始终激活？

**设计决策**: SafetyReflexLayer 在所有模式下始终激活。

**原因:**
1. **安全第一**: 防跌倒和紧急停止是机器人的基本要求
2. **最高优先级**: 可以覆盖其他层的输出
3. **低延迟**: <10ms响应时间，不影响整体性能

**代码体现:**
```python
def should_activate(self, inputs, context=None):
    """安全层始终激活"""
    return True
```

### 8.3 为什么使用混合架构？

**设计决策**: 不同层使用不同的网络架构（GRU、Transformer、混合）。

**原因:**
1. **响应时间需求不同**: 安全层需要极快（GRU），规划层可以慢（大Transformer）
2. **任务性质不同**: 步态需要序列建模（GRU+Transformer），操作需要全局注意力（Transformer）
3. **参数效率**: 根据层的重要性分配不同的模型容量

| 层 | 架构 | 参数量 | 响应时间 |
|---|---|---|---|
| Safety | 1层GRU | ~10K | <10ms |
| Gait | 2层GRU + 2层Transformer | ~200K | ~20ms |
| Manipulation | 3层Transformer | ~2M | ~100ms |
| Planning | 4层Transformer | ~10M | ~500ms |

### 8.4 为什么需要任务特定训练？

**设计决策**: 支持多任务场景，每个任务有特定的层权重和课程学习。

**原因:**
1. **任务多样性**: 动态抓取、称重、摆放、分拣需要不同的能力组合
2. **防止灾难性遗忘**: 通过加权采样和重放机制
3. **渐进式学习**: 从简单任务到复杂任务，逐步扩展能力

**效果:**
- 任务1（抓取）: 强化 manipulation 层
- 任务2（称重）: 强化 gait 和 manipulation 层
- 任务3（摆放）: 强化 manipulation 和 planning 层
- 任务4（分拣）: 平衡所有层

### 8.5 为什么使用加权损失聚合？

**设计决策**: 总损失 = Diffusion损失 + Σ(层权重 × 层损失)

**原因:**
1. **灵活性**: 可以动态调整各层的训练强度
2. **课程学习**: 只计算激活层的损失
3. **任务特定**: 不同任务使用不同的层权重

**代码体现:**
```python
def _aggregate_hierarchical_loss(self, diffusion_loss, layer_outputs, use_task_weights=False):
    total_loss = diffusion_loss

    for layer_name, layer_output in layer_outputs.items():
        if layer_name in self.enabled_layers:  # 只计算激活层
            layer_weight = self.task_layer_weights.get(layer_name, 1.0)
            layer_loss = layer_output['loss']
            total_loss = total_loss + layer_weight * layer_loss

    return total_loss
```

---

## 附录

### A. 文件结构

```
kuavo_data_challenge/
├── kuavo_train/
│   ├── train_hierarchical_policy.py  # 训练主入口
│   └── wrapper/
│       └── policy/
│           ├── diffusion/
│           │   ├── DiffusionPolicyWrapper.py
│           │   ├── DiffusionConfigWrapper.py
│           │   └── DiffusionModelWrapper.py
│           └── humanoid/
│               ├── HumanoidDiffusionPolicy.py  # 主策略
│               ├── HierarchicalScheduler.py    # 层调度器
│               ├── HierarchicalDiffusionModel.py
│               ├── TaskSpecificTrainingManager.py
│               └── layers/
│                   ├── BaseLayer.py
│                   ├── SafetyReflexLayer.py
│                   ├── GaitControlLayer.py
│                   ├── ManipulationLayer.py
│                   └── GlobalPlanningLayer.py
├── configs/
│   └── policy/
│       └── humanoid_diffusion_config.yaml
└── kuavo_deploy/
    └── examples/
        └── eval/
            └── eval_kuavo.py  # 推理示例
```

### B. 训练命令

```bash
# 基础训练（单任务，课程学习）
python kuavo_train/train_hierarchical_policy.py \
  --config-name=humanoid_diffusion_config

# 任务特定训练（需要在配置中设置 task_specific_training.enable=True）
python kuavo_train/train_hierarchical_policy.py \
  --config-name=humanoid_diffusion_config

# 测试模式（快速验证）
python kuavo_train/train_hierarchical_policy.py \
  --config-name=humanoid_diffusion_config \
  training.test_training_mode=True \
  training.test_training_epochs=2
```

### C. 推理命令

```bash
# 使用分层架构推理
python kuavo_deploy/examples/eval/eval_kuavo.py \
  --checkpoint path/to/checkpoint \
  --use-hierarchical \
  --latency-budget 50  # 50ms延迟预算
```

### D. 性能指标

| 指标 | 值 | 说明 |
|---|---|---|
| 总参数量 | ~12M | 包含所有层 |
| 训练batch size | 64 | 可根据GPU调整 |
| 训练epochs | 500 | 课程学习后继续训练 |
| 课程学习阶段数 | 4 | 可配置 |
| 推理延迟（全层） | ~600ms | 所有层激活 |
| 推理延迟（核心层） | ~130ms | safety+gait+manipulation |
| 推理延迟（最小） | <10ms | 仅safety层 |

---

## 联系与支持

如有问题，请查阅：
- 项目README: `/Users/HarowrdLiu/learn/robot/kuavo_data_challenge/README.md`
- 代码注释: 各模块均有详细注释
- 训练日志: `hierarchical_training.log`

---

**文档版本**: 1.0
**最后更新**: 2025-10-10
**作者**: AI Assistant

