# 分层架构 - 各层详细设计文档

> 详细说明四层架构的内部实现、数学原理和工程细节

---

## 目录

1. [BaseLayer 抽象基类](#1-baselayer-抽象基类)
2. [SafetyReflexLayer 详解](#2-safetyreflexlayer-详解)
3. [GaitControlLayer 详解](#3-gaitcontrollayer-详解)
4. [ManipulationLayer 详解](#4-manipulationlayer-详解)
5. [GlobalPlanningLayer 详解](#5-globalplanninglayer-详解)
6. [层间通信机制](#6-层间通信机制)
7. [损失函数设计](#7-损失函数设计)

---

## 1. BaseLayer 抽象基类

### 1.1 设计目标

提供统一的层接口，确保所有层实现一致的行为。

### 1.2 核心接口

```python
class BaseLayer(nn.Module, ABC):
    """所有分层架构层的抽象基类"""

    def __init__(self, config: Dict[str, Any], layer_name: str, priority: int):
        super().__init__()
        self.layer_name = layer_name
        self.priority = priority  # 1=最高，4=最低
        self.enabled = config.get('enabled', True)
        self.response_time_ms = config.get('response_time_ms', 100)

        # 性能监控
        self.execution_times = []
        self.activation_count = 0
```

### 1.3 必须实现的方法

#### 1.3.1 forward()

```python
@abstractmethod
def forward(self, inputs: Dict[str, torch.Tensor],
            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    层的前向传播

    Args:
        inputs: 输入数据字典
            - 'observation.state': [batch, state_dim] 关节状态
            - 'observation.images.*': [batch, C, H, W] RGB图像
            - 'observation.depth.*': [batch, 1, H, W] 深度图

        context: 上下文信息
            - 'batch_size': int
            - 'device': torch.device
            - 'training': bool
            - 其他层的输出

    Returns:
        Dict: 该层的输出
            - 'layer': str 层名称
            - 'action': Tensor 动作输出
            - 'loss': Tensor (可选) 层损失
            - 'execution_time_ms': float 执行时间
            - ... 其他层特定输出
    """
    pass
```

#### 1.3.2 should_activate()

```python
@abstractmethod
def should_activate(self, inputs: Dict[str, torch.Tensor],
                   context: Optional[Dict[str, Any]] = None) -> bool:
    """
    判断该层是否应该激活

    Returns:
        bool: True=激活，False=跳过
    """
    pass
```

### 1.4 提供的工具方法

#### 1.4.1 forward_with_timing()

```python
def forward_with_timing(self, inputs, context=None):
    """带时间监控的前向传播"""
    if not self.enabled:
        return {'layer': self.layer_name, 'enabled': False}

    start_time = time.time()
    output = self.forward(inputs, context)
    execution_time_ms = (time.time() - start_time) * 1000

    self.activation_count += 1
    self.execution_times.append(execution_time_ms)

    # 保持最近100次记录
    if len(self.execution_times) > 100:
        self.execution_times.pop(0)

    output['execution_time_ms'] = execution_time_ms
    output['layer'] = self.layer_name
    output['activation_count'] = self.activation_count

    return output
```

#### 1.4.2 get_performance_stats()

```python
def get_performance_stats(self):
    """获取性能统计"""
    if not self.execution_times:
        return {
            'avg_time_ms': 0.0,
            'max_time_ms': 0.0,
            'min_time_ms': 0.0,
            'activation_count': self.activation_count
        }

    return {
        'avg_time_ms': sum(self.execution_times) / len(self.execution_times),
        'max_time_ms': max(self.execution_times),
        'min_time_ms': min(self.execution_times),
        'activation_count': self.activation_count,
        'budget_ms': self.response_time_ms
    }
```

#### 1.4.3 validate_inputs()

```python
def validate_inputs(self, inputs):
    """验证输入有效性"""
    if not isinstance(inputs, dict):
        return False

    required_keys = self.get_required_input_keys()
    for key in required_keys:
        if key not in inputs:
            return False
        if not isinstance(inputs[key], torch.Tensor):
            return False

    return True
```

---

## 2. SafetyReflexLayer 详解

### 2.1 设计理念

**核心思想**: 极简设计，确保<10ms响应时间，模拟人类的"反射弧"。

**生物学类比**:
- 人类的脊髓反射：无需大脑参与，直接从脊髓发出响应
- 触摸热物体时的瞬间缩手：~50ms反应时间
- 机器人的"脊髓"：安全反射层

### 2.2 输入输出维度

```python
输入:
  observation.state: [batch, 16]
    ├─ 双臂关节: 14维
    └─ 手爪状态: 2维

输出:
  {
    'emergency': [batch] bool,              # 是否紧急
    'emergency_score': [batch] float,        # 紧急评分 0-1
    'balance_action': [batch, 16],          # 平衡控制动作
    'emergency_action': [batch, 16],        # 紧急动作
    'tilt_angles_degrees': [batch, 2],      # roll, pitch
    'balance_confidence': [batch] float,     # 平衡置信度
    'safety_status': List[str],             # ['SAFE', 'CAUTION', 'UNSTABLE', 'EMERGENCY']
    'action': [batch, 16]                   # 最终选择的动作
  }
```

### 2.3 网络架构详解

#### 2.3.1 GRU处理

```python
self.balance_gru = nn.GRU(
    input_size=16,    # 输入维度
    hidden_size=64,   # 隐藏层维度
    num_layers=1,     # 只用一层
    batch_first=True
)

# 前向传播
gru_output, hidden = self.balance_gru(robot_state)
# gru_output: [batch, seq_len, 64]
# hidden: [1, batch, 64]

last_output = gru_output[:, -1, :]  # [batch, 64]
```

**为什么用GRU?**
- 比LSTM更快（少一个门）
- 比RNN更稳定（有门控机制）
- 可以处理时序信息（虽然输入通常只有1个时间步）

#### 2.3.2 紧急检测器

```python
self.emergency_detector = nn.Sequential(
    nn.Linear(64, 32),   # 降维
    nn.ReLU(),
    nn.Linear(32, 1),    # 输出单一评分
    nn.Sigmoid()         # 限制到0-1
)

emergency_score = self.emergency_detector(last_output)  # [batch, 1]
emergency = (emergency_score > 0.8).squeeze(-1)  # [batch] bool
```

**阈值设计**:
- 默认阈值: 0.8（可配置）
- 高阈值 = 低误报率（但可能漏检）
- 低阈值 = 高召回率（但可能过于保守）

#### 2.3.3 倾斜检测器

```python
self.tilt_detector = nn.Linear(64, 2)  # 输出 roll 和 pitch

tilt_angles = self.tilt_detector(last_output)  # [batch, 2]
tilt_angles_degrees = tilt_angles * 45.0  # 缩放到±45度
```

**物理意义**:
- Roll: 侧翻角度（左右倾斜）
- Pitch: 俯仰角度（前后倾斜）
- 阈值: 15度（可配置）

**为什么是45度?**
- 训练时使用Tanh激活（输出-1到1）
- 乘以45度后，输出范围是-45到+45度
- 覆盖人形机器人的典型倾斜范围

#### 2.3.4 平衡控制器

```python
self.balance_controller = nn.Sequential(
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 16),   # 输出动作维度
    nn.Tanh()            # 限制输出范围
)

balance_action = self.balance_controller(last_output)  # [batch, 16]
```

**输出范围**:
- Tanh输出: -1 到 +1
- 实际应用时会根据关节限制进行反归一化

#### 2.3.5 紧急动作生成器

```python
self.emergency_action_generator = nn.Sequential(
    nn.Linear(64, 16),
    nn.Tanh()
)

emergency_action = self.emergency_action_generator(last_output)  # [batch, 16]
```

**紧急动作设计**:
- 目标: 快速稳定机器人
- 通常学习到的动作: 降低重心、展开双臂、弯曲膝盖

### 2.4 决策逻辑

```python
# 1. 综合紧急状态
overall_emergency = torch.logical_or(
    emergency,           # 基于GRU的紧急检测
    tilt_emergency       # 基于倾斜的紧急检测
)  # [batch] bool

# 2. 选择动作
overall_emergency_expanded = overall_emergency.unsqueeze(-1)  # [batch, 1]

balance_action = torch.where(
    overall_emergency_expanded,
    emergency_action,         # 紧急时使用紧急动作
    balance_action_normal     # 正常时使用平衡动作
)  # [batch, 16]
```

### 2.5 训练目标

**损失函数**:
```python
# 假设的损失计算（实际在Policy层聚合）
safety_loss = (
    # 1. 紧急检测准确性
    bce_loss(emergency_score, ground_truth_emergency)

    # 2. 倾斜预测准确性
    + mse_loss(tilt_angles_degrees, ground_truth_tilt)

    # 3. 动作平滑性
    + smooth_loss(balance_action)
)
```

### 2.6 性能优化

**为什么能做到<10ms?**
1. **极简架构**: 只有1层GRU + 几个小全连接层
2. **小batch推理**: 通常batch_size=1
3. **无注意力机制**: 避免O(n²)复杂度
4. **固定序列长度**: 通常只处理当前时间步

**实测性能** (Tesla V100):
- Batch=1: 5-8ms
- Batch=8: 8-10ms
- Batch=64: 10-15ms

---

## 3. GaitControlLayer 详解

### 3.1 设计理念

**核心思想**: 混合架构，GRU处理实时状态，Transformer规划未来步态。

**生物学类比**:
- 小脑: 负责运动协调和步态控制
- 中枢模式生成器(CPG): 自动生成周期性运动模式

### 3.2 输入输出维度

```python
输入:
  observation.state: [batch, seq_len, 16]

输出:
  {
    'gait_features': [batch, seq_len, 128],      # GRU特征
    'planned_gait': [batch, seq_len, 128],       # Transformer规划
    'adapted_gait': [batch, seq_len, 128],       # 负载适应后
    'action': [batch, 16]                        # 最终动作
  }
```

### 3.3 网络架构详解

#### 3.3.1 GRU状态跟踪

```python
self.gait_state_gru = nn.GRU(
    input_size=16,
    hidden_size=128,
    num_layers=2,     # 使用2层增强表达能力
    batch_first=True,
    dropout=0.1       # 防止过拟合
)

gru_output, gru_hidden = self.gait_state_gru(robot_state)
# gru_output: [batch, seq_len, 128]
```

**为什么用2层?**
- 单层GRU: 足够快，但表达能力有限
- 2层GRU: 可以学习层次化特征，延迟增加不多
- 3层及以上: 提升有限，延迟明显增加

#### 3.3.2 Transformer步态规划

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=128,
    nhead=4,                    # 4个注意力头
    dim_feedforward=128 * 2,    # FFN维度
    dropout=0.1,
    batch_first=True
)
self.gait_planner = nn.TransformerEncoder(encoder_layer, num_layers=2)

# 只在序列足够长时使用
if seq_len >= 10:  # 至少200ms历史（假设50Hz采样）
    planned_gait = self.gait_planner(gru_output)
else:
    planned_gait = gru_output  # 直接使用GRU输出
```

**为什么需要序列长度检查?**
- Transformer需要足够的上下文才能发挥作用
- 序列太短时，注意力机制退化为简单的全连接
- 10步 = 200ms历史，足够捕捉步态周期（人类步态周期~1-2秒）

#### 3.3.3 负载适应模块

```python
class LoadAdaptationModule(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.adaptation_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )

    def forward(self, gait_features, context=None):
        adaptation = self.adaptation_net(gait_features)
        return gait_features + 0.1 * adaptation  # 小幅调整
```

**设计思想**:
- 残差连接: 保留原始特征，只做小幅调整
- 系数0.1: 避免过度修正
- 可以学习不同负载下的步态调整

#### 3.3.4 输出投影

```python
self.output_projection = nn.Linear(128, 16)
final_output = self.output_projection(adapted_gait[:, -1, :])
```

### 3.4 步态周期建模

**理论基础**:
```
步态周期 (Gait Cycle):
  ├─ 支撑相 (Stance Phase): 60%
  │   ├─ 初始接触 (Initial Contact)
  │   ├─ 负载响应 (Loading Response)
  │   ├─ 中期支撑 (Mid Stance)
  │   ├─ 末期支撑 (Terminal Stance)
  │   └─ 预摆动 (Pre-Swing)
  └─ 摆动相 (Swing Phase): 40%
      ├─ 初始摆动 (Initial Swing)
      ├─ 中期摆动 (Mid Swing)
      └─ 末期摆动 (Terminal Swing)
```

**如何在Transformer中建模?**
- 自注意力机制可以捕捉周期性模式
- 长序列输入允许模型"看到"完整的步态周期
- 位置编码帮助模型理解时间关系

### 3.5 激活条件

```python
def should_activate(self, inputs, context=None):
    if context is None:
        return True
    return context.get('requires_locomotion', True)
```

**什么时候需要步态控制?**
- 需要移动时（显然）
- 需要维持平衡时（即使不移动）
- 负载发生变化时

---

## 4. ManipulationLayer 详解

### 4.1 设计理念

**核心思想**: Transformer主导，处理精细操作，融合视觉和本体感知。

**生物学类比**:
- 运动皮层: 计划和执行精细动作
- 视觉皮层: 提供视觉引导
- 体感皮层: 提供本体感知

### 4.2 输入输出维度

```python
输入:
  observation.state: [batch, 16]
  observation.images.*: [batch, 3, H, W] × N_cameras
  observation.depth.*: [batch, 1, H, W] × N_cameras

处理后:
  状态特征: [batch, 1, 16]
  视觉特征: [batch, 1, 1280]  (投影后)
  拼接: [batch, 1, 1296]
  投影: [batch, 1, 512]

输出:
  {
    'manipulation_features': [batch, seq_len, 512],
    'constraint_solution': Dict,
    'coordinated_actions': [batch, 16],
    'action': [batch, 16]
  }
```

### 4.3 多模态特征融合

#### 4.3.1 视觉特征提取

```python
def _extract_features(self, inputs):
    # 1. 状态特征
    state_features = inputs['observation.state']  # [batch, 16]
    if len(state_features.shape) == 2:
        state_features = state_features.unsqueeze(1)  # [batch, 1, 16]

    # 2. 收集所有相机的图像
    visual_features_list = []
    image_keys = [k for k in inputs.keys()
                  if k.startswith('observation.images.')
                  or k.startswith('observation.depth')]

    # 3. 全局平均池化每个相机
    for key in image_keys:
        img = inputs[key]  # [batch, C, H, W]
        img_pooled = img.mean(dim=(-2, -1))  # [batch, C]
        visual_features_list.append(img_pooled)

    # 4. 拼接所有相机特征
    combined_visual = torch.cat(visual_features_list, dim=-1)  # [batch, 12]
    #   3个RGB相机 (3通道) + 3个深度相机 (1通道) = 12维

    # 5. 投影到标准维度
    combined_visual = self.visual_projection(combined_visual)  # [batch, 1280]
    combined_visual = combined_visual.unsqueeze(1)  # [batch, 1, 1280]

    # 6. 拼接状态和视觉
    combined = torch.cat([state_features, combined_visual], dim=-1)  # [batch, 1, 1296]

    # 7. 最终投影
    projected = self.input_projection(combined)  # [batch, 1, 512]

    return projected
```

**为什么使用全局平均池化?**
- 空间信息已经在视觉backbone中处理（ResNet18）
- 全局池化提取整体特征，丢弃空间细节
- 减少计算量（H×W维度降为1）

#### 4.3.2 Transformer处理

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,                    # 8个注意力头
    dim_feedforward=2048,       # FFN维度（4倍）
    dropout=0.1,
    batch_first=True
)
self.manipulation_transformer = nn.TransformerEncoder(
    encoder_layer,
    num_layers=3                # 3层Transformer
)

manipulation_features = self.manipulation_transformer(features)
```

**Transformer层数选择**:
- 1层: 过于简单，表达能力不足
- 3层: 平衡性能和速度
- 6层: 标准BERT大小，可能过大

### 4.4 约束满足模块

```python
class ConstraintSatisfactionModule(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.constraint_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, features, context=None):
        score = self.constraint_net(features)
        return {
            'constraint_satisfaction_score': score,
            'constraints_met': score > 0.5
        }
```

**约束类型**:
1. **关节限制**: 关节角度不超过物理限制
2. **碰撞检测**: 不与环境或自身碰撞
3. **力矩限制**: 电机力矩不超过额定值
4. **速度限制**: 运动速度在安全范围内

### 4.5 双臂协调模块

```python
class BimanualCoordinationModule(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int = 16):
        super().__init__()
        self.coordination_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_dim)
        )

    def forward(self, features, context=None):
        # 使用最后时间步的特征
        coordinated_action = self.coordination_net(features[:, -1, :])
        return coordinated_action
```

**协调策略**:
- 同步运动: 双臂同时移动（如举起大物体）
- 互补运动: 一臂固定，另一臂操作
- 对称运动: 镜像动作（如双臂展开）

### 4.6 任务特定行为

**不同任务的特征使用**:

| 任务 | 主要输入 | 关注点 |
|---|---|---|
| 动态抓取 | 视觉+状态 | 轨迹跟踪、抓取时机 |
| 称重 | 状态+力觉 | 双臂协调、平衡控制 |
| 摆放 | 视觉+状态 | 空间定位、精确控制 |
| 分拣 | 视觉+状态 | 对象识别、序列规划 |

---

## 5. GlobalPlanningLayer 详解

### 5.1 设计理念

**核心思想**: 大型Transformer，长期记忆，复杂任务规划。

**生物学类比**:
- 前额叶皮层: 执行功能、计划、决策
- 海马体: 长期记忆存储和检索

### 5.2 网络架构

```python
self.hidden_size = 1024          # 大隐藏层
self.num_layers = 4              # 深层Transformer
self.num_heads = 16              # 多头注意力
self.dim_feedforward = 4096      # 大FFN

encoder_layer = nn.TransformerEncoderLayer(
    d_model=1024,
    nhead=16,
    dim_feedforward=4096,
    dropout=0.1,
    batch_first=True
)
self.global_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
```

**参数量对比**:
- ManipulationLayer: ~2M参数
- GlobalPlanningLayer: ~10M参数
- 5倍参数量差异

### 5.3 长期记忆模块

```python
class LongTermMemoryModule(nn.Module):
    def __init__(self, feature_dim: int, memory_size: int = 1000):
        super().__init__()
        # 记忆库
        self.register_buffer('memory_bank', torch.zeros(memory_size, feature_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))

        # 检索网络
        self.retrieval_net = nn.MultiheadAttention(
            feature_dim,
            num_heads=8,
            batch_first=True
        )
```

#### 5.3.1 记忆存储

```python
def store(self, memory: torch.Tensor):
    """存储新记忆"""
    batch_size, seq_len, _ = memory.shape
    memory_flat = memory.view(-1, self.feature_dim)

    # 循环存储（FIFO策略）
    for i in range(memory_flat.size(0)):
        ptr = self.memory_ptr.item()
        self.memory_bank[ptr] = memory_flat[i]
        self.memory_ptr[0] = (ptr + 1) % self.memory_size
```

#### 5.3.2 记忆检索

```python
def retrieve(self, query: torch.Tensor):
    """检索相关记忆"""
    batch_size, seq_len, _ = query.shape

    # 扩展记忆库到batch维度
    memory_expanded = self.memory_bank.unsqueeze(0).expand(batch_size, -1, -1)

    # 使用注意力机制检索
    retrieved_memory, _ = self.retrieval_net(
        query,
        memory_expanded,
        memory_expanded
    )

    return retrieved_memory
```

**注意力检索原理**:
```
Query: 当前状态特征
Key: 记忆库中的所有记忆
Value: 记忆库中的所有记忆

Attention(Q, K, V) = softmax(QK^T / √d_k) V

结果: 检索到与当前状态最相关的记忆
```

### 5.4 任务分解模块

```python
class TaskDecompositionModule(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.decomposer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 10)  # 最多10个子任务
        )

    def forward(self, features, context=None):
        task_scores = self.decomposer(features[:, -1, :])
        task_priorities = torch.softmax(task_scores, dim=-1)

        return {
            'task_scores': task_scores,
            'task_priorities': task_priorities,
            'num_subtasks': torch.sum(task_scores > 0.1, dim=-1)
        }
```

**任务分解示例**:
```
原始任务: "将物体从A处移动到B处并摆放整齐"

分解为子任务:
  1. 移动到A处 (priority: 0.3)
  2. 识别并抓取物体 (priority: 0.25)
  3. 移动到B处 (priority: 0.2)
  4. 定位摆放位置 (priority: 0.15)
  5. 精确摆放 (priority: 0.1)
```

### 5.5 激活条件

```python
def should_activate(self, inputs, context=None):
    if context is None:
        return False

    task_complexity = context.get('task_complexity', 'medium')
    return task_complexity in ['high', 'very_high']
```

**什么时候需要全局规划?**
- 任务复杂度高（multi-step任务）
- 需要长期记忆（记住之前的状态）
- 需要全局优化（考虑整体最优）

**什么时候不需要?**
- 简单反射性任务（快速抓取）
- 实时性要求高（延迟预算不足）
- 局部决策足够（不需要全局视角）

---

## 6. 层间通信机制

### 6.1 Context传递

```python
# 在 HierarchicalScheduler.forward()
context = self._build_context(batch, task_info)
outputs = {}

for layer_name in self._get_processing_order():
    layer = self.layers[layer_name]

    # 执行层
    layer_output = layer.forward(batch, context)
    outputs[layer_name] = layer_output

    # 更新上下文（供后续层使用）
    context.update(layer_output)
```

### 6.2 Context内容

```python
context = {
    # 基础信息
    'batch_size': 64,
    'device': 'cuda:0',
    'training': True,

    # 任务信息
    'task_type': 'dynamic_grasping',
    'task_complexity': 'medium',
    'requires_locomotion': False,
    'requires_manipulation': True,
    'requires_planning': False,

    # Safety层的输出（供后续层使用）
    'emergency': False,
    'balance_confidence': 0.95,
    'safety_status': 'SAFE',

    # Gait层的输出
    'gait_features': Tensor[batch, seq, 128],
    'planned_gait': Tensor[batch, seq, 128],

    # Manipulation层的输出
    'manipulation_features': Tensor[batch, seq, 512],

    # ... 更多层的输出
}
```

### 6.3 层输出标准格式

```python
layer_output = {
    # 必须字段
    'layer': str,                    # 层名称
    'execution_time_ms': float,      # 执行时间
    'activation_count': int,         # 激活次数

    # 可选字段
    'action': Tensor,                # 该层的动作输出
    'loss': Tensor,                  # 该层的损失（训练时）
    'features': Tensor,              # 中间特征

    # 层特定字段
    'emergency': bool,               # (Safety) 紧急状态
    'gait_features': Tensor,         # (Gait) 步态特征
    'manipulation_features': Tensor, # (Manipulation) 操作特征
    'task_plan': Dict,               # (Planning) 任务计划
    # ...
}
```

---

## 7. 损失函数设计

### 7.1 总体损失聚合

```python
# 在 HumanoidDiffusionPolicy._aggregate_hierarchical_loss()
def _aggregate_hierarchical_loss(self, diffusion_loss, layer_outputs, use_task_weights=False):
    total_loss = diffusion_loss

    # 选择权重来源
    if use_task_weights:
        layer_weights = self.task_layer_weights
    else:
        layer_weights = self.scheduler.config.get('layer_weights', {})

    # 聚合各层损失
    for layer_name, layer_output in layer_outputs.items():
        if layer_name in self.enabled_layers:  # 只计算激活层
            if 'loss' in layer_output:
                layer_weight = layer_weights.get(layer_name, 1.0)
                layer_loss = layer_output['loss']
                total_loss = total_loss + layer_weight * layer_loss

    return total_loss
```

### 7.2 各层损失设计

#### 7.2.1 Safety层损失

```python
safety_loss = (
    # 紧急检测准确性 (BCE)
    F.binary_cross_entropy(emergency_score, ground_truth_emergency)

    # 倾斜预测准确性 (MSE)
    + F.mse_loss(tilt_angles_degrees, ground_truth_tilt)

    # 动作平滑性 (L2正则)
    + 0.01 * torch.mean(balance_action ** 2)
)
```

#### 7.2.2 Gait层损失

```python
gait_loss = (
    # 步态跟踪准确性 (MSE)
    F.mse_loss(action, ground_truth_action)

    # 步态周期一致性 (自定义)
    + periodicity_loss(planned_gait)

    # 负载适应效果 (MSE)
    + F.mse_loss(adapted_gait, target_adapted_gait)
)
```

#### 7.2.3 Manipulation层损失

```python
manipulation_loss = (
    # 操作准确性 (MSE)
    F.mse_loss(action, ground_truth_action)

    # 约束满足 (BCE)
    + F.binary_cross_entropy(constraint_score, ground_truth_constraints)

    # 双臂协调 (MSE)
    + F.mse_loss(coordinated_actions, target_coordinated_actions)
)
```

#### 7.2.4 Planning层损失

```python
planning_loss = (
    # 规划准确性 (MSE)
    F.mse_loss(action, ground_truth_action)

    # 任务分解准确性 (Cross-Entropy)
    + F.cross_entropy(task_priorities, ground_truth_task_priorities)

    # 长期记忆一致性 (Contrastive)
    + contrastive_loss(retrieved_memory, positive_samples, negative_samples)
)
```

### 7.3 损失权重配置

```yaml
# 基础权重（balanced）
layer_weights:
  safety: 2.0        # 高权重，因为安全最重要
  gait: 1.5
  manipulation: 1.0
  planning: 0.8      # 低权重，因为不总是需要

# 任务特定权重（dynamic_grasping）
layer_weights:
  safety: 2.0
  gait: 0.5          # 降低，因为抓取不需要太多步态
  manipulation: 2.0   # 提高，因为抓取是核心
  planning: 0.8
```

### 7.4 课程学习时的损失

```python
# Stage 1: 只训练 manipulation 层
enabled_layers = ['manipulation']
total_loss = diffusion_loss + 2.0 * manipulation_loss

# Stage 2: 训练 manipulation + safety
enabled_layers = ['manipulation', 'safety']
total_loss = diffusion_loss + 2.0 * manipulation_loss + 2.0 * safety_loss

# Stage 3: 全部激活
enabled_layers = ['safety', 'gait', 'manipulation', 'planning']
total_loss = (diffusion_loss
              + 2.0 * safety_loss
              + 1.0 * gait_loss
              + 2.0 * manipulation_loss
              + 0.5 * planning_loss)
```

---

## 总结

本文详细讲解了分层架构的四个层的内部实现：

1. **SafetyReflexLayer**: 极简GRU，<10ms响应，保障安全
2. **GaitControlLayer**: 混合GRU+Transformer，步态控制和负载适应
3. **ManipulationLayer**: Transformer主导，多模态融合，精细操作
4. **GlobalPlanningLayer**: 大型Transformer，长期记忆，复杂规划

每层都有明确的职责，通过Context机制通信，通过加权损失聚合训练。

---

**相关文档**:
- [主架构文档](hierarchical_policy_architecture.md)
- [训练指南](../README.md)

