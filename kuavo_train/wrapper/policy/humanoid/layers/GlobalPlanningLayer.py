"""
GlobalPlanningLayer: 全局规划层 - 最复杂的长期规划
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from .BaseLayer import BaseLayer


class GlobalPlanningLayer(BaseLayer):
    """
    全局规划层 - 优先级4（最低，最复杂）

    特点：
    - 大型Transformer架构
    - 长期记忆和复杂任务规划
    - 任务分解和全局优化
    - 约500ms响应时间
    """

    def __init__(self, config: Dict[str, Any], base_config: Any):
        super().__init__(config, "planning", priority=4)

        self.base_config = base_config

        # 配置参数
        self.hidden_size = config.get('hidden_size', 1024)
        self.num_layers = config.get('layers', 4)
        self.num_heads = config.get('heads', 16)
        self.dim_feedforward = config.get('dim_feedforward', 4096)

        # 特征维度计算 - 适配实际机器人配置
        visual_dim = 1280  # EfficientNet-B0输出
        state_shape = getattr(base_config, 'robot_state_feature', None)
        if state_shape and hasattr(state_shape, 'shape'):
            self.state_dim = state_shape.shape[0]
        else:
            # 默认配置：only_arm=true时的双臂+手爪配置
            self.state_dim = 16
        self.input_projection = nn.Linear(visual_dim + self.state_dim, self.hidden_size)

        # 大型Transformer用于复杂推理
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.global_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # 长期记忆模块
        self.memory_bank = LongTermMemoryModule(self.hidden_size)

        # 任务分解模块
        self.task_decomposer = TaskDecompositionModule(self.hidden_size)

        # 输出投影 - 动作维度应该与状态维度一致
        self.action_head = nn.Linear(self.hidden_size, self.state_dim)
        self.plan_head = nn.Linear(self.hidden_size, 64)  # 规划输出

    def should_activate(self, inputs: Dict[str, torch.Tensor], context: Optional[Dict[str, Any]] = None) -> bool:
        """当需要复杂规划时激活"""
        if context is None:
            return False  # 默认不激活最复杂的层

        task_complexity = context.get('task_complexity', 'medium')
        return task_complexity in ['high', 'very_high']

    def get_required_input_keys(self) -> List[str]:
        return ['observation.state']

    def forward(self, inputs: Dict[str, torch.Tensor], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """全局规划前向传播"""
        # 提取全局状态
        global_state = self._encode_global_state(inputs, context)
        if global_state is None:
            return self._generate_default_output(inputs)

        batch_size, seq_len, _ = global_state.shape

        # 记忆检索
        relevant_memory = self.memory_bank.retrieve(global_state)

        # 全局推理
        enhanced_state = torch.cat([global_state, relevant_memory], dim=-1)
        planning_output = self.global_transformer(enhanced_state)

        # 任务分解
        task_plan = self.task_decomposer(planning_output, context)

        # 输出动作和规划
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

    def _encode_global_state(self, inputs: Dict[str, torch.Tensor], context: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
        """编码全局状态"""
        features_list = []

        # 状态特征
        if 'observation.state' in inputs:
            state_features = inputs['observation.state']
            features_list.append(state_features)

        # 视觉特征
        if 'observation.images' in inputs:
            visual_features = inputs['observation.images']
            if len(visual_features.shape) > 3:
                visual_features = visual_features.mean(dim=(-2, -1))
            features_list.append(visual_features)

        if not features_list:
            return None

        # 特征融合
        combined_features = torch.cat(features_list, dim=-1)
        global_state = self.input_projection(combined_features)

        return global_state

    def _generate_default_output(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """生成默认输出"""
        batch_size = list(inputs.values())[0].size(0)
        device = list(inputs.values())[0].device

        zero_features = torch.zeros(batch_size, 10, self.hidden_size, device=device)
        zero_action = torch.zeros(batch_size, self.state_dim, device=device)
        zero_plan = torch.zeros(batch_size, 64, device=device)

        return {
            'global_features': zero_features,
            'task_plan': {},
            'relevant_memory': zero_features,
            'global_plan': zero_plan,
            'action': zero_action,
            'layer': 'planning'
        }


class LongTermMemoryModule(nn.Module):
    """长期记忆模块"""

    def __init__(self, feature_dim: int, memory_size: int = 1000):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size

        # 简化的记忆存储
        self.register_buffer('memory_bank', torch.zeros(memory_size, feature_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))

        # 记忆检索网络
        self.retrieval_net = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """检索相关记忆"""
        batch_size, seq_len, _ = query.shape

        # 使用注意力机制检索记忆
        memory_expanded = self.memory_bank.unsqueeze(0).expand(batch_size, -1, -1)
        retrieved_memory, _ = self.retrieval_net(query, memory_expanded, memory_expanded)

        return retrieved_memory

    def store(self, memory: torch.Tensor):
        """存储新记忆"""
        batch_size, seq_len, _ = memory.shape
        memory_flat = memory.view(-1, self.feature_dim)

        for i in range(memory_flat.size(0)):
            ptr = self.memory_ptr.item()
            self.memory_bank[ptr] = memory_flat[i]
            self.memory_ptr[0] = (ptr + 1) % self.memory_size


class TaskDecompositionModule(nn.Module):
    """任务分解模块"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.decomposer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 10)  # 假设最多10个子任务
        )

    def forward(self, features: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """任务分解处理"""
        task_scores = self.decomposer(features[:, -1, :])
        task_priorities = torch.softmax(task_scores, dim=-1)

        return {
            'task_scores': task_scores,
            'task_priorities': task_priorities,
            'num_subtasks': torch.sum(task_scores > 0.1, dim=-1)
        }