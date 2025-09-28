"""
BaseLayer: Abstract base class for all hierarchical layers
"""
import torch
import torch.nn as nn
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List


class BaseLayer(nn.Module, ABC):
    """所有分层架构层的抽象基类"""

    def __init__(self, config: Dict[str, Any], layer_name: str, priority: int):
        super().__init__()
        self.config = config
        self.layer_name = layer_name
        self.priority = priority  # 优先级：1最高，4最低
        self.enabled = config.get('enabled', True)
        self.response_time_ms = config.get('response_time_ms', 100)

        # 性能监控
        self.execution_times = []
        self.activation_count = 0

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        层的前向传播

        Args:
            inputs: 输入数据字典，包含各种传感器数据
            context: 上下文信息，包含其他层的输出和任务信息

        Returns:
            Dict: 该层的输出结果
        """
        pass

    @abstractmethod
    def should_activate(self, inputs: Dict[str, torch.Tensor], context: Optional[Dict[str, Any]] = None) -> bool:
        """
        判断该层是否应该激活

        Args:
            inputs: 输入数据
            context: 上下文信息

        Returns:
            bool: 是否应该激活该层
        """
        pass

    def get_latency_budget(self) -> float:
        """获取该层的延迟预算（毫秒）"""
        return self.response_time_ms

    def get_priority(self) -> int:
        """获取层的优先级"""
        return self.priority

    def is_enabled(self) -> bool:
        """检查层是否启用"""
        return self.enabled

    def set_enabled(self, enabled: bool):
        """设置层的启用状态"""
        self.enabled = enabled

    def forward_with_timing(self, inputs: Dict[str, torch.Tensor], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        带时间监控的前向传播

        Args:
            inputs: 输入数据
            context: 上下文信息

        Returns:
            Dict: 包含执行时间的输出结果
        """
        if not self.enabled:
            return {'layer': self.layer_name, 'enabled': False}

        start_time = time.time()

        try:
            output = self.forward(inputs, context)
            self.activation_count += 1
        except Exception as e:
            return {
                'layer': self.layer_name,
                'error': str(e),
                'execution_time_ms': 0
            }

        execution_time_ms = (time.time() - start_time) * 1000
        self.execution_times.append(execution_time_ms)

        # 保持最近100次执行时间记录
        if len(self.execution_times) > 100:
            self.execution_times.pop(0)

        # 在输出中添加性能信息
        output['execution_time_ms'] = execution_time_ms
        output['layer'] = self.layer_name
        output['activation_count'] = self.activation_count

        return output

    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计信息"""
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

    def check_latency_budget(self) -> bool:
        """检查是否在延迟预算内"""
        if not self.execution_times:
            return True

        avg_time = sum(self.execution_times) / len(self.execution_times)
        return avg_time <= self.response_time_ms

    def reset_performance_stats(self):
        """重置性能统计"""
        self.execution_times.clear()
        self.activation_count = 0

    def validate_inputs(self, inputs: Dict[str, torch.Tensor]) -> bool:
        """
        验证输入数据的有效性

        Args:
            inputs: 输入数据

        Returns:
            bool: 输入是否有效
        """
        if not isinstance(inputs, dict):
            return False

        # 检查是否有必要的输入
        required_keys = self.get_required_input_keys()
        for key in required_keys:
            if key not in inputs:
                return False
            if not isinstance(inputs[key], torch.Tensor):
                return False

        return True

    def get_required_input_keys(self) -> List[str]:
        """
        获取该层需要的输入key列表
        子类可以重写此方法

        Returns:
            List[str]: 必需的输入key列表
        """
        return []

    def get_output_keys(self) -> List[str]:
        """
        获取该层输出的key列表
        子类可以重写此方法

        Returns:
            List[str]: 输出key列表
        """
        return ['layer', 'execution_time_ms']

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(layer={self.layer_name}, priority={self.priority}, enabled={self.enabled})"