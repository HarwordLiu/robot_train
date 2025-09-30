"""
HierarchicalScheduler: 分层架构的核心调度器
"""
import torch
import torch.nn as nn
import time
from typing import Dict, Any, Optional, List
from collections import OrderedDict

from .layers import BaseLayer, SafetyReflexLayer, GaitControlLayer, ManipulationLayer, GlobalPlanningLayer


class HierarchicalScheduler(nn.Module):
    """
    分层架构的核心调度器

    负责管理四个层次的激活、调度和输出聚合
    """

    def __init__(self, hierarchical_config: Dict[str, Any], base_config: Any):
        super().__init__()
        self.config = hierarchical_config
        self.base_config = base_config

        # 构建四个层次
        self.layers = self._build_layers()

        # 调度配置
        self.layer_priorities = {name: layer.get_priority() for name, layer in self.layers.items()}
        self.layer_weights = hierarchical_config.get('layer_weights', {})

        # 性能监控
        self.total_forward_calls = 0
        self.layer_activation_stats = {name: 0 for name in self.layers.keys()}

        print(f"🏗️ HierarchicalScheduler initialized with layers: {list(self.layers.keys())}")

    def _build_layers(self) -> nn.ModuleDict:
        """构建四个层次的网络"""
        layers = nn.ModuleDict()

        layer_configs = self.config.get('layers', {})
        print(f"🔍 Available layer configs: {list(layer_configs.keys())}")

        # 按优先级顺序构建层
        layer_builders = {
            'safety': SafetyReflexLayer,
            'gait': GaitControlLayer,
            'manipulation': ManipulationLayer,
            'planning': GlobalPlanningLayer
        }

        for layer_name, layer_class in layer_builders.items():
            if layer_name in layer_configs:
                layer_config = layer_configs[layer_name]
                try:
                    layer = layer_class(layer_config, self.base_config)
                    layers[layer_name] = layer
                    print(f"✅ {layer_name} layer created successfully")
                except Exception as e:
                    print(f"❌ Failed to create {layer_name} layer: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️ No config found for {layer_name} layer")

        print(f"🏗️ Total layers built: {len(layers)} - {list(layers.keys())}")
        return layers

    def forward(self, batch: Dict[str, torch.Tensor], task_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        分层处理前向传播

        Args:
            batch: 输入批次数据
            task_info: 任务信息

        Returns:
            Dict: 各层的输出结果
        """
        self.total_forward_calls += 1
        outputs = {}
        context = self._build_context(batch, task_info)

        print(f"🔄 Starting hierarchical forward pass with {len(self.layers)} layers available")
        processing_order = self._get_processing_order()
        print(f"🔄 Processing order: {processing_order}")

        # 按优先级顺序处理各层
        for layer_name in processing_order:
            if layer_name not in self.layers:
                print(f"⚠️ Layer {layer_name} not found in available layers")
                continue
            layer = self.layers[layer_name]

            # 检查是否应该激活该层
            should_activate = layer.should_activate(batch, context)
            print(f"🔍 Layer {layer_name} should_activate: {should_activate}")
            if not should_activate:
                continue

            # 执行层的前向传播（带时间监控）
            try:
                print(f"🚀 Executing layer {layer_name}")
                layer_output = layer.forward_with_timing(batch, context)
                outputs[layer_name] = layer_output
                self.layer_activation_stats[layer_name] += 1
                print(f"✅ Layer {layer_name} executed successfully, output keys: {list(layer_output.keys())}")

                # 更新上下文
                context.update(layer_output)

                # 安全层可以立即返回（紧急情况）
                if layer_name == 'safety' and layer_output.get('emergency', False):
                    print(f"🚨 Emergency stop triggered by safety layer")
                    return {layer_name: layer_output}

            except Exception as e:
                print(f"❌ Error in {layer_name} layer: {e}")
                import traceback
                traceback.print_exc()
                outputs[layer_name] = {
                    'layer': layer_name,
                    'error': str(e),
                    'execution_time_ms': 0
                }

        print(f"🎯 Forward pass completed with {len(outputs)} active layers: {list(outputs.keys())}")
        return outputs

    def inference_mode(self,
                      batch: Dict[str, torch.Tensor],
                      task_info: Optional[Dict[str, Any]] = None,
                      latency_budget_ms: float = 50.0) -> Dict[str, Any]:
        """
        推理模式：根据延迟预算自适应激活层

        Args:
            batch: 输入数据
            task_info: 任务信息
            latency_budget_ms: 延迟预算（毫秒）

        Returns:
            Dict: 在预算内完成的层输出
        """
        start_time = time.time()
        outputs = {}
        context = self._build_context(batch, task_info)
        remaining_budget = latency_budget_ms

        # 按优先级顺序处理，在预算内尽可能多地激活层
        for layer_name in self._get_processing_order():
            if layer_name not in self.layers:
                continue
            layer = self.layers[layer_name]

            # 检查是否有足够的时间预算
            layer_budget = layer.get_latency_budget()
            if remaining_budget < layer_budget:
                print(f"⏰ Skipping {layer_name} due to time budget ({remaining_budget:.1f}ms < {layer_budget}ms)")
                continue

            # 检查是否应该激活
            if not layer.should_activate(batch, context):
                continue

            # 执行层推理
            layer_start = time.time()
            try:
                layer_output = layer.forward_with_timing(batch, context)
                outputs[layer_name] = layer_output
                context.update(layer_output)

                # 更新剩余预算
                layer_time = (time.time() - layer_start) * 1000
                remaining_budget -= layer_time

                # 安全层紧急情况
                if layer_name == 'safety' and layer_output.get('emergency', False):
                    break

            except Exception as e:
                print(f"❌ Inference error in {layer_name}: {e}")
                remaining_budget -= 1  # 小惩罚

        total_time = (time.time() - start_time) * 1000
        outputs['_inference_stats'] = {
            'total_time_ms': total_time,
            'budget_ms': latency_budget_ms,
            'within_budget': total_time <= latency_budget_ms
        }

        return outputs

    def _build_context(self, batch: Dict[str, torch.Tensor], task_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """构建上下文信息"""
        context = {
            'batch_size': batch.get('observation.state', list(batch.values())[0]).size(0),
            'device': list(batch.values())[0].device,
            'training': self.training
        }

        if task_info:
            context.update(task_info)

        return context

    def _get_processing_order(self) -> List[str]:
        """获取层的处理顺序（按优先级）"""
        return sorted(self.layers.keys(), key=lambda x: self.layer_priorities.get(x, 999))

    def set_layer_enabled(self, layer_name: str, enabled: bool):
        """设置特定层的启用状态"""
        if layer_name in self.layers:
            self.layers[layer_name].set_enabled(enabled)
            print(f"📝 Layer {layer_name} {'enabled' if enabled else 'disabled'}")

    def get_active_layers(self) -> List[str]:
        """获取当前启用的层列表"""
        return [name for name, layer in self.layers.items() if layer.is_enabled()]

    def get_layer_performance(self, layer_name: str) -> Dict[str, Any]:
        """获取特定层的性能统计"""
        if layer_name in self.layers:
            return self.layers[layer_name].get_performance_stats()
        return {}

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取整体性能统计"""
        stats = {
            'total_forward_calls': self.total_forward_calls,
            'layer_activation_stats': self.layer_activation_stats.copy(),
            'layer_performance': {}
        }

        for layer_name, layer in self.layers.items():
            stats['layer_performance'][layer_name] = layer.get_performance_stats()

        return stats

    def reset_performance_stats(self):
        """重置所有性能统计"""
        self.total_forward_calls = 0
        self.layer_activation_stats = {name: 0 for name in self.layers.keys()}

        for layer in self.layers.values():
            layer.reset_performance_stats()

    def check_layer_health(self) -> Dict[str, bool]:
        """检查各层的健康状态（延迟预算等）"""
        health_status = {}

        for layer_name, layer in self.layers.items():
            health_status[layer_name] = {
                'enabled': layer.is_enabled(),
                'within_budget': layer.check_latency_budget(),
                'activation_count': layer.activation_count
            }

        return health_status

    def auto_tune_layers(self, target_latency_ms: float = 50.0):
        """根据目标延迟自动调整层的启用状态"""
        print(f"🔧 Auto-tuning layers for target latency: {target_latency_ms}ms")

        # 获取各层的平均执行时间
        layer_times = {}
        for layer_name, layer in self.layers.items():
            stats = layer.get_performance_stats()
            layer_times[layer_name] = stats.get('avg_time_ms', 0)

        # 按优先级和执行时间调整
        cumulative_time = 0
        for layer_name in self._get_processing_order():
            layer_time = layer_times.get(layer_name, 0)

            if cumulative_time + layer_time <= target_latency_ms:
                self.set_layer_enabled(layer_name, True)
                cumulative_time += layer_time
            else:
                self.set_layer_enabled(layer_name, False)
                print(f"⚠️ Disabled {layer_name} to meet latency target")

        print(f"✅ Auto-tuning complete. Estimated latency: {cumulative_time:.1f}ms")

    def __repr__(self) -> str:
        layer_info = ", ".join([f"{name}({'✓' if layer.is_enabled() else '✗'})"
                               for name, layer in self.layers.items()])
        return f"HierarchicalScheduler({layer_info})"