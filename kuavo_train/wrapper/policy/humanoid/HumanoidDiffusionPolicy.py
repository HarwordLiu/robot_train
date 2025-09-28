"""
HumanoidDiffusionPolicy: 分层人形机器人Diffusion Policy主入口
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

# 导入原有的组件
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from kuavo_train.wrapper.policy.diffusion.DiffusionConfigWrapper import CustomDiffusionConfigWrapper

# 导入分层架构组件
from .HierarchicalScheduler import HierarchicalScheduler
from .HierarchicalDiffusionModel import HierarchicalDiffusionModel


class HumanoidDiffusionPolicyWrapper(CustomDiffusionPolicyWrapper):
    """
    分层人形机器人Diffusion Policy

    继承自CustomDiffusionPolicyWrapper以保持向后兼容性
    根据配置自动选择使用分层架构或传统架构
    """

    def __init__(self,
                 config: CustomDiffusionConfigWrapper,
                 dataset_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
                 use_hierarchical: Optional[bool] = None,
                 **kwargs):
        """
        初始化分层Diffusion Policy

        Args:
            config: 配置对象
            dataset_stats: 数据集统计信息
            use_hierarchical: 是否启用分层架构（可选，优先使用config中的设置）
            **kwargs: 其他参数（用于Hydra兼容性）
        """
        # 检查是否启用分层架构（优先使用参数，其次使用config）
        if use_hierarchical is not None:
            self.use_hierarchical = use_hierarchical
        else:
            self.use_hierarchical = getattr(config, 'use_hierarchical', False)

        if self.use_hierarchical:
            # 使用分层架构
            super().__init__(config, dataset_stats)
            self._init_hierarchical_components(config)

            # 初始化任务条件权重系统
            self._init_task_conditional_weights(config)
        else:
            # 向后兼容：使用原有架构
            super().__init__(config, dataset_stats)
            self.scheduler = None
            self.task_layer_weights = None
            self.current_curriculum_stage = None

    def _init_hierarchical_components(self, config):
        """初始化分层架构组件"""
        try:
            # 替换原有的diffusion模型为分层版本
            self.diffusion = HierarchicalDiffusionModel(config)

            # 创建分层调度器
            hierarchical_config = getattr(config, 'hierarchical', {})
            self.scheduler = HierarchicalScheduler(hierarchical_config, config)

            print(f"✅ Hierarchical architecture initialized with {len(self.scheduler.layers)} layers")

        except Exception as e:
            print(f"❌ Failed to initialize hierarchical components: {e}")
            print("🔄 Falling back to traditional architecture")
            self.use_hierarchical = False
            self.scheduler = None

    def _init_task_conditional_weights(self, config):
        """初始化任务条件权重系统"""
        try:
            # 默认层权重
            hierarchical_config = getattr(config, 'hierarchical', {})
            self.default_layer_weights = hierarchical_config.get('layer_weights', {
                'safety': 2.0,
                'gait': 1.5,
                'manipulation': 1.0,
                'planning': 0.8
            })

            # 当前激活的任务特定权重
            self.task_layer_weights = self.default_layer_weights.copy()

            # 课程学习状态
            self.current_curriculum_stage = None
            self.enabled_layers = list(self.default_layer_weights.keys())

            print("✅ 任务条件权重系统初始化完成")

        except Exception as e:
            print(f"⚠️  任务条件权重系统初始化失败: {e}")
            self.task_layer_weights = self.default_layer_weights
            self.current_curriculum_stage = None

    def forward(self, batch: Dict[str, torch.Tensor],
               curriculum_info: Optional[Dict[str, Any]] = None,
               task_weights: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        前向传播，根据架构类型选择处理方式

        Args:
            batch: 输入批次数据
            curriculum_info: 课程学习信息
            task_weights: 任务特定权重

        Returns:
            Tuple[loss, outputs]: 损失和输出结果
        """
        if self.use_hierarchical and self.scheduler is not None:
            return self._hierarchical_forward(batch, curriculum_info, task_weights)
        else:
            return super().forward(batch)

    def _hierarchical_forward(self, batch: Dict[str, torch.Tensor],
                             curriculum_info: Optional[Dict[str, Any]] = None,
                             task_weights: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """分层架构的前向传播"""
        # 更新任务条件权重
        if task_weights is not None:
            self._update_task_weights(task_weights)

        # 更新课程学习状态
        if curriculum_info is not None:
            self._update_curriculum_state(curriculum_info)

        # 图像预处理（保持与原有逻辑一致）
        batch = self._preprocess_batch(batch)

        # 归一化输入和目标
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # 任务识别（增强版，考虑课程学习信息）
        task_info = self._identify_task(batch, curriculum_info)

        # 分层处理
        layer_outputs = self.scheduler(batch, task_info)

        # Diffusion损失计算
        diffusion_loss = self.diffusion.compute_loss(batch, layer_outputs)

        # 分层损失聚合（使用任务特定权重）
        total_loss = self._aggregate_hierarchical_loss(diffusion_loss, layer_outputs, use_task_weights=True)

        # 添加课程学习和任务特定信息到输出
        hierarchical_info = {
            'curriculum_stage': self.current_curriculum_stage,
            'enabled_layers': self.enabled_layers.copy(),
            'task_weights': self.task_layer_weights.copy(),
            'layer_performance': self._get_layer_performance_metrics(layer_outputs)
        }

        return total_loss, hierarchical_info

    def _preprocess_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预处理批次数据（图像裁剪、缩放等）"""
        # 复用原有的图像预处理逻辑
        random_crop = self.config.crop_is_random and self.training
        crop_position = None

        # RGB图像预处理
        if self.config.image_features:
            batch = dict(batch)  # shallow copy
            for key in self.config.image_features:
                from kuavo_train.utils.augmenter import crop_image, resize_image
                batch[key], crop_position = crop_image(
                    batch[key],
                    target_range=self.config.crop_shape,
                    random_crop=random_crop
                )
                batch[key] = resize_image(
                    batch[key],
                    target_size=self.config.resize_shape,
                    image_type="rgb"
                )

            # 堆叠RGB特征
            from lerobot.constants import OBS_IMAGES
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        # 深度图像预处理
        if self.config.use_depth and self.config.depth_features:
            batch = dict(batch)
            for key in self.config.depth_features:
                import torchvision.transforms.functional as TF
                if len(crop_position) == 4:
                    batch[key] = TF.crop(batch[key], *crop_position)
                else:
                    batch[key] = TF.center_crop(batch[key], crop_position)

                from kuavo_train.utils.augmenter import resize_image
                batch[key] = resize_image(
                    batch[key],
                    target_size=self.config.resize_shape,
                    image_type="depth"
                )

            # 堆叠深度特征
            OBS_DEPTH = "observation.depth"
            batch[OBS_DEPTH] = torch.stack([batch[key] for key in self.config.depth_features], dim=-4)

        return batch

    def _update_task_weights(self, task_weights: Dict[str, float]):
        """更新任务特定的层权重"""
        if task_weights:
            self.task_layer_weights.update(task_weights)
            print(f"🎯 更新任务权重: {task_weights}")

    def _update_curriculum_state(self, curriculum_info: Dict[str, Any]):
        """更新课程学习状态"""
        stage_changed = False
        layers_changed = False
        
        if 'stage' in curriculum_info:
            new_stage = curriculum_info['stage']
            if new_stage != self.current_curriculum_stage:
                self.current_curriculum_stage = new_stage
                stage_changed = True

        if 'enabled_layers' in curriculum_info:
            new_layers = curriculum_info['enabled_layers'].copy()
            if new_layers != self.enabled_layers:
                self.enabled_layers = new_layers
                layers_changed = True
        
        # 只在状态真正改变时输出日志，避免进度条重复渲染
        if stage_changed or layers_changed:
            print(f"🎓 课程学习阶段: {self.current_curriculum_stage}, 激活层: {self.enabled_layers}")

    def _identify_task(self, batch: Dict[str, torch.Tensor],
                      curriculum_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """任务识别（增强版，考虑课程学习信息）"""
        # 基础任务信息
        task_info = {
            'task_type': 'general',
            'task_complexity': 'medium',
            'requires_locomotion': True,
            'requires_manipulation': True,
            'requires_planning': False
        }

        # 根据课程学习阶段调整任务信息
        if curriculum_info:
            target_task = curriculum_info.get('target_task')
            if target_task == 1:  # 动态抓取
                task_info.update({
                    'task_type': 'dynamic_grasping',
                    'task_complexity': 'medium',
                    'requires_locomotion': False,  # 主要是操作任务
                    'requires_manipulation': True,
                    'requires_planning': False
                })
            elif target_task == 2:  # 称重
                task_info.update({
                    'task_type': 'package_weighing',
                    'task_complexity': 'high',
                    'requires_locomotion': True,  # 需要平衡控制
                    'requires_manipulation': True,
                    'requires_planning': True
                })
            elif target_task == 3:  # 摆放
                task_info.update({
                    'task_type': 'precise_placement',
                    'task_complexity': 'high',
                    'requires_locomotion': False,
                    'requires_manipulation': True,
                    'requires_planning': True  # 需要空间规划
                })
            elif target_task == 4:  # 分拣
                task_info.update({
                    'task_type': 'full_process_sorting',
                    'task_complexity': 'very_high',
                    'requires_locomotion': True,
                    'requires_manipulation': True,
                    'requires_planning': True
                })

        return task_info

    def _get_layer_performance_metrics(self, layer_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """获取层性能指标"""
        performance = {}

        for layer_name, layer_output in layer_outputs.items():
            if isinstance(layer_output, dict):
                metrics = {}

                # 执行时间
                if 'execution_time' in layer_output:
                    metrics['execution_time'] = layer_output['execution_time']

                # 损失值
                if 'loss' in layer_output:
                    metrics['loss'] = layer_output['loss'].item() if torch.is_tensor(layer_output['loss']) else layer_output['loss']

                # 激活状态
                metrics['active'] = layer_name in self.enabled_layers

                # 权重
                metrics['weight'] = self.task_layer_weights.get(layer_name, 1.0)

                performance[layer_name] = metrics

        return performance

    def _aggregate_hierarchical_loss(self,
                                   diffusion_loss: torch.Tensor,
                                   layer_outputs: Dict[str, Any],
                                   use_task_weights: bool = False) -> torch.Tensor:
        """聚合分层损失"""
        total_loss = diffusion_loss

        # 选择权重来源
        if use_task_weights and hasattr(self, 'task_layer_weights'):
            layer_weights = self.task_layer_weights
        else:
            layer_weights = getattr(self.scheduler.config, 'layer_weights', {}) if self.scheduler else {}

        # 聚合各层的损失（只计算激活层的损失）
        for layer_name, layer_output in layer_outputs.items():
            if isinstance(layer_output, dict) and 'loss' in layer_output:
                # 检查层是否在当前激活列表中
                if layer_name in self.enabled_layers:
                    layer_weight = layer_weights.get(layer_name, 1.0)
                    layer_loss = layer_output['loss']
                    total_loss = total_loss + layer_weight * layer_loss

        return total_loss

    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        选择动作（推理时使用）

        Args:
            batch: 观测数据

        Returns:
            torch.Tensor: 选择的动作
        """
        if self.use_hierarchical and self.scheduler is not None:
            return self._hierarchical_select_action(batch)
        else:
            return super().select_action(batch)

    def _hierarchical_select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """分层架构的动作选择"""
        # 预处理
        batch = self._preprocess_batch(batch)
        batch = self.normalize_inputs(batch)

        # 任务识别
        task_info = self._identify_task(batch)

        # 分层推理（可能只激活部分层以满足实时性要求）
        with torch.no_grad():
            layer_outputs = self.scheduler.inference_mode(batch, task_info)

        # 从分层输出中提取最终动作
        return self._extract_action_from_layers(layer_outputs, batch)

    def _extract_action_from_layers(self,
                                  layer_outputs: Dict[str, Any],
                                  batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """从分层输出中提取最终动作"""
        # 优先级处理：安全层可以覆盖其他层的输出
        if 'safety' in layer_outputs and layer_outputs['safety'].get('emergency', False):
            return layer_outputs['safety'].get('emergency_action', torch.zeros_like(batch.get('action', torch.zeros(1, 32))))

        # 正常情况下，使用最高级别可用层的输出
        for layer_name in ['planning', 'manipulation', 'gait', 'safety']:
            if layer_name in layer_outputs and 'action' in layer_outputs[layer_name]:
                return layer_outputs[layer_name]['action']

        # 回退：使用传统diffusion输出
        return super().select_action(batch)

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        if self.use_hierarchical and self.scheduler is not None:
            return self.scheduler.get_performance_stats()
        else:
            return {'architecture': 'traditional', 'hierarchical_enabled': False}

    def set_layer_enabled(self, layer_name: str, enabled: bool):
        """动态启用/禁用特定层"""
        if self.use_hierarchical and self.scheduler is not None:
            self.scheduler.set_layer_enabled(layer_name, enabled)

    def get_active_layers(self) -> List[str]:
        """获取当前激活的层列表"""
        if self.use_hierarchical and self.scheduler is not None:
            return self.scheduler.get_active_layers()
        else:
            return ['traditional']

    def set_task_layer_weights(self, task_weights: Dict[str, float]):
        """设置任务特定的层权重"""
        if self.use_hierarchical:
            self._update_task_weights(task_weights)

    def set_curriculum_stage(self, enabled_layers: List[str]):
        """设置课程学习阶段"""
        if self.use_hierarchical:
            self.enabled_layers = enabled_layers.copy()
            self.current_curriculum_stage = f"layers_{'_'.join(enabled_layers)}"
            print(f"🎓 设置课程学习阶段: 激活层 {enabled_layers}")

    def get_layer_states(self) -> Dict[str, Any]:
        """获取层状态信息"""
        if not self.use_hierarchical:
            return {}

        return {
            'current_curriculum_stage': self.current_curriculum_stage,
            'enabled_layers': self.enabled_layers.copy(),
            'task_layer_weights': self.task_layer_weights.copy(),
            'default_layer_weights': self.default_layer_weights.copy()
        }

    def load_layer_states(self, layer_states: Dict[str, Any]):
        """加载层状态信息"""
        if not self.use_hierarchical:
            return

        if 'current_curriculum_stage' in layer_states:
            self.current_curriculum_stage = layer_states['current_curriculum_stage']

        if 'enabled_layers' in layer_states:
            self.enabled_layers = layer_states['enabled_layers'].copy()

        if 'task_layer_weights' in layer_states:
            self.task_layer_weights = layer_states['task_layer_weights'].copy()

        print(f"✅ 已恢复层状态: 阶段={self.current_curriculum_stage}, 激活层={self.enabled_layers}")

    def print_architecture_summary(self):
        """打印架构摘要"""
        if not self.use_hierarchical:
            print("📊 使用传统Diffusion Policy架构")
            return

        print("🏗️  分层人形机器人Diffusion Policy架构摘要")
        print("=" * 50)
        print(f"当前课程学习阶段: {self.current_curriculum_stage}")
        print(f"激活层: {self.enabled_layers}")
        print(f"层权重配置:")
        for layer_name, weight in self.task_layer_weights.items():
            status = "✅" if layer_name in self.enabled_layers else "⏸️ "
            print(f"  {status} {layer_name}: {weight}")

        if self.scheduler:
            print(f"分层调度器: {type(self.scheduler).__name__}")
            print(f"总层数: {len(self.scheduler.layers) if hasattr(self.scheduler, 'layers') else 'unknown'}")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """从预训练模型加载（保持与父类接口兼容）"""
        # TODO: 实现分层架构的模型加载逻辑
        return super().from_pretrained(*args, **kwargs)


# 为了向后兼容性，创建别名
HumanoidDiffusionPolicy = HumanoidDiffusionPolicyWrapper