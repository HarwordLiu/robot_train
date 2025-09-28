"""
HumanoidDiffusionPolicy: 分层人形机器人Diffusion Policy主入口
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
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
                 dataset_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None):
        """
        初始化分层Diffusion Policy

        Args:
            config: 配置对象
            dataset_stats: 数据集统计信息
        """
        # 检查是否启用分层架构
        self.use_hierarchical = getattr(config, 'use_hierarchical', False)

        if self.use_hierarchical:
            # 使用分层架构
            super().__init__(config, dataset_stats)
            self._init_hierarchical_components(config)
        else:
            # 向后兼容：使用原有架构
            super().__init__(config, dataset_stats)
            self.scheduler = None

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

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        前向传播，根据架构类型选择处理方式

        Args:
            batch: 输入批次数据

        Returns:
            Tuple[loss, outputs]: 损失和输出结果
        """
        if self.use_hierarchical and self.scheduler is not None:
            return self._hierarchical_forward(batch)
        else:
            return super().forward(batch)

    def _hierarchical_forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """分层架构的前向传播"""
        # 图像预处理（保持与原有逻辑一致）
        batch = self._preprocess_batch(batch)

        # 归一化输入和目标
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # 任务识别
        task_info = self._identify_task(batch)

        # 分层处理
        layer_outputs = self.scheduler(batch, task_info)

        # Diffusion损失计算
        diffusion_loss = self.diffusion.compute_loss(batch, layer_outputs)

        # 分层损失聚合
        total_loss = self._aggregate_hierarchical_loss(diffusion_loss, layer_outputs)

        return total_loss, layer_outputs

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

    def _identify_task(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """任务识别（目前简化为基于配置的静态识别）"""
        # TODO: 实现基于输入数据的动态任务识别
        return {
            'task_type': 'general',
            'task_complexity': 'medium',
            'requires_locomotion': True,
            'requires_manipulation': True,
            'requires_planning': False  # 默认不启用最复杂的规划层
        }

    def _aggregate_hierarchical_loss(self,
                                   diffusion_loss: torch.Tensor,
                                   layer_outputs: Dict[str, Any]) -> torch.Tensor:
        """聚合分层损失"""
        total_loss = diffusion_loss

        # 获取层权重配置
        layer_weights = getattr(self.scheduler.config, 'layer_weights', {})

        # 聚合各层的损失
        for layer_name, layer_output in layer_outputs.items():
            if isinstance(layer_output, dict) and 'loss' in layer_output:
                layer_weight = layer_weights.get(layer_name, 1.0)
                total_loss = total_loss + layer_weight * layer_output['loss']

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

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """从预训练模型加载（保持与父类接口兼容）"""
        # TODO: 实现分层架构的模型加载逻辑
        return super().from_pretrained(*args, **kwargs)