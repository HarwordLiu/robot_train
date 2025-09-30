# -*- coding: utf-8 -*-
"""
分层架构评估器模块

专门用于评估HumanoidDiffusionPolicy分层架构模型
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import lerobot_patches.custom_patches

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter

from .base_evaluator import BaseEvaluator, EvaluationResults
from kuavo_train.wrapper.policy.humanoid.HumanoidDiffusionPolicy import HumanoidDiffusionPolicy

class HierarchicalEvaluator(BaseEvaluator):
    """
    分层架构专用评估器

    提供分层架构特有的评估功能，包括：
    - 层激活分析
    - 推理延迟分析
    - 层一致性检查
    - 任务特化分析
    """

    def __init__(self, config):
        super().__init__(config)

        # 分层架构特有的统计数据
        self.layer_activation_stats = defaultdict(int)
        self.layer_timing_stats = defaultdict(list)
        self.layer_conflict_count = 0
        self.safety_override_count = 0
        self.budget_violation_count = 0
        self.total_inference_steps = 0

        self.logger.info("Initialized HierarchicalEvaluator for humanoid_diffusion model")

    def load_model(self) -> None:
        """加载分层架构模型"""
        self.logger.info(f"Loading hierarchical model from {self.config.model.checkpoint_path}")

        try:
            # 传递分层架构配置信息
            load_kwargs = {
                'strict': True,
                'use_hierarchical': True,
            }

            # 如果配置中有 hierarchical 信息，传递给模型
            if hasattr(self.config, 'hierarchical'):
                load_kwargs['hierarchical'] = self.config.hierarchical

            self.model = HumanoidDiffusionPolicy.from_pretrained(
                self.config.model.checkpoint_path,
                **load_kwargs
            )
            self.model.eval()
            self.model.to(self.device)
            self.model.reset()

            self.logger.info("Hierarchical model loaded successfully")

            # 打印模型架构信息
            if hasattr(self.model, 'print_architecture_summary'):
                self.model.print_architecture_summary()

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _model_inference(self, observation: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        分层架构模型推理

        Args:
            observation: 观测数据

        Returns:
            Tuple[预测动作, 推理信息]
        """
        inference_info = {}

        # 构建任务信息（根据配置）
        task_info = {
            'task_complexity': 'medium',
            'requires_locomotion': False,  # 仅手臂任务
            'requires_manipulation': True,
            'safety_priority': True,
            'enabled_layers': self.config.hierarchical_evaluation.enabled_layers
            # 注意：不传递latency_budget_ms，以使用标准forward模式进行离线评估
        }

        with torch.no_grad():
            if hasattr(self.model, 'scheduler') and self.model.scheduler:
                # 分层架构推理
                start_time = time.time()

                try:
                    outputs = self.model.scheduler(observation, task_info)
                    inference_time = (time.time() - start_time) * 1000  # 转换为毫秒

                    print(f"🔥 DEBUG: Scheduler outputs: {list(outputs.keys())}")
                    for layer_name, layer_output in outputs.items():
                        if isinstance(layer_output, dict):
                            print(f"🔥 DEBUG: Layer {layer_name} output keys: {list(layer_output.keys())}")
                        else:
                            print(f"🔥 DEBUG: Layer {layer_name} output type: {type(layer_output)}")

                    # 提取最终动作
                    if 'final_action' in outputs:
                        action = outputs['final_action']
                        print(f"🔥 DEBUG: Using final_action")
                    else:
                        # 使用最高优先级层的输出
                        found_action = False
                        for layer_name in ['safety', 'gait', 'manipulation', 'planning']:
                            if layer_name in outputs and isinstance(outputs[layer_name], dict) and 'action' in outputs[layer_name]:
                                action = outputs[layer_name]['action']
                                print(f"🔥 DEBUG: Using action from layer: {layer_name}")
                                found_action = True
                                break

                        if not found_action:
                            print(f"🔥 DEBUG: No valid action found in outputs: {outputs}")
                            raise RuntimeError("No valid action output from hierarchical layers")

                    # 提取分层信息
                    inference_info.update({
                        'active_layers': list(outputs.keys()) if isinstance(outputs, dict) else [],
                        'inference_time': inference_time,
                        'within_budget': inference_time <= self.config.hierarchical_evaluation.latency_budget_ms,
                        'hierarchical_outputs': outputs
                    })

                    # 更新统计
                    self._update_hierarchical_stats(inference_info)

                except Exception as e:
                    self.logger.error(f"Hierarchical inference failed: {e}")
                    # 回退到传统推理
                    action = self.model.select_action(observation)
                    inference_info.update({
                        'active_layers': ['fallback'],
                        'inference_time': (time.time() - start_time) * 1000,
                        'within_budget': True,
                        'fallback_used': True
                    })
            else:
                # 传统推理模式
                start_time = time.time()
                action = self.model.select_action(observation)
                inference_time = (time.time() - start_time) * 1000

                inference_info.update({
                    'active_layers': ['main'],
                    'inference_time': inference_time,
                    'within_budget': True,
                    'traditional_mode': True
                })

        return action, inference_info

    def _update_hierarchical_stats(self, inference_info: Dict[str, Any]) -> None:
        """更新分层架构统计信息"""
        self.total_inference_steps += 1

        # 更新层激活统计
        active_layers = inference_info.get('active_layers', [])
        for layer in active_layers:
            self.layer_activation_stats[layer] += 1

        # 更新延迟统计
        inference_time = inference_info.get('inference_time', 0)
        if not inference_info.get('within_budget', True):
            self.budget_violation_count += 1

        # 层级特定统计
        hierarchical_outputs = inference_info.get('hierarchical_outputs', {})
        if isinstance(hierarchical_outputs, dict):
            for layer_name, layer_output in hierarchical_outputs.items():
                if isinstance(layer_output, dict) and 'execution_time' in layer_output:
                    self.layer_timing_stats[layer_name].append(layer_output['execution_time'])

    def _calculate_hierarchical_metrics(self) -> Dict[str, float]:
        """计算分层架构特有指标"""
        metrics = {}

        if self.total_inference_steps == 0:
            return metrics

        # 层激活率
        for layer, count in self.layer_activation_stats.items():
            activation_rate = count / self.total_inference_steps
            metrics[f'{layer}_activation_rate'] = activation_rate

        # 预算遵从率
        budget_compliance_rate = 1.0 - (self.budget_violation_count / self.total_inference_steps)
        metrics['budget_compliance_rate'] = budget_compliance_rate

        # 平均层执行时间
        for layer, times in self.layer_timing_stats.items():
            if times:
                metrics[f'{layer}_avg_execution_time'] = np.mean(times)
                metrics[f'{layer}_std_execution_time'] = np.std(times)

        # 安全覆盖率
        safety_activation = self.layer_activation_stats.get('safety', 0)
        if safety_activation > 0:
            metrics['safety_override_rate'] = self.safety_override_count / safety_activation

        return metrics

    def _analyze_layer_consistency(self, episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """分析层激活一致性"""
        if not self.config.hierarchical_evaluation.layer_consistency_check.enable:
            return {}

        consistency_metrics = {}
        window_size = self.config.hierarchical_evaluation.layer_consistency_check.temporal_consistency_window

        # 提取层激活序列
        layer_sequences = defaultdict(list)
        for result in episode_results:
            inference_info = result.get('inference_info', {})
            active_layers = inference_info.get('active_layers', [])

            for layer in ['safety', 'gait', 'manipulation', 'planning']:
                layer_sequences[layer].append(1 if layer in active_layers else 0)

        # 计算时序一致性
        for layer, sequence in layer_sequences.items():
            if len(sequence) >= window_size:
                # 计算层切换频率
                switches = sum(1 for i in range(1, len(sequence)) if sequence[i] != sequence[i-1])
                switch_rate = switches / (len(sequence) - 1)
                consistency_metrics[f'{layer}_switch_rate'] = switch_rate

                # 计算连续激活长度的标准差（稳定性指标）
                continuous_lengths = []
                current_length = 1
                for i in range(1, len(sequence)):
                    if sequence[i] == sequence[i-1]:
                        current_length += 1
                    else:
                        continuous_lengths.append(current_length)
                        current_length = 1
                continuous_lengths.append(current_length)

                if continuous_lengths:
                    consistency_metrics[f'{layer}_stability'] = 1.0 / (1.0 + np.std(continuous_lengths))

        return consistency_metrics

    def _check_expected_activation_rates(self, hierarchical_metrics: Dict[str, float]) -> Dict[str, float]:
        """检查层激活率是否符合预期"""
        check_results = {}
        expected_rates = self.config.hierarchical_evaluation.layer_activation_analysis.expected_rates

        for layer, expected_rate in expected_rates.items():
            actual_rate = hierarchical_metrics.get(f'{layer}_activation_rate', 0.0)
            deviation = abs(actual_rate - expected_rate)
            check_results[f'{layer}_rate_deviation'] = deviation
            check_results[f'{layer}_rate_check'] = 1.0 if deviation < 0.1 else 0.0  # 10%容差

        return check_results

    def evaluate(self) -> EvaluationResults:
        """执行分层架构评估"""
        self.logger.info("Starting hierarchical evaluation...")

        # 加载模型和数据
        self.load_model()
        self.prepare_data()

        # 重置统计
        self.layer_activation_stats.clear()
        self.layer_timing_stats.clear()
        self.layer_conflict_count = 0
        self.safety_override_count = 0
        self.budget_violation_count = 0
        self.total_inference_steps = 0

        all_episode_results = []
        all_action_metrics = []

        # 按episode组织数据
        current_episode = None
        episode_data = []

        for i, sample in enumerate(self.test_dataset):
            episode_idx = sample['episode_index'].item()

            # 检查是否开始新episode
            if current_episode is None:
                current_episode = episode_idx
            elif episode_idx != current_episode:
                # 处理完整的episode
                if episode_data:
                    episode_result = self.evaluate_single_episode(episode_data)
                    all_episode_results.append(episode_result)

                    # 检查是否达到最大episode数
                    if len(all_episode_results) >= self.config.test_data.max_episodes:
                        break

                # 开始新episode
                current_episode = episode_idx
                episode_data = []

            episode_data.append(sample)

            # 检查episode内步数限制
            if len(episode_data) >= self.config.test_data.max_steps_per_episode:
                episode_result = self.evaluate_single_episode(episode_data)
                all_episode_results.append(episode_result)
                episode_data = []
                current_episode = None

                if len(all_episode_results) >= self.config.test_data.max_episodes:
                    break

        # 处理最后一个episode
        if episode_data and len(all_episode_results) < self.config.test_data.max_episodes:
            episode_result = self.evaluate_single_episode(episode_data)
            all_episode_results.append(episode_result)

        # 计算总体指标
        action_metrics = {}
        performance_metrics = {}

        # 汇总动作指标
        for metric_name in self.config.evaluation.action_metrics:
            values = []
            for episode_result in all_episode_results:
                if f'avg_{metric_name}' in episode_result:
                    values.append(episode_result[f'avg_{metric_name}'])

            if values:
                action_metrics[f'overall_avg_{metric_name}'] = np.mean(values)
                action_metrics[f'overall_std_{metric_name}'] = np.std(values)

        # 计算分层架构指标
        hierarchical_metrics = self._calculate_hierarchical_metrics()
        performance_metrics.update(hierarchical_metrics)

        # 层一致性分析
        consistency_metrics = self._analyze_layer_consistency(all_episode_results)
        performance_metrics.update(consistency_metrics)

        # 检查预期激活率
        rate_check_results = self._check_expected_activation_rates(hierarchical_metrics)
        performance_metrics.update(rate_check_results)

        # 计算总体性能指标
        if all_episode_results:
            avg_inference_time = np.mean([r['average_inference_time'] for r in all_episode_results])
            performance_metrics['overall_avg_inference_time'] = avg_inference_time

            total_steps = sum(r['num_steps'] for r in all_episode_results)
            performance_metrics['total_evaluation_steps'] = total_steps
            performance_metrics['total_episodes'] = len(all_episode_results)

        # 生成摘要
        summary = {
            'model_type': 'humanoid_diffusion',
            'total_episodes_evaluated': len(all_episode_results),
            'total_inference_steps': self.total_inference_steps,
            'budget_violation_rate': self.budget_violation_count / max(self.total_inference_steps, 1),
            'most_active_layer': max(self.layer_activation_stats.items(), key=lambda x: x[1])[0] if self.layer_activation_stats else 'none',
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        results = EvaluationResults(
            action_metrics=action_metrics,
            performance_metrics=performance_metrics,
            episode_results=all_episode_results,
            summary=summary,
            timing_info={'total_evaluation_time': time.time()}
        )

        self.logger.info(f"Hierarchical evaluation completed: {len(all_episode_results)} episodes, {self.total_inference_steps} steps")

        return results

    def _generate_plots(self, results: EvaluationResults) -> None:
        """生成分层架构特有的可视化图表"""
        try:
            import matplotlib.pyplot as plt

            # 调用基类的绘图方法
            super()._generate_plots(results)

            # 分层架构特有的图表
            if self.config.hierarchical_visualization.plot_layer_activation:
                self._plot_layer_activation(results)

            if self.config.hierarchical_visualization.plot_latency_distribution:
                self._plot_latency_distribution(results)

            if self.config.hierarchical_visualization.plot_coordination_matrix:
                self._plot_coordination_matrix(results)

        except ImportError:
            self.logger.warning("Matplotlib not available, skipping hierarchical plot generation")
        except Exception as e:
            self.logger.error(f"Error generating hierarchical plots: {e}")

    def _plot_layer_activation(self, results: EvaluationResults) -> None:
        """绘制层激活图"""
        import matplotlib.pyplot as plt

        layers = list(self.layer_activation_stats.keys())
        activations = [self.layer_activation_stats[layer] / max(self.total_inference_steps, 1) for layer in layers]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(layers, activations, color=['red', 'blue', 'green', 'orange'][:len(layers)])
        plt.xlabel('Layers')
        plt.ylabel('Activation Rate')
        plt.title('Hierarchical Layer Activation Rates')
        plt.ylim(0, 1.0)

        # 添加数值标签
        for bar, activation in zip(bars, activations):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{activation:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'layer_activation_rates.png',
                   dpi=self.config.visualization.dpi)
        plt.close()

    def _plot_latency_distribution(self, results: EvaluationResults) -> None:
        """绘制延迟分布图"""
        import matplotlib.pyplot as plt

        inference_times = []
        for episode_result in results.episode_results:
            if 'average_inference_time' in episode_result:
                inference_times.append(episode_result['average_inference_time'])

        if not inference_times:
            return

        plt.figure(figsize=(10, 6))
        plt.hist(inference_times, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(self.config.hierarchical_evaluation.latency_budget_ms,
                   color='red', linestyle='--', label=f'Budget ({self.config.hierarchical_evaluation.latency_budget_ms}ms)')
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Inference Time Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'latency_distribution.png',
                   dpi=self.config.visualization.dpi)
        plt.close()

    def _plot_coordination_matrix(self, results: EvaluationResults) -> None:
        """绘制层协调矩阵"""
        import matplotlib.pyplot as plt
        import numpy as np

        # 这里可以实现层之间协调关系的可视化
        # 由于需要更复杂的层交互数据，这里提供一个基础框架

        layers = ['safety', 'gait', 'manipulation', 'planning']

        # 创建一个示例协调矩阵
        coordination_matrix = np.zeros((len(layers), len(layers)))

        plt.figure(figsize=(8, 6))
        plt.imshow(coordination_matrix, cmap='Blues')
        plt.colorbar(label='Coordination Strength')
        plt.xticks(range(len(layers)), layers, rotation=45)
        plt.yticks(range(len(layers)), layers)
        plt.title('Layer Coordination Matrix')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'coordination_matrix.png',
                   dpi=self.config.visualization.dpi)
        plt.close()