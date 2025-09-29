# -*- coding: utf-8 -*-
"""
传统Diffusion Policy评估器模块

专门用于评估传统DiffusionPolicy模型
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
from collections import defaultdict

from .base_evaluator import BaseEvaluator, EvaluationResults
from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper

class DiffusionEvaluator(BaseEvaluator):
    """
    传统Diffusion Policy专用评估器

    提供传统diffusion模型特有的评估功能，包括：
    - 去噪过程分析
    - 推理步数优化分析
    - 轨迹平滑度分析
    - 动作一致性分析
    """

    def __init__(self, config):
        super().__init__(config)

        # Diffusion特有的统计数据
        self.denoising_stats = []
        self.inference_steps_stats = defaultdict(list)
        self.smoothness_stats = []
        self.consistency_stats = []

        self.logger.info("Initialized DiffusionEvaluator for traditional diffusion model")

    def load_model(self) -> None:
        """加载传统diffusion模型"""
        self.logger.info(f"Loading diffusion model from {self.config.model.checkpoint_path}")

        try:
            self.model = CustomDiffusionPolicyWrapper.from_pretrained(
                self.config.model.checkpoint_path,
                strict=True
            )
            self.model.eval()
            self.model.to(self.device)
            self.model.reset()

            self.logger.info("Diffusion model loaded successfully")

            # 打印模型信息
            self.logger.info(f"Model horizon: {self.model.config.horizon}")
            self.logger.info(f"Action steps: {self.model.config.n_action_steps}")
            self.logger.info(f"Observation steps: {self.model.config.n_obs_steps}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _model_inference(self, observation: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        传统diffusion模型推理

        Args:
            observation: 观测数据

        Returns:
            Tuple[预测动作, 推理信息]
        """
        inference_info = {}

        start_time = time.time()

        with torch.no_grad():
            try:
                # 传统diffusion推理
                action = self.model.select_action(observation)
                inference_time = (time.time() - start_time) * 1000  # 转换为毫秒

                # 获取diffusion特有信息
                inference_info.update({
                    'inference_time': inference_time,
                    'model_type': 'diffusion',
                    'denoising_steps': getattr(self.model, 'num_inference_steps', None),
                })

                # 如果配置了去噪分析，获取去噪过程信息
                if self.config.diffusion_evaluation.denoising_analysis.enable:
                    denoising_info = self._analyze_denoising_process(observation)
                    inference_info.update(denoising_info)

            except Exception as e:
                self.logger.error(f"Diffusion inference failed: {e}")
                raise

        return action, inference_info

    def _analyze_denoising_process(self, observation: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """分析去噪过程"""
        denoising_info = {}

        try:
            # 这里可以添加去噪过程的详细分析
            # 由于具体实现依赖于diffusion模型的内部结构，这里提供基础框架

            if hasattr(self.model, 'scheduler'):
                denoising_info['scheduler_type'] = type(self.model.scheduler).__name__
                denoising_info['num_train_timesteps'] = getattr(self.model.scheduler, 'num_train_timesteps', None)

            if hasattr(self.model, 'noise_pred_net'):
                # 可以添加网络特有的分析
                denoising_info['noise_predictor_available'] = True

        except Exception as e:
            self.logger.debug(f"Denoising analysis failed: {e}")

        return denoising_info

    def _calculate_smoothness_metrics(self, actions: List[torch.Tensor]) -> Dict[str, float]:
        """计算轨迹平滑度指标"""
        if not self.config.diffusion_evaluation.trajectory_smoothness.enable:
            return {}

        if len(actions) < 2:
            return {}

        metrics = {}
        action_tensor = torch.stack(actions)  # [seq_len, action_dim]

        # 速度变化（一阶差分）
        velocities = action_tensor[1:] - action_tensor[:-1]
        velocity_variation = torch.std(velocities, dim=0).mean()
        metrics['velocity_variation'] = float(velocity_variation.item())

        # 加速度变化（二阶差分）
        if len(velocities) > 1:
            accelerations = velocities[1:] - velocities[:-1]
            acceleration_variation = torch.std(accelerations, dim=0).mean()
            metrics['acceleration_variation'] = float(acceleration_variation.item())

            # Jerk（三阶差分）
            if len(accelerations) > 1:
                jerks = accelerations[1:] - accelerations[:-1]
                jerk_variation = torch.std(jerks, dim=0).mean()
                metrics['jerk_variation'] = float(jerk_variation.item())

        return metrics

    def _calculate_consistency_metrics(self, episode_predictions: List[torch.Tensor],
                                     episode_ground_truth: List[torch.Tensor]) -> Dict[str, float]:
        """计算动作一致性指标"""
        if not self.config.diffusion_evaluation.action_consistency.enable:
            return {}

        metrics = {}
        window_size = self.config.diffusion_evaluation.action_consistency.consistency_window

        if len(episode_predictions) < window_size:
            return metrics

        # 时序相关性分析
        if self.config.diffusion_evaluation.action_consistency.temporal_correlation:
            pred_tensor = torch.stack(episode_predictions)
            true_tensor = torch.stack(episode_ground_truth)

            # 计算相邻时刻的相关性
            correlations = []
            for i in range(len(pred_tensor) - 1):
                pred_corr = torch.corrcoef(torch.stack([pred_tensor[i], pred_tensor[i+1]]))[0, 1]
                true_corr = torch.corrcoef(torch.stack([true_tensor[i], true_tensor[i+1]]))[0, 1]

                if not torch.isnan(pred_corr) and not torch.isnan(true_corr):
                    correlations.append(abs(pred_corr - true_corr))

            if correlations:
                metrics['temporal_correlation_error'] = float(np.mean(correlations))

        return metrics

    def _analyze_inference_speed(self, inference_times: List[float]) -> Dict[str, float]:
        """分析推理速度"""
        if not inference_times:
            return {}

        metrics = {
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'steps_per_second': 1000.0 / np.mean(inference_times)  # 从毫秒转换
        }

        return metrics

    def evaluate_single_episode(self, episode_data: List[Dict]) -> Dict[str, Any]:
        """评估单个episode（覆盖基类方法以添加diffusion特有分析）"""
        # 调用基类方法获取基础结果
        episode_result = super().evaluate_single_episode(episode_data)

        # 添加diffusion特有的分析
        episode_predictions = []
        episode_ground_truth = []
        inference_times = []

        for step_data in episode_data:
            # 重新进行推理以获取diffusion特有信息
            observation = {}
            for key, value in step_data.items():
                if key.startswith('observation.'):
                    observation[key] = value.unsqueeze(0).to(self.device)

            true_action = step_data['action'].to(self.device)

            try:
                predicted_action, inference_info = self._model_inference(observation)
                episode_predictions.append(predicted_action.squeeze(0))
                episode_ground_truth.append(true_action)
                inference_times.append(inference_info.get('inference_time', 0))

            except Exception as e:
                self.logger.error(f"Failed to re-infer for diffusion analysis: {e}")
                continue

        # 计算diffusion特有指标
        if episode_predictions:
            # 平滑度分析
            smoothness_metrics = self._calculate_smoothness_metrics(episode_predictions)
            episode_result.update(smoothness_metrics)

            # 一致性分析
            consistency_metrics = self._calculate_consistency_metrics(
                episode_predictions, episode_ground_truth
            )
            episode_result.update(consistency_metrics)

            # 推理速度分析
            speed_metrics = self._analyze_inference_speed(inference_times)
            episode_result.update(speed_metrics)

        return episode_result

    def _test_different_inference_steps(self, sample_observation: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """测试不同推理步数的效果"""
        if not self.config.diffusion_evaluation.inference_steps_analysis.enable:
            return {}

        test_steps = self.config.diffusion_evaluation.inference_steps_analysis.test_different_steps
        step_results = {}

        for num_steps in test_steps:
            try:
                # 保存原始设置
                original_steps = getattr(self.model, 'num_inference_steps', None)

                # 设置新的推理步数
                if hasattr(self.model, 'scheduler'):
                    self.model.scheduler.set_timesteps(num_steps)

                start_time = time.time()
                with torch.no_grad():
                    action = self.model.select_action(sample_observation)
                inference_time = (time.time() - start_time) * 1000

                step_results[f'steps_{num_steps}'] = {
                    'inference_time': inference_time,
                    'action_norm': float(torch.norm(action).item())
                }

                # 恢复原始设置
                if original_steps and hasattr(self.model, 'scheduler'):
                    self.model.scheduler.set_timesteps(original_steps)

            except Exception as e:
                self.logger.warning(f"Failed to test {num_steps} inference steps: {e}")

        return step_results

    def evaluate(self) -> EvaluationResults:
        """执行传统diffusion评估"""
        self.logger.info("Starting diffusion evaluation...")

        # 加载模型和数据
        self.load_model()
        self.prepare_data()

        # 重置统计
        self.denoising_stats.clear()
        self.inference_steps_stats.clear()
        self.smoothness_stats.clear()
        self.consistency_stats.clear()

        all_episode_results = []

        # 测试不同推理步数（使用第一个样本）
        if self.config.diffusion_evaluation.inference_steps_analysis.enable and len(self.test_dataset) > 0:
            sample = self.test_dataset[0]
            sample_observation = {}
            for key, value in sample.items():
                if key.startswith('observation.'):
                    sample_observation[key] = value.unsqueeze(0).to(self.device)

            steps_analysis = self._test_different_inference_steps(sample_observation)
            self.logger.info(f"Inference steps analysis: {steps_analysis}")

        # 按episode组织数据并评估
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

        # 汇总diffusion特有指标
        diffusion_metric_names = ['velocity_variation', 'acceleration_variation', 'jerk_variation',
                                'temporal_correlation_error', 'avg_inference_time', 'steps_per_second']

        for metric_name in diffusion_metric_names:
            values = []
            for episode_result in all_episode_results:
                if metric_name in episode_result:
                    values.append(episode_result[metric_name])

            if values:
                performance_metrics[f'overall_{metric_name}'] = np.mean(values)
                performance_metrics[f'overall_{metric_name}_std'] = np.std(values)

        # 生成摘要
        summary = {
            'model_type': 'diffusion',
            'total_episodes_evaluated': len(all_episode_results),
            'total_steps': sum(r['num_steps'] for r in all_episode_results),
            'avg_episode_length': np.mean([r['num_steps'] for r in all_episode_results]) if all_episode_results else 0,
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        results = EvaluationResults(
            action_metrics=action_metrics,
            performance_metrics=performance_metrics,
            episode_results=all_episode_results,
            summary=summary,
            timing_info={'total_evaluation_time': time.time()}
        )

        self.logger.info(f"Diffusion evaluation completed: {len(all_episode_results)} episodes")

        return results

    def _generate_plots(self, results: EvaluationResults) -> None:
        """生成diffusion特有的可视化图表"""
        try:
            import matplotlib.pyplot as plt

            # 调用基类的绘图方法
            super()._generate_plots(results)

            # Diffusion特有的图表
            if self.config.diffusion_visualization.plot_denoising_process:
                self._plot_denoising_process(results)

            if self.config.diffusion_visualization.plot_action_trajectories:
                self._plot_action_trajectories(results)

            if self.config.diffusion_visualization.plot_smoothness_analysis:
                self._plot_smoothness_analysis(results)

            if self.config.diffusion_visualization.plot_inference_speed:
                self._plot_inference_speed(results)

        except ImportError:
            self.logger.warning("Matplotlib not available, skipping diffusion plot generation")
        except Exception as e:
            self.logger.error(f"Error generating diffusion plots: {e}")

    def _plot_denoising_process(self, results: EvaluationResults) -> None:
        """绘制去噪过程图"""
        # 这里可以实现去噪过程的可视化
        # 由于需要去噪过程的中间状态，这里提供基础框架
        pass

    def _plot_action_trajectories(self, results: EvaluationResults) -> None:
        """绘制动作轨迹图"""
        import matplotlib.pyplot as plt

        # 选择前几个episode的轨迹进行可视化
        num_episodes_to_plot = min(3, len(results.episode_results))

        plt.figure(figsize=(12, 8))

        for i in range(num_episodes_to_plot):
            plt.subplot(num_episodes_to_plot, 1, i+1)
            # 这里需要从episode结果中提取轨迹数据
            # 由于没有保存完整轨迹，这里提供框架
            plt.title(f'Episode {i+1} Action Trajectory')
            plt.xlabel('Time Step')
            plt.ylabel('Action Value')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'action_trajectories.png',
                   dpi=self.config.visualization.dpi)
        plt.close()

    def _plot_smoothness_analysis(self, results: EvaluationResults) -> None:
        """绘制平滑度分析图"""
        import matplotlib.pyplot as plt

        # 提取平滑度指标
        velocity_variations = []
        acceleration_variations = []

        for episode_result in results.episode_results:
            if 'velocity_variation' in episode_result:
                velocity_variations.append(episode_result['velocity_variation'])
            if 'acceleration_variation' in episode_result:
                acceleration_variations.append(episode_result['acceleration_variation'])

        if velocity_variations or acceleration_variations:
            plt.figure(figsize=(12, 5))

            if velocity_variations:
                plt.subplot(1, 2, 1)
                plt.hist(velocity_variations, bins=15, alpha=0.7, edgecolor='black')
                plt.xlabel('Velocity Variation')
                plt.ylabel('Frequency')
                plt.title('Velocity Variation Distribution')
                plt.grid(True, alpha=0.3)

            if acceleration_variations:
                plt.subplot(1, 2, 2)
                plt.hist(acceleration_variations, bins=15, alpha=0.7, edgecolor='black')
                plt.xlabel('Acceleration Variation')
                plt.ylabel('Frequency')
                plt.title('Acceleration Variation Distribution')
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'smoothness_analysis.png',
                       dpi=self.config.visualization.dpi)
            plt.close()

    def _plot_inference_speed(self, results: EvaluationResults) -> None:
        """绘制推理速度分析图"""
        import matplotlib.pyplot as plt

        inference_times = []
        for episode_result in results.episode_results:
            if 'avg_inference_time' in episode_result:
                inference_times.append(episode_result['avg_inference_time'])

        if inference_times:
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(inference_times)), inference_times, 'b-', alpha=0.7)
            plt.axhline(np.mean(inference_times), color='red', linestyle='--',
                       label=f'Mean: {np.mean(inference_times):.2f}ms')
            plt.xlabel('Episode')
            plt.ylabel('Average Inference Time (ms)')
            plt.title('Inference Speed Over Episodes')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'inference_speed.png',
                       dpi=self.config.visualization.dpi)
            plt.close()