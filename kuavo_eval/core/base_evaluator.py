# -*- coding: utf-8 -*-
"""
基础评估器模块

定义所有评估器的通用接口和基础功能
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import lerobot_patches.custom_patches

import torch
import numpy as np
import time
import json
import csv
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
import logging
from datetime import datetime

from kuavo_train.wrapper.dataset.LeRobotDatasetWrapper import CustomLeRobotDataset
from lerobot.utils.random_utils import set_seed

@dataclass
class EvaluationResults:
    """评估结果数据类"""
    action_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    episode_results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    timing_info: Dict[str, float]

class BaseEvaluator(ABC):
    """
    基础评估器抽象类

    定义所有评估器必须实现的接口和通用功能
    """

    def __init__(self, config: DictConfig):
        """
        初始化评估器

        Args:
            config: 评估配置
        """
        self.config = config
        self.model = None
        self.test_dataset = None
        self.device = torch.device(config.common.device)
        self.output_dir = Path(config.common.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置随机种子
        set_seed(config.common.seed)

        # 设置日志
        self.logger = self._setup_logger()

        # 初始化结果存储
        self.results = {
            'action_metrics': {},
            'performance_metrics': {},
            'episode_results': [],
            'timing_info': {}
        }

        self.logger.info(f"Initialized {self.__class__.__name__} with config: {config.model.type}")

    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(getattr(logging, self.config.logging.level))

        # 清除现有的handlers
        logger.handlers.clear()

        # 控制台handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # 文件handler（如果配置了保存日志）
        if self.config.logging.save_logs:
            log_file = self.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)

        return logger

    @abstractmethod
    def load_model(self) -> None:
        """
        加载模型（子类必须实现）
        """
        pass

    def prepare_data(self) -> None:
        """
        准备测试数据
        """
        self.logger.info("Preparing test dataset...")

        # 构建episodes列表
        start_ep, end_ep = self.config.test_data.episodes_range
        episodes = list(range(start_ep, min(end_ep + 1, start_ep + self.config.test_data.max_episodes)))

        self.logger.info(f"Loading episodes {episodes} from {self.config.test_data.root}")

        # 创建数据集
        self.test_dataset = CustomLeRobotDataset(
            repo_id=self.config.test_data.repo_id,
            root=self.config.test_data.root,
            episodes=episodes,
        )

        self.logger.info(f"Dataset loaded with {len(self.test_dataset)} samples")

    def _calculate_action_metrics(self, predicted_actions: torch.Tensor,
                                true_actions: torch.Tensor) -> Dict[str, float]:
        """
        计算动作精度指标

        Args:
            predicted_actions: 预测动作 [batch_size, action_dim]
            true_actions: 真实动作 [batch_size, action_dim]

        Returns:
            指标字典
        """
        metrics = {}

        # 确保在CPU上计算
        pred = predicted_actions.detach().cpu()
        true = true_actions.detach().cpu()

        # MSE
        if 'mse' in self.config.evaluation.action_metrics:
            mse = torch.nn.functional.mse_loss(pred, true)
            metrics['mse'] = float(mse.item())

        # MAE
        if 'mae' in self.config.evaluation.action_metrics:
            mae = torch.nn.functional.l1_loss(pred, true)
            metrics['mae'] = float(mae.item())

        # 余弦相似度
        if 'cosine_sim' in self.config.evaluation.action_metrics:
            cosine_sim = torch.nn.functional.cosine_similarity(pred, true, dim=-1).mean()
            metrics['cosine_sim'] = float(cosine_sim.item())

        # L2范数差异
        if 'l2_norm' in self.config.evaluation.action_metrics:
            l2_norm = torch.norm(pred - true, p=2, dim=-1).mean()
            metrics['l2_norm'] = float(l2_norm.item())

        return metrics

    def _calculate_joint_metrics(self, predicted_actions: torch.Tensor,
                               true_actions: torch.Tensor) -> Dict[str, float]:
        """
        计算关节特定指标

        Args:
            predicted_actions: 预测动作
            true_actions: 真实动作

        Returns:
            关节指标字典
        """
        if not self.config.evaluation.joint_analysis.enable:
            return {}

        metrics = {}
        pred = predicted_actions.detach().cpu()
        true = true_actions.detach().cpu()

        # 每个关节的误差
        joint_errors = torch.abs(pred - true)  # [batch_size, num_joints]
        joint_weights = torch.tensor(self.config.evaluation.joint_analysis.joint_weights)

        # 加权误差
        weighted_errors = joint_errors * joint_weights.unsqueeze(0)
        metrics['weighted_joint_error'] = float(weighted_errors.mean().item())

        # 关键关节误差
        critical_joints = self.config.evaluation.joint_analysis.critical_joints
        if critical_joints:
            critical_errors = joint_errors[:, critical_joints].mean()
            metrics['critical_joint_error'] = float(critical_errors.item())

        return metrics

    def _calculate_temporal_metrics(self, episode_predictions: List[torch.Tensor],
                                  episode_ground_truth: List[torch.Tensor]) -> Dict[str, float]:
        """
        计算时序一致性指标

        Args:
            episode_predictions: episode中的预测序列
            episode_ground_truth: episode中的真实序列

        Returns:
            时序指标字典
        """
        if not self.config.evaluation.temporal_consistency.enable:
            return {}

        metrics = {}
        window_size = self.config.evaluation.temporal_consistency.window_size

        if len(episode_predictions) < window_size:
            return metrics

        # 计算连续动作的平滑度
        pred_actions = torch.stack(episode_predictions)  # [seq_len, action_dim]
        true_actions = torch.stack(episode_ground_truth)

        # 速度平滑度（相邻时刻动作差异）
        pred_velocities = pred_actions[1:] - pred_actions[:-1]
        true_velocities = true_actions[1:] - true_actions[:-1]

        velocity_error = torch.nn.functional.mse_loss(pred_velocities, true_velocities)
        metrics['velocity_smoothness'] = float(velocity_error.item())

        # 加速度平滑度
        if len(pred_velocities) > 1:
            pred_accelerations = pred_velocities[1:] - pred_velocities[:-1]
            true_accelerations = true_velocities[1:] - true_velocities[:-1]

            acceleration_error = torch.nn.functional.mse_loss(pred_accelerations, true_accelerations)
            metrics['acceleration_smoothness'] = float(acceleration_error.item())

        return metrics

    @abstractmethod
    def _model_inference(self, observation: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        模型推理（子类必须实现）

        Args:
            observation: 观测数据

        Returns:
            Tuple[预测动作, 推理信息]
        """
        pass

    def evaluate_single_episode(self, episode_data: List[Dict]) -> Dict[str, Any]:
        """
        评估单个episode

        Args:
            episode_data: episode数据列表

        Returns:
            episode评估结果
        """
        episode_predictions = []
        episode_ground_truth = []
        episode_metrics = []
        inference_times = []

        self.logger.debug(f"Evaluating episode with {len(episode_data)} steps")

        for step_idx, step_data in enumerate(episode_data):
            # 准备观测数据
            observation = {}
            for key, value in step_data.items():
                if key.startswith('observation.'):
                    observation[key] = value.unsqueeze(0).to(self.device)

            # 获取真实动作
            true_action = step_data['action'].to(self.device)

            # 模型推理
            start_time = time.time()
            try:
                predicted_action, inference_info = self._model_inference(observation)
                inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            except Exception as e:
                self.logger.error(f"Inference failed at step {step_idx}: {e}")
                continue

            # 记录结果
            episode_predictions.append(predicted_action.squeeze(0))
            episode_ground_truth.append(true_action)
            inference_times.append(inference_time)

            # 计算步骤级指标
            step_metrics = self._calculate_action_metrics(predicted_action, true_action.unsqueeze(0))
            step_metrics.update(self._calculate_joint_metrics(predicted_action, true_action.unsqueeze(0)))
            step_metrics['inference_time'] = inference_time
            step_metrics.update(inference_info)

            episode_metrics.append(step_metrics)

        # 计算episode级指标
        episode_result = {
            'num_steps': len(episode_predictions),
            'average_inference_time': np.mean(inference_times) if inference_times else 0,
            'total_inference_time': np.sum(inference_times) if inference_times else 0,
        }

        # 汇总动作指标
        if episode_metrics:
            for metric_name in self.config.evaluation.action_metrics:
                if metric_name in episode_metrics[0]:
                    values = [m[metric_name] for m in episode_metrics if metric_name in m]
                    episode_result[f'avg_{metric_name}'] = np.mean(values)
                    episode_result[f'std_{metric_name}'] = np.std(values)

        # 计算时序指标
        temporal_metrics = self._calculate_temporal_metrics(episode_predictions, episode_ground_truth)
        episode_result.update(temporal_metrics)

        return episode_result

    @abstractmethod
    def evaluate(self) -> EvaluationResults:
        """
        执行完整评估（子类必须实现）
        """
        pass

    def save_results(self, results: EvaluationResults) -> None:
        """
        保存评估结果

        Args:
            results: 评估结果
        """
        if not self.config.common.save_results:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存JSON格式
        if 'json' in self.config.report.format:
            json_file = self.output_dir / f"evaluation_results_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                # 转换为可序列化的格式
                serializable_results = self._make_serializable(results.__dict__)
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Results saved to {json_file}")

        # 保存CSV格式
        if 'csv' in self.config.report.format:
            csv_file = self.output_dir / f"evaluation_summary_{timestamp}.csv"
            self._save_csv_summary(results, csv_file)
            self.logger.info(f"Summary saved to {csv_file}")

    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    def _save_csv_summary(self, results: EvaluationResults, csv_file: Path) -> None:
        """保存CSV格式的摘要"""
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 写入标题行
            writer.writerow(['Metric', 'Value'])

            # 写入动作指标
            writer.writerow(['=== Action Metrics ===', ''])
            for metric, value in results.action_metrics.items():
                writer.writerow([metric, value])

            # 写入性能指标
            writer.writerow(['=== Performance Metrics ===', ''])
            for metric, value in results.performance_metrics.items():
                writer.writerow([metric, value])

            # 写入摘要
            writer.writerow(['=== Summary ===', ''])
            for metric, value in results.summary.items():
                writer.writerow([metric, value])

    def generate_report(self, results: EvaluationResults) -> None:
        """
        生成评估报告

        Args:
            results: 评估结果
        """
        self.logger.info("Generating evaluation report...")

        # 打印摘要到控制台
        self._print_summary(results)

        # 保存结果
        self.save_results(results)

        # 生成可视化（如果配置了）
        if self.config.report.include_plots:
            self._generate_plots(results)

    def _print_summary(self, results: EvaluationResults) -> None:
        """打印评估摘要"""
        print("\n" + "="*60)
        print(f"EVALUATION SUMMARY - {self.__class__.__name__}")
        print("="*60)

        print("\n📊 Action Metrics:")
        for metric, value in results.action_metrics.items():
            print(f"  {metric}: {value:.6f}")

        print("\n⚡ Performance Metrics:")
        for metric, value in results.performance_metrics.items():
            print(f"  {metric}: {value:.3f}")

        print("\n📈 Summary:")
        for metric, value in results.summary.items():
            print(f"  {metric}: {value}")

        print("="*60 + "\n")

    def _generate_plots(self, results: EvaluationResults) -> None:
        """生成可视化图表（基础版本，子类可以覆盖）"""
        try:
            import matplotlib.pyplot as plt

            if self.config.visualization.plot_action_comparison:
                self._plot_action_comparison(results)

            if self.config.visualization.plot_error_distribution:
                self._plot_error_distribution(results)

        except ImportError:
            self.logger.warning("Matplotlib not available, skipping plot generation")
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")

    def _plot_action_comparison(self, results: EvaluationResults) -> None:
        """绘制动作对比图（基础版本）"""
        # 子类可以实现具体的绘图逻辑
        pass

    def _plot_error_distribution(self, results: EvaluationResults) -> None:
        """绘制误差分布图（基础版本）"""
        # 子类可以实现具体的绘图逻辑
        pass