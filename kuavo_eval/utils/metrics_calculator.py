# -*- coding: utf-8 -*-
"""
评估指标计算工具模块

提供各种评估指标的计算功能
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
import warnings

class MetricsCalculator:
    """评估指标计算器"""

    @staticmethod
    def calculate_action_accuracy_metrics(predicted: torch.Tensor,
                                        ground_truth: torch.Tensor,
                                        metrics_list: List[str]) -> Dict[str, float]:
        """
        计算动作精度指标

        Args:
            predicted: 预测动作 [batch_size, action_dim] 或 [action_dim]
            ground_truth: 真实动作 [batch_size, action_dim] 或 [action_dim]
            metrics_list: 要计算的指标列表

        Returns:
            指标字典
        """
        metrics = {}

        # 确保张量形状一致
        if predicted.dim() == 1:
            predicted = predicted.unsqueeze(0)
        if ground_truth.dim() == 1:
            ground_truth = ground_truth.unsqueeze(0)

        pred = predicted.detach().cpu().float()
        true = ground_truth.detach().cpu().float()

        # MSE (均方误差)
        if 'mse' in metrics_list:
            mse = torch.nn.functional.mse_loss(pred, true)
            metrics['mse'] = float(mse.item())

        # MAE (平均绝对误差)
        if 'mae' in metrics_list:
            mae = torch.nn.functional.l1_loss(pred, true)
            metrics['mae'] = float(mae.item())

        # RMSE (均方根误差)
        if 'rmse' in metrics_list:
            rmse = torch.sqrt(torch.nn.functional.mse_loss(pred, true))
            metrics['rmse'] = float(rmse.item())

        # 余弦相似度
        if 'cosine_sim' in metrics_list:
            # 对每个样本计算余弦相似度，然后取平均
            cosine_sims = []
            for i in range(pred.shape[0]):
                cos_sim = torch.nn.functional.cosine_similarity(
                    pred[i:i+1], true[i:i+1], dim=-1
                )
                if not torch.isnan(cos_sim):
                    cosine_sims.append(cos_sim.item())

            if cosine_sims:
                metrics['cosine_sim'] = float(np.mean(cosine_sims))
            else:
                metrics['cosine_sim'] = 0.0

        # L2范数差异
        if 'l2_norm' in metrics_list:
            l2_norm = torch.norm(pred - true, p=2, dim=-1).mean()
            metrics['l2_norm'] = float(l2_norm.item())

        # 相对误差 (百分比)
        if 'relative_error' in metrics_list:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                relative_error = torch.abs((pred - true) / (true + 1e-8)).mean()
                metrics['relative_error'] = float(relative_error.item()) * 100

        # 最大误差
        if 'max_error' in metrics_list:
            max_error = torch.max(torch.abs(pred - true))
            metrics['max_error'] = float(max_error.item())

        return metrics

    @staticmethod
    def calculate_joint_specific_metrics(predicted: torch.Tensor,
                                       ground_truth: torch.Tensor,
                                       joint_weights: List[float],
                                       critical_joints: List[int]) -> Dict[str, float]:
        """
        计算关节特定指标

        Args:
            predicted: 预测动作
            ground_truth: 真实动作
            joint_weights: 关节权重
            critical_joints: 关键关节索引列表

        Returns:
            关节特定指标字典
        """
        metrics = {}

        if predicted.dim() == 1:
            predicted = predicted.unsqueeze(0)
        if ground_truth.dim() == 1:
            ground_truth = ground_truth.unsqueeze(0)

        pred = predicted.detach().cpu().float()
        true = ground_truth.detach().cpu().float()

        # 每个关节的绝对误差
        joint_errors = torch.abs(pred - true)  # [batch_size, num_joints]

        # 加权关节误差
        if joint_weights and len(joint_weights) == joint_errors.shape[-1]:
            weights = torch.tensor(joint_weights, dtype=torch.float32)
            weighted_errors = joint_errors * weights.unsqueeze(0)
            metrics['weighted_joint_error'] = float(weighted_errors.mean().item())

            # 每个关节的平均误差
            for i, weight in enumerate(joint_weights):
                metrics[f'joint_{i}_error'] = float(joint_errors[:, i].mean().item())

        # 关键关节误差
        if critical_joints:
            valid_joints = [j for j in critical_joints if j < joint_errors.shape[-1]]
            if valid_joints:
                critical_errors = joint_errors[:, valid_joints]
                metrics['critical_joint_error'] = float(critical_errors.mean().item())
                metrics['critical_joint_max_error'] = float(critical_errors.max().item())

        # 关节误差标准差 (关节间一致性)
        joint_error_std = torch.std(joint_errors, dim=-1).mean()
        metrics['joint_consistency'] = float(joint_error_std.item())

        return metrics

    @staticmethod
    def calculate_temporal_metrics(action_sequence: List[torch.Tensor],
                                 true_sequence: List[torch.Tensor],
                                 window_size: int = 5) -> Dict[str, float]:
        """
        计算时序指标

        Args:
            action_sequence: 预测动作序列
            true_sequence: 真实动作序列
            window_size: 时间窗口大小

        Returns:
            时序指标字典
        """
        metrics = {}

        if len(action_sequence) < 2 or len(true_sequence) < 2:
            return metrics

        if len(action_sequence) != len(true_sequence):
            min_len = min(len(action_sequence), len(true_sequence))
            action_sequence = action_sequence[:min_len]
            true_sequence = true_sequence[:min_len]

        pred_tensor = torch.stack([a.detach().cpu().float() for a in action_sequence])
        true_tensor = torch.stack([a.detach().cpu().float() for a in true_sequence])

        # 速度计算 (一阶差分)
        pred_velocities = pred_tensor[1:] - pred_tensor[:-1]
        true_velocities = true_tensor[1:] - true_tensor[:-1]

        # 速度误差
        velocity_error = torch.nn.functional.mse_loss(pred_velocities, true_velocities)
        metrics['velocity_mse'] = float(velocity_error.item())

        # 速度平滑度 (速度变化的标准差)
        pred_velocity_smoothness = torch.std(pred_velocities, dim=0).mean()
        true_velocity_smoothness = torch.std(true_velocities, dim=0).mean()
        metrics['velocity_smoothness_error'] = float(abs(pred_velocity_smoothness - true_velocity_smoothness).item())

        # 加速度计算 (二阶差分)
        if len(pred_velocities) > 1:
            pred_accelerations = pred_velocities[1:] - pred_velocities[:-1]
            true_accelerations = true_velocities[1:] - true_velocities[:-1]

            acceleration_error = torch.nn.functional.mse_loss(pred_accelerations, true_accelerations)
            metrics['acceleration_mse'] = float(acceleration_error.item())

            # 加速度平滑度
            pred_acc_smoothness = torch.std(pred_accelerations, dim=0).mean()
            true_acc_smoothness = torch.std(true_accelerations, dim=0).mean()
            metrics['acceleration_smoothness_error'] = float(abs(pred_acc_smoothness - true_acc_smoothness).item())

        # 时序相关性
        if len(action_sequence) >= window_size:
            correlations = []
            for i in range(len(action_sequence) - window_size + 1):
                pred_window = pred_tensor[i:i+window_size].flatten()
                true_window = true_tensor[i:i+window_size].flatten()

                try:
                    corr, _ = stats.pearsonr(pred_window.numpy(), true_window.numpy())
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    pass

            if correlations:
                metrics['temporal_correlation'] = float(np.mean(correlations))
                metrics['temporal_correlation_std'] = float(np.std(correlations))

        # Jerk计算 (三阶差分)
        if len(pred_tensor) >= 4:  # 至少需要4个点来计算jerk
            pred_jerk = pred_tensor[3:] - 3*pred_tensor[2:-1] + 3*pred_tensor[1:-2] - pred_tensor[:-3]
            true_jerk = true_tensor[3:] - 3*true_tensor[2:-1] + 3*true_tensor[1:-2] - true_tensor[:-3]

            jerk_error = torch.nn.functional.mse_loss(pred_jerk, true_jerk)
            metrics['jerk_mse'] = float(jerk_error.item())

        return metrics

    @staticmethod
    def calculate_distribution_metrics(values: List[float]) -> Dict[str, float]:
        """
        计算数值分布的统计指标

        Args:
            values: 数值列表

        Returns:
            分布指标字典
        """
        if not values:
            return {}

        metrics = {}
        values_array = np.array(values)

        # 基础统计量
        metrics['mean'] = float(np.mean(values_array))
        metrics['std'] = float(np.std(values_array))
        metrics['min'] = float(np.min(values_array))
        metrics['max'] = float(np.max(values_array))
        metrics['median'] = float(np.median(values_array))

        # 百分位数
        metrics['q25'] = float(np.percentile(values_array, 25))
        metrics['q75'] = float(np.percentile(values_array, 75))
        metrics['q95'] = float(np.percentile(values_array, 95))

        # 分布形状
        if len(values_array) > 1:
            metrics['skewness'] = float(stats.skew(values_array))
            metrics['kurtosis'] = float(stats.kurtosis(values_array))

        # 变异系数
        if metrics['mean'] != 0:
            metrics['coefficient_of_variation'] = metrics['std'] / abs(metrics['mean'])

        return metrics

    @staticmethod
    def calculate_performance_metrics(inference_times: List[float],
                                    budget_ms: Optional[float] = None) -> Dict[str, float]:
        """
        计算性能指标

        Args:
            inference_times: 推理时间列表 (毫秒)
            budget_ms: 时间预算 (毫秒)

        Returns:
            性能指标字典
        """
        if not inference_times:
            return {}

        metrics = {}
        times_array = np.array(inference_times)

        # 基础性能指标
        metrics['avg_inference_time'] = float(np.mean(times_array))
        metrics['std_inference_time'] = float(np.std(times_array))
        metrics['min_inference_time'] = float(np.min(times_array))
        metrics['max_inference_time'] = float(np.max(times_array))
        metrics['median_inference_time'] = float(np.median(times_array))

        # 吞吐量指标
        metrics['steps_per_second'] = 1000.0 / metrics['avg_inference_time']  # 从毫秒转换

        # 预算相关指标
        if budget_ms is not None:
            within_budget = times_array <= budget_ms
            metrics['budget_compliance_rate'] = float(np.mean(within_budget))
            metrics['budget_violation_count'] = int(np.sum(~within_budget))
            metrics['avg_budget_excess'] = float(np.mean(np.maximum(0, times_array - budget_ms)))

            # 预算利用率
            budget_utilization = times_array / budget_ms
            metrics['avg_budget_utilization'] = float(np.mean(budget_utilization))
            metrics['max_budget_utilization'] = float(np.max(budget_utilization))

        return metrics

    @staticmethod
    def calculate_robustness_metrics(predictions: List[torch.Tensor],
                                   ground_truth: List[torch.Tensor],
                                   noise_levels: List[float] = None) -> Dict[str, float]:
        """
        计算鲁棒性指标

        Args:
            predictions: 预测结果列表
            ground_truth: 真实值列表
            noise_levels: 噪声水平列表

        Returns:
            鲁棒性指标字典
        """
        metrics = {}

        if not predictions or not ground_truth:
            return metrics

        # 预测一致性 (多次预测的方差)
        if len(predictions) > 1:
            pred_tensor = torch.stack([p.detach().cpu().float() for p in predictions])
            prediction_variance = torch.var(pred_tensor, dim=0).mean()
            metrics['prediction_variance'] = float(prediction_variance.item())

            # 预测稳定性 (相邻预测的差异)
            pred_diffs = []
            for i in range(len(predictions) - 1):
                diff = torch.nn.functional.mse_loss(
                    predictions[i].detach().cpu().float(),
                    predictions[i+1].detach().cpu().float()
                )
                pred_diffs.append(diff.item())

            if pred_diffs:
                metrics['prediction_stability'] = float(np.mean(pred_diffs))

        # 误差分布的稳定性
        errors = []
        for pred, true in zip(predictions, ground_truth):
            error = torch.nn.functional.mse_loss(
                pred.detach().cpu().float(),
                true.detach().cpu().float()
            )
            errors.append(error.item())

        if errors:
            error_std = np.std(errors)
            error_mean = np.mean(errors)
            if error_mean != 0:
                metrics['error_coefficient_of_variation'] = error_std / error_mean

        return metrics