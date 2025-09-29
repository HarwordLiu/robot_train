# -*- coding: utf-8 -*-
"""
评估报告生成器模块

提供详细的评估报告生成功能
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available, plotting disabled")

class EvaluationReportGenerator:
    """评估报告生成器"""

    def __init__(self, output_dir: Path, config: Optional[Dict] = None):
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录
            config: 配置信息
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}

        # 报告设置
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.report_data = {}

        # 绘图设置
        if PLOTTING_AVAILABLE:
            plt.style.use('default')
            sns.set_palette("husl")

    def generate_comprehensive_report(self, results: Dict[str, Any],
                                    model_type: str = 'unknown') -> Dict[str, Path]:
        """
        生成全面的评估报告

        Args:
            results: 评估结果
            model_type: 模型类型

        Returns:
            生成的文件路径字典
        """
        generated_files = {}

        # 生成JSON报告
        json_file = self._generate_json_report(results, model_type)
        generated_files['json'] = json_file

        # 生成CSV摘要
        csv_file = self._generate_csv_summary(results, model_type)
        generated_files['csv'] = csv_file

        # 生成Markdown报告
        md_file = self._generate_markdown_report(results, model_type)
        generated_files['markdown'] = md_file

        # 生成可视化图表
        if PLOTTING_AVAILABLE and self.config.get('include_plots', True):
            plot_files = self._generate_plots(results, model_type)
            generated_files.update(plot_files)

        return generated_files

    def _generate_json_report(self, results: Dict[str, Any], model_type: str) -> Path:
        """生成JSON格式的详细报告"""
        json_file = self.output_dir / f"{model_type}_evaluation_report_{self.timestamp}.json"

        # 准备JSON数据
        json_data = {
            'metadata': {
                'timestamp': self.timestamp,
                'model_type': model_type,
                'evaluation_config': self.config
            },
            'summary': results.get('summary', {}),
            'action_metrics': results.get('action_metrics', {}),
            'performance_metrics': results.get('performance_metrics', {}),
            'episode_results': results.get('episode_results', []),
            'timing_info': results.get('timing_info', {})
        }

        # 确保数据可序列化
        json_data = self._make_serializable(json_data)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        return json_file

    def _generate_csv_summary(self, results: Dict[str, Any], model_type: str) -> Path:
        """生成CSV格式的摘要报告"""
        csv_file = self.output_dir / f"{model_type}_evaluation_summary_{self.timestamp}.csv"

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 标题
            writer.writerow(['Evaluation Summary', f'{model_type} Model'])
            writer.writerow(['Timestamp', self.timestamp])
            writer.writerow([])

            # 动作指标
            writer.writerow(['=== Action Metrics ==='])
            action_metrics = results.get('action_metrics', {})
            for metric, value in action_metrics.items():
                writer.writerow([metric, self._format_value(value)])
            writer.writerow([])

            # 性能指标
            writer.writerow(['=== Performance Metrics ==='])
            performance_metrics = results.get('performance_metrics', {})
            for metric, value in performance_metrics.items():
                writer.writerow([metric, self._format_value(value)])
            writer.writerow([])

            # 摘要信息
            writer.writerow(['=== Summary ==='])
            summary = results.get('summary', {})
            for key, value in summary.items():
                writer.writerow([key, self._format_value(value)])

            # Episode详细信息
            writer.writerow([])
            writer.writerow(['=== Episode Details ==='])
            episode_results = results.get('episode_results', [])
            if episode_results:
                # 获取所有可能的列名
                all_columns = set()
                for episode in episode_results:
                    all_columns.update(episode.keys())

                # 写入标题行
                columns = ['episode_idx'] + sorted(list(all_columns))
                writer.writerow(columns)

                # 写入数据行
                for i, episode in enumerate(episode_results):
                    row = [i]
                    for col in columns[1:]:
                        value = episode.get(col, '')
                        row.append(self._format_value(value))
                    writer.writerow(row)

        return csv_file

    def _generate_markdown_report(self, results: Dict[str, Any], model_type: str) -> Path:
        """生成Markdown格式的报告"""
        md_file = self.output_dir / f"{model_type}_evaluation_report_{self.timestamp}.md"

        with open(md_file, 'w', encoding='utf-8') as f:
            # 标题
            f.write(f"# {model_type.upper()} Model Evaluation Report\n\n")
            f.write(f"**Generated:** {self.timestamp}\n\n")

            # 执行摘要
            f.write("## Executive Summary\n\n")
            summary = results.get('summary', {})
            for key, value in summary.items():
                f.write(f"- **{key.replace('_', ' ').title()}:** {self._format_value(value)}\n")
            f.write("\n")

            # 动作精度指标
            f.write("## Action Accuracy Metrics\n\n")
            action_metrics = results.get('action_metrics', {})
            if action_metrics:
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for metric, value in action_metrics.items():
                    f.write(f"| {metric.replace('_', ' ').title()} | {self._format_value(value)} |\n")
            f.write("\n")

            # 性能指标
            f.write("## Performance Metrics\n\n")
            performance_metrics = results.get('performance_metrics', {})
            if performance_metrics:
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for metric, value in performance_metrics.items():
                    f.write(f"| {metric.replace('_', ' ').title()} | {self._format_value(value)} |\n")
            f.write("\n")

            # 分层架构特有分析（如果是hierarchical模型）
            if model_type == 'humanoid_diffusion':
                f.write("## Hierarchical Architecture Analysis\n\n")
                self._write_hierarchical_analysis(f, results)

            # Diffusion特有分析（如果是diffusion模型）
            elif model_type == 'diffusion':
                f.write("## Diffusion Process Analysis\n\n")
                self._write_diffusion_analysis(f, results)

            # Episode详细分析
            f.write("## Episode Analysis\n\n")
            episode_results = results.get('episode_results', [])
            if episode_results:
                f.write(f"Total episodes evaluated: {len(episode_results)}\n\n")

                # 统计分析
                self._write_episode_statistics(f, episode_results)

            # 建议和结论
            f.write("## Recommendations\n\n")
            self._write_recommendations(f, results, model_type)

        return md_file

    def _write_hierarchical_analysis(self, f, results: Dict[str, Any]) -> None:
        """写入分层架构分析"""
        performance_metrics = results.get('performance_metrics', {})

        # 层激活分析
        f.write("### Layer Activation Analysis\n\n")
        layer_activations = {k: v for k, v in performance_metrics.items() if 'activation_rate' in k}
        if layer_activations:
            f.write("| Layer | Activation Rate |\n")
            f.write("|-------|----------------|\n")
            for layer, rate in layer_activations.items():
                layer_name = layer.replace('_activation_rate', '')
                f.write(f"| {layer_name.title()} | {rate:.3f} |\n")
        f.write("\n")

        # 延迟分析
        f.write("### Latency Analysis\n\n")
        budget_compliance = performance_metrics.get('budget_compliance_rate')
        if budget_compliance is not None:
            f.write(f"- **Budget Compliance Rate:** {budget_compliance:.3f}\n")

        avg_inference_time = performance_metrics.get('overall_avg_inference_time')
        if avg_inference_time is not None:
            f.write(f"- **Average Inference Time:** {avg_inference_time:.2f}ms\n")
        f.write("\n")

    def _write_diffusion_analysis(self, f, results: Dict[str, Any]) -> None:
        """写入diffusion分析"""
        performance_metrics = results.get('performance_metrics', {})

        # 平滑度分析
        f.write("### Trajectory Smoothness Analysis\n\n")
        smoothness_metrics = {k: v for k, v in performance_metrics.items() if 'smoothness' in k or 'variation' in k}
        if smoothness_metrics:
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for metric, value in smoothness_metrics.items():
                f.write(f"| {metric.replace('_', ' ').title()} | {self._format_value(value)} |\n")
        f.write("\n")

        # 推理速度分析
        f.write("### Inference Speed Analysis\n\n")
        steps_per_second = performance_metrics.get('overall_steps_per_second')
        if steps_per_second is not None:
            f.write(f"- **Inference Speed:** {steps_per_second:.2f} steps/second\n")
        f.write("\n")

    def _write_episode_statistics(self, f, episode_results: List[Dict]) -> None:
        """写入episode统计分析"""
        if not episode_results:
            return

        # 计算统计量
        num_steps = [ep.get('num_steps', 0) for ep in episode_results]
        inference_times = [ep.get('average_inference_time', 0) for ep in episode_results]

        f.write("### Episode Statistics\n\n")
        f.write("| Statistic | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| Average Episode Length | {np.mean(num_steps):.1f} steps |\n")
        f.write(f"| Episode Length Std | {np.std(num_steps):.1f} steps |\n")
        f.write(f"| Average Inference Time | {np.mean(inference_times):.2f}ms |\n")
        f.write(f"| Inference Time Std | {np.std(inference_times):.2f}ms |\n")
        f.write("\n")

    def _write_recommendations(self, f, results: Dict[str, Any], model_type: str) -> None:
        """写入建议和结论"""
        action_metrics = results.get('action_metrics', {})
        performance_metrics = results.get('performance_metrics', {})

        f.write("Based on the evaluation results:\n\n")

        # 动作精度建议
        mse = action_metrics.get('overall_avg_mse')
        if mse is not None:
            if mse < 0.01:
                f.write("✅ **Action Accuracy:** Excellent - MSE < 0.01\n")
            elif mse < 0.1:
                f.write("⚠️ **Action Accuracy:** Good - MSE < 0.1, but could be improved\n")
            else:
                f.write("❌ **Action Accuracy:** Poor - MSE > 0.1, requires attention\n")

        # 模型特定建议
        if model_type == 'humanoid_diffusion':
            budget_compliance = performance_metrics.get('budget_compliance_rate', 0)
            if budget_compliance < 0.8:
                f.write("⚠️ **Latency:** Budget compliance < 80%, consider optimizing inference\n")
            else:
                f.write("✅ **Latency:** Good budget compliance\n")

        elif model_type == 'diffusion':
            velocity_var = performance_metrics.get('overall_velocity_variation')
            if velocity_var is not None and velocity_var > 0.1:
                f.write("⚠️ **Smoothness:** High velocity variation, consider smoothing techniques\n")
            else:
                f.write("✅ **Smoothness:** Trajectory smoothness is acceptable\n")

        f.write("\n")

    def _generate_plots(self, results: Dict[str, Any], model_type: str) -> Dict[str, Path]:
        """生成可视化图表"""
        if not PLOTTING_AVAILABLE:
            return {}

        plot_files = {}

        # 基础图表
        plot_files.update(self._plot_action_metrics(results, model_type))
        plot_files.update(self._plot_episode_trends(results, model_type))

        # 模型特定图表
        if model_type == 'humanoid_diffusion':
            plot_files.update(self._plot_hierarchical_metrics(results))
        elif model_type == 'diffusion':
            plot_files.update(self._plot_diffusion_metrics(results))

        return plot_files

    def _plot_action_metrics(self, results: Dict[str, Any], model_type: str) -> Dict[str, Path]:
        """绘制动作指标图表"""
        plot_files = {}
        action_metrics = results.get('action_metrics', {})

        if not action_metrics:
            return plot_files

        # 动作指标条形图
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics_to_plot = ['overall_avg_mse', 'overall_avg_mae', 'overall_avg_l2_norm']
        values = []
        labels = []

        for metric in metrics_to_plot:
            if metric in action_metrics:
                values.append(action_metrics[metric])
                labels.append(metric.replace('overall_avg_', '').upper())

        if values:
            bars = ax.bar(labels, values, color=['skyblue', 'lightcoral', 'lightgreen'][:len(values)])
            ax.set_ylabel('Error Value')
            ax.set_title(f'{model_type.upper()} Model - Action Accuracy Metrics')
            ax.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values)*0.01,
                       f'{value:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plot_file = self.output_dir / f'{model_type}_action_metrics_{self.timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        plot_files['action_metrics'] = plot_file
        return plot_files

    def _plot_episode_trends(self, results: Dict[str, Any], model_type: str) -> Dict[str, Path]:
        """绘制episode趋势图"""
        plot_files = {}
        episode_results = results.get('episode_results', [])

        if not episode_results:
            return plot_files

        # 提取时序数据
        episodes = list(range(len(episode_results)))
        inference_times = [ep.get('average_inference_time', 0) for ep in episode_results]
        episode_lengths = [ep.get('num_steps', 0) for ep in episode_results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 推理时间趋势
        ax1.plot(episodes, inference_times, 'b-', alpha=0.7, marker='o', markersize=3)
        ax1.set_ylabel('Inference Time (ms)')
        ax1.set_title(f'{model_type.upper()} Model - Inference Time Trend')
        ax1.grid(True, alpha=0.3)

        # Episode长度趋势
        ax2.plot(episodes, episode_lengths, 'g-', alpha=0.7, marker='s', markersize=3)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Episode Length (steps)')
        ax2.set_title('Episode Length Trend')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = self.output_dir / f'{model_type}_episode_trends_{self.timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        plot_files['episode_trends'] = plot_file
        return plot_files

    def _plot_hierarchical_metrics(self, results: Dict[str, Any]) -> Dict[str, Path]:
        """绘制分层架构特有图表"""
        plot_files = {}
        performance_metrics = results.get('performance_metrics', {})

        # 层激活率饼图
        layer_activations = {k.replace('_activation_rate', ''): v
                           for k, v in performance_metrics.items()
                           if 'activation_rate' in k and v > 0}

        if layer_activations:
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

            wedges, texts, autotexts = ax.pie(layer_activations.values(),
                                             labels=list(layer_activations.keys()),
                                             autopct='%1.1f%%',
                                             colors=colors[:len(layer_activations)])

            ax.set_title('Hierarchical Layer Activation Distribution')

            plt.tight_layout()
            plot_file = self.output_dir / f'hierarchical_layer_activation_{self.timestamp}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            plot_files['layer_activation'] = plot_file

        return plot_files

    def _plot_diffusion_metrics(self, results: Dict[str, Any]) -> Dict[str, Path]:
        """绘制diffusion特有图表"""
        plot_files = {}
        episode_results = results.get('episode_results', [])

        # 平滑度指标分布
        velocity_variations = [ep.get('velocity_variation', 0) for ep in episode_results if 'velocity_variation' in ep]

        if velocity_variations:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(velocity_variations, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
            ax.set_xlabel('Velocity Variation')
            ax.set_ylabel('Frequency')
            ax.set_title('Trajectory Smoothness Distribution')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = self.output_dir / f'diffusion_smoothness_{self.timestamp}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            plot_files['smoothness_distribution'] = plot_file

        return plot_files

    def _format_value(self, value: Any) -> str:
        """格式化数值显示"""
        if isinstance(value, float):
            if value == 0:
                return "0.000"
            elif abs(value) < 0.001:
                return f"{value:.6f}"
            elif abs(value) < 1:
                return f"{value:.4f}"
            else:
                return f"{value:.3f}"
        elif isinstance(value, (int, np.integer)):
            return str(value)
        else:
            return str(value)

    def _make_serializable(self, obj: Any) -> Any:
        """转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj