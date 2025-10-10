"""
推理日志记录器
用于记录模型推理过程中的详细信息，包括：
- 每步的推理结果
- 层激活情况
- 执行时间统计
- 任务相关信息
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torch
from datetime import datetime


class InferenceLogger:
    """推理日志记录器"""

    def __init__(self,
                 output_dir: Path,
                 episode_idx: int,
                 log_every_n_steps: int = 1,
                 save_detailed_layers: bool = True):
        """
        初始化推理日志记录器

        Args:
            output_dir: 输出目录
            episode_idx: 当前回合索引
            log_every_n_steps: 每N步记录一次详细信息
            save_detailed_layers: 是否保存详细的层信息
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.episode_idx = episode_idx
        self.log_every_n_steps = log_every_n_steps
        self.save_detailed_layers = save_detailed_layers

        # 创建回合特定的日志文件
        self.log_file = self.output_dir / \
            f"inference_episode_{episode_idx}.jsonl"
        self.summary_file = self.output_dir / \
            f"inference_episode_{episode_idx}_summary.json"

        # 初始化统计信息
        self.step_count = 0
        self.total_inference_time = 0.0
        self.layer_activation_counts = {}
        self.layer_execution_times = {}

        # 缓存当前回合的所有步骤记录
        self.step_records = []

        # 记录开始时间
        self.episode_start_time = datetime.now()

    def log_step(self,
                 step: int,
                 action: np.ndarray,
                 observation_shapes: Dict[str, tuple],
                 layer_outputs: Optional[Dict[str, Any]] = None,
                 inference_time: float = 0.0,
                 additional_info: Optional[Dict[str, Any]] = None):
        """
        记录单步推理信息

        Args:
            step: 当前步数
            action: 推理得到的动作
            observation_shapes: 观测数据的形状信息
            layer_outputs: 层输出信息（分层架构）
            inference_time: 推理耗时
            additional_info: 额外信息
        """
        self.step_count += 1
        self.total_inference_time += inference_time

        # 构建基础记录
        step_record = {
            "timestamp": datetime.now().isoformat(),
            "episode": self.episode_idx,
            "step": step,
            "inference_time_ms": inference_time * 1000,
            "observation_shapes": observation_shapes,
        }

        # 记录动作信息
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        step_record["action"] = {
            "shape": action.shape,
            "mean": float(np.mean(action)),
            "std": float(np.std(action)),
            "min": float(np.min(action)),
            "max": float(np.max(action)),
        }

        # 每N步记录完整动作值
        if step % self.log_every_n_steps == 0:
            step_record["action"]["values"] = action.tolist()

        # 记录层激活信息（分层架构）
        if layer_outputs is not None and self.save_detailed_layers:
            layer_info = self._process_layer_outputs(layer_outputs)
            step_record["hierarchical_layers"] = layer_info

        # 添加额外信息
        if additional_info:
            step_record["additional_info"] = additional_info

        # 保存到缓存
        self.step_records.append(step_record)

        # 实时写入JSONL文件（每步追加一行）
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(step_record, ensure_ascii=False) + '\n')

    def _process_layer_outputs(self, layer_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """处理层输出信息，提取关键统计数据"""
        layer_info = {}

        for layer_name, layer_output in layer_outputs.items():
            # 跳过内部统计信息
            if layer_name.startswith('_'):
                continue

            # 更新激活计数
            if layer_name not in self.layer_activation_counts:
                self.layer_activation_counts[layer_name] = 0
            self.layer_activation_counts[layer_name] += 1

            layer_record = {
                "activated": True,
                "activation_count": self.layer_activation_counts[layer_name],
            }

            if isinstance(layer_output, dict):
                # 记录执行时间
                if 'execution_time_ms' in layer_output:
                    exec_time = layer_output['execution_time_ms']
                    layer_record["execution_time_ms"] = exec_time

                    if layer_name not in self.layer_execution_times:
                        self.layer_execution_times[layer_name] = []
                    self.layer_execution_times[layer_name].append(exec_time)

                # 记录损失值（训练模式）
                if 'loss' in layer_output:
                    loss_value = layer_output['loss']
                    if torch.is_tensor(loss_value):
                        loss_value = loss_value.item()
                    layer_record["loss"] = loss_value

                # 记录紧急状态（安全层）
                if 'emergency' in layer_output:
                    layer_record["emergency"] = layer_output['emergency']

                # 记录动作信息
                if 'action' in layer_output:
                    action = layer_output['action']
                    if torch.is_tensor(action):
                        action = action.cpu().numpy()
                    layer_record["action_shape"] = action.shape
                    layer_record["action_norm"] = float(np.linalg.norm(action))

                # 记录其他可序列化的字段
                for key, value in layer_output.items():
                    if key not in ['action', 'loss', 'execution_time_ms', 'emergency']:
                        if isinstance(value, (int, float, str, bool)):
                            layer_record[key] = value
                        elif isinstance(value, (list, tuple)) and len(value) < 10:
                            layer_record[key] = value

            layer_info[layer_name] = layer_record

        return layer_info

    def save_episode_summary(self,
                             success: bool,
                             total_reward: float,
                             additional_stats: Optional[Dict[str, Any]] = None):
        """
        保存回合总结信息

        Args:
            success: 是否成功
            total_reward: 总奖励
            additional_stats: 额外统计信息
        """
        episode_end_time = datetime.now()
        episode_duration = (episode_end_time -
                            self.episode_start_time).total_seconds()

        summary = {
            "episode_index": self.episode_idx,
            "start_time": self.episode_start_time.isoformat(),
            "end_time": episode_end_time.isoformat(),
            "episode_duration_sec": episode_duration,
            "success": success,
            "total_reward": total_reward,
            "total_steps": self.step_count,
            "total_inference_time_sec": self.total_inference_time,
            "avg_inference_time_ms": (self.total_inference_time / self.step_count * 1000) if self.step_count > 0 else 0,
        }

        # 分层架构统计
        if self.layer_activation_counts:
            summary["hierarchical_stats"] = {
                "layer_activation_counts": self.layer_activation_counts,
                "layer_avg_execution_times_ms": {
                    layer: np.mean(times)
                    for layer, times in self.layer_execution_times.items()
                },
                "layer_total_execution_times_ms": {
                    layer: np.sum(times)
                    for layer, times in self.layer_execution_times.items()
                },
            }

        # 添加额外统计信息
        if additional_stats:
            summary["additional_stats"] = additional_stats

        # 保存总结
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"✅ 推理日志已保存:")
        print(f"   详细记录: {self.log_file}")
        print(f"   回合总结: {self.summary_file}")
        print(f"   总步数: {self.step_count}, 成功: {success}")
        if self.layer_activation_counts:
            print(f"   层激活次数: {self.layer_activation_counts}")

    @staticmethod
    def create_aggregated_report(output_dir: Path, task_name: str = ""):
        """
        创建聚合报告，汇总所有回合的统计信息

        Args:
            output_dir: 输出目录
            task_name: 任务名称
        """
        output_dir = Path(output_dir)
        summary_files = list(output_dir.glob(
            "inference_episode_*_summary.json"))

        if not summary_files:
            print("⚠️  未找到任何回合总结文件")
            return

        # 读取所有总结文件
        all_summaries = []
        for summary_file in summary_files:
            with open(summary_file, 'r', encoding='utf-8') as f:
                all_summaries.append(json.load(f))

        # 计算聚合统计
        total_episodes = len(all_summaries)
        success_count = sum(
            1 for s in all_summaries if s.get('success', False))
        success_rate = success_count / total_episodes if total_episodes > 0 else 0

        avg_steps = np.mean([s['total_steps'] for s in all_summaries])
        avg_inference_time = np.mean(
            [s['avg_inference_time_ms'] for s in all_summaries])

        aggregated_report = {
            "task_name": task_name,
            "generated_at": datetime.now().isoformat(),
            "total_episodes": total_episodes,
            "success_count": success_count,
            "success_rate": success_rate,
            "avg_steps_per_episode": avg_steps,
            "avg_inference_time_ms": avg_inference_time,
            "episodes": all_summaries,
        }

        # 聚合分层架构统计
        hierarchical_stats = {}
        for summary in all_summaries:
            if 'hierarchical_stats' in summary:
                h_stats = summary['hierarchical_stats']
                for layer, count in h_stats.get('layer_activation_counts', {}).items():
                    if layer not in hierarchical_stats:
                        hierarchical_stats[layer] = {
                            'total_activations': 0,
                            'avg_execution_times': []
                        }
                    hierarchical_stats[layer]['total_activations'] += count

                    layer_avg_time = h_stats.get(
                        'layer_avg_execution_times_ms', {}).get(layer)
                    if layer_avg_time is not None:
                        hierarchical_stats[layer]['avg_execution_times'].append(
                            layer_avg_time)

        # 计算每层的平均执行时间
        if hierarchical_stats:
            aggregated_report['hierarchical_aggregated_stats'] = {
                layer: {
                    'total_activations': stats['total_activations'],
                    'avg_execution_time_ms': np.mean(stats['avg_execution_times']) if stats['avg_execution_times'] else 0,
                }
                for layer, stats in hierarchical_stats.items()
            }

        # 保存聚合报告
        report_file = output_dir / "aggregated_inference_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated_report, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"📊 聚合推理报告已生成: {report_file}")
        print(f"{'='*60}")
        print(f"任务: {task_name}")
        print(f"总回合数: {total_episodes}")
        print(f"成功率: {success_rate:.2%} ({success_count}/{total_episodes})")
        print(f"平均步数: {avg_steps:.1f}")
        print(f"平均推理时间: {avg_inference_time:.2f}ms")

        if 'hierarchical_aggregated_stats' in aggregated_report:
            print(f"\n分层架构统计:")
            for layer, stats in aggregated_report['hierarchical_aggregated_stats'].items():
                print(f"  {layer}:")
                print(f"    总激活次数: {stats['total_activations']}")
                print(f"    平均执行时间: {stats['avg_execution_time_ms']:.2f}ms")

        print(f"{'='*60}\n")

        return aggregated_report


class InferenceLoggerContext:
    """推理日志记录器上下文管理器"""

    def __init__(self, logger: InferenceLogger, step: int):
        self.logger = logger
        self.step = step
        self.start_time = None
        self.observation_shapes = {}
        self.layer_outputs = None
        self.action = None
        self.additional_info = {}

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self.action is not None:
            inference_time = time.time() - self.start_time
            self.logger.log_step(
                step=self.step,
                action=self.action,
                observation_shapes=self.observation_shapes,
                layer_outputs=self.layer_outputs,
                inference_time=inference_time,
                additional_info=self.additional_info
            )
