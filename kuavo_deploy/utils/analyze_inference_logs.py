"""
推理日志分析工具

用于分析推理日志文件，生成统计报告和可视化图表
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """读取JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def analyze_inference_time(log_dir: Path):
    """分析推理时间统计"""
    print("\n" + "="*60)
    print("📊 推理时间分析")
    print("="*60)

    log_files = list(log_dir.glob("inference_episode_*.jsonl"))
    if not log_files:
        print("❌ 未找到日志文件")
        return

    all_times = []
    episode_times = {}

    for log_file in log_files:
        episode_idx = int(log_file.stem.split('_')[-1])
        steps = read_jsonl(log_file)
        times = [step['inference_time_ms'] for step in steps]
        all_times.extend(times)
        episode_times[episode_idx] = {
            'mean': np.mean(times),
            'min': np.min(times),
            'max': np.max(times),
            'std': np.std(times),
        }

    print(f"\n总体统计 (基于 {len(log_files)} 个回合, {len(all_times)} 步):")
    print(f"  平均推理时间: {np.mean(all_times):.2f}ms")
    print(f"  最小推理时间: {np.min(all_times):.2f}ms")
    print(f"  最大推理时间: {np.max(all_times):.2f}ms")
    print(f"  标准差: {np.std(all_times):.2f}ms")
    print(f"  中位数: {np.median(all_times):.2f}ms")
    print(f"  95分位数: {np.percentile(all_times, 95):.2f}ms")
    print(f"  99分位数: {np.percentile(all_times, 99):.2f}ms")

    # 显示推理时间分布
    print(f"\n推理时间分布:")
    bins = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, float('inf')]
    bin_labels = ['0-10ms', '10-20ms', '20-30ms', '30-40ms', '40-50ms',
                  '50-75ms', '75-100ms', '100-150ms', '150-200ms', '>200ms']

    hist, _ = np.histogram(all_times, bins=bins)
    for label, count in zip(bin_labels, hist):
        percentage = count / len(all_times) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {label:>12}: {bar} {count:5d} ({percentage:5.1f}%)")


def analyze_action_statistics(log_dir: Path):
    """分析动作统计"""
    print("\n" + "="*60)
    print("🎯 动作统计分析")
    print("="*60)

    log_files = list(log_dir.glob("inference_episode_*.jsonl"))
    if not log_files:
        return

    all_action_means = []
    all_action_stds = []
    all_action_mins = []
    all_action_maxs = []

    for log_file in log_files[:5]:  # 只分析前5个回合作为示例
        steps = read_jsonl(log_file)
        for step in steps:
            action_info = step.get('action', {})
            all_action_means.append(action_info.get('mean', 0))
            all_action_stds.append(action_info.get('std', 0))
            all_action_mins.append(action_info.get('min', 0))
            all_action_maxs.append(action_info.get('max', 0))

    print(f"\n基于前 {min(5, len(log_files))} 个回合的动作统计:")
    print(f"  动作均值的平均值: {np.mean(all_action_means):.4f}")
    print(f"  动作标准差的平均值: {np.mean(all_action_stds):.4f}")
    print(f"  动作最小值的平均值: {np.min(all_action_mins):.4f}")
    print(f"  动作最大值的平均值: {np.max(all_action_maxs):.4f}")


def analyze_layer_activation(log_dir: Path):
    """分析层激活情况"""
    print("\n" + "="*60)
    print("🏗️  分层架构激活分析")
    print("="*60)

    # 读取聚合报告
    aggregated_file = log_dir / "aggregated_inference_report.json"
    if not aggregated_file.exists():
        print("⚠️  未找到聚合报告，跳过层激活分析")
        return

    with open(aggregated_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'hierarchical_aggregated_stats' not in data:
        print("ℹ️  当前策略不是分层架构，无层激活信息")
        return

    h_stats = data['hierarchical_aggregated_stats']

    print(f"\n层激活统计 (总回合数: {data['total_episodes']}):")
    for layer, stats in h_stats.items():
        print(f"\n  📍 {layer.upper()} 层:")
        print(f"    总激活次数: {stats['total_activations']}")
        print(f"    平均执行时间: {stats['avg_execution_time_ms']:.2f}ms")

        # 计算激活率（相对于总步数）
        total_steps = sum(ep['total_steps'] for ep in data['episodes'])
        activation_rate = stats['total_activations'] / total_steps * 100
        print(f"    激活率: {activation_rate:.1f}%")

    # 分析层执行时间占比
    print(f"\n层执行时间占比:")
    total_layer_time = sum(stats['avg_execution_time_ms'] * stats['total_activations']
                           for stats in h_stats.values())

    for layer, stats in h_stats.items():
        layer_total_time = stats['avg_execution_time_ms'] * \
            stats['total_activations']
        percentage = layer_total_time / total_layer_time * 100
        bar = '█' * int(percentage / 5)
        print(f"  {layer:>12}: {bar} {percentage:5.1f}%")


def analyze_success_rate(log_dir: Path):
    """分析成功率"""
    print("\n" + "="*60)
    print("🎯 成功率分析")
    print("="*60)

    summary_files = list(log_dir.glob("inference_episode_*_summary.json"))
    if not summary_files:
        print("❌ 未找到总结文件")
        return

    successes = []
    episode_steps = []
    episode_durations = []

    for summary_file in sorted(summary_files):
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            successes.append(data.get('success', False))
            episode_steps.append(data.get('total_steps', 0))
            episode_durations.append(data.get('episode_duration_sec', 0))

    success_count = sum(successes)
    total_episodes = len(successes)
    success_rate = success_count / total_episodes if total_episodes > 0 else 0

    print(f"\n总体成功率: {success_rate:.2%} ({success_count}/{total_episodes})")
    print(f"平均回合步数: {np.mean(episode_steps):.1f}")
    print(f"平均回合时长: {np.mean(episode_durations):.1f}秒")

    # 成功和失败的对比
    if success_count > 0 and success_count < total_episodes:
        success_steps = [steps for success, steps in zip(
            successes, episode_steps) if success]
        fail_steps = [steps for success, steps in zip(
            successes, episode_steps) if not success]

        print(f"\n成功回合平均步数: {np.mean(success_steps):.1f}")
        print(f"失败回合平均步数: {np.mean(fail_steps):.1f}")


def generate_report(log_dir: Path, output_file: Path = None):
    """生成完整的分析报告"""
    log_dir = Path(log_dir)

    if not log_dir.exists():
        print(f"❌ 日志目录不存在: {log_dir}")
        return

    print("\n" + "="*60)
    print(f"📝 推理日志分析报告")
    print(f"📁 日志目录: {log_dir}")
    print("="*60)

    # 运行各项分析
    analyze_success_rate(log_dir)
    analyze_inference_time(log_dir)
    analyze_action_statistics(log_dir)
    analyze_layer_activation(log_dir)

    print("\n" + "="*60)
    print("✅ 分析完成!")
    print("="*60 + "\n")

    # 如果指定了输出文件，保存报告（这里只是打印，实际可以保存为文件）
    if output_file:
        print(f"💾 报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="推理日志分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python analyze_inference_logs.py \\
    --log_dir outputs/eval/task_400_episodes/humanoid_hierarchical/task_specific_run_20251009_033145/epoch30/inference_logs

  python analyze_inference_logs.py \\
    --log_dir outputs/eval/.../inference_logs \\
    --output report.txt
        """
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        required=True,
        help='推理日志目录路径'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='输出报告文件路径（可选）'
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_file = Path(args.output) if args.output else None

    generate_report(log_dir, output_file)


if __name__ == "__main__":
    main()
