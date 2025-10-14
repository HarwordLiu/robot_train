#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集质量分析脚本

用于分析任务1数据集的质量，帮助诊断训练问题：
1. 物品位置分布分析
2. 动作序列统计
3. 轨迹可视化
4. 成功率估计

使用方法:
    python analyze_dataset_quality.py --data_root /root/robot/data/task-1/1-2000/lerobot/ --episodes 0-50
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm

# 导入lerobot数据集加载工具
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def analyze_action_distribution(dataset, num_episodes=50):
    """分析动作分布"""
    print("\n" + "="*70)
    print("📊 动作分布分析")
    print("="*70)

    actions = []
    for i in tqdm(range(min(num_episodes * 100, len(dataset))), desc="加载动作数据"):
        try:
            sample = dataset[i]
            action = sample['action'].numpy()
            actions.append(action)
        except:
            continue

    actions = np.array(actions)

    print(f"\n动作维度: {actions.shape}")
    print(f"\n各维度统计:")
    print(f"{'维度':<6} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12}")
    print("-" * 60)

    for i in range(min(16, actions.shape[1])):  # 只显示前16维（Kuavo的实际维度）
        mean = np.mean(actions[:, i])
        std = np.std(actions[:, i])
        min_val = np.min(actions[:, i])
        max_val = np.max(actions[:, i])
        print(f"{i:<6} {mean:>11.4f} {std:>11.4f} {min_val:>11.4f} {max_val:>11.4f}")

    return actions


def analyze_state_distribution(dataset, num_episodes=50):
    """分析状态分布（包括末端执行器位置）"""
    print("\n" + "="*70)
    print("🤖 状态分布分析")
    print("="*70)

    states = []
    for i in tqdm(range(min(num_episodes * 100, len(dataset))), desc="加载状态数据"):
        try:
            sample = dataset[i]
            state = sample['observation.state'].numpy()
            states.append(state)
        except:
            continue

    states = np.array(states)

    print(f"\n状态维度: {states.shape}")
    print(f"\n各维度统计:")
    print(f"{'维度':<6} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12}")
    print("-" * 60)

    for i in range(min(16, states.shape[1])):  # 只显示前16维
        mean = np.mean(states[:, i])
        std = np.std(states[:, i])
        min_val = np.min(states[:, i])
        max_val = np.max(states[:, i])
        print(f"{i:<6} {mean:>11.4f} {std:>11.4f} {min_val:>11.4f} {max_val:>11.4f}")

    # 重点分析末端位置（假设前3维是xyz位置）
    if states.shape[1] >= 3:
        print("\n🎯 末端执行器位置分析（假设前3维为xyz）:")
        print(
            f"X轴范围: [{np.min(states[:, 0]):.4f}, {np.max(states[:, 0]):.4f}], 均值: {np.mean(states[:, 0]):.4f}")
        print(
            f"Y轴范围: [{np.min(states[:, 1]):.4f}, {np.max(states[:, 1]):.4f}], 均值: {np.mean(states[:, 1]):.4f}")
        print(
            f"Z轴范围: [{np.min(states[:, 2]):.4f}, {np.max(states[:, 2]):.4f}], 均值: {np.mean(states[:, 2]):.4f}")

        # 检查位置分布是否集中
        x_std = np.std(states[:, 0])
        y_std = np.std(states[:, 1])
        z_std = np.std(states[:, 2])

        print(f"\n位置标准差:")
        print(f"X: {x_std:.4f}, Y: {y_std:.4f}, Z: {z_std:.4f}")

        if x_std < 0.05 or y_std < 0.05:
            print("⚠️  警告: 位置分布过于集中，可能导致泛化能力差！")
            print("   建议: 增加更多不同位置的演示数据")

    return states


def analyze_episode_statistics(dataset, metadata):
    """分析episode统计信息"""
    print("\n" + "="*70)
    print("📈 Episode统计分析")
    print("="*70)

    # 获取episode信息
    episode_data_index = dataset.episode_data_index
    num_episodes = len(episode_data_index)

    print(f"\nEpisode总数: {num_episodes}")
    print(f"总帧数: {len(dataset)}")

    # 分析每个episode的长度
    episode_lengths = []
    for episode_id in range(num_episodes):
        episode_indices = episode_data_index[episode_id]
        episode_length = len(episode_indices)
        episode_lengths.append(episode_length)

    episode_lengths = np.array(episode_lengths)

    print(f"\nEpisode长度统计:")
    print(f"  平均长度: {np.mean(episode_lengths):.1f} 帧")
    print(f"  标准差: {np.std(episode_lengths):.1f} 帧")
    print(f"  最短: {np.min(episode_lengths)} 帧")
    print(f"  最长: {np.max(episode_lengths)} 帧")

    # 分析episode长度分布
    short_episodes = np.sum(episode_lengths < 50)
    medium_episodes = np.sum((episode_lengths >= 50) & (episode_lengths < 100))
    long_episodes = np.sum(episode_lengths >= 100)

    print(f"\nEpisode长度分布:")
    print(
        f"  短(<50帧): {short_episodes} ({short_episodes/num_episodes*100:.1f}%)")
    print(
        f"  中(50-100帧): {medium_episodes} ({medium_episodes/num_episodes*100:.1f}%)")
    print(
        f"  长(>100帧): {long_episodes} ({long_episodes/num_episodes*100:.1f}%)")

    if short_episodes > num_episodes * 0.3:
        print("⚠️  警告: 短episode占比过高，可能包含失败的演示")
        print("   建议: 检查短episode的质量，考虑过滤掉失败的演示")

    return episode_lengths


def visualize_trajectory(dataset, episode_id=0, save_path=None):
    """可视化单个episode的轨迹"""
    print(f"\n📍 可视化Episode {episode_id}的轨迹...")

    # 获取episode数据
    episode_indices = dataset.episode_data_index[episode_id]

    states = []
    actions = []
    for idx in episode_indices:
        sample = dataset[idx]
        states.append(sample['observation.state'].numpy())
        actions.append(sample['action'].numpy())

    states = np.array(states)
    actions = np.array(actions)

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Episode {episode_id} 轨迹分析', fontsize=16)

    # 1. XY平面轨迹（假设前2维是XY位置）
    ax = axes[0, 0]
    if states.shape[1] >= 2:
        ax.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='轨迹')
        ax.scatter(states[0, 0], states[0, 1], c='green',
                   s=100, marker='o', label='起点')
        ax.scatter(states[-1, 0], states[-1, 1], c='red',
                   s=100, marker='x', label='终点')
        ax.set_xlabel('X位置')
        ax.set_ylabel('Y位置')
        ax.set_title('XY平面轨迹')
        ax.legend()
        ax.grid(True)

    # 2. Z高度随时间变化
    ax = axes[0, 1]
    if states.shape[1] >= 3:
        ax.plot(states[:, 2], 'b-', linewidth=2)
        ax.set_xlabel('时间步')
        ax.set_ylabel('Z位置（高度）')
        ax.set_title('Z高度变化')
        ax.grid(True)

    # 3. 动作幅度（前3维）
    ax = axes[1, 0]
    if actions.shape[1] >= 3:
        ax.plot(actions[:, 0], label='动作维度0', alpha=0.7)
        ax.plot(actions[:, 1], label='动作维度1', alpha=0.7)
        ax.plot(actions[:, 2], label='动作维度2', alpha=0.7)
        ax.set_xlabel('时间步')
        ax.set_ylabel('动作值')
        ax.set_title('动作序列（前3维）')
        ax.legend()
        ax.grid(True)

    # 4. 状态变化率
    ax = axes[1, 1]
    if states.shape[1] >= 3:
        velocity = np.diff(states[:, :3], axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        ax.plot(speed, 'r-', linewidth=2)
        ax.set_xlabel('时间步')
        ax.set_ylabel('速度')
        ax.set_title('末端执行器速度')
        ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 轨迹图已保存到: {save_path}")
    else:
        plt.savefig(f'episode_{episode_id}_trajectory.png',
                    dpi=150, bbox_inches='tight')
        print(f"✅ 轨迹图已保存到: episode_{episode_id}_trajectory.png")

    plt.close()


def detect_potential_issues(states, actions, episode_lengths):
    """检测潜在问题"""
    print("\n" + "="*70)
    print("⚠️  潜在问题检测")
    print("="*70)

    issues = []

    # 1. 检查位置分布
    if states.shape[1] >= 3:
        x_std = np.std(states[:, 0])
        y_std = np.std(states[:, 1])
        z_std = np.std(states[:, 2])

        if x_std < 0.05 or y_std < 0.05:
            issues.append("❌ 问题1: 物品位置分布过于集中")
            print("\n❌ 问题1: 物品位置分布过于集中")
            print(f"   XY位置标准差: X={x_std:.4f}, Y={y_std:.4f}")
            print("   这可能导致模型无法泛化到不同位置")
            print("   建议: 收集更多不同位置的演示数据")
        else:
            print("\n✅ 物品位置分布良好")

    # 2. 检查动作幅度
    action_std = np.std(actions, axis=0)
    if np.any(action_std < 0.01):
        issues.append("❌ 问题2: 部分动作维度几乎不变")
        print("\n❌ 问题2: 部分动作维度几乎不变")
        print(f"   静止维度: {np.where(action_std < 0.01)[0]}")
        print("   这可能表示数据中缺乏某些动作")
    else:
        print("\n✅ 动作维度分布良好")

    # 3. 检查episode长度
    if np.std(episode_lengths) > np.mean(episode_lengths) * 0.5:
        issues.append("⚠️  问题3: Episode长度差异较大")
        print("\n⚠️  问题3: Episode长度差异较大")
        print(
            f"   标准差({np.std(episode_lengths):.1f}) > 均值({np.mean(episode_lengths):.1f}) * 0.5")
        print("   这可能表示数据质量不一致或包含失败的演示")
        print("   建议: 检查异常短或长的episodes")
    else:
        print("\n✅ Episode长度分布一致")

    # 4. 检查动作平滑性
    action_diff = np.diff(actions, axis=0)
    action_jerk = np.mean(np.abs(action_diff), axis=0)
    if np.any(action_jerk > 0.5):
        issues.append("⚠️  问题4: 动作序列有较大跳变")
        print("\n⚠️  问题4: 动作序列有较大跳变")
        print(f"   高跳变维度: {np.where(action_jerk > 0.5)[0]}")
        print("   这可能影响策略学习的平滑性")
    else:
        print("\n✅ 动作序列平滑")

    # 总结
    print("\n" + "="*70)
    if len(issues) == 0:
        print("✅ 数据集质量检查通过，未发现明显问题")
    else:
        print(f"⚠️  发现 {len(issues)} 个潜在问题:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    print("="*70)

    return issues


def generate_report(dataset, metadata, output_dir):
    """生成完整的分析报告"""
    print("\n" + "="*70)
    print("📝 生成分析报告")
    print("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 动作分布分析
    actions = analyze_action_distribution(dataset, num_episodes=50)

    # 2. 状态分布分析
    states = analyze_state_distribution(dataset, num_episodes=50)

    # 3. Episode统计
    episode_lengths = analyze_episode_statistics(dataset, metadata)

    # 4. 可视化几个典型轨迹
    print("\n📊 可视化轨迹...")
    num_episodes_to_viz = min(5, len(dataset.episode_data_index))
    for ep_id in range(num_episodes_to_viz):
        visualize_trajectory(
            dataset,
            episode_id=ep_id,
            save_path=output_dir / f"trajectory_episode_{ep_id}.png"
        )

    # 5. 问题检测
    issues = detect_potential_issues(states, actions, episode_lengths)

    # 6. 保存数值统计
    report = {
        "dataset_info": {
            "total_episodes": len(dataset.episode_data_index),
            "total_frames": len(dataset),
            "fps": metadata.fps,
            "state_dim": states.shape[1],
            "action_dim": actions.shape[1],
        },
        "action_statistics": {
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
        },
        "state_statistics": {
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).tolist(),
            "min": states.min(axis=0).tolist(),
            "max": states.max(axis=0).tolist(),
        },
        "episode_statistics": {
            "mean_length": float(episode_lengths.mean()),
            "std_length": float(episode_lengths.std()),
            "min_length": int(episode_lengths.min()),
            "max_length": int(episode_lengths.max()),
        },
        "detected_issues": issues,
    }

    report_file = output_dir / "dataset_analysis_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 分析报告已保存到: {report_file}")
    print(f"✅ 可视化图表已保存到: {output_dir}")

    return report


def main():
    parser = argparse.ArgumentParser(description='分析LeRobot数据集质量')
    parser.add_argument('--data_root', type=str,
                        default='/root/robot/data/task-1/1-2000/lerobot/',
                        help='数据集根目录')
    parser.add_argument('--repo_id', type=str,
                        default='lerobot/task1_moving_grasp',
                        help='数据集repo ID')
    parser.add_argument('--episodes', type=str,
                        default='0-199',
                        help='要分析的episode范围，格式: 0-199')
    parser.add_argument('--output_dir', type=str,
                        default='./dataset_analysis',
                        help='输出目录')

    args = parser.parse_args()

    # 解析episode范围
    episode_start, episode_end = map(int, args.episodes.split('-'))
    episodes = list(range(episode_start, episode_end + 1))

    print("="*70)
    print("🔍 任务1数据集质量分析")
    print("="*70)
    print(f"数据集路径: {args.data_root}")
    print(f"Repo ID: {args.repo_id}")
    print(f"分析Episodes: {episode_start} - {episode_end}")
    print(f"输出目录: {args.output_dir}")
    print("="*70)

    # 加载数据集
    print("\n📂 加载数据集...")
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.data_root)
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.data_root,
        episodes=episodes[:50]  # 只分析前50个episodes，避免太慢
    )
    print(f"✅ 数据集加载完成: {len(dataset)} 帧")

    # 生成分析报告
    report = generate_report(dataset, metadata, args.output_dir)

    print("\n" + "="*70)
    print("✅ 分析完成！")
    print("="*70)
    print(f"\n请查看输出目录获取详细报告: {args.output_dir}")
    print("\n建议:")
    print("1. 查看 dataset_analysis_report.json 获取数值统计")
    print("2. 查看 trajectory_episode_*.png 了解轨迹模式")
    print("3. 根据检测到的问题优化数据收集策略")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
