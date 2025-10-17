#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练状态检查脚本 - 简单版本

快速检查训练进度，打印评估报告

用法:
    # 自动找最新训练
    python kuavo_train/check_training.py

    # 指定训练目录
    python kuavo_train/check_training.py --run-dir outputs/train/.../run_xxx
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False


def find_latest_run():
    """查找最新的训练运行"""
    base_dir = Path(__file__).parent.parent / "outputs" / "train"
    if not base_dir.exists():
        return None

    run_dirs = []
    for task_dir in base_dir.iterdir():
        if task_dir.is_dir():
            for method_dir in task_dir.iterdir():
                if method_dir.is_dir():
                    for run_dir in method_dir.glob("run_*"):
                        if run_dir.is_dir():
                            run_dirs.append(run_dir)

    if not run_dirs:
        return None

    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return run_dirs[0]


def load_metrics(run_dir):
    """从TensorBoard事件文件加载指标"""
    if not HAS_TB:
        return None

    try:
        ea = EventAccumulator(str(run_dir))
        ea.Reload()

        metrics = {}

        # 加载训练loss
        if 'train/loss' in ea.Tags()['scalars']:
            loss_events = ea.Scalars('train/loss')
            metrics['loss'] = [(e.step, e.value) for e in loss_events]

        # 加载学习率
        if 'train/lr' in ea.Tags()['scalars']:
            lr_events = ea.Scalars('train/lr')
            metrics['lr'] = [(e.step, e.value) for e in lr_events]

        # 加载epoch耗时
        if 'train/epoch_duration_minutes' in ea.Tags()['scalars']:
            duration_events = ea.Scalars('train/epoch_duration_minutes')
            metrics['duration'] = [(e.step, e.value) for e in duration_events]

        # 加载验证loss
        metrics['validation'] = {}
        for tag in ea.Tags()['scalars']:
            if tag.startswith('validation/'):
                val_events = ea.Scalars(tag)
                task_name = tag.replace('validation/', '').replace('_loss', '')
                metrics['validation'][task_name] = [
                    (e.step, e.value) for e in val_events]

        return metrics
    except Exception as e:
        print(f"⚠️  读取TensorBoard数据失败: {e}")
        return None


def get_checkpoint_info(run_dir):
    """获取checkpoint信息"""
    info = {
        'best_exists': (run_dir / 'best').exists(),
        'saved_epochs': []
    }

    epoch_dirs = sorted(run_dir.glob('epoch*'))
    if epoch_dirs:
        info['saved_epochs'] = [int(d.name.replace('epoch', ''))
                                for d in epoch_dirs]

    return info


def print_report(run_dir, metrics, checkpoint_info):
    """打印训练评估报告"""
    print("\n" + "=" * 80)
    print("🤖 训练状态评估报告")
    print("=" * 80)
    print(f"📁 训练目录: {run_dir}")
    print(f"🕐 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    if metrics is None or not metrics:
        print("\n❌ 无法读取训练数据")
        print("   原因: TensorBoard未安装或事件文件不存在")
        print("   解决: pip install tensorboard")
        return

    # ========== 基本信息 ==========
    print("\n📊 训练进度")
    print("-" * 80)

    if 'loss' in metrics and metrics['loss']:
        total_epochs = len(metrics['loss'])
        current_loss = metrics['loss'][-1][1]
        print(f"已完成Epoch: {total_epochs}")
        print(f"当前Loss: {current_loss:.6f}")

        # Loss变化趋势
        if len(metrics['loss']) >= 2:
            prev_loss = metrics['loss'][-2][1]
            change = current_loss - prev_loss
            change_pct = (change / prev_loss) * 100 if prev_loss != 0 else 0
            trend = "📉 下降" if change < 0 else "📈 上升" if change > 0 else "➡️  持平"
            print(f"Loss变化: {change:+.6f} ({change_pct:+.2f}%) {trend}")

        # 最近10个epoch的趋势
        if len(metrics['loss']) >= 10:
            recent = metrics['loss'][-10:]
            min_loss = min(v for s, v in recent)
            max_loss = max(v for s, v in recent)
            print(f"\n最近10个Epoch Loss范围: [{min_loss:.6f}, {max_loss:.6f}]")

    if 'lr' in metrics and metrics['lr']:
        current_lr = metrics['lr'][-1][1]
        print(f"\n当前学习率: {current_lr:.2e}")

    if 'duration' in metrics and metrics['duration']:
        avg_duration = sum(
            v for s, v in metrics['duration']) / len(metrics['duration'])
        latest_duration = metrics['duration'][-1][1]
        print(f"\nEpoch平均耗时: {avg_duration:.2f} 分钟")
        print(f"最新Epoch耗时: {latest_duration:.2f} 分钟")

    # ========== 验证指标 ==========
    if 'validation' in metrics and metrics['validation']:
        print("\n📈 验证指标")
        print("-" * 80)
        for task_name, val_data in metrics['validation'].items():
            if val_data:
                latest_val_loss = val_data[-1][1]
                print(f"{task_name}: {latest_val_loss:.6f}")

    # ========== Checkpoint状态 ==========
    print("\n💾 Checkpoint状态")
    print("-" * 80)
    print(f"最佳模型: {'✅ 已保存' if checkpoint_info['best_exists'] else '❌ 未保存'}")
    if checkpoint_info['saved_epochs']:
        print(
            f"已保存Epoch: {', '.join(map(str, checkpoint_info['saved_epochs']))}")
    else:
        print("已保存Epoch: 无")

    # ========== 训练评估 ==========
    print("\n🔍 训练状态分析")
    print("-" * 80)

    warnings = []
    suggestions = []

    # 分析loss趋势
    if 'loss' in metrics and len(metrics['loss']) >= 5:
        recent_losses = [v for s, v in metrics['loss'][-5:]]

        # 检查是否持续上升
        if all(recent_losses[i] <= recent_losses[i+1] for i in range(len(recent_losses)-1)):
            warnings.append("⚠️  Loss持续上升，可能学习率过大或数据有问题")

        # 检查震荡
        mean_loss = sum(recent_losses) / len(recent_losses)
        std_loss = (sum((x - mean_loss)**2 for x in recent_losses) /
                    len(recent_losses)) ** 0.5
        if mean_loss > 0 and std_loss / mean_loss > 0.5:
            warnings.append("⚠️  Loss震荡较大，建议降低学习率")

        # 检查是否收敛
        if std_loss < 0.01:
            print("状态: ✅ 已收敛")
        elif recent_losses[-1] < recent_losses[0]:
            print("状态: ✅ 正常下降")
        else:
            print("状态: ⚠️  需要关注")

    # 检查学习率
    if 'lr' in metrics and metrics['lr']:
        current_lr = metrics['lr'][-1][1]
        if current_lr < 1e-7:
            warnings.append("⚠️  学习率过小，训练可能停滞")
        elif current_lr > 1e-2:
            warnings.append("⚠️  学习率过大，可能导致不稳定")

    # 检查过拟合
    if 'loss' in metrics and 'validation' in metrics and metrics['validation']:
        train_loss = metrics['loss'][-1][1]
        val_losses = [data[-1][1]
                      for data in metrics['validation'].values() if data]
        if val_losses:
            avg_val_loss = sum(val_losses) / len(val_losses)
            if avg_val_loss > train_loss * 1.5:
                warnings.append("⚠️  验证loss明显高于训练loss，可能过拟合")
                suggestions.append("💡 建议增加数据增强或使用正则化")

    # 打印警告和建议
    if warnings:
        print("\n⚠️  警告:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("\n✅ 未发现明显问题")

    if suggestions:
        print("\n💡 优化建议:")
        for s in suggestions:
            print(f"  {s}")

    # ========== Loss趋势图 ==========
    if 'loss' in metrics and len(metrics['loss']) >= 5:
        print("\n📉 Loss趋势 (最近10个Epoch)")
        print("-" * 80)
        recent = metrics['loss'][-10:]
        min_loss = min(v for s, v in recent)
        max_loss = max(v for s, v in recent)
        loss_range = max_loss - min_loss if max_loss != min_loss else 1

        for step, loss in recent:
            bar_length = int(((loss - min_loss) / loss_range) * 40)
            bar = "█" * bar_length
            print(f"Epoch {int(step)+1:3d}: {loss:.6f} {bar}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="训练状态检查")
    parser.add_argument("--run-dir", type=str, help="训练运行目录")
    args = parser.parse_args()

    # 确定运行目录
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"❌ 目录不存在: {run_dir}")
            sys.exit(1)
    else:
        print("🔍 查找最新的训练运行...")
        run_dir = find_latest_run()
        if run_dir is None:
            print("❌ 未找到训练运行")
            print("   请使用 --run-dir 指定目录")
            sys.exit(1)

    # 检查TensorBoard
    if not HAS_TB:
        print("\n⚠️  TensorBoard未安装，将无法读取训练数据")
        print("   安装命令: pip install tensorboard\n")

    # 加载数据
    metrics = load_metrics(run_dir)
    checkpoint_info = get_checkpoint_info(run_dir)

    # 打印报告
    print_report(run_dir, metrics, checkpoint_info)


if __name__ == "__main__":
    main()
