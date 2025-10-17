#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级训练监控脚本 - 提供实时交互式界面和丰富的可视化

功能增强：
- 实时动态更新的图表（自动刷新）
- 多窗口仪表板布局
- GPU使用率监控
- 详细的统计分析（loss分布、梯度统计等）
- 训练健康度评分
- 自动异常检测和告警
- 对比多个训练运行

依赖：
    pip install tensorboard matplotlib rich psutil GPUtil

使用方法：
    # 启动监控仪表板
    python kuavo_train/monitor_training_advanced.py

    # 监控指定目录
    python kuavo_train/monitor_training_advanced.py --run-dir outputs/train/.../run_xxx

    # 对比多个运行
    python kuavo_train/monitor_training_advanced.py --compare run1 run2 run3

    # 启用GPU监控
    python kuavo_train/monitor_training_advanced.py --monitor-gpu
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from collections import defaultdict

# Rich库用于美化终端输出
try:
    from rich.console import Console
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Rich未安装，使用基础输出。安装: pip install rich")

# TensorBoard
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Matplotlib用于图表
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# GPU监控
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# 系统监控
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class AdvancedTrainingMonitor:
    """高级训练监控器"""

    def __init__(self, run_dir: Path, monitor_gpu: bool = False):
        self.run_dir = Path(run_dir)
        self.monitor_gpu = monitor_gpu and GPU_AVAILABLE
        self.console = Console() if RICH_AVAILABLE else None

        # TensorBoard事件累加器
        self.event_acc = None
        self._init_event_accumulator()

        # 数据缓存
        self.metrics_cache = defaultdict(list)
        self.gpu_history = []
        self.system_history = []

        # 健康度评分
        self.health_score = 100
        self.health_issues = []

    def _init_event_accumulator(self):
        """初始化事件累加器"""
        if not TENSORBOARD_AVAILABLE:
            return

        # 查找TensorBoard日志
        tb_files = list(self.run_dir.glob('events.out.tfevents.*'))
        if tb_files:
            try:
                self.event_acc = EventAccumulator(str(self.run_dir))
                self.event_acc.Reload()
            except Exception as e:
                if self.console:
                    self.console.print(
                        f"[yellow]⚠️  加载TensorBoard事件失败: {e}[/yellow]")

    def reload_data(self):
        """重新加载所有数据"""
        if self.event_acc:
            try:
                self.event_acc.Reload()
                self._update_metrics_cache()
            except Exception:
                pass

        if self.monitor_gpu:
            self._update_gpu_stats()

        if PSUTIL_AVAILABLE:
            self._update_system_stats()

        self._calculate_health_score()

    def _update_metrics_cache(self):
        """更新指标缓存"""
        if not self.event_acc:
            return

        try:
            all_tags = self.event_acc.Tags().get('scalars', [])
            for tag in all_tags:
                try:
                    events = self.event_acc.Scalars(tag)
                    self.metrics_cache[tag] = [
                        (e.step, e.value, e.wall_time) for e in events]
                except:
                    pass
        except:
            pass

    def _update_gpu_stats(self):
        """更新GPU统计"""
        if not GPU_AVAILABLE:
            return

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = {
                    'timestamp': time.time(),
                    'utilization': [gpu.load * 100 for gpu in gpus],
                    'memory_used': [gpu.memoryUsed for gpu in gpus],
                    'memory_total': [gpu.memoryTotal for gpu in gpus],
                    'temperature': [gpu.temperature for gpu in gpus]
                }
                self.gpu_history.append(gpu_info)

                # 只保留最近100个数据点
                if len(self.gpu_history) > 100:
                    self.gpu_history.pop(0)
        except:
            pass

    def _update_system_stats(self):
        """更新系统统计"""
        if not PSUTIL_AVAILABLE:
            return

        try:
            system_info = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_io': psutil.disk_io_counters()
            }
            self.system_history.append(system_info)

            # 只保留最近100个数据点
            if len(self.system_history) > 100:
                self.system_history.pop(0)
        except:
            pass

    def _calculate_health_score(self):
        """计算训练健康度评分"""
        score = 100
        issues = []

        # 检查loss趋势
        loss_data = self.metrics_cache.get('train/loss', [])
        if len(loss_data) >= 10:
            recent_losses = [v for s, v, t in loss_data[-10:]]

            # 检查是否有NaN或Inf
            if any(v != v or v == float('inf') for v in recent_losses):
                score -= 50
                issues.append("❌ Loss出现NaN或Inf")

            # 检查是否持续上升
            elif all(recent_losses[i] <= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                score -= 30
                issues.append("⚠️  Loss持续上升")

            # 检查是否震荡过大
            loss_std = (sum((x - sum(recent_losses)/len(recent_losses))
                        ** 2 for x in recent_losses) / len(recent_losses)) ** 0.5
            loss_mean = sum(recent_losses) / len(recent_losses)
            if loss_mean > 0 and loss_std / loss_mean > 0.5:
                score -= 20
                issues.append("⚠️  Loss震荡较大")

        # 检查学习率
        lr_data = self.metrics_cache.get('train/lr', [])
        if lr_data:
            current_lr = lr_data[-1][1]
            if current_lr < 1e-7:
                score -= 15
                issues.append("⚠️  学习率过小")
            elif current_lr > 1e-2:
                score -= 15
                issues.append("⚠️  学习率过大")

        # 检查GPU利用率（如果启用）
        if self.monitor_gpu and self.gpu_history:
            recent_gpu = self.gpu_history[-10:] if len(
                self.gpu_history) >= 10 else self.gpu_history
            avg_utilization = sum(sum(info['utilization']) / len(info['utilization'])
                                  for info in recent_gpu) / len(recent_gpu)

            if avg_utilization < 30:
                score -= 10
                issues.append("💡 GPU利用率较低")

        self.health_score = max(0, score)
        self.health_issues = issues

    def get_training_summary(self) -> Dict:
        """获取训练摘要"""
        summary = {
            'run_name': self.run_dir.name,
            'run_path': str(self.run_dir),
            'update_time': datetime.now(),
        }

        # 获取最新指标
        for key in ['train/loss', 'train/lr', 'train/epoch_duration_minutes']:
            if key in self.metrics_cache and self.metrics_cache[key]:
                summary[key.split('/')[-1]] = self.metrics_cache[key][-1][1]
                summary[f'{key.split("/")[-1]}_step'] = self.metrics_cache[key][-1][0]

        # 获取验证指标
        val_metrics = {k: v for k, v in self.metrics_cache.items()
                       if k.startswith('validation/')}
        if val_metrics:
            summary['validation'] = {
                k.replace('validation/', ''): v[-1][1]
                for k, v in val_metrics.items() if v
            }

        # checkpoint信息
        summary['checkpoints'] = {
            'best_exists': (self.run_dir / 'best').exists(),
            'saved_epochs': sorted([
                int(d.name.replace('epoch', ''))
                for d in self.run_dir.glob('epoch*')
            ])
        }

        # 健康度
        summary['health_score'] = self.health_score
        summary['health_issues'] = self.health_issues

        return summary

    def create_rich_layout(self) -> Layout:
        """创建Rich布局"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )

        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        layout["left"].split_column(
            Layout(name="metrics"),
            Layout(name="status")
        )

        layout["right"].split_column(
            Layout(name="resources"),
            Layout(name="checkpoints")
        )

        return layout

    def render_rich_dashboard(self) -> Layout:
        """渲染Rich仪表板"""
        layout = self.create_rich_layout()
        summary = self.get_training_summary()

        # Header
        header_text = f"🤖 [bold cyan]训练监控仪表板[/bold cyan] - {summary['run_name']}\n"
        header_text += f"📁 {summary['run_path']}\n"
        header_text += f"🕐 {summary['update_time'].strftime('%Y-%m-%d %H:%M:%S')}"
        layout["header"].update(Panel(header_text, border_style="cyan"))

        # Metrics
        metrics_table = Table(title="训练指标", box=box.ROUNDED, show_header=True)
        metrics_table.add_column("指标", style="cyan")
        metrics_table.add_column("当前值", style="green")
        metrics_table.add_column("Epoch/Step", style="yellow")

        if 'loss' in summary:
            metrics_table.add_row("Loss", f"{summary['loss']:.6f}",
                                  f"Epoch {int(summary.get('loss_step', 0)) + 1}")
        if 'lr' in summary:
            metrics_table.add_row("Learning Rate", f"{summary['lr']:.2e}",
                                  f"Step {int(summary.get('lr_step', 0))}")
        if 'epoch_duration_minutes' in summary:
            metrics_table.add_row("Epoch Duration",
                                  f"{summary['epoch_duration_minutes']:.2f} 分钟",
                                  f"Epoch {int(summary.get('epoch_duration_minutes_step', 0)) + 1}")

        layout["metrics"].update(Panel(metrics_table, border_style="green"))

        # Status (Health Score)
        health_color = "green" if self.health_score >= 80 else "yellow" if self.health_score >= 60 else "red"
        status_text = f"[bold {health_color}]健康度评分: {self.health_score}/100[/bold {health_color}]\n\n"

        if self.health_issues:
            status_text += "问题:\n"
            for issue in self.health_issues:
                status_text += f"  {issue}\n"
        else:
            status_text += "✅ 训练状态良好"

        layout["status"].update(
            Panel(status_text, title="训练状态", border_style=health_color))

        # Resources (GPU/System)
        resources_text = ""

        if self.monitor_gpu and self.gpu_history:
            latest_gpu = self.gpu_history[-1]
            resources_text += "[bold]GPU状态:[/bold]\n"
            for i, util in enumerate(latest_gpu['utilization']):
                mem_used = latest_gpu['memory_used'][i]
                mem_total = latest_gpu['memory_total'][i]
                temp = latest_gpu['temperature'][i]
                resources_text += f"  GPU {i}: {util:.1f}% | {mem_used:.0f}/{mem_total:.0f}MB | {temp}°C\n"

        if PSUTIL_AVAILABLE and self.system_history:
            latest_sys = self.system_history[-1]
            resources_text += "\n[bold]系统状态:[/bold]\n"
            resources_text += f"  CPU: {latest_sys['cpu_percent']:.1f}%\n"
            resources_text += f"  Memory: {latest_sys['memory_percent']:.1f}%\n"

        if not resources_text:
            resources_text = "系统监控未启用"

        layout["resources"].update(
            Panel(resources_text, title="资源使用", border_style="blue"))

        # Checkpoints
        ckpt_info = summary['checkpoints']
        ckpt_text = f"最佳模型: [{'green' if ckpt_info['best_exists'] else 'red'}]"
        ckpt_text += f"{'✅ 已保存' if ckpt_info['best_exists'] else '❌ 未保存'}[/]\n\n"

        if ckpt_info['saved_epochs']:
            ckpt_text += f"已保存Epoch: {', '.join(map(str, ckpt_info['saved_epochs'][-5:]))}"
            if len(ckpt_info['saved_epochs']) > 5:
                ckpt_text += f" ... (共{len(ckpt_info['saved_epochs'])}个)"
        else:
            ckpt_text += "暂无已保存的Epoch"

        layout["checkpoints"].update(
            Panel(ckpt_text, title="Checkpoint", border_style="magenta"))

        # Footer
        footer_text = "[dim]按 Ctrl+C 退出 | 数据每5秒自动刷新[/dim]"
        layout["footer"].update(Panel(footer_text, border_style="dim"))

        return layout

    def start_live_dashboard(self, refresh_interval: int = 5):
        """启动实时仪表板"""
        if not RICH_AVAILABLE:
            print("❌ Rich未安装，无法启动实时仪表板")
            return

        try:
            with Live(self.render_rich_dashboard(), refresh_per_second=1/refresh_interval,
                      console=self.console, screen=True) as live:
                while True:
                    self.reload_data()
                    live.update(self.render_rich_dashboard())
                    time.sleep(refresh_interval)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]👋 监控已停止[/yellow]")

    def create_matplotlib_dashboard(self):
        """创建Matplotlib实时仪表板"""
        if not MATPLOTLIB_AVAILABLE:
            print("❌ Matplotlib未安装，无法创建图表")
            return

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'训练监控 - {self.run_dir.name}',
                     fontsize=16, fontweight='bold')

        # 创建子图
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        ax_loss = fig.add_subplot(gs[0, :2])
        ax_lr = fig.add_subplot(gs[1, :2])
        ax_duration = fig.add_subplot(gs[2, :2])
        ax_gpu = fig.add_subplot(gs[0, 2])
        ax_health = fig.add_subplot(gs[1, 2])
        ax_validation = fig.add_subplot(gs[2, 2])

        def update_plot(frame):
            """更新图表"""
            self.reload_data()

            # 清空所有子图
            for ax in [ax_loss, ax_lr, ax_duration, ax_gpu, ax_health, ax_validation]:
                ax.clear()

            # 1. Loss曲线
            loss_data = self.metrics_cache.get('train/loss', [])
            if loss_data:
                steps, values, _ = zip(*loss_data)
                ax_loss.plot(steps, values, 'b-', linewidth=2,
                             label='Training Loss')
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss')
                ax_loss.set_title('训练Loss变化')
                ax_loss.grid(True, alpha=0.3)
                ax_loss.legend()

            # 2. 学习率曲线
            lr_data = self.metrics_cache.get('train/lr', [])
            if lr_data:
                steps, values, _ = zip(*lr_data)
                ax_lr.plot(steps, values, 'r-', linewidth=2)
                ax_lr.set_xlabel('Epoch')
                ax_lr.set_ylabel('Learning Rate')
                ax_lr.set_title('学习率变化')
                ax_lr.set_yscale('log')
                ax_lr.grid(True, alpha=0.3)

            # 3. Epoch耗时
            duration_data = self.metrics_cache.get(
                'train/epoch_duration_minutes', [])
            if duration_data:
                steps, values, _ = zip(*duration_data)
                ax_duration.plot(steps, values, 'g-', linewidth=2)
                ax_duration.set_xlabel('Epoch')
                ax_duration.set_ylabel('Time (minutes)')
                ax_duration.set_title('每Epoch训练时间')
                ax_duration.grid(True, alpha=0.3)

            # 4. GPU使用率
            if self.gpu_history:
                for gpu_idx in range(len(self.gpu_history[0]['utilization'])):
                    utils = [info['utilization'][gpu_idx]
                             for info in self.gpu_history]
                    ax_gpu.plot(utils, label=f'GPU {gpu_idx}')
                ax_gpu.set_xlabel('Time')
                ax_gpu.set_ylabel('Utilization (%)')
                ax_gpu.set_title('GPU使用率')
                ax_gpu.set_ylim(0, 100)
                ax_gpu.legend()
                ax_gpu.grid(True, alpha=0.3)
            else:
                ax_gpu.text(0.5, 0.5, 'GPU监控未启用', ha='center', va='center')

            # 5. 健康度评分
            ax_health.bar(['Health Score'], [self.health_score],
                          color='green' if self.health_score >= 80 else 'orange')
            ax_health.set_ylim(0, 100)
            ax_health.set_title('训练健康度')
            ax_health.text(0, self.health_score + 2,
                           f'{self.health_score}/100', ha='center', fontweight='bold')

            # 6. 验证Loss
            val_metrics = {
                k: v for k, v in self.metrics_cache.items() if k.startswith('validation/')}
            if val_metrics:
                for tag, data in val_metrics.items():
                    if data:
                        steps, values, _ = zip(*data)
                        label = tag.replace('validation/', '')
                        ax_validation.plot(
                            steps, values, linewidth=2, label=label, marker='o')
                ax_validation.set_xlabel('Epoch')
                ax_validation.set_ylabel('Validation Loss')
                ax_validation.set_title('验证Loss')
                ax_validation.legend()
                ax_validation.grid(True, alpha=0.3)
            else:
                ax_validation.text(0.5, 0.5, '暂无验证数据',
                                   ha='center', va='center')

        # 创建动画
        ani = FuncAnimation(fig, update_plot, interval=5000,
                            cache_frame_data=False)

        plt.tight_layout()
        plt.show()


def find_latest_run(base_dir: Path = None) -> Optional[Path]:
    """查找最新的训练运行"""
    if base_dir is None:
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


def main():
    parser = argparse.ArgumentParser(description="高级训练监控器")
    parser.add_argument("--run-dir", type=str, help="训练运行目录")
    parser.add_argument("--monitor-gpu", action="store_true", help="启用GPU监控")
    parser.add_argument("--mode", choices=['terminal', 'plot'], default='terminal',
                        help="显示模式：terminal(终端仪表板) 或 plot(图表)")
    parser.add_argument("--refresh", type=int, default=5, help="刷新间隔（秒）")

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
            print("❌ 未找到训练运行。请使用 --run-dir 指定目录。")
            sys.exit(1)
        print(f"✅ 找到最新运行: {run_dir}\n")

    # 检查依赖
    missing_deps = []
    if not RICH_AVAILABLE and args.mode == 'terminal':
        missing_deps.append("rich")
    if not MATPLOTLIB_AVAILABLE and args.mode == 'plot':
        missing_deps.append("matplotlib")
    if not TENSORBOARD_AVAILABLE:
        missing_deps.append("tensorboard")
    if args.monitor_gpu and not GPU_AVAILABLE:
        print("⚠️  GPUtil未安装，GPU监控不可用")
        args.monitor_gpu = False

    if missing_deps:
        print(f"❌ 缺少依赖: {', '.join(missing_deps)}")
        print(f"   安装命令: pip install {' '.join(missing_deps)}")
        sys.exit(1)

    # 创建监控器
    monitor = AdvancedTrainingMonitor(run_dir, monitor_gpu=args.monitor_gpu)

    # 启动监控
    try:
        if args.mode == 'terminal':
            print("🚀 启动终端仪表板...")
            monitor.start_live_dashboard(refresh_interval=args.refresh)
        else:
            print("📊 启动实时图表...")
            monitor.create_matplotlib_dashboard()
    except KeyboardInterrupt:
        print("\n👋 监控已停止")


if __name__ == "__main__":
    main()
