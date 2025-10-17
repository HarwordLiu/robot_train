#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练监控脚本 - 实时显示训练进度和关键指标

功能：
- 实时监控训练日志
- 解析TensorBoard事件文件
- 显示关键指标（loss、lr、epoch进度等）
- 可视化训练趋势
- 评估训练状态（正常/过拟合/学习率异常等）
- 支持多种训练脚本（SmolVLA、Diffusion、Hierarchical等）

使用方法：
    # 监控最新的训练运行
    python kuavo_train/monitor_training.py
    
    # 监控指定的训练运行
    python kuavo_train/monitor_training.py --run-dir outputs/train/task1_moving_grasp/smolvla_sequential/run_20251017_120000
    
    # 显示可视化图表
    python kuavo_train/monitor_training.py --plot
    
    # 自动刷新模式（每N秒刷新一次）
    python kuavo_train/monitor_training.py --refresh 5
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import re

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard未安装，无法解析事件文件。请运行: pip install tensorboard")


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, run_dir: Path, enable_plot: bool = False):
        self.run_dir = Path(run_dir)
        self.enable_plot = enable_plot
        
        # 查找TensorBoard日志目录
        self.tb_log_dir = self._find_tensorboard_dir()
        
        # 初始化事件累加器
        self.event_acc = None
        if self.tb_log_dir and TENSORBOARD_AVAILABLE:
            self._init_event_accumulator()
        
        # 用于存储解析的数据
        self.metrics_history = {
            'train/loss': [],
            'train/lr': [],
            'train/epoch_duration_minutes': [],
        }
        
        self.last_update_time = None
        
    def _find_tensorboard_dir(self) -> Optional[Path]:
        """查找TensorBoard日志目录"""
        # TensorBoard日志默认在run_dir本身
        if (self.run_dir / 'events.out.tfevents').exists() or \
           any(f.name.startswith('events.out.tfevents') for f in self.run_dir.glob('events.out.tfevents.*')):
            return self.run_dir
        return None
    
    def _init_event_accumulator(self):
        """初始化事件累加器"""
        try:
            self.event_acc = EventAccumulator(str(self.tb_log_dir))
            self.event_acc.Reload()
        except Exception as e:
            print(f"⚠️  无法加载TensorBoard事件: {e}")
            self.event_acc = None
    
    def reload_events(self):
        """重新加载事件数据"""
        if self.event_acc:
            try:
                self.event_acc.Reload()
            except Exception as e:
                print(f"⚠️  重新加载事件失败: {e}")
    
    def get_scalar_data(self, tag: str) -> List[Tuple[float, float]]:
        """获取标量数据 (step, value)"""
        if not self.event_acc:
            return []
        
        try:
            events = self.event_acc.Scalars(tag)
            return [(e.step, e.value) for e in events]
        except KeyError:
            return []
    
    def get_latest_metrics(self) -> Dict:
        """获取最新的指标"""
        self.reload_events()
        
        metrics = {}
        
        # 获取训练loss
        loss_data = self.get_scalar_data('train/loss')
        if loss_data:
            self.metrics_history['train/loss'] = loss_data
            metrics['loss'] = loss_data[-1][1]
            metrics['epoch'] = int(loss_data[-1][0])
        
        # 获取学习率
        lr_data = self.get_scalar_data('train/lr')
        if lr_data:
            self.metrics_history['train/lr'] = lr_data
            metrics['lr'] = lr_data[-1][1]
        
        # 获取epoch耗时
        duration_data = self.get_scalar_data('train/epoch_duration_minutes')
        if duration_data:
            self.metrics_history['train/epoch_duration_minutes'] = duration_data
            metrics['epoch_duration'] = duration_data[-1][1]
        
        # 获取验证指标（可能有多个任务）
        if self.event_acc:
            all_tags = self.event_acc.Tags().get('scalars', [])
            validation_tags = [tag for tag in all_tags if tag.startswith('validation/')]
            
            if validation_tags:
                metrics['validation'] = {}
                for tag in validation_tags:
                    val_data = self.get_scalar_data(tag)
                    if val_data:
                        task_name = tag.replace('validation/', '').replace('_loss', '')
                        metrics['validation'][task_name] = val_data[-1][1]
        
        self.last_update_time = datetime.now()
        return metrics
    
    def get_checkpoint_info(self) -> Dict:
        """获取checkpoint信息"""
        info = {
            'best_exists': (self.run_dir / 'best').exists(),
            'latest_epoch': None,
            'saved_epochs': []
        }
        
        # 查找所有epoch checkpoints
        epoch_dirs = sorted(self.run_dir.glob('epoch*'))
        if epoch_dirs:
            info['saved_epochs'] = [int(d.name.replace('epoch', '')) for d in epoch_dirs]
            info['latest_epoch'] = max(info['saved_epochs'])
        
        return info
    
    def get_training_status(self, metrics: Dict) -> Dict:
        """评估训练状态"""
        status = {
            'status': '未知',
            'warnings': [],
            'suggestions': []
        }
        
        # 检查loss趋势
        loss_history = self.metrics_history.get('train/loss', [])
        if len(loss_history) >= 5:
            recent_losses = [v for s, v in loss_history[-5:]]
            
            # 检查是否收敛
            loss_std = sum((x - sum(recent_losses)/len(recent_losses))**2 for x in recent_losses) / len(recent_losses)
            loss_std = loss_std ** 0.5
            
            if loss_std < 0.01:
                status['status'] = '已收敛'
            elif recent_losses[-1] < recent_losses[0]:
                status['status'] = '正常下降'
            elif recent_losses[-1] > recent_losses[0] * 1.5:
                status['status'] = '异常上升'
                status['warnings'].append('⚠️  Loss异常上升，可能学习率过高或数据有问题')
            else:
                status['status'] = '震荡中'
        
        # 检查学习率
        lr_history = self.metrics_history.get('train/lr', [])
        if lr_history:
            current_lr = lr_history[-1][1]
            if current_lr < 1e-7:
                status['warnings'].append('⚠️  学习率过小，训练可能停滞')
            elif current_lr > 1e-2:
                status['warnings'].append('⚠️  学习率过大，可能导致不稳定')
        
        # 检查训练时间
        if len(loss_history) >= 2:
            total_epochs = len(loss_history)
            current_epoch = metrics.get('epoch', 0)
            
            if 'epoch_duration' in metrics:
                avg_duration = metrics['epoch_duration']
                estimated_total_time = avg_duration * total_epochs
                
                if estimated_total_time > 24 * 60:  # 超过24小时
                    status['suggestions'].append(f'💡 预计总训练时间: {estimated_total_time/60:.1f}小时，建议调整batch size或减少epoch')
        
        # 检查过拟合
        if 'validation' in metrics and loss_history:
            train_loss = metrics.get('loss')
            val_losses = list(metrics['validation'].values())
            if val_losses and train_loss:
                avg_val_loss = sum(val_losses) / len(val_losses)
                if avg_val_loss > train_loss * 1.5:
                    status['warnings'].append('⚠️  验证loss明显高于训练loss，可能过拟合')
                    status['suggestions'].append('💡 建议: 增加数据增强、使用正则化或减少模型复杂度')
        
        return status
    
    def format_time_delta(self, seconds: float) -> str:
        """格式化时间差"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分钟"
        else:
            return f"{seconds/3600:.1f}小时"
    
    def display_metrics(self, clear_screen: bool = True):
        """显示指标"""
        if clear_screen:
            os.system('clear' if os.name != 'nt' else 'cls')
        
        print("=" * 80)
        print("🤖 训练监控器")
        print("=" * 80)
        print(f"📁 运行目录: {self.run_dir}")
        print(f"🕐 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # 获取最新指标
        metrics = self.get_latest_metrics()
        
        if not metrics:
            print("\n⚠️  暂无训练数据，等待训练开始...")
            return
        
        # 显示训练进度
        print("\n📊 训练进度:")
        print("-" * 80)
        
        if 'epoch' in metrics:
            print(f"当前Epoch: {metrics['epoch'] + 1}")
        
        if 'loss' in metrics:
            # 计算loss变化
            loss_history = self.metrics_history.get('train/loss', [])
            if len(loss_history) >= 2:
                prev_loss = loss_history[-2][1]
                curr_loss = metrics['loss']
                loss_change = curr_loss - prev_loss
                loss_change_pct = (loss_change / prev_loss) * 100 if prev_loss != 0 else 0
                
                change_symbol = "📉" if loss_change < 0 else "📈"
                print(f"训练Loss: {metrics['loss']:.6f} {change_symbol} ({loss_change_pct:+.2f}%)")
            else:
                print(f"训练Loss: {metrics['loss']:.6f}")
        
        if 'lr' in metrics:
            print(f"学习率: {metrics['lr']:.2e}")
        
        if 'epoch_duration' in metrics:
            print(f"Epoch耗时: {metrics['epoch_duration']:.2f}分钟")
            
            # 估算剩余时间（假设总共要训练的epoch数）
            loss_history = self.metrics_history.get('train/loss', [])
            if len(loss_history) >= 2:
                current_epoch = metrics['epoch']
                # 尝试从配置文件推断总epoch数（简化处理，可以改进）
                # 这里假设一个默认值
                total_epochs_estimate = max(50, current_epoch + 1)
                remaining_epochs = total_epochs_estimate - (current_epoch + 1)
                estimated_remaining_time = remaining_epochs * metrics['epoch_duration']
                
                if remaining_epochs > 0:
                    print(f"预计剩余时间: {estimated_remaining_time:.1f}分钟 ({estimated_remaining_time/60:.1f}小时)")
        
        # 显示验证指标
        if 'validation' in metrics and metrics['validation']:
            print("\n📈 验证指标:")
            print("-" * 80)
            for task_name, val_loss in metrics['validation'].items():
                print(f"{task_name}: {val_loss:.6f}")
        
        # 显示Loss趋势
        loss_history = self.metrics_history.get('train/loss', [])
        if len(loss_history) >= 5:
            print("\n📉 Loss趋势 (最近10个epoch):")
            print("-" * 80)
            recent_history = loss_history[-10:]
            
            # 简单的ASCII图表
            min_loss = min(v for s, v in recent_history)
            max_loss = max(v for s, v in recent_history)
            loss_range = max_loss - min_loss if max_loss != min_loss else 1
            
            for step, loss in recent_history:
                bar_length = int(((loss - min_loss) / loss_range) * 40)
                bar = "█" * bar_length
                print(f"Epoch {int(step)+1:3d}: {loss:.6f} {bar}")
        
        # 显示checkpoint信息
        checkpoint_info = self.get_checkpoint_info()
        print("\n💾 Checkpoint状态:")
        print("-" * 80)
        print(f"最佳模型: {'✅ 已保存' if checkpoint_info['best_exists'] else '❌ 未保存'}")
        if checkpoint_info['saved_epochs']:
            print(f"已保存Epoch: {', '.join(map(str, checkpoint_info['saved_epochs']))}")
            print(f"最新Epoch: {checkpoint_info['latest_epoch']}")
        
        # 显示训练状态评估
        status = self.get_training_status(metrics)
        print("\n🔍 训练状态评估:")
        print("-" * 80)
        print(f"状态: {status['status']}")
        
        if status['warnings']:
            print("\n警告:")
            for warning in status['warnings']:
                print(f"  {warning}")
        
        if status['suggestions']:
            print("\n建议:")
            for suggestion in status['suggestions']:
                print(f"  {suggestion}")
        
        print("\n" + "=" * 80)
    
    def plot_metrics(self):
        """绘制指标图表"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('TkAgg')  # 使用交互式后端
        except ImportError:
            print("⚠️  Matplotlib未安装，无法绘制图表。请运行: pip install matplotlib")
            return
        
        self.reload_events()
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'训练监控 - {self.run_dir.name}', fontsize=16)
        
        # 1. Loss曲线
        loss_data = self.metrics_history.get('train/loss', [])
        if loss_data:
            steps, values = zip(*loss_data)
            axes[0, 0].plot(steps, values, 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('训练Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 学习率曲线
        lr_data = self.metrics_history.get('train/lr', [])
        if lr_data:
            steps, values = zip(*lr_data)
            axes[0, 1].plot(steps, values, 'r-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('学习率变化')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Epoch耗时
        duration_data = self.metrics_history.get('train/epoch_duration_minutes', [])
        if duration_data:
            steps, values = zip(*duration_data)
            axes[1, 0].plot(steps, values, 'g-', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Time (minutes)')
            axes[1, 0].set_title('每Epoch训练时间')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 验证Loss（如果有）
        if self.event_acc:
            all_tags = self.event_acc.Tags().get('scalars', [])
            validation_tags = [tag for tag in all_tags if tag.startswith('validation/')]
            
            if validation_tags:
                for tag in validation_tags:
                    val_data = self.get_scalar_data(tag)
                    if val_data:
                        steps, values = zip(*val_data)
                        task_name = tag.replace('validation/', '')
                        axes[1, 1].plot(steps, values, linewidth=2, label=task_name)
                
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Validation Loss')
                axes[1, 1].set_title('验证Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, '暂无验证数据', 
                               ha='center', va='center', 
                               transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('验证Loss')
        
        plt.tight_layout()
        plt.show()
    
    def save_report(self, output_file: Optional[Path] = None):
        """保存训练报告"""
        if output_file is None:
            output_file = self.run_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        metrics = self.get_latest_metrics()
        checkpoint_info = self.get_checkpoint_info()
        status = self.get_training_status(metrics)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("训练监控报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"运行目录: {self.run_dir}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("训练指标:\n")
            f.write("-" * 80 + "\n")
            for key, value in metrics.items():
                if key != 'validation':
                    f.write(f"{key}: {value}\n")
            
            if 'validation' in metrics:
                f.write("\n验证指标:\n")
                f.write("-" * 80 + "\n")
                for task, loss in metrics['validation'].items():
                    f.write(f"{task}: {loss}\n")
            
            f.write("\nCheckpoint状态:\n")
            f.write("-" * 80 + "\n")
            f.write(f"最佳模型: {'已保存' if checkpoint_info['best_exists'] else '未保存'}\n")
            if checkpoint_info['saved_epochs']:
                f.write(f"已保存Epoch: {', '.join(map(str, checkpoint_info['saved_epochs']))}\n")
            
            f.write("\n训练状态评估:\n")
            f.write("-" * 80 + "\n")
            f.write(f"状态: {status['status']}\n")
            
            if status['warnings']:
                f.write("\n警告:\n")
                for warning in status['warnings']:
                    f.write(f"  {warning}\n")
            
            if status['suggestions']:
                f.write("\n建议:\n")
                for suggestion in status['suggestions']:
                    f.write(f"  {suggestion}\n")
        
        print(f"✅ 报告已保存到: {output_file}")


def find_latest_run(base_dir: Path = None) -> Optional[Path]:
    """查找最新的训练运行"""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "outputs" / "train"
    
    if not base_dir.exists():
        return None
    
    # 查找所有run_*目录
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
    
    # 按修改时间排序，返回最新的
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return run_dirs[0]


def main():
    parser = argparse.ArgumentParser(description="训练监控器")
    parser.add_argument("--run-dir", type=str, help="训练运行目录")
    parser.add_argument("--plot", action="store_true", help="显示可视化图表")
    parser.add_argument("--refresh", type=int, default=0, 
                       help="自动刷新间隔（秒），0表示不刷新")
    parser.add_argument("--save-report", action="store_true", help="保存训练报告")
    
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
    
    # 创建监控器
    monitor = TrainingMonitor(run_dir, enable_plot=args.plot)
    
    # 检查TensorBoard可用性
    if not TENSORBOARD_AVAILABLE:
        print("⚠️  TensorBoard未安装，部分功能不可用")
        print("   安装命令: pip install tensorboard\n")
    
    # 显示指标
    try:
        if args.refresh > 0:
            # 自动刷新模式
            print(f"🔄 自动刷新模式 (每{args.refresh}秒刷新一次，按Ctrl+C退出)\n")
            while True:
                monitor.display_metrics(clear_screen=True)
                time.sleep(args.refresh)
        else:
            # 单次显示
            monitor.display_metrics(clear_screen=False)
        
        # 保存报告
        if args.save_report:
            monitor.save_report()
        
        # 显示图表
        if args.plot:
            print("\n📊 正在生成可视化图表...")
            monitor.plot_metrics()
    
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")
        sys.exit(0)


if __name__ == "__main__":
    main()

