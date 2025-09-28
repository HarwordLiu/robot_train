# -*- coding: utf-8 -*-
"""
分层架构自动测试脚本

基于分层架构的仿真自动测试，支持多回合评估和性能分析

使用示例:
  python script_hierarchical_auto_test.py --task auto_test --config /path/to/hierarchical_config.yaml
"""

import rospy
import time
import argparse
from pathlib import Path
import gymnasium as gym
import numpy as np
import signal
import sys, os
import threading
import traceback
import json
from datetime import datetime

from kuavo_deploy.utils.logging_utils import setup_logger
from configs.deploy.config_inference import load_inference_config
from configs.deploy.config_kuavo_env import load_kuavo_env_config

# 使用分层架构的评估模块
from kuavo_deploy.examples.eval.eval_kuavo_hierarchical import main as hierarchical_main

# 配置日志
log_model = setup_logger("hierarchical_auto_test", "DEBUG")
log_robot = setup_logger("hierarchical_robot", "DEBUG")

class HierarchicalAutoTestController:
    """分层架构自动测试控制器"""

    def __init__(self):
        self.paused = False
        self.should_stop = False
        self.lock = threading.Lock()
        self.test_results = []
        self.current_episode = 0
        self.start_time = None

    def pause(self):
        with self.lock:
            self.paused = True
            log_model.info("⏸️  Auto test paused")

    def resume(self):
        with self.lock:
            self.paused = False
            log_model.info("▶️  Auto test resumed")

    def stop(self):
        with self.lock:
            self.should_stop = True
            log_model.info("🛑 Auto test stopping")

    def is_paused(self):
        with self.lock:
            return self.paused

    def should_terminate(self):
        with self.lock:
            return self.should_stop

    def add_result(self, result):
        with self.lock:
            self.test_results.append(result)

    def get_results(self):
        with self.lock:
            return self.test_results.copy()

    def set_current_episode(self, episode):
        with self.lock:
            self.current_episode = episode

    def get_current_episode(self):
        with self.lock:
            return self.current_episode

    def start_timer(self):
        with self.lock:
            self.start_time = time.time()

    def get_elapsed_time(self):
        with self.lock:
            if self.start_time:
                return time.time() - self.start_time
            return 0

# 全局控制器
auto_test_controller = HierarchicalAutoTestController()

def signal_handler_pause(signum, frame):
    """暂停/恢复信号处理"""
    if auto_test_controller.is_paused():
        auto_test_controller.resume()
    else:
        auto_test_controller.pause()

def signal_handler_stop(signum, frame):
    """停止信号处理"""
    auto_test_controller.stop()

# 设置信号处理
signal.signal(signal.SIGUSR1, signal_handler_pause)  # 暂停/恢复
signal.signal(signal.SIGUSR2, signal_handler_stop)   # 停止

def analyze_hierarchical_performance(results):
    """分析分层架构性能"""

    if not results:
        log_model.warning("⚠️  No results to analyze")
        return

    total_episodes = len(results)
    successful_episodes = sum(1 for r in results if r.get('success', False))
    success_rate = successful_episodes / total_episodes

    rewards = [r.get('reward', 0) for r in results]
    lengths = [r.get('length', 0) for r in results]

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_length = np.mean(lengths)
    std_length = np.std(lengths)

    log_model.info("🏆 === Hierarchical Performance Analysis ===")
    log_model.info(f"📊 Total Episodes: {total_episodes}")
    log_model.info(f"✅ Success Rate: {success_rate:.1%} ({successful_episodes}/{total_episodes})")
    log_model.info(f"🎯 Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
    log_model.info(f"📏 Average Length: {avg_length:.1f} ± {std_length:.1f}")

    # 分层架构特有分析
    hierarchical_stats = {
        'total_inference_time': 0,
        'total_budget_violations': 0,
        'total_steps': 0,
        'layer_activations': {},
        'layer_performance': {}
    }

    for result in results:
        h_stats = result.get('hierarchical_stats', {})
        hierarchical_stats['total_inference_time'] += h_stats.get('total_inference_time', 0)
        hierarchical_stats['total_budget_violations'] += h_stats.get('budget_violations', 0)
        hierarchical_stats['total_steps'] += result.get('length', 0)

        # 累积层激活统计
        layer_activations = h_stats.get('layer_activations', {})
        for layer, count in layer_activations.items():
            hierarchical_stats['layer_activations'][layer] = hierarchical_stats['layer_activations'].get(layer, 0) + count

    # 计算平均性能指标
    if hierarchical_stats['total_steps'] > 0:
        avg_inference_time = hierarchical_stats['total_inference_time'] / hierarchical_stats['total_steps']
        budget_violation_rate = hierarchical_stats['total_budget_violations'] / hierarchical_stats['total_steps']

        log_model.info("⚡ === Hierarchical Architecture Analysis ===")
        log_model.info(f"⏱️  Average Inference Time: {avg_inference_time:.2f}ms")
        log_model.info(f"⚠️  Budget Violation Rate: {budget_violation_rate:.1%}")
        log_model.info(f"🏗️  Layer Usage Statistics:")

        total_activations = sum(hierarchical_stats['layer_activations'].values())
        for layer, count in hierarchical_stats['layer_activations'].items():
            usage_rate = count / total_activations if total_activations > 0 else 0
            log_model.info(f"   📊 {layer}: {count} activations ({usage_rate:.1%})")

    return {
        'total_episodes': total_episodes,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'hierarchical_stats': hierarchical_stats
    }

def save_test_results(results, analysis, config_path, output_dir="outputs/hierarchical_test_results"):
    """保存测试结果"""

    try:
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hierarchical_test_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        # 准备保存数据
        save_data = {
            'timestamp': timestamp,
            'config_path': config_path,
            'results': results,
            'analysis': analysis,
            'metadata': {
                'total_episodes': len(results),
                'test_duration': auto_test_controller.get_elapsed_time(),
                'hierarchical_architecture': True
            }
        }

        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        log_model.info(f"💾 Test results saved to: {filepath}")

        # 创建简化的摘要文件
        summary_filename = f"hierarchical_summary_{timestamp}.txt"
        summary_filepath = os.path.join(output_dir, summary_filename)

        with open(summary_filepath, 'w') as f:
            f.write("🤖 Hierarchical Architecture Test Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Config: {config_path}\n")
            f.write(f"Total Episodes: {analysis['total_episodes']}\n")
            f.write(f"Success Rate: {analysis['success_rate']:.1%}\n")
            f.write(f"Average Reward: {analysis['avg_reward']:.3f}\n")
            f.write(f"Average Length: {analysis['avg_length']:.1f}\n\n")

            f.write("🏗️  Hierarchical Performance:\n")
            h_stats = analysis['hierarchical_stats']
            if h_stats['total_steps'] > 0:
                avg_time = h_stats['total_inference_time'] / h_stats['total_steps']
                violation_rate = h_stats['total_budget_violations'] / h_stats['total_steps']
                f.write(f"Average Inference Time: {avg_time:.2f}ms\n")
                f.write(f"Budget Violation Rate: {violation_rate:.1%}\n")

            f.write("\n📊 Layer Usage:\n")
            total_activations = sum(h_stats['layer_activations'].values())
            for layer, count in h_stats['layer_activations'].items():
                usage_rate = count / total_activations if total_activations > 0 else 0
                f.write(f"{layer}: {count} ({usage_rate:.1%})\n")

        log_model.info(f"📋 Test summary saved to: {summary_filepath}")

    except Exception as e:
        log_model.error(f"❌ Failed to save test results: {e}")

def progress_monitor():
    """进度监控线程"""

    while not auto_test_controller.should_terminate():
        time.sleep(10)  # 每10秒更新一次

        if not auto_test_controller.is_paused():
            current_episode = auto_test_controller.get_current_episode()
            elapsed_time = auto_test_controller.get_elapsed_time()
            results = auto_test_controller.get_results()

            if results:
                recent_success_rate = np.mean([r.get('success', False) for r in results[-10:]])
                log_model.info(f"📈 Progress - Episode: {current_episode}, "
                             f"Recent success rate: {recent_success_rate:.1%}, "
                             f"Elapsed: {elapsed_time:.1f}s")

def task_auto_test(config_path: str):
    """分层架构自动测试任务"""

    log_model.info("🧪 Hierarchical Task: AUTO_TEST - Starting automated testing")

    try:
        # 加载配置
        cfg = load_kuavo_env_config(config_path)
        inference_cfg = load_inference_config(config_path)

        log_model.info(f"🎯 Test Episodes: {inference_cfg.eval_episodes}")
        log_model.info(f"🤖 Policy Type: {inference_cfg.policy_type}")
        log_model.info(f"🌍 Environment: {cfg.env_name}")

        # 启动进度监控
        auto_test_controller.start_timer()
        monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
        monitor_thread.start()

        # 创建环境
        env = gym.make(cfg.env_name, config_path=config_path, render_mode="rgb_array")
        log_robot.info(f"✅ Environment created: {cfg.env_name}")

        # 运行分层架构测试
        log_model.info("🚀 Starting hierarchical inference testing...")
        results = hierarchical_main(config_path, env)

        env.close()

        if results:
            # 分析结果
            analysis = analyze_hierarchical_performance(results)

            # 保存结果
            save_test_results(results, analysis, config_path)

            log_model.info("✅ Hierarchical auto test completed successfully")
            return analysis
        else:
            log_model.error("❌ No test results generated")
            return None

    except Exception as e:
        log_model.error(f"❌ Auto test failed: {e}")
        traceback.print_exc()
        return None

def main():
    """分层架构自动测试主函数"""

    parser = argparse.ArgumentParser(description="Hierarchical Architecture Auto Test Script")
    parser.add_argument("--task", type=str, required=True,
                       choices=["auto_test"],
                       help="Task to execute (currently only auto_test)")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to hierarchical configuration file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--save_results", action="store_true", default=True,
                       help="Save test results to file")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        log_model.error(f"❌ Config file not found: {args.config}")
        return

    # 初始化ROS
    rospy.init_node('hierarchical_kuavo_auto_test', anonymous=True)

    log_model.info("🧪 Hierarchical Kuavo Auto Test Starting...")
    log_model.info(f"📝 Task: {args.task}")
    log_model.info(f"⚙️  Config: {args.config}")
    log_model.info(f"📢 Verbose: {args.verbose}")

    # 设置详细日志
    if args.verbose:
        log_model.setLevel("DEBUG")
        log_robot.setLevel("DEBUG")

    try:
        if args.task == "auto_test":
            analysis = task_auto_test(args.config)

            if analysis:
                log_model.info("🎉 Hierarchical auto test completed successfully!")
                log_model.info(f"📊 Final Success Rate: {analysis['success_rate']:.1%}")
                log_model.info(f"🎯 Final Average Reward: {analysis['avg_reward']:.3f}")
            else:
                log_model.error("❌ Auto test failed to produce results")

    except KeyboardInterrupt:
        log_model.info("⏹️  Auto test interrupted by user")
        auto_test_controller.stop()
    except Exception as e:
        log_model.error(f"❌ Auto test failed: {e}")
        traceback.print_exc()
    finally:
        log_model.info("🏁 Hierarchical auto test finished")

if __name__ == "__main__":
    main()