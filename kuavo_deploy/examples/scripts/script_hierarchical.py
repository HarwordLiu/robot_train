# -*- coding: utf-8 -*-
"""
分层架构机器人控制脚本

提供基于分层架构的机械臂运动控制、轨迹回放等功能

使用示例:
  python script_hierarchical.py --task go --config /path/to/hierarchical_config.yaml
  python script_hierarchical.py --task run --config /path/to/hierarchical_config.yaml
  python script_hierarchical.py --task go_run --config /path/to/hierarchical_config.yaml
"""

import rospy
import rosbag
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.kuavo_env.KuavoBaseRosEnv import KuavoBaseRosEnv
from configs.deploy.config_inference import load_inference_config
from configs.deploy.config_kuavo_env import load_kuavo_env_config
import gymnasium as gym

import numpy as np
import signal
import sys, os
import threading
import subprocess
import traceback

from std_msgs.msg import Bool

# 使用分层架构的评估模块
from kuavo_deploy.examples.eval.eval_kuavo_hierarchical import main as hierarchical_main

# 配置日志
log_model = setup_logger("hierarchical_model", "DEBUG")
log_robot = setup_logger("hierarchical_robot", "DEBUG")

# 控制变量
class HierarchicalArmMoveController:
    def __init__(self):
        self.paused = False
        self.should_stop = False
        self.lock = threading.Lock()
        self.current_layer_stats = {}

    def pause(self):
        with self.lock:
            self.paused = True
            log_model.info("⏸️  Hierarchical controller paused")

    def resume(self):
        with self.lock:
            self.paused = False
            log_model.info("▶️  Hierarchical controller resumed")

    def stop(self):
        with self.lock:
            self.should_stop = True
            log_model.info("🛑 Hierarchical controller stopping")

    def is_paused(self):
        with self.lock:
            return self.paused

    def should_terminate(self):
        with self.lock:
            return self.should_stop

    def update_layer_stats(self, stats):
        """更新层统计信息"""
        with self.lock:
            self.current_layer_stats = stats

    def get_layer_stats(self):
        with self.lock:
            return self.current_layer_stats.copy()

# 全局控制器
controller = HierarchicalArmMoveController()

def signal_handler_pause(signum, frame):
    """暂停/恢复信号处理"""
    if controller.is_paused():
        controller.resume()
    else:
        controller.pause()

def signal_handler_stop(signum, frame):
    """停止信号处理"""
    controller.stop()

# 设置信号处理
signal.signal(signal.SIGUSR1, signal_handler_pause)  # 暂停/恢复
signal.signal(signal.SIGUSR2, signal_handler_stop)   # 停止

def print_layer_performance(stats):
    """打印层性能信息"""
    if stats:
        log_model.info("🏗️  Layer Performance:")
        for layer_name, layer_stats in stats.items():
            if isinstance(layer_stats, dict):
                exec_time = layer_stats.get('execution_time', 0)
                active = layer_stats.get('active', False)
                status = "🟢" if active else "🔴"
                log_model.info(f"   {status} {layer_name}: {exec_time:.2f}ms")

def enhanced_bag_playback(bag_path: str, env: gym.Env, speed_factor: float = 1.0,
                         reverse: bool = False, target_frame: Optional[int] = None):
    """增强的bag回放，支持分层架构监控"""

    if not os.path.exists(bag_path):
        log_robot.error(f"❌ Bag file not found: {bag_path}")
        return False

    log_robot.info(f"🎬 Starting hierarchical bag playback: {bag_path}")
    log_robot.info(f"⚙️  Speed factor: {speed_factor}, Reverse: {reverse}, Target frame: {target_frame}")

    try:
        bag = rosbag.Bag(bag_path, 'r')

        # 获取状态消息
        state_msgs = []
        for topic, msg, t in bag.read_messages(topics=['/joint_cmd']):
            state_msgs.append((msg, t))

        if reverse:
            state_msgs.reverse()

        if target_frame:
            state_msgs = state_msgs[:target_frame]

        log_robot.info(f"📊 Found {len(state_msgs)} frames to replay")

        # 回放控制
        frame_count = 0
        for i, (msg, timestamp) in enumerate(state_msgs):

            # 检查控制信号
            while controller.is_paused():
                log_robot.info("⏸️  Playback paused...")
                time.sleep(0.1)

            if controller.should_terminate():
                log_robot.info("🛑 Playback terminated")
                break

            # 执行动作
            try:
                env.step(np.array(msg.data))
                frame_count += 1

                # 每100帧显示进度
                if frame_count % 100 == 0:
                    progress = (i + 1) / len(state_msgs) * 100
                    log_robot.info(f"📈 Playback progress: {progress:.1f}% ({frame_count} frames)")

                    # 显示层统计信息
                    layer_stats = controller.get_layer_stats()
                    print_layer_performance(layer_stats)

            except Exception as e:
                log_robot.error(f"❌ Error executing frame {i}: {e}")
                continue

            # 控制播放速度
            if speed_factor < 1.0:
                time.sleep((1.0 / speed_factor - 1.0) * 0.1)

        bag.close()
        log_robot.info(f"✅ Playback completed: {frame_count} frames executed")
        return True

    except Exception as e:
        log_robot.error(f"❌ Bag playback failed: {e}")
        traceback.print_exc()
        return False

def task_go(config_path: str):
    """分层架构：到达工作位置任务"""
    log_model.info("🎯 Hierarchical Task: GO - Moving to work position")

    # 加载配置
    cfg = load_kuavo_env_config(config_path)

    try:
        # 创建环境
        env = gym.make(cfg.env_name, config_path=config_path, render_mode="rgb_array")
        log_robot.info(f"🌍 Environment created: {cfg.env_name}")

        # 回放到工作位置
        if hasattr(cfg, 'go_bag_path') and cfg.go_bag_path:
            success = enhanced_bag_playback(cfg.go_bag_path, env, speed_factor=1.0)
            if success:
                log_model.info("✅ Successfully reached work position using hierarchical control")
            else:
                log_model.error("❌ Failed to reach work position")
        else:
            log_model.warning("⚠️  No go_bag_path specified, skipping movement")

        env.close()

    except Exception as e:
        log_model.error(f"❌ Task GO failed: {e}")
        traceback.print_exc()

def task_run(config_path: str):
    """分层架构：运行模型任务"""
    log_model.info("🤖 Hierarchical Task: RUN - Running hierarchical model inference")

    try:
        # 创建环境
        cfg = load_kuavo_env_config(config_path)
        env = gym.make(cfg.env_name, config_path=config_path, render_mode="rgb_array")
        log_robot.info(f"🌍 Environment created: {cfg.env_name}")

        # 监控线程
        def monitor_performance():
            while not controller.should_terminate():
                time.sleep(5)  # 每5秒检查一次
                if not controller.is_paused():
                    layer_stats = controller.get_layer_stats()
                    if layer_stats:
                        print_layer_performance(layer_stats)

        monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
        monitor_thread.start()

        # 运行分层架构推理
        results = hierarchical_main(config_path, env)

        # 输出结果摘要
        if results:
            success_rate = np.mean([r['success'] for r in results])
            avg_reward = np.mean([r['reward'] for r in results])
            log_model.info(f"🏆 Hierarchical inference completed - Success rate: {success_rate:.1%}, Avg reward: {avg_reward:.3f}")

        env.close()

    except Exception as e:
        log_model.error(f"❌ Task RUN failed: {e}")
        traceback.print_exc()

def task_go_run(config_path: str):
    """分层架构：到达+运行任务"""
    log_model.info("🎯🤖 Hierarchical Task: GO_RUN - Move to position then run model")

    log_model.info("📍 Phase 1: Moving to work position...")
    task_go(config_path)

    if not controller.should_terminate():
        log_model.info("🤖 Phase 2: Starting hierarchical inference...")
        task_run(config_path)

    log_model.info("✅ Hierarchical GO_RUN task completed")

def task_here_run(config_path: str):
    """分层架构：插值到最后一帧+运行"""
    log_model.info("📐🤖 Hierarchical Task: HERE_RUN - Interpolate to last frame then run")

    cfg = load_kuavo_env_config(config_path)

    try:
        env = gym.make(cfg.env_name, config_path=config_path, render_mode="rgb_array")

        # 插值到最后一帧
        if hasattr(cfg, 'go_bag_path') and cfg.go_bag_path:
            log_robot.info("📐 Interpolating to last frame...")
            # 只回放最后几帧以到达目标位置
            enhanced_bag_playback(cfg.go_bag_path, env, target_frame=-5)

        if not controller.should_terminate():
            log_model.info("🤖 Starting hierarchical inference from last frame...")
            results = hierarchical_main(config_path, env)

            if results:
                success_rate = np.mean([r['success'] for r in results])
                log_model.info(f"🎯 HERE_RUN success rate: {success_rate:.1%}")

        env.close()

    except Exception as e:
        log_model.error(f"❌ Task HERE_RUN failed: {e}")
        traceback.print_exc()

def task_back_to_zero(config_path: str):
    """分层架构：回到零位"""
    log_model.info("🏠 Hierarchical Task: BACK_TO_ZERO - Returning to zero position")

    cfg = load_kuavo_env_config(config_path)

    try:
        env = gym.make(cfg.env_name, config_path=config_path, render_mode="rgb_array")

        # 反向回放回到零位
        if hasattr(cfg, 'go_bag_path') and cfg.go_bag_path:
            log_robot.info("🔄 Reversing to zero position...")
            enhanced_bag_playback(cfg.go_bag_path, env, reverse=True)
            log_model.info("✅ Returned to zero position")
        else:
            log_model.warning("⚠️  No go_bag_path specified for return movement")

        env.close()

    except Exception as e:
        log_model.error(f"❌ Task BACK_TO_ZERO failed: {e}")
        traceback.print_exc()

def main():
    """分层架构主函数"""
    parser = argparse.ArgumentParser(description="Hierarchical Humanoid Robot Control Script")
    parser.add_argument("--task", type=str, required=True,
                       choices=["go", "run", "go_run", "here_run", "back_to_zero"],
                       help="Task to execute")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to hierarchical configuration file")
    parser.add_argument("--dry_run", action="store_true",
                       help="Dry run mode - only show what would be executed")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        log_model.error(f"❌ Config file not found: {args.config}")
        return

    # 初始化ROS
    rospy.init_node('hierarchical_kuavo_control', anonymous=True)

    log_model.info("🤖 Hierarchical Kuavo Robot Control Starting...")
    log_model.info(f"📝 Task: {args.task}")
    log_model.info(f"⚙️  Config: {args.config}")
    log_model.info(f"🔍 Dry run: {args.dry_run}")
    log_model.info(f"📢 Verbose: {args.verbose}")

    if args.dry_run:
        log_model.info("🔍 DRY RUN MODE - No actual execution")
        log_model.info(f"Would execute hierarchical task: {args.task}")
        return

    # 设置详细日志
    if args.verbose:
        log_model.setLevel("DEBUG")
        log_robot.setLevel("DEBUG")

    try:
        # 执行任务
        task_functions = {
            "go": task_go,
            "run": task_run,
            "go_run": task_go_run,
            "here_run": task_here_run,
            "back_to_zero": task_back_to_zero
        }

        task_func = task_functions[args.task]
        log_model.info(f"🚀 Starting hierarchical task: {args.task}")

        task_func(args.config)

        if not controller.should_terminate():
            log_model.info("✅ Hierarchical task completed successfully")
        else:
            log_model.info("⏹️  Hierarchical task terminated by user")

    except KeyboardInterrupt:
        log_model.info("⏹️  Hierarchical control interrupted by user")
        controller.stop()
    except Exception as e:
        log_model.error(f"❌ Hierarchical control failed: {e}")
        traceback.print_exc()
    finally:
        log_model.info("🏁 Hierarchical control finished")

if __name__ == "__main__":
    main()