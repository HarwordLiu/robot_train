"""
SmolVLA Diffusion 实时部署脚本

在仿真或真实机器人上实时运行 SmolVLA Diffusion 模型
"""

import os
import sys
import time
import torch
import numpy as np
import threading
import queue
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
from collections import deque

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from kuavo_train.wrapper.policy.smolvla import SmolVLADiffusionPolicyWrapper
from kuavo_sim_env.envs.kuavo_sim_env import KuavoSimEnv


class SmolVLADiffusionRealtime:
    """
    SmolVLA Diffusion 实时运行器
    """

    def __init__(self, config_path: str, model_path: str, device: str = "cuda"):
        """
        初始化
        """
        self.config_path = config_path
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 加载配置
        self.cfg = self._load_config(config_path)

        # 初始化组件
        self.env = None
        self.policy = None
        self.running = False

        # 性能优化
        self.action_queue = queue.Queue(maxsize=1)
        self.obs_buffer = deque(maxlen=1)
        self.inference_times = deque(maxlen=100)

        # 统计信息
        self.stats = {
            'total_steps': 0,
            'avg_inference_time': 0.0,
            'fps': 0.0,
        }

        print(f"🚀 SmolVLA Diffusion 实时部署初始化")
        print(f"   - 配置: {config_path}")
        print(f"   - 模型: {model_path}")
        print(f"   - 设备: {self.device}")

    def _load_config(self, config_path: str):
        """加载配置"""
        from omegaconf import OmegaConf
        return OmegaConf.load(config_path)

    def initialize(self):
        """初始化环境和模型"""
        print("\n📦 初始化环境和模型...")

        # 初始化环境
        self.env = KuavoSimEnv(
            host=self.cfg.env.host,
            port=self.cfg.env.port
        )
        print("✅ 环境初始化完成")

        # 加载模型
        self.policy = SmolVLADiffusionPolicyWrapper.from_pretrained(
            pretrained_name_or_path=self.model_path,
            apply_freezing=False
        )
        self.policy.to(self.device)
        self.policy.eval()

        # 预热模型
        self._warmup()
        print("✅ 模型初始化和预热完成")

    def _warmup(self):
        """模型预热"""
        print("🔥 模型预热中...")
        with torch.no_grad():
            # 创建虚拟输入
            dummy_batch = {
                'observation.images.h': torch.randn(1, 3, 512, 512).to(self.device),
                'observation.state': torch.randn(1, 32).to(self.device),
                'task': ['Warm up task'],
            }

            # 预热几次
            for _ in range(3):
                _ = self.policy.select_action(dummy_batch, num_inference_steps=5)

    def prepare_observation(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """准备观测数据"""
        batch = {}

        # 处理图像
        for key in self.cfg.observation.images:
            if key in obs:
                img = obs[key]
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()
                    if len(img.shape) == 3:
                        img = img.permute(2, 0, 1)  # HWC -> CHW
                    img = img / 255.0
                batch[key] = img.unsqueeze(0).to(self.device)

        # 处理深度
        for key in self.cfg.observation.depth:
            if key in obs:
                depth = obs[key]
                if isinstance(depth, np.ndarray):
                    depth = torch.from_numpy(depth).float()
                    if len(depth.shape) == 2:
                        depth = depth.unsqueeze(0)
                batch[key] = depth.unsqueeze(0).to(self.device)

        # 处理状态
        if 'observation.state' in obs:
            state = obs['observation.state']
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()

            # 填充到32维
            if len(state) < 32:
                padding = torch.zeros(32 - len(state))
                state = torch.cat([state, padding], dim=-1)

            batch['observation.state'] = state.unsqueeze(0).to(self.device)

        # 添加语言指令
        batch['task'] = [self.cfg.task.language_instruction]

        return batch

    def inference_worker(self):
        """推理工作线程"""
        while self.running:
            try:
                # 获取最新观测
                if not self.obs_buffer:
                    time.sleep(0.001)
                    continue

                obs = self.obs_buffer[-1]

                # 准备输入
                batch = self.prepare_observation(obs)

                # 推理
                start_time = time.time()
                with torch.no_grad():
                    if self.cfg.optimization.use_amp:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            actions = self.policy.select_action(
                                batch,
                                num_inference_steps=self.cfg.policy.inference.num_inference_steps
                            )
                    else:
                        actions = self.policy.select_action(
                            batch,
                            num_inference_steps=self.cfg.policy.inference.num_inference_steps
                        )

                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)

                # 提取第一个动作
                action = actions[0, 0].cpu().numpy()[:16]  # 裁剪到16维

                # 放入队列
                try:
                    self.action_queue.put(action, timeout=0.01)
                except queue.Full:
                    # 队列满了，丢弃旧动作
                    try:
                        self.action_queue.get_nowait()
                        self.action_queue.put(action, timeout=0.01)
                    except:
                        pass

            except Exception as e:
                print(f"⚠️ 推理错误: {e}")
                continue

    def run(self):
        """运行主循环"""
        print("\n🏃 开始实时运行...")
        print("按 Ctrl+C 停止")

        # 重置环境
        obs = self.env.reset()
        self.running = True

        # 启动推理线程
        inference_thread = threading.Thread(target=self.inference_worker, daemon=True)
        inference_thread.start()

        # 控制循环
        control_freq = self.cfg.task.action.control_frequency
        control_period = 1.0 / control_freq

        last_print_time = time.time()
        step_count = 0

        try:
            while self.running:
                loop_start = time.time()

                # 获取观测
                obs = self.env.get_observation()
                self.obs_buffer.append(obs)

                # 获取动作
                action = None
                try:
                    action = self.action_queue.get(timeout=0.01)
                except queue.Empty:
                    # 没有新动作，使用零动作或保持上一动作
                    action = np.zeros(16)  # Kuavo 16自由度

                # 执行动作
                self.env.step(action)

                # 统计
                step_count += 1
                self.stats['total_steps'] = step_count

                # 打印统计信息（每5秒）
                current_time = time.time()
                if current_time - last_print_time > 5.0:
                    self._print_stats(current_time - last_print_time)
                    last_print_time = current_time

                # 控制频率
                elapsed = time.time() - loop_start
                if elapsed < control_period:
                    time.sleep(control_period - elapsed)

        except KeyboardInterrupt:
            print("\n⏹️ 停止运行")
        finally:
            self.running = False

    def _print_stats(self, duration: float):
        """打印统计信息"""
        # 计算FPS
        fps = self.stats['total_steps'] / duration if duration > 0 else 0

        # 计算平均推理时间
        if self.inference_times:
            avg_inference_time = np.mean(list(self.inference_times)) * 1000
            min_inference_time = np.min(list(self.inference_times)) * 1000
            max_inference_time = np.max(list(self.inference_times)) * 1000
        else:
            avg_inference_time = min_inference_time = max_inference_time = 0

        print(f"\n📊 运行统计 (最近{duration:.1f}秒):")
        print(f"   - FPS: {fps:.1f}")
        print(f"   - 总步数: {self.stats['total_steps']}")
        print(f"   - 推理时间: 平均 {avg_inference_time:.1f}ms, "
              f"最快 {min_inference_time:.1f}ms, "
              f"最慢 {max_inference_time:.1f}ms")
        print(f"   - 队列大小: {self.action_queue.qsize()}")

        # 重置统计
        self.stats['total_steps'] = 0
        self.inference_times.clear()


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="SmolVLA Diffusion 实时部署")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/deploy/kuavo_smolvla_diffusion_sim_env.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (cuda/cpu)"
    )

    args = parser.parse_args()

    # 创建运行器
    runner = SmolVLADiffusionRealtime(
        config_path=args.config,
        model_path=args.model,
        device=args.device
    )

    # 初始化
    runner.initialize()

    # 运行
    runner.run()


if __name__ == "__main__":
    main()