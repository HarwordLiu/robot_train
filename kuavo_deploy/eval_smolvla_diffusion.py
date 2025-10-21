"""
SmolVLA Diffusion 部署评估脚本

在仿真环境中评估 SmolVLA Diffusion 模型的性能
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import hydra
from omegaconf import OmegaConf

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from kuavo_train.wrapper.policy.smolvla import SmolVLADiffusionPolicyWrapper
from kuavo_sim_env.envs.kuavo_sim_env import KuavoSimEnv


class SmolVLADiffusionEvaluator:
    """
    SmolVLA Diffusion 评估器
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.policy.device if torch.cuda.is_available() else 'cpu')

        # 初始化环境
        self.env = KuavoSimEnv(
            host=cfg.env.host,
            port=cfg.env.port
        )

        # 初始化策略
        self.policy = self._load_policy()

        # 统计信息
        self.stats = {
            'episodes': [],
            'success_rate': 0.0,
            'avg_inference_time': 0.0,
            'placement_accuracy': 0.0,
        }

        print(f"\n🚀 SmolVLA Diffusion 评估器初始化完成")
        print(f"   - 设备: {self.device}")
        print(f"   - 推理步数: {cfg.policy.inference.num_inference_steps}")
        print(f"   - 使用 DDIM: {cfg.policy.inference.use_ddim_sampling}")

    def _load_policy(self) -> SmolVLADiffusionPolicyWrapper:
        """
        加载 SmolVLA Diffusion 模型
        """
        print("\n📦 加载 SmolVLA Diffusion 模型...")

        policy = SmolVLADiffusionPolicyWrapper.from_pretrained(
            pretrained_name_or_path=self.cfg.policy.pretrained_name_or_path,
            apply_freezing=False  # 推理模式不需要冻结
        )

        policy.to(self.device)
        policy.eval()

        # 优化推理
        if self.cfg.optimization.use_amp:
            policy = policy.half()  # 使用半精度

        print(f"✅ 模型加载成功")

        return policy

    def prepare_observation(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        准备观测数据
        """
        batch = {}

        # 处理图像
        for key in self.cfg.observation.images:
            if key in obs:
                img = obs[key]
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()
                    img = img.permute(2, 0, 1)  # HWC -> CHW
                    img = img / 255.0  # 归一化到 [0, 1]
                batch[key] = img.unsqueeze(0).to(self.device)  # 添加 batch 维

        # 处理深度
        for key in self.cfg.observation.depth:
            if key in obs:
                depth = obs[key]
                if isinstance(depth, np.ndarray):
                    depth = torch.from_numpy(depth).float()
                    depth = depth.unsqueeze(0)  # 添加通道维
                batch[key] = depth.unsqueeze(0).to(self.device)

        # 处理状态
        if 'observation.state' in obs:
            state = obs['observation.state']
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()

            # 填充到32维
            if self.cfg.preprocessing.state.padding and state.shape[-1] < 32:
                padding = torch.zeros(32 - state.shape[-1])
                state = torch.cat([state, padding], dim=-1)

            batch['observation.state'] = state.unsqueeze(0).to(self.device)

        # 添加语言指令
        batch['task'] = [self.cfg.task.language_instruction]

        return batch

    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        选择动作
        """
        # 准备观测
        batch = self.prepare_observation(obs)

        # 推理
        with torch.no_grad():
            start_time = time.time()

            # 使用 Diffusion 采样
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

        # 提取第一个动作
        action = actions[0, 0].cpu().numpy()

        # 裁剪到16维（Kuavo 实际维度）
        action = action[:16]

        return action, inference_time

    def evaluate_episode(self, episode_idx: int) -> Dict[str, Any]:
        """
        评估单个回合
        """
        print(f"\n📊 评估回合 {episode_idx + 1}/{self.cfg.evaluation.num_episodes}")

        # 重置环境
        obs = self.env.reset()
        done = False
        steps = 0
        inference_times = []
        placements = []

        while not done and steps < self.cfg.task.action.chunk_size:
            # 选择动作
            action, inference_time = self.select_action(obs)
            inference_times.append(inference_time)

            # 执行动作
            obs, reward, done, info = self.env.step(action)

            # 记录放置位置
            if 'placement_position' in info:
                placements.append(info['placement_position'])

            steps += 1

            # 检查超时
            if steps * self.cfg.task.action.control_frequency > self.cfg.evaluation.timeout.per_episode:
                print(f"   ⚠️ 回合超时")
                break

        # 计算成功率
        success = done and info.get('is_success', False)

        # 计算放置精度
        placement_accuracy = 0.0
        if placements:
            target_positions = info.get('target_positions', [])
            if target_positions:
                errors = [np.linalg.norm(p - t) for p, t in zip(placements, target_positions)]
                placement_accuracy = np.mean(errors)

        # 计算平均推理时间
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0

        episode_stats = {
            'success': success,
            'steps': steps,
            'inference_times': inference_times,
            'avg_inference_time': avg_inference_time,
            'placement_accuracy': placement_accuracy,
            'total_reward': info.get('total_reward', 0.0),
        }

        print(f"   - 成功: {'✅' if success else '❌'}")
        print(f"   - 步数: {steps}")
        print(f"   - 平均推理时间: {avg_inference_time*1000:.2f} ms")
        if placement_accuracy > 0:
            print(f"   - 放置精度: {placement_accuracy*100:.2f} cm")

        return episode_stats

    def evaluate(self):
        """
        执行完整评估
        """
        print(f"\n{'='*70}")
        print("🎯 开始 SmolVLA Diffusion 评估")
        print(f"{'='*70}")
        print(f"📋 评估配置:")
        print(f"   - 回合数: {self.cfg.evaluation.num_episodes}")
        print(f"   - 模型路径: {self.cfg.policy.pretrained_name_or_path}")
        print(f"   - 推理步数: {self.cfg.policy.inference.num_inference_steps}")
        print(f"{'='*70}\n")

        # 评估所有回合
        all_episodes = []
        for i in range(self.cfg.evaluation.num_episodes):
            episode_stats = self.evaluate_episode(i)
            all_episodes.append(episode_stats)
            self.stats['episodes'].append(episode_stats)

        # 计算总体统计
        successes = sum(e['success'] for e in all_episodes)
        self.stats['success_rate'] = successes / len(all_episodes)

        avg_inference_times = [e['avg_inference_time'] for e in all_episodes]
        self.stats['avg_inference_time'] = np.mean(avg_inference_times)

        placement_accuracies = [e['placement_accuracy'] for e in all_episodes if e['placement_accuracy'] > 0]
        if placement_accuracies:
            self.stats['placement_accuracy'] = np.mean(placement_accuracies)

        # 打印结果
        self.print_results()

        # 保存结果
        self.save_results()

        return self.stats

    def print_results(self):
        """
        打印评估结果
        """
        print(f"\n{'='*70}")
        print("📊 评估结果汇总")
        print(f"{'='*70}")
        print(f"✅ 成功率: {self.stats['success_rate']*100:.1f}%")
        print(f"⚡ 平均推理时间: {self.stats['avg_inference_time']*1000:.2f} ms")
        if self.stats['placement_accuracy'] > 0:
            print(f"🎯 平均放置精度: {self.stats['placement_accuracy']*100:.2f} cm")

        # 详细统计
        print(f"\n📈 详细统计:")
        episodes = self.stats['episodes']
        print(f"   - 成功回合: {sum(e['success'] for e in episodes)}/{len(episodes)}")
        print(f"   - 平均步数: {np.mean([e['steps'] for e in episodes]):.1f}")
        print(f"   - 最快推理: {min([e['avg_inference_time'] for e in episodes])*1000:.2f} ms")
        print(f"   - 最慢推理: {max([e['avg_inference_time'] for e in episodes])*1000:.2f} ms")

    def save_results(self):
        """
        保存评估结果
        """
        import json
        from datetime import datetime

        # 创建输出目录
        output_dir = Path(self.cfg.logging.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"eval_results_{timestamp}.json"

        # 准备保存数据
        save_data = {
            'config': OmegaConf.to_container(self.cfg, resolve=True),
            'stats': {
                'success_rate': float(self.stats['success_rate']),
                'avg_inference_time': float(self.stats['avg_inference_time']),
                'placement_accuracy': float(self.stats['placement_accuracy']),
            },
            'episodes': self.stats['episodes'],
            'timestamp': timestamp,
        }

        # 保存到文件
        with open(result_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"\n💾 结果已保存到: {result_file}")


@hydra.main(
    version_base=None,
    config_path="../configs/deploy",
    config_name="kuavo_smolvla_diffusion_sim_env"
)
def main(cfg):
    """
    主评估函数
    """
    # 创建评估器
    evaluator = SmolVLADiffusionEvaluator(cfg)

    # 执行评估
    stats = evaluator.evaluate()

    print(f"\n🎉 评估完成!")

    return stats


if __name__ == "__main__":
    main()