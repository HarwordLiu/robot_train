# -*- coding: utf-8 -*-
"""
分层架构部署模块

支持分层人形机器人Diffusion Policy的部署和推理
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import lerobot_patches.custom_patches

from dataclasses import dataclass, field
import hydra
import gymnasium as gym
import imageio
import numpy
import torch
from tqdm import tqdm
import datetime
import time
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from torchvision.transforms.functional import to_tensor
from std_msgs.msg import Bool
import rospy
import threading

# 导入分层架构模块
from kuavo_train.wrapper.policy.humanoid.HumanoidDiffusionPolicy import HumanoidDiffusionPolicy
from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.random_utils import set_seed

from configs.deploy.config_inference import load_inference_config
from kuavo_deploy.utils.logging_utils import setup_logger

log_model = setup_logger("model")
log_robot = setup_logger("robot")

def pause_callback(msg):
    if msg.data:
        pause_flag.set()
    else:
        pause_flag.clear()

def stop_callback(msg):
    if msg.data:
        stop_flag.set()

pause_sub = rospy.Subscriber('/kuavo/pause_state', Bool, pause_callback, queue_size=10)
stop_sub = rospy.Subscriber('/kuavo/stop_state', Bool, stop_callback, queue_size=10)
stop_flag = threading.Event()
pause_flag = threading.Event()

def check_control_signals():
    """检查控制信号"""
    log_model.info(f"🔍 检查控制信号 - 暂停状态: {pause_flag}, 停止状态: {stop_flag}")

def img_preprocess(image, device="cpu"):
    return to_tensor(image).unsqueeze(0).to(device, non_blocking=True)

def depth_preprocess(depth, device="cpu", depth_range=[0, 1000]):
    """预处理深度图像"""
    depth = np.array(depth)
    depth = np.clip(depth, depth_range[0], depth_range[1])
    depth = (depth - depth_range[0]) / (depth_range[1] - depth_range[0])
    return torch.tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)

def setup_hierarchical_policy(pretrained_path, policy_type, device=torch.device("cuda")):
    """
    设置并加载分层架构策略模型

    Args:
        pretrained_path: 检查点路径
        policy_type: 策略类型 ('diffusion', 'act', 'hierarchical_diffusion')
        device: 设备

    Returns:
        加载的策略模型
    """

    if device.type == 'cpu':
        log_model.warning("Warning: Using CPU for inference, this may be slow.")
        time.sleep(3)

    if policy_type == 'hierarchical_diffusion':
        # 加载分层架构模型
        log_model.info("🤖 Loading Hierarchical Diffusion Policy...")
        policy = HumanoidDiffusionPolicy.from_pretrained(Path(pretrained_path), strict=True)

        # 打印分层架构信息
        if hasattr(policy, 'print_architecture_summary'):
            policy.print_architecture_summary()

        # 打印层性能统计
        if hasattr(policy, 'get_performance_stats'):
            stats = policy.get_performance_stats()
            log_model.info(f"📊 Layer performance stats: {stats}")

    elif policy_type == 'diffusion':
        # 传统diffusion模型
        log_model.info("📝 Loading Traditional Diffusion Policy...")
        policy = CustomDiffusionPolicyWrapper.from_pretrained(Path(pretrained_path), strict=True)

    elif policy_type == 'act':
        # ACT模型
        log_model.info("🎭 Loading ACT Policy...")
        policy = ACTPolicy.from_pretrained(Path(pretrained_path), strict=True)
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}. Supported: 'diffusion', 'act', 'hierarchical_diffusion'")

    policy.eval()
    policy.to(device)
    policy.reset()

    # Log model info
    log_model.info(f"✅ Model loaded from {pretrained_path}")
    log_model.info(f"📋 Model n_obs_steps: {policy.config.n_obs_steps}")
    log_model.info(f"🖥️  Model device: {device}")
    log_model.info(f"🔧 Policy type: {policy_type}")

    return policy

def hierarchical_inference_step(policy, observation, task_info=None):
    """
    分层架构推理步骤

    Args:
        policy: 分层策略模型
        observation: 观测数据
        task_info: 任务信息（用于层选择）

    Returns:
        action: 动作输出
        hierarchical_info: 分层架构特有信息
    """

    if hasattr(policy, 'scheduler') and policy.scheduler:
        # 分层架构推理
        with torch.no_grad():
            # 构建任务上下文
            if task_info is None:
                task_info = {
                    'task_complexity': 'medium',  # 默认中等复杂度
                    'requires_locomotion': False,  # 仅手臂任务
                    'requires_manipulation': True,
                    'safety_priority': True
                }

            # 分层推理
            outputs = policy.scheduler(observation, task_info)

            # 提取最终动作
            if 'final_action' in outputs:
                action = outputs['final_action']
            else:
                # 如果没有final_action，使用最高优先级的层输出
                for layer_name in ['safety', 'gait', 'manipulation', 'planning']:
                    if layer_name in outputs and 'action' in outputs[layer_name]:
                        action = outputs[layer_name]['action']
                        break
                else:
                    raise RuntimeError("No valid action output from hierarchical layers")

            # 构建分层信息
            hierarchical_info = {
                'active_layers': list(outputs.keys()),
                'layer_outputs': {k: v for k, v in outputs.items() if k != 'final_action'},
                'inference_time': outputs.get('_inference_stats', {}).get('total_time', 0),
                'within_budget': outputs.get('_inference_stats', {}).get('within_budget', True)
            }

            return action, hierarchical_info
    else:
        # 传统推理模式
        with torch.no_grad():
            action = policy.select_action(observation)
            hierarchical_info = {
                'mode': 'traditional',
                'active_layers': ['main'],
                'layer_outputs': {},
                'inference_time': 0,
                'within_budget': True
            }
            return action, hierarchical_info

def log_hierarchical_performance(hierarchical_info, step):
    """记录分层架构性能信息"""

    if hierarchical_info['mode'] == 'hierarchical':
        active_layers = hierarchical_info.get('active_layers', [])
        inference_time = hierarchical_info.get('inference_time', 0)
        within_budget = hierarchical_info.get('within_budget', True)

        budget_status = "✅" if within_budget else "❌"
        log_model.info(f"Step {step}: {budget_status} Layers: {active_layers}, Time: {inference_time:.2f}ms")

        # 每100步记录详细统计
        if step % 100 == 0:
            layer_outputs = hierarchical_info.get('layer_outputs', {})
            for layer_name, layer_output in layer_outputs.items():
                log_model.info(f"  📊 {layer_name}: {layer_output.get('execution_time', 0):.2f}ms")

def main(config_path: str, env: gym.Env):
    """分层架构主推理循环"""

    # 加载配置
    cfg = load_inference_config(config_path)

    use_delta = cfg.use_delta
    eval_episodes = cfg.eval_episodes
    device = torch.device(cfg.device)

    # 设置随机种子
    set_seed(cfg.seed)

    # 构建模型路径
    pretrained_path = f"outputs/train/{cfg.task}/{cfg.method}/{cfg.timestamp}/epoch{cfg.epoch}"

    # 加载分层策略
    policy = setup_hierarchical_policy(pretrained_path, cfg.policy_type, device)

    # 检查模型类型
    is_hierarchical = cfg.policy_type == 'hierarchical_diffusion'
    log_model.info(f"🔄 Hierarchical mode: {is_hierarchical}")

    # 推理循环
    results = []
    for episode in range(eval_episodes):

        log_model.info(f"🎯 Episode {episode + 1}/{eval_episodes}")

        # 重置环境和策略
        obs, info = env.reset()
        policy.reset()

        episode_reward = 0
        episode_length = 0
        success = False
        hierarchical_stats = {
            'total_inference_time': 0,
            'layer_activations': {},
            'budget_violations': 0
        }

        while True:
            # 检查控制信号
            if stop_flag.is_set():
                log_model.info("🛑 Stop signal received, terminating...")
                break

            while pause_flag.is_set():
                log_model.info("⏸️  Paused, waiting for resume...")
                time.sleep(0.1)

            # 预处理观测
            observation = {}

            # 处理图像观测
            for key in obs.keys():
                if 'image' in key.lower() or 'cam' in key.lower():
                    observation[f"observation.{key}"] = img_preprocess(obs[key], device)
                elif 'depth' in key.lower():
                    observation[f"observation.{key}"] = depth_preprocess(obs[key], device, cfg.depth_range)
                elif 'state' in key.lower():
                    observation[f"observation.{key}"] = torch.tensor(obs[key], dtype=torch.float32).unsqueeze(0).to(device)

            # 构建任务信息（针对分层架构）
            task_info = {
                'task_complexity': 'medium',
                'requires_locomotion': False,  # 仅手臂操作
                'requires_manipulation': True,
                'safety_priority': True,
                'episode': episode,
                'step': episode_length
            }

            # 分层推理
            if is_hierarchical:
                action, hierarchical_info = hierarchical_inference_step(policy, observation, task_info)

                # 更新统计
                hierarchical_stats['total_inference_time'] += hierarchical_info.get('inference_time', 0)
                if not hierarchical_info.get('within_budget', True):
                    hierarchical_stats['budget_violations'] += 1

                # 统计层激活
                for layer in hierarchical_info.get('active_layers', []):
                    hierarchical_stats['layer_activations'][layer] = hierarchical_stats['layer_activations'].get(layer, 0) + 1

                # 记录性能
                log_hierarchical_performance(hierarchical_info, episode_length)
            else:
                # 传统推理
                action = policy.select_action(observation)
                hierarchical_info = {'mode': 'traditional'}

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())

            episode_reward += reward
            episode_length += 1

            # 检查回合结束
            if terminated or truncated:
                success = info.get('is_success', False)
                break

        # 记录回合结果
        results.append({
            'episode': episode,
            'reward': episode_reward,
            'length': episode_length,
            'success': success,
            'hierarchical_stats': hierarchical_stats.copy()
        })

        # 记录回合统计
        log_model.info(f"📈 Episode {episode + 1} - Reward: {episode_reward:.3f}, Length: {episode_length}, Success: {success}")

        if is_hierarchical:
            avg_inference_time = hierarchical_stats['total_inference_time'] / max(episode_length, 1)
            log_model.info(f"⏱️  Avg inference time: {avg_inference_time:.2f}ms")
            log_model.info(f"🎯 Layer activations: {hierarchical_stats['layer_activations']}")
            log_model.info(f"⚠️  Budget violations: {hierarchical_stats['budget_violations']}")

        if stop_flag.is_set():
            break

    # 计算总体统计
    if results:
        avg_reward = np.mean([r['reward'] for r in results])
        success_rate = np.mean([r['success'] for r in results])
        avg_length = np.mean([r['length'] for r in results])

        log_model.info(f"🏆 Final Results - Episodes: {len(results)}")
        log_model.info(f"📊 Average Reward: {avg_reward:.3f}")
        log_model.info(f"✅ Success Rate: {success_rate:.1%}")
        log_model.info(f"📏 Average Length: {avg_length:.1f}")

        if is_hierarchical:
            # 分层架构特有统计
            total_budget_violations = sum(r['hierarchical_stats']['budget_violations'] for r in results)
            total_steps = sum(r['length'] for r in results)
            budget_violation_rate = total_budget_violations / max(total_steps, 1)

            log_model.info(f"⚡ Hierarchical Performance:")
            log_model.info(f"   Budget violation rate: {budget_violation_rate:.1%}")

            # 层激活统计
            all_activations = {}
            for r in results:
                for layer, count in r['hierarchical_stats']['layer_activations'].items():
                    all_activations[layer] = all_activations.get(layer, 0) + count

            log_model.info(f"   Layer usage: {all_activations}")

    return results

# 为了保持与原有接口的兼容性，提供setup_policy别名
def setup_policy(pretrained_path, policy_type, device=torch.device("cuda")):
    """兼容性接口"""
    return setup_hierarchical_policy(pretrained_path, policy_type, device)