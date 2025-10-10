# -*- coding: utf-8 -*-
"""
VLA Transformer Policy部署模块

支持VLA策略的部署和推理
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

# 导入VLA模块
from kuavo_train.wrapper.policy.vla.VLAPolicyWrapper import VLAPolicyWrapper
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

def img_preprocess(image, device="cpu"):
    return to_tensor(image).unsqueeze(0).to(device, non_blocking=True)

def depth_preprocess(depth, device="cpu", depth_range=[0, 1000]):
    """预处理深度图像"""
    depth = np.array(depth)
    depth = np.clip(depth, depth_range[0], depth_range[1])
    depth = (depth - depth_range[0]) / (depth_range[1] - depth_range[0])
    return torch.tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)

def setup_vla_policy(pretrained_path, device=torch.device("cuda")):
    """
    设置并加载VLA策略模型

    Args:
        pretrained_path: 检查点路径
        device: 设备

    Returns:
        加载的策略模型
    """

    if device.type == 'cpu':
        log_model.warning("Warning: Using CPU for inference, this may be slow.")
        time.sleep(3)

    log_model.info("🤖 Loading VLA Transformer Policy...")
    policy = VLAPolicyWrapper.from_pretrained(Path(pretrained_path), strict=True)

    policy.eval()
    policy.to(device)
    policy.reset()

    # Log model info
    log_model.info(f"✅ Model loaded from {pretrained_path}")
    log_model.info(f"📋 Model n_obs_steps: {policy.config.n_obs_steps}")
    log_model.info(f"🖥️  Model device: {device}")
    log_model.info(f"🔧 Policy type: VLA Transformer")
    log_model.info(f"📊 Token dim: {policy.config.token_embed_dim}")
    log_model.info(f"📊 Transformer depth: {policy.config.transformer_depth}")

    return policy

def main(config_path: str, env: gym.Env):
    """VLA主推理循环"""

    # 加载配置
    cfg = load_inference_config(config_path)
    from omegaconf import OmegaConf
    full_cfg = OmegaConf.load(config_path)

    use_delta = cfg.use_delta
    eval_episodes = cfg.eval_episodes
    device = torch.device(cfg.device)

    # 设置随机种子
    set_seed(cfg.seed)

    # 构建模型路径
    pretrained_path = f"outputs/train/{cfg.task}/{cfg.method}/{cfg.timestamp}/epoch{cfg.epoch}"

    # 加载VLA策略
    policy = setup_vla_policy(pretrained_path, device)

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

        # 记录推理时间
        inference_times = []

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

            # VLA推理
            start_time = time.time()
            with torch.no_grad():
                action = policy.select_action(observation)
            inference_time = (time.time() - start_time) * 1000  # 转为毫秒

            inference_times.append(inference_time)

            # 每100步记录一次推理时间
            if episode_length % 100 == 0:
                avg_time = np.mean(inference_times[-100:]) if len(inference_times) >= 100 else np.mean(inference_times)
                log_model.info(f"Step {episode_length}: Avg inference time: {avg_time:.2f}ms")

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())

            episode_reward += reward
            episode_length += 1

            # 检查回合结束
            if terminated or truncated:
                success = info.get('is_success', False)
                break

        # 记录回合结果
        avg_inference_time = np.mean(inference_times)
        results.append({
            'episode': episode,
            'reward': episode_reward,
            'length': episode_length,
            'success': success,
            'avg_inference_time_ms': avg_inference_time
        })

        # 记录回合统计
        log_model.info(f"📈 Episode {episode + 1} - Reward: {episode_reward:.3f}, Length: {episode_length}, Success: {success}")
        log_model.info(f"⏱️  Average inference time: {avg_inference_time:.2f}ms")

        if stop_flag.is_set():
            break

    # 计算总体统计
    if results:
        avg_reward = np.mean([r['reward'] for r in results])
        success_rate = np.mean([r['success'] for r in results])
        avg_length = np.mean([r['length'] for r in results])
        avg_inference = np.mean([r['avg_inference_time_ms'] for r in results])

        log_model.info(f"🏆 Final Results - Episodes: {len(results)}")
        log_model.info(f"📊 Average Reward: {avg_reward:.3f}")
        log_model.info(f"✅ Success Rate: {success_rate:.1%}")
        log_model.info(f"📏 Average Length: {avg_length:.1f}")
        log_model.info(f"⏱️  Average Inference Time: {avg_inference:.2f}ms")

    return results

# 为了保持与原有接口的兼容性，提供setup_policy别名
def setup_policy(pretrained_path, policy_type, device=torch.device("cuda")):
    """兼容性接口"""
    if policy_type != 'vla_transformer':
        raise ValueError(f"This script only supports 'vla_transformer' policy, got '{policy_type}'")
    return setup_vla_policy(pretrained_path, device)
