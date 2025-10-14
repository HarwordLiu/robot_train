# -*- coding: utf-8 -*-
"""
SmolVLA Policy Deployment Module (Enhanced)

增强版SmolVLA部署脚本 - 集成推理优化（无需重新训练）：
1. Action后处理：平滑滤波 + 精细操作增益
2. 精确Language Instruction：更准确的任务描述
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
from torchvision.transforms.functional import to_tensor, resize
from torchvision.transforms import InterpolationMode
from std_msgs.msg import Bool
import rospy
import threading

# Import SmolVLA modules
from kuavo_train.wrapper.policy.smolvla.SmolVLAPolicyWrapper import SmolVLAPolicyWrapper
from lerobot.utils.random_utils import set_seed

# ✨ 导入推理后处理模块
from kuavo_deploy.utils.action_postprocessing import ActionPostProcessor

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

def img_preprocess_smolvla(image, target_size=(512, 512), device="cpu"):
    """Preprocess image for SmolVLA (512x512)"""
    tensor_img = to_tensor(image)
    h, w = tensor_img.shape[-2:]
    target_h, target_w = target_size

    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    tensor_img = resize(tensor_img, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    tensor_img = torch.nn.functional.pad(
        tensor_img,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode='constant',
        value=0
    )

    return tensor_img.unsqueeze(0).to(device, non_blocking=True)

def depth_preprocess(depth, device="cpu", depth_range=[0, 1000]):
    """Preprocess depth image"""
    depth = np.array(depth)
    depth = np.clip(depth, depth_range[0], depth_range[1])
    depth = (depth - depth_range[0]) / (depth_range[1] - depth_range[0])
    return torch.tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)

def setup_smolvla_policy(pretrained_path, language_instruction, device=torch.device("cuda")):
    """Setup and load SmolVLA policy model"""

    if device.type == 'cpu':
        log_model.warning("Warning: Using CPU for inference, this may be slow.")
        time.sleep(3)

    log_model.info("🤖 Loading SmolVLA Policy (Enhanced)...")
    log_model.info(f"📝 Task Instruction: {language_instruction}")

    policy = SmolVLAPolicyWrapper.from_pretrained(Path(pretrained_path))

    policy.eval()
    policy.to(device)
    policy.reset()

    log_model.info(f"✅ Model loaded from {pretrained_path}")
    log_model.info(f"📋 Model n_obs_steps: {policy.config.n_obs_steps}")
    log_model.info(f"🖥️  Model device: {device}")
    log_model.info(f"🔧 Policy type: SmolVLA Sequential (Enhanced)")
    log_model.info(f"📊 VLM: {policy.config.vlm_model_name}")
    log_model.info(f"📊 Action dim: {policy.config.max_action_dim} (Kuavo uses first 16)")
    log_model.info(f"📊 Chunk size: {policy.config.chunk_size}")
    log_model.info(f"📊 Action steps: {policy.config.n_action_steps}")

    return policy

def main(config_path: str, env: gym.Env):
    """SmolVLA enhanced inference loop"""

    # Load config
    cfg = load_inference_config(config_path)
    from omegaconf import OmegaConf
    full_cfg = OmegaConf.load(config_path)

    use_delta = cfg.use_delta
    eval_episodes = cfg.eval_episodes
    device = torch.device(cfg.device)

    # ✨ 使用更精确的language instruction
    # 如果配置文件中有精确描述，使用配置中的；否则使用默认的
    language_instruction = cfg.get('language_instruction_enhanced', cfg.language_instruction)

    # 默认的精确instruction（如果配置文件没有）
    if language_instruction == cfg.language_instruction and 'push' in language_instruction:
        language_instruction = 'Pick up the moving object from the conveyor belt, place it precisely at the first target position on the table, then pick it up again and place it precisely at the second target position'
        log_model.info("📝 Using enhanced language instruction (precise placement)")

    # Set random seed
    set_seed(cfg.seed)

    # Build model path
    pretrained_path = f"outputs/train/{cfg.task}/{cfg.method}/{cfg.timestamp}/epoch{cfg.epoch}"

    # Load SmolVLA policy
    policy = setup_smolvla_policy(pretrained_path, language_instruction, device)

    # ✨ 初始化Action后处理器
    enable_postprocessing = cfg.get('enable_postprocessing', True)
    if enable_postprocessing:
        postprocessor = ActionPostProcessor(
            action_dim=16,
            enable_smoothing=cfg.get('enable_smoothing', True),
            enable_fine_gain=cfg.get('enable_fine_gain', True),
            enable_workspace_limit=cfg.get('enable_workspace_limit', True),
            enable_velocity_limit=cfg.get('enable_velocity_limit', True),
            smooth_alpha=cfg.get('smooth_alpha', 0.3),
            fine_motion_gain=cfg.get('fine_motion_gain', 1.5),
            max_velocity=cfg.get('max_velocity', 0.2),
            control_frequency=cfg.get('control_frequency', 10.0)
        )
        log_model.info("✨ Action Postprocessing Enabled:")
        log_model.info(f"   - Smoothing: {cfg.get('enable_smoothing', True)} (alpha={cfg.get('smooth_alpha', 0.3)})")
        log_model.info(f"   - Fine Gain: {cfg.get('enable_fine_gain', True)} (gain={cfg.get('fine_motion_gain', 1.5)}x)")
        log_model.info(f"   - Workspace Limit: {cfg.get('enable_workspace_limit', True)}")
        log_model.info(f"   - Velocity Limit: {cfg.get('enable_velocity_limit', True)} (max={cfg.get('max_velocity', 0.2)} rad/s)")
    else:
        postprocessor = None
        log_model.info("⚠️  Action Postprocessing Disabled")

    # Inference loop
    results = []
    for episode in range(eval_episodes):

        log_model.info(f"🎯 Episode {episode + 1}/{eval_episodes}")

        # Reset environment and policy
        obs, info = env.reset()
        policy.reset()

        # ✨ Reset postprocessor
        if postprocessor:
            postprocessor.reset()

        episode_reward = 0
        episode_length = 0
        success = False

        # Track inference times and action stats
        inference_times = []
        raw_action_magnitudes = []
        processed_action_magnitudes = []

        # 获取初始状态
        current_state = obs.get('state', np.zeros(16))

        while True:
            # Check control signals
            if stop_flag.is_set():
                log_model.info("🛑 Stop signal received, terminating...")
                break

            while pause_flag.is_set():
                log_model.info("⏸️  Paused, waiting for resume...")
                time.sleep(0.1)

            # Preprocess observations
            observation = {}

            # Process image observations (resize to 512x512 for SmolVLA)
            for key in obs.keys():
                if 'image' in key.lower() or 'cam' in key.lower():
                    observation[f"observation.{key}"] = img_preprocess_smolvla(
                        obs[key], target_size=(512, 512), device=device
                    )
                elif 'depth' in key.lower():
                    observation[f"observation.{key}"] = depth_preprocess(
                        obs[key], device, cfg.depth_range
                    )
                elif 'state' in key.lower():
                    observation[f"observation.{key}"] = torch.tensor(
                        obs[key], dtype=torch.float32
                    ).unsqueeze(0).to(device)
                    current_state = obs[key]  # 保存当前状态用于后处理

            # Add language instruction to batch
            observation['task'] = [language_instruction]

            # SmolVLA inference
            start_time = time.time()
            with torch.no_grad():
                action_chunk = policy.select_action(observation)
                action = action_chunk[:, 0, :]
                raw_action = action.squeeze(0).cpu().numpy()[:16]

            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)

            # ✨ Action后处理
            if postprocessor:
                processed_action = postprocessor.process(raw_action, current_state)

                # 记录统计信息
                raw_magnitude = np.linalg.norm(raw_action[:14])
                processed_magnitude = np.linalg.norm(processed_action[:14])
                raw_action_magnitudes.append(raw_magnitude)
                processed_action_magnitudes.append(processed_magnitude)

                final_action = processed_action
            else:
                final_action = raw_action

            # Log inference time and action stats every 100 steps
            if episode_length % 100 == 0:
                avg_time = np.mean(inference_times[-100:]) if len(inference_times) >= 100 else np.mean(inference_times)
                log_model.info(f"Step {episode_length}: Avg inference time: {avg_time:.2f}ms")

                if postprocessor and len(raw_action_magnitudes) > 0:
                    avg_raw = np.mean(raw_action_magnitudes[-100:]) if len(raw_action_magnitudes) >= 100 else np.mean(raw_action_magnitudes)
                    avg_processed = np.mean(processed_action_magnitudes[-100:]) if len(processed_action_magnitudes) >= 100 else np.mean(processed_action_magnitudes)
                    gain = avg_processed / avg_raw if avg_raw > 1e-6 else 1.0
                    log_model.info(f"   Raw action: {avg_raw:.4f}, Processed: {avg_processed:.4f}, Gain: {gain:.2f}x")

            # Execute action
            obs, reward, terminated, truncated, info = env.step(final_action)

            episode_reward += reward
            episode_length += 1

            # Check episode end
            if terminated or truncated:
                success = info.get('is_success', False)
                break

        # Record episode results
        avg_inference_time = np.mean(inference_times)

        result = {
            'episode': episode,
            'reward': episode_reward,
            'length': episode_length,
            'success': success,
            'avg_inference_time_ms': avg_inference_time
        }

        if postprocessor and len(raw_action_magnitudes) > 0:
            result['avg_raw_action'] = np.mean(raw_action_magnitudes)
            result['avg_processed_action'] = np.mean(processed_action_magnitudes)
            result['avg_gain'] = result['avg_processed_action'] / result['avg_raw_action'] if result['avg_raw_action'] > 1e-6 else 1.0

        results.append(result)

        # Log episode statistics
        log_model.info(f"📈 Episode {episode + 1} - Reward: {episode_reward:.3f}, Length: {episode_length}, Success: {success}")
        log_model.info(f"⏱️  Average inference time: {avg_inference_time:.2f}ms")

        if postprocessor and 'avg_gain' in result:
            log_model.info(f"🔧 Action postprocessing - Avg gain: {result['avg_gain']:.2f}x")

        if stop_flag.is_set():
            break

    # Calculate overall statistics
    if results:
        avg_reward = np.mean([r['reward'] for r in results])
        success_rate = np.mean([r['success'] for r in results])
        avg_length = np.mean([r['length'] for r in results])
        avg_inference = np.mean([r['avg_inference_time_ms'] for r in results])

        log_model.info(f"\n{'='*70}")
        log_model.info(f"🏆 Final Results - Episodes: {len(results)}")
        log_model.info(f"📊 Average Reward: {avg_reward:.3f}")
        log_model.info(f"✅ Success Rate: {success_rate:.1%}")
        log_model.info(f"📏 Average Length: {avg_length:.1f}")
        log_model.info(f"⏱️  Average Inference Time: {avg_inference:.2f}ms")

        if postprocessor and 'avg_gain' in results[0]:
            avg_gain = np.mean([r['avg_gain'] for r in results])
            log_model.info(f"🔧 Average Action Gain: {avg_gain:.2f}x")

        log_model.info(f"{'='*70}\n")

    return results

# Compatibility interface
def setup_policy(pretrained_path, policy_type, device=torch.device("cuda"), language_instruction=""):
    """Compatibility interface"""
    if policy_type != 'smolvla':
        raise ValueError(f"This script only supports 'smolvla' policy, got '{policy_type}'")
    return setup_smolvla_policy(pretrained_path, language_instruction, device)
