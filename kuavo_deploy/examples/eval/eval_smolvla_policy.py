# -*- coding: utf-8 -*-
"""
SmolVLA Policy Deployment Module

Support for SmolVLA sequential multi-task policy deployment and inference
"""

from kuavo_deploy.utils.logging_utils import setup_logger
from configs.deploy.config_inference import load_inference_config
from lerobot.utils.random_utils import set_seed
from kuavo_train.wrapper.policy.smolvla.SmolVLAPolicyWrapper import SmolVLAPolicyWrapper
import threading
import rospy
from std_msgs.msg import Bool
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_tensor, resize
from omegaconf import DictConfig, ListConfig, OmegaConf
import numpy as np
import time
import datetime
from tqdm import tqdm
import torch
import numpy
import imageio
import gymnasium as gym
import hydra
from dataclasses import dataclass, field
import lerobot_patches.custom_patches
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


# Import SmolVLA modules


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


pause_sub = rospy.Subscriber(
    '/kuavo/pause_state', Bool, pause_callback, queue_size=10)
stop_sub = rospy.Subscriber(
    '/kuavo/stop_state', Bool, stop_callback, queue_size=10)
stop_flag = threading.Event()
pause_flag = threading.Event()


def img_preprocess_smolvla(image, target_size=(512, 512), device="cpu"):
    """
    Preprocess image for SmolVLA

    SmolVLA uses 512x512 input images (not 224x224 like VLA)

    Args:
        image: Input image (numpy array or PIL Image)
        target_size: Target size (512, 512) for SmolVLA
        device: Device to place tensor on

    Returns:
        Preprocessed tensor [1, 3, 512, 512]
    """
    # Convert to tensor [3, H, W]
    tensor_img = to_tensor(image)

    # Resize to 512x512 with padding (matching training)
    # SmolVLA config uses resize_imgs_with_padding: [512, 512]
    h, w = tensor_img.shape[-2:]
    target_h, target_w = target_size

    # Calculate scaling factor (maintain aspect ratio)
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    tensor_img = resize(
        tensor_img, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)

    # Pad to target size
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

    # Add batch dimension [1, 3, 512, 512]
    return tensor_img.unsqueeze(0).to(device, non_blocking=True)


def depth_preprocess(depth, device="cpu", depth_range=[0, 1000]):
    """Preprocess depth image"""
    depth = np.array(depth)
    depth = np.clip(depth, depth_range[0], depth_range[1])
    depth = (depth - depth_range[0]) / (depth_range[1] - depth_range[0])
    return torch.tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)


def setup_smolvla_policy(pretrained_path, language_instruction, device=torch.device("cuda"), warmup_iterations=2):
    """
    Setup and load SmolVLA policy model

    Args:
        pretrained_path: Path to checkpoint
        language_instruction: Task language instruction
        device: Device
        warmup_iterations: Number of warmup inference iterations (default: 2)

    Returns:
        Loaded policy model
    """

    if device.type == 'cpu':
        log_model.warning(
            "Warning: Using CPU for inference, this may be slow.")
        time.sleep(3)

    log_model.info("🤖 Loading SmolVLA Policy...")
    log_model.info(f"📝 Task Instruction: {language_instruction}")

    policy = SmolVLAPolicyWrapper.from_pretrained(Path(pretrained_path))

    policy.eval()
    policy.to(device)
    policy.reset()

    # Log model info
    log_model.info(f"✅ Model loaded from {pretrained_path}")
    log_model.info(f"📋 Model n_obs_steps: {policy.config.n_obs_steps}")
    log_model.info(f"🖥️  Model device: {device}")
    log_model.info(f"🔧 Policy type: SmolVLA Sequential")
    log_model.info(f"📊 VLM: {policy.config.vlm_model_name}")
    log_model.info(
        f"📊 Action dim: {policy.config.max_action_dim} (Kuavo uses first 16)")
    log_model.info(f"📊 Chunk size: {policy.config.chunk_size}")
    log_model.info(f"📊 Action steps: {policy.config.n_action_steps}")

    # Warmup inference to initialize CUDA kernels and reduce first inference latency
    if warmup_iterations > 0:
        log_model.info(
            f"🔥 Warming up model with {warmup_iterations} dummy inferences...")

        # Create dummy observations matching SmolVLA input format
        # Automatically get the correct feature keys from policy config
        dummy_obs = {}

        # Add dummy images for each expected image feature
        for feature_name, feature_info in policy.config.input_features.items():
            if 'image' in feature_name.lower():
                # Get expected shape from config or use default (512, 512)
                if hasattr(feature_info, 'shape') and len(feature_info.shape) == 3:
                    channels, height, width = feature_info.shape
                else:
                    channels, height, width = 3, 512, 512
                dummy_obs[feature_name] = torch.zeros(
                    (1, channels, height, width), dtype=torch.float32, device=device)
                log_model.debug(
                    f"  Created dummy image for {feature_name}: [{1}, {channels}, {height}, {width}]")
            elif 'state' in feature_name.lower():
                # SmolVLA uses 32D state
                dummy_obs[feature_name] = torch.zeros(
                    (1, 32), dtype=torch.float32, device=device)
                log_model.debug(
                    f"  Created dummy state for {feature_name}: [1, 32]")

        # Add language instruction
        dummy_obs['task'] = [language_instruction]

        warmup_times = []
        with torch.no_grad():
            for i in range(warmup_iterations):
                start_time = time.time()
                _ = policy.select_action(dummy_obs)
                warmup_time = (time.time() - start_time) * \
                    1000  # Convert to ms
                warmup_times.append(warmup_time)
                log_model.info(
                    f"  Warmup {i+1}/{warmup_iterations}: {warmup_time:.2f}ms")

        avg_warmup_time = np.mean(warmup_times)
        log_model.info(
            f"✅ Warmup completed! Average time: {avg_warmup_time:.2f}ms")
        log_model.info(f"💡 First inference should now be faster")

        # Reset policy state after warmup
        policy.reset()

    return policy


def main(config_path: str, env: gym.Env):
    """SmolVLA main inference loop"""

    # Load config
    cfg = load_inference_config(config_path)
    from omegaconf import OmegaConf
    full_cfg = OmegaConf.load(config_path)

    use_delta = cfg.use_delta
    eval_episodes = cfg.eval_episodes
    device = torch.device(cfg.device)
    language_instruction = cfg.language_instruction

    # Set random seed
    set_seed(cfg.seed)

    # Build model path
    pretrained_path = f"outputs/train/{cfg.task}/{cfg.method}/{cfg.timestamp}/epoch{cfg.epoch}"

    # Load SmolVLA policy
    policy = setup_smolvla_policy(
        pretrained_path, language_instruction, device)

    # Inference loop
    results = []
    for episode in range(eval_episodes):

        log_model.info(f"🎯 Episode {episode + 1}/{eval_episodes}")

        # Reset environment and policy
        obs, info = env.reset()
        policy.reset()

        episode_reward = 0
        episode_length = 0
        success = False

        # Track inference times
        inference_times = []

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

            # Add language instruction to batch
            observation['task'] = [language_instruction]

            # SmolVLA inference
            start_time = time.time()
            with torch.no_grad():
                # SmolVLA outputs action chunks of shape [B, chunk_size, action_dim]
                # For Kuavo: [1, 50, 32], we take first step and first 16 dims
                action_chunk = policy.select_action(observation)

                # Extract first action step: [1, 50, 32] -> [1, 32]
                action = action_chunk[:, 0, :]

                # Convert to numpy and take first 16 dimensions (Kuavo DOF)
                numpy_action = action.squeeze(0).cpu().numpy()[:16]

            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            inference_times.append(inference_time)

            # Log inference time every 100 steps
            if episode_length % 100 == 0:
                avg_time = np.mean(
                    inference_times[-100:]) if len(inference_times) >= 100 else np.mean(inference_times)
                log_model.info(
                    f"Step {episode_length}: Avg inference time: {avg_time:.2f}ms")

            # Execute action
            obs, reward, terminated, truncated, info = env.step(numpy_action)

            episode_reward += reward
            episode_length += 1

            # Check episode end
            if terminated or truncated:
                success = info.get('is_success', False)
                break

        # Record episode results
        avg_inference_time = np.mean(inference_times)
        results.append({
            'episode': episode,
            'reward': episode_reward,
            'length': episode_length,
            'success': success,
            'avg_inference_time_ms': avg_inference_time
        })

        # Log episode statistics
        log_model.info(
            f"📈 Episode {episode + 1} - Reward: {episode_reward:.3f}, Length: {episode_length}, Success: {success}")
        log_model.info(
            f"⏱️  Average inference time: {avg_inference_time:.2f}ms")

        if stop_flag.is_set():
            break

    # Calculate overall statistics
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

# Compatibility interface


def setup_policy(pretrained_path, policy_type, device=torch.device("cuda"), language_instruction=""):
    """Compatibility interface"""
    if policy_type != 'smolvla':
        raise ValueError(
            f"This script only supports 'smolvla' policy, got '{policy_type}'")
    return setup_smolvla_policy(pretrained_path, language_instruction, device)
