# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Kuavo Data Challenge** repository, a robotics imitation learning framework based on [Lerobot](https://github.com/huggingface/lerobot). It provides complete pipeline for:
- Converting Kuavo robot rosbag data to Lerobot parquet format
- Training imitation learning models (Diffusion Policy, ACT)
- Testing in Mujoco simulator
- Real robot deployment

## Environment Setup

**Required Dependencies:**
- Python 3.10 (recommended)
- ROS Noetic (required for data conversion and deployment)
- CUDA-enabled GPU (recommended for training)

**Installation:**
```bash
# Install Python dependencies
pip install -r requirements_total.txt  # Full functionality (requires ROS)
# OR
pip install -r requirements_ilcode.txt  # Training only (no ROS dependency)

# Initialize third-party submodules
git submodule init
git submodule update --recursive
```

## Key Architecture

### Core Modules
- **kuavo_data/**: Data conversion from rosbag to Lerobot format
- **kuavo_train/**: Imitation learning training pipeline
- **kuavo_deploy/**: Model deployment for simulation and real robot
- **lerobot_patches/**: Custom patches for Lerobot framework extending RGB+Depth support

### Important Patches
The `lerobot_patches/` directory contains critical extensions to Lerobot:
- Extended `FeatureType` for RGB and Depth images
- Custom statistics computation for multi-modal data
- Feature mapping for Kuavo-specific data types

**CRITICAL:** Always import patches at the top of main scripts:
```python
import lerobot_patches.custom_patches  # DON'T REMOVE THIS LINE!
```

## Common Commands

### Data Conversion
```bash
python kuavo_data/CvtRosbag2Lerobot.py \
  --config-path=../configs/data/ \
  --config-name=KuavoRosbag2Lerobot.yaml \
  rosbag.rosbag_dir=/path/to/rosbag \
  rosbag.lerobot_dir=/path/to/lerobot_data
```

### Training
```bash
# Diffusion Policy
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=diffusion_config.yaml \
  task=your_task_name \
  method=your_method_name \
  root=/path/to/lerobot_data/lerobot \
  policy_name=diffusion

# ACT Policy
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=act_config.yaml \
  task=your_task_name \
  method=your_method_name \
  root=/path/to/lerobot_data/lerobot \
  policy_name=act
```

### Deployment & Testing
```bash
# Interactive deployment script
bash kuavo_deploy/eval_kuavo.sh

# Direct simulation testing
python kuavo_deploy/examples/scripts/script_auto_test.py \
  --task auto_test --config /path/to/config.yaml
```

## Configuration System

Uses Hydra for configuration management:

- **Data configs:** `configs/data/KuavoRosbag2Lerobot.yaml`
- **Training configs:** `configs/policy/{diffusion_config,act_config}.yaml`
- **Deployment configs:** `configs/deploy/{kuavo_sim_env,kuavo_real_env}.yaml`

### Key Parameters
- `task`: Task name (should match across data conversion and training)
- `method`: Training method identifier
- `root`: Path to converted Lerobot data directory
- `policy_name`: "diffusion" or "act"

## Output Structure
```
outputs/
├── train/<task>/<method>/run_<timestamp>/   # Training checkpoints
├── eval/<task>/<method>/run_<timestamp>/    # Evaluation results
```

## ROS Topics (Reference)

**Simulation Topics:**
- `/cam_h/color/image_raw/compressed` - Top camera RGB
- `/cam_h/depth/image_raw/compressedDepth` - Top camera depth
- `/cam_l/color/image_raw/compressed` - Left camera RGB
- `/cam_r/color/image_raw/compressed` - Right camera RGB
- `/joint_cmd` - Joint control commands
- `/sensors_data_raw` - All sensor data

**Real Robot Topics:**
- Same camera topics as simulation
- `/control_robot_hand_position` - Dexterous hand control
- `/dexhand/state` - Hand joint states
- `/leju_claw_command` - Gripper control

## Development Notes

- **Joint Control Only:** Current implementation supports joint angle control, not end-effector control
- **Depth Support:** Custom patches enable RGB + Depth training (see diffusion_config.yaml custom settings)
- **Multi-Camera:** Supports multiple camera configurations (head, left, right cameras)
- **Data Augmentation:** Configurable RGB augmentation pipeline in training configs
- **Mixed Precision:** Support for AMP training with `use_amp` parameter

## Testing

No specific test framework is defined. Model evaluation is performed through:
1. Simulation testing with Mujoco
2. Real robot deployment validation
3. Manual evaluation using deployment scripts