# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the Kuavo Data Challenge repository - a comprehensive robotics framework for embodied AI manipulation tasks. It extends the HuggingFace LeRobot framework with support for the Kuavo humanoid robot, providing data conversion (rosbag → parquet), imitation learning training, simulation testing, and real robot deployment capabilities.

## Environment Setup Commands

### Python Environment
```bash
# Create virtual environment (Python 3.10 recommended)
conda create -n kdc python=3.10
conda activate kdc

# Install dependencies
pip install -r requirements_total.txt    # Full installation (requires ROS Noetic)
# OR
pip install -r requirements_ilcode.txt   # Training only (no ROS required)
```

### ROS Environment
- **Required**: ROS Noetic on Ubuntu 20.04
- **Alternative**: Use Docker with provided Dockerfile for other Ubuntu versions

### Git Submodules
```bash
git submodule init
git submodule update --recursive
```

## Common Development Commands

### Data Conversion (rosbag → LeRobot format)
```bash
python kuavo_data/CvtRosbag2Lerobot.py \
  --config-path=../configs/data/ \
  --config-name=KuavoRosbag2Lerobot.yaml \
  rosbag.rosbag_dir=/path/to/rosbag \
  rosbag.lerobot_dir=/path/to/lerobot_data
```

### Training
```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=diffusion_config.yaml \
  task=your_task_name \
  method=your_method_name \
  root=/path/to/lerobot_data/lerobot \
  training.batch_size=128 \
  policy_name=diffusion
```

### Deployment and Testing
```bash
# Interactive deployment script
bash kuavo_deploy/eval_kuavo.sh

# Direct deployment
python kuavo_deploy/examples/scripts/script.py --task go_run --config /path/to/config.yaml
python kuavo_deploy/examples/scripts/script_auto_test.py --task auto_test --config /path/to/config.yaml
```

## Architecture Overview

### Core Modules

1. **kuavo_data/**: Data processing and conversion
   - `CvtRosbag2Lerobot.py`: Main conversion script (rosbag → parquet format)
   - Handles RGB and depth camera data, sensor states, joint angles

2. **kuavo_train/**: Imitation learning training
   - `train_policy.py`: Main training script with Hydra configuration
   - Supports Diffusion Policy and ACT (Action Chunking Transformer)
   - Custom dataset wrappers and augmentation utilities

3. **kuavo_deploy/**: Deployment and evaluation
   - `eval_kuavo.sh`: Interactive deployment script with process management
   - Supports both simulation (Mujoco) and real robot deployment
   - Task modes: go, run, go_run, here_run, back_to_zero, auto_test

4. **lerobot_patches/**: LeRobot framework extensions
   - `custom_patches.py`: Adds RGB/DEPTH FeatureTypes and custom statistics
   - **Important**: Always imported first in entry scripts

5. **configs/**: Hydra configuration files
   - `data/`: Data conversion configurations
   - `policy/`: Training configurations (diffusion, ACT)
   - `deploy/`: Deployment configurations (sim/real environments)

### Key Dependencies

- **LeRobot**: Base framework (extended via third_party/lerobot submodule)
- **ROS Noetic**: Required for rosbag processing and robot communication
- **PyTorch**: Deep learning framework
- **Hydra**: Configuration management
- **Mujoco**: Physics simulation

### Data Flow

1. **Raw Data**: ROS bags with camera feeds, sensor data, joint states
2. **Conversion**: rosbag → LeRobot parquet format with RGB/depth support
3. **Training**: Imitation learning on converted data
4. **Deployment**: Model inference on simulation or real robot

## Important Notes

### Configuration Management
- All scripts use Hydra for configuration management
- Config files are hierarchical: base configs can be overridden via command line
- Model paths follow pattern: `outputs/train/{task}/{method}/{timestamp}/epoch{epoch}`

### ROS Topics (Key ones for development)
**Simulation:**
- `/cam_h/color/image_raw/compressed`: Top camera RGB
- `/cam_h/depth/image_raw/compressedDepth`: Top camera depth
- `/joint_cmd`: Joint control commands
- `/kuavo_arm_traj`: Arm trajectory control

**Real Robot:**
- Same camera topics as simulation
- `/control_robot_hand_position`: Dexterous hand control
- `/leju_claw_command`: Leju gripper control

### Output Structure
```
outputs/
├── train/<task>/<method>/run_<timestamp>/   # Training models and logs
├── eval/<task>/<method>/run_<timestamp>/    # Evaluation logs and videos
```

### Custom Patches
- **Critical**: `lerobot_patches.custom_patches` must be imported first in all entry scripts
- Extends LeRobot with RGB/DEPTH feature types and custom statistics computation
- Enables depth data processing and multi-modal training

### Deployment Tasks
- `go`: Navigate to working position via bag replay
- `run`: Execute model from current position
- `go_run`: Navigate then execute model
- `here_run`: Interpolate to bag final state then execute
- `back_to_zero`: Return to zero position
- `auto_test`: Automated simulation testing with specified episodes