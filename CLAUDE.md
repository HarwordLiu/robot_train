# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Kuavo Data Challenge** repository, a robotics imitation learning framework based on [Lerobot](https://github.com/huggingface/lerobot). It provides a complete pipeline for:
- Converting Kuavo robot rosbag data to Lerobot parquet format
- Training imitation learning models (Diffusion Policy, ACT, Hierarchical Diffusion)
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
  - Standard policies: Diffusion Policy, ACT
  - Hierarchical framework: Four-layer architecture for humanoid robots
- **kuavo_deploy/**: Model deployment for simulation and real robot
- **kuavo_eval/**: Offline evaluation framework for trained models
- **lerobot_patches/**: Custom patches for Lerobot framework extending RGB+Depth support

### Hierarchical Framework Architecture

The repository includes an advanced **four-layer hierarchical Diffusion Policy** specifically designed for humanoid robots:

#### Four Layers (by priority)
1. **SafetyReflexLayer** (Priority 1): <10ms response, fall prevention, emergency stop
2. **GaitControlLayer** (Priority 2): ~20ms response, gait planning and terrain adaptation (GRU + Transformer hybrid)
3. **ManipulationLayer** (Priority 3): ~100ms response, fine manipulation and dual-arm coordination
4. **GlobalPlanningLayer** (Priority 4): ~500ms response, long-term planning and global optimization

#### Key Components
- `kuavo_train/wrapper/policy/humanoid/HumanoidDiffusionPolicy.py`: Main entry point
- `kuavo_train/wrapper/policy/humanoid/HierarchicalScheduler.py`: Core scheduler managing layer activation
- `kuavo_train/wrapper/policy/humanoid/HierarchicalDiffusionModel.py`: Hierarchical diffusion model
- `kuavo_train/wrapper/policy/humanoid/layers/`: Individual layer implementations
- `kuavo_train/wrapper/policy/humanoid/TaskSpecificTrainingManager.py`: Multi-task training coordination

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

#### Standard Diffusion Policy
```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=diffusion_config.yaml \
  task=your_task_name \
  method=your_method_name \
  root=/path/to/lerobot_data/lerobot \
  policy_name=diffusion
```

#### ACT Policy
```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=act_config.yaml \
  task=your_task_name \
  method=your_method_name \
  root=/path/to/lerobot_data/lerobot \
  policy_name=act
```

#### Hierarchical Diffusion Policy (Humanoid)
```bash
# Quick start with validation
python start_hierarchical_training.py --validate-first

# Direct training
python kuavo_train/train_hierarchical_policy.py \
  --config-name=humanoid_diffusion_config

# Task-specific training (multi-task)
python kuavo_train/train_hierarchical_task_specific.py \
  --config-name=humanoid_diffusion_config
```

### Validation & Testing
```bash
# Validate hierarchical framework
python validate_hierarchical_framework.py

# Test hierarchical training
python test_hierarchical_training.py

# Test hierarchical deployment
python test_hierarchical_deployment.py
```

### Deployment & Evaluation

#### Interactive Deployment
```bash
# Standard deployment
bash kuavo_deploy/eval_kuavo.sh

# Hierarchical deployment
bash kuavo_deploy/eval_hierarchical_kuavo.sh
```

#### Direct Simulation Testing
```bash
python kuavo_deploy/examples/scripts/script_auto_test.py \
  --task auto_test --config /path/to/config.yaml
```

#### Offline Evaluation
```bash
# Evaluate standard diffusion policy
python kuavo_eval/scripts/offline_eval.py \
  --config-path=../configs/eval \
  --config-name=offline_diffusion_eval.yaml

# Evaluate hierarchical policy
python kuavo_eval/scripts/offline_eval.py \
  --config-path=../configs/eval \
  --config-name=offline_hierarchical_eval.yaml
```

## Configuration System

Uses Hydra for configuration management:

- **Data configs:** `configs/data/KuavoRosbag2Lerobot.yaml`
- **Training configs:**
  - Standard: `configs/policy/{diffusion_config,act_config}.yaml`
  - Hierarchical: `configs/policy/humanoid_diffusion_config.yaml`
- **Deployment configs:**
  - Standard: `configs/deploy/{kuavo_sim_env,kuavo_real_env}.yaml`
  - Hierarchical: `configs/deploy/kuavo_hierarchical_sim_env.yaml`
- **Evaluation configs:** `configs/eval/offline_*_eval.yaml`

### Key Parameters
- `task`: Task name (should match across data conversion and training)
- `method`: Training method identifier
- `root`: Path to converted Lerobot data directory
- `policy_name`: "diffusion", "act", or "humanoid_diffusion"

### Hierarchical Framework Configuration

Enable hierarchical architecture in `humanoid_diffusion_config.yaml`:
```yaml
policy:
  use_hierarchical: True  # Enable hierarchical architecture

hierarchical:
  layers:
    safety:
      enabled: True
      priority: 1
      response_time_ms: 10
    gait:
      enabled: True
      priority: 2
    manipulation:
      enabled: True
      priority: 3
    planning:
      enabled: True  # Can be disabled for simpler tasks
      priority: 4

  # Layer weights for training
  layer_weights:
    safety: 2.0
    gait: 1.5
    manipulation: 1.0
    planning: 0.8

  # Curriculum learning
  curriculum_learning:
    enable: True
    # Progressive training: safety -> safety+gait -> ... -> full

  # Task-specific training
  task_specialization:
    enable: True
    modules: ['dynamic_grasping', 'weighing', 'placement', 'sorting']
```

## Output Structure
```
outputs/
├── train/<task>/<method>/run_<timestamp>/   # Training checkpoints
├── eval/<task>/<method>/run_<timestamp>/    # Evaluation results
├── train_hydra_save/                        # Hydra config snapshots
```

## ROS Topics (Reference)

**Simulation Topics:**
- `/cam_h/color/image_raw/compressed` - Top camera RGB
- `/cam_h/depth/image_raw/compressedDepth` - Top camera depth
- `/cam_l/color/image_raw/compressed` - Left camera RGB
- `/cam_r/color/image_raw/compressed` - Right camera RGB
- `/joint_cmd` - Joint control commands
- `/gripper/command` - Simulated RQ2F85 gripper control
- `/sensors_data_raw` - All sensor data

**Real Robot Topics:**
- Same camera topics as simulation
- `/control_robot_hand_position` - Dexterous hand control
- `/dexhand/state` - Hand joint states
- `/leju_claw_command` - Gripper control
- `/leju_claw_state` - Gripper state

## Development Notes

### General
- **Joint Control Only:** Current implementation supports joint angle control, not end-effector control
- **Depth Support:** Custom patches enable RGB + Depth training (see diffusion_config.yaml custom settings)
- **Multi-Camera:** Supports multiple camera configurations (head, left, right cameras)
- **Data Augmentation:** Configurable RGB augmentation pipeline in training configs
- **Mixed Precision:** Support for AMP training with `use_amp` parameter

### Hierarchical Framework Specifics
- **Backward Compatible:** Set `use_hierarchical: False` to use traditional architecture
- **Curriculum Learning:** Progressive training from simple to complex layers
- **Multi-Task Training:** Support for 4 robot tasks (dynamic grasping, weighing, placement, sorting)
- **Real-Time Performance:** Adaptive computation budget allocation based on latency requirements
- **Performance Monitoring:** Built-in layer performance statistics and health checks
- **Task Specialization:** Dedicated modules for specific manipulation tasks

### Performance Benchmarks
- Safety Layer: 1-5ms (batch size 1-8)
- Gait Layer: 5-20ms (depends on sequence length)
- Manipulation Layer: 20-100ms (depends on complexity)
- Planning Layer: 100-500ms (most complex tasks)

## Validation and Testing

### Framework Validation
```bash
# Run complete framework validation
python validate_hierarchical_framework.py
```

Expected output:
- ✅ SafetyReflexLayer: <10ms response time
- ✅ GaitControlLayer: <50ms response time
- ✅ ManipulationLayer: <200ms response time
- ✅ GlobalPlanningLayer: <1000ms response time
- ✅ HierarchicalScheduler: Correct layer activation

### Model Evaluation
No specific test framework is defined. Model evaluation is performed through:
1. Simulation testing with Mujoco
2. Real robot deployment validation
3. Manual evaluation using deployment scripts
4. Offline evaluation with recorded data

## Troubleshooting

### Hierarchical Framework Issues

**CUDA Out of Memory:**
```yaml
# Reduce batch size
training:
  batch_size: 32  # from 64

# Disable complex layers
hierarchical:
  layers:
    planning:
      enabled: False
```

**Training Not Converging:**
```yaml
# Enable curriculum learning
curriculum_learning:
  enable: True

# Adjust layer weights
layer_weights:
  safety: 1.0
  gait: 1.0
  manipulation: 1.0
  planning: 0.5  # Reduce complex layer weight
```

**High Inference Latency:**
```python
# Use inference mode with latency budget
scheduler.inference_mode(batch, task_info, latency_budget_ms=30.0)

# Auto-tune layers
scheduler.auto_tune_layers(target_latency_ms=50.0)
```

### Common Issues

**Import Errors:**
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=$PYTHONPATH:/path/to/kuavo_data_challenge
```

**FFmpeg/Torchcodec Errors:**
```bash
conda install ffmpeg==6.1.1
# OR
pip uninstall torchcodec
```

## Additional Documentation

- **Hierarchical Framework Guide:** `HIERARCHICAL_FRAMEWORK_GUIDE.md`
- **Implementation Summary:** `IMPLEMENTATION_SUMMARY.md`
- **Deployment Instructions:** `kuavo_deploy/readme.md`
- **Evaluation Guide:** `kuavo_eval/README.md`
