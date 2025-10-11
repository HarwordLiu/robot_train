#!/bin/bash
# SmolVLA Sequential Multi-Task Policy Evaluation Script
# This script provides easy deployment for SmolVLA trained models

# Load ROS Docker environment variables
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$SCRIPT_ROOT/.ros_docker_desktop_env" ]; then
    source "$SCRIPT_ROOT/.ros_docker_desktop_env"
    export ROS_MASTER_URI
    export ROS_IP
    echo "✅ Loaded ROS environment configuration"
    echo "   ROS_MASTER_URI=$ROS_MASTER_URI"
    echo "   ROS_IP=$ROS_IP"
fi

cleanup() {
    echo "⏹️ Caught Ctrl+C, terminating task"
    if [ -n "$current_pid" ] && kill -0 "$current_pid" 2>/dev/null; then
        echo "⏹️ Terminating task (PID: $current_pid)..."
        kill -9 "$current_pid"
        wait "$current_pid" 2>/dev/null
    fi
    exit 130
}

# Catch Ctrl+C
trap cleanup SIGINT SIGTERM

echo "=========================================="
echo "🤖 SmolVLA Sequential Multi-Task Evaluation"
echo "=========================================="
echo ""
echo "This script evaluates SmolVLA models trained with sequential learning"
echo "Supports 4 robot tasks with language instruction switching"
echo ""

# Get script directory
if [ -n "$BASH_SOURCE" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
fi

# Script file path
AUTO_TEST_SCRIPT="$SCRIPT_DIR/examples/scripts/script_auto_test.py"

# Create log directory
LOG_DIR="$(dirname "$SCRIPT_ROOT")/log"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi
LOG_DIR="$LOG_DIR/kuavo_deploy"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# Check if script exists
if [ ! -f "$AUTO_TEST_SCRIPT" ]; then
    echo "❌ Error: Cannot find script_auto_test.py: $AUTO_TEST_SCRIPT"
    exit 1
fi

echo "📋 SmolVLA Task Selection"
echo "=========================================="
echo "1. Task 1: Moving Target Grasping"
echo "   - Pick from conveyor belt, place on table, push to area"
echo ""
echo "2. Task 2: Package Weighing"
echo "   - Pick from belt, weigh on scale, place in container"
echo ""
echo "3. Task 3: Product Placement"
echo "   - Pick bottle, transfer hands, place with label up"
echo ""
echo "4. Task 4: Full Process Sorting"
echo "   - Move, pick workpiece, place at designated location"
echo ""
echo "5. Multi-Task Evaluation (All 4 Tasks)"
echo "   - Sequentially evaluate all tasks"
echo ""
echo "6. Custom Config Path"
echo "   - Use your own config file"
echo ""
echo "=========================================="

# Task configurations
declare -A TASK_NAMES=(
    [1]="task1_moving_grasp"
    [2]="task2_weighing"
    [3]="task3_placement"
    [4]="task4_sorting"
)

declare -A TASK_INSTRUCTIONS=(
    [1]="Pick up the moving object from the conveyor belt, place it on the table, and push it to the designated area"
    [2]="Pick up the package from the conveyor belt, weigh it on the electronic scale, then pick it up again and place it in the designated storage container"
    [3]="Pick up the daily chemical product bottle, transfer it to the other hand, and place it in the designated location with the label facing up"
    [4]="Move from the starting position, pick up the workpiece, move to the designated location and place it precisely"
)

echo "Please select a task (1-6):"
read -r task_choice

# Default config file
CONFIG_FILE="$SCRIPT_ROOT/configs/deploy/kuavo_smolvla_sim_env.yaml"

case $task_choice in
    1|2|3|4)
        TASK_ID=$task_choice
        TASK_NAME="${TASK_NAMES[$TASK_ID]}"
        TASK_INSTRUCTION="${TASK_INSTRUCTIONS[$TASK_ID]}"

        echo ""
        echo "✅ Selected Task $TASK_ID: $TASK_NAME"
        echo "📝 Language Instruction:"
        echo "   \"$TASK_INSTRUCTION\""
        echo ""

        # Ask for training timestamp
        echo "Please enter the training timestamp (e.g., run_20251011_100000):"
        echo "Or press Enter to use default from config"
        read -r timestamp

        if [ -z "$timestamp" ]; then
            echo "Using default timestamp from config file"
        else
            echo "Using timestamp: $timestamp"
            # Update config with sed (temporary modification)
            sed -i.bak "s/timestamp: .*/timestamp: '$timestamp'/" "$CONFIG_FILE"
        fi

        # Ask for epoch
        echo "Please enter the epoch to evaluate (e.g., best, 10, 20):"
        echo "Or press Enter to use 'best'"
        read -r epoch

        if [ -z "$epoch" ]; then
            epoch="best"
        fi
        echo "Using epoch: $epoch"

        # Show model path
        MODEL_PATH="outputs/train/smolvla_sequential/${TASK_NAME}/${epoch}"
        echo ""
        echo "📂 Model Path: $MODEL_PATH"

        # Check if model exists
        if [ -d "$MODEL_PATH" ]; then
            echo "✅ Model path exists"
        else
            echo "⚠️  Warning: Model path does not exist yet"
            echo "   Make sure you have trained the model first"
        fi

        echo ""
        echo "🚀 Starting SmolVLA evaluation for Task $TASK_ID..."
        echo "📊 Config: $CONFIG_FILE"
        echo "📝 Task: $TASK_NAME"
        echo "🗣️  Instruction: $TASK_INSTRUCTION"
        echo ""

        # Run evaluation
        python "$AUTO_TEST_SCRIPT" --task auto_test --config "$CONFIG_FILE"

        # Restore config if modified
        if [ -f "${CONFIG_FILE}.bak" ]; then
            mv "${CONFIG_FILE}.bak" "$CONFIG_FILE"
        fi
        ;;

    5)
        echo ""
        echo "🔄 Multi-Task Evaluation Mode"
        echo "Will evaluate all 4 tasks sequentially"
        echo ""

        # Ask for training timestamp
        echo "Please enter the training timestamp for Task 4 (final model):"
        read -r timestamp

        if [ -z "$timestamp" ]; then
            echo "❌ Timestamp is required for multi-task evaluation"
            exit 1
        fi

        # Evaluate each task
        for TASK_ID in 1 2 3 4; do
            TASK_NAME="${TASK_NAMES[$TASK_ID]}"
            TASK_INSTRUCTION="${TASK_INSTRUCTIONS[$TASK_ID]}"

            echo ""
            echo "=========================================="
            echo "📊 Evaluating Task $TASK_ID: $TASK_NAME"
            echo "=========================================="
            echo "📝 Instruction: $TASK_INSTRUCTION"
            echo ""

            # Update config for this task
            cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"

            # Use Python to update YAML config
            python3 << EOF
import yaml
with open("$CONFIG_FILE", 'r') as f:
    config = yaml.safe_load(f)

config['task'] = "smolvla_sequential/${TASK_NAME}"
config['timestamp'] = "$timestamp"
config['epoch'] = 'best'
config['language_instruction'] = "$TASK_INSTRUCTION"

with open("$CONFIG_FILE", 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
EOF

            # Run evaluation
            python "$AUTO_TEST_SCRIPT" --task auto_test --config "$CONFIG_FILE"

            # Restore config
            mv "${CONFIG_FILE}.backup" "$CONFIG_FILE"

            echo ""
            echo "✅ Task $TASK_ID evaluation completed"
            echo ""
        done

        echo ""
        echo "=========================================="
        echo "🎉 Multi-Task Evaluation Completed!"
        echo "=========================================="
        ;;

    6)
        echo ""
        echo "📁 Custom Config Mode"
        echo "Please enter the full path to your config file:"
        read -r custom_config

        if [ ! -f "$custom_config" ]; then
            echo "❌ Config file not found: $custom_config"
            exit 1
        fi

        echo "✅ Using config: $custom_config"
        echo ""
        echo "🚀 Starting evaluation..."

        python "$AUTO_TEST_SCRIPT" --task auto_test --config "$custom_config"
        ;;

    "")
        echo "Exiting"
        exit 0
        ;;

    *)
        echo "❌ Invalid selection: $task_choice"
        echo "Please select 1-6"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ Evaluation completed!"
echo "📊 Check logs in: $LOG_DIR"
echo "=========================================="
