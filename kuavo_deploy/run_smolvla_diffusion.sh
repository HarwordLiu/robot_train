#!/bin/bash

# SmolVLA Diffusion 实时运行脚本
# 用于实时运行 SmolVLA Diffusion 模型

# 设置默认参数
CONFIG_PATH="configs/deploy/kuavo_smolvla_diffusion_sim_env.yaml"
MODEL_PATH=""
DEVICE="cuda"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -c, --config PATH     配置文件路径 (默认: $CONFIG_PATH)"
            echo "  -m, --model PATH       模型路径 (必需)"
            echo "  -d, --device DEVICE    设备 (cuda/cpu, 默认: cuda)"
            echo "  -h, --help             显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 -m outputs/train/task1_moving_grasp/smolvla_diffusion/best"
            echo "  $0 -m model_checkpoint -d cpu"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [[ -z "$MODEL_PATH" ]]; then
    echo "错误: 必须指定模型路径"
    echo "使用 -m 或 --model 指定模型路径"
    echo "使用 -h 或 --help 查看帮助"
    exit 1
fi

# 检查模型路径是否存在
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 打印配置信息
echo "=========================================="
echo "🚀 SmolVLA Diffusion 实时运行"
echo "=========================================="
echo "📋 配置信息:"
echo "   - 配置文件: $CONFIG_PATH"
echo "   - 模型路径: $MODEL_PATH"
echo "   - 设备: $DEVICE"
echo "=========================================="
echo ""
echo "⚠️  注意事项:"
echo "   1. 确保仿真服务器正在运行"
echo "   2. 按 Ctrl+C 停止运行"
echo "   3. 首次运行会进行模型预热"
echo ""

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# 运行实时部署
python kuavo_deploy/run_smolvla_diffusion_realtime.py \
    --config "$CONFIG_PATH" \
    --model "$MODEL_PATH" \
    --device "$DEVICE"