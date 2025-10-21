#!/bin/bash

# SmolVLA Diffusion 评估脚本
# 用于在仿真环境中评估 SmolVLA Diffusion 模型

# 设置默认参数
CONFIG_PATH="configs/deploy/kuavo_smolvla_diffusion_sim_env.yaml"
MODEL_PATH=""
OUTPUT_DIR="outputs/eval/smolvla_diffusion"
NUM_EPISODES=10

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
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -c, --config PATH     配置文件路径 (默认: $CONFIG_PATH)"
            echo "  -m, --model PATH       模型路径 (必需)"
            echo "  -o, --output PATH      输出目录 (默认: $OUTPUT_DIR)"
            echo "  -n, --episodes NUM    评估回合数 (默认: $NUM_EPISODES)"
            echo "  -h, --help             显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 -m outputs/train/task1_moving_grasp/smolvla_diffusion/best"
            echo "  $0 -m model_checkpoint -n 20 -o results"
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
echo "🚀 SmolVLA Diffusion 评估"
echo "=========================================="
echo "📋 配置信息:"
echo "   - 配置文件: $CONFIG_PATH"
echo "   - 模型路径: $MODEL_PATH"
echo "   - 输出目录: $OUTPUT_DIR"
echo "   - 评估回合: $NUM_EPISODES"
echo "=========================================="

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行评估
python kuavo_deploy/eval_smolvla_diffusion.py \
    --config-path="$(dirname "$CONFIG_PATH")" \
    --config-name="$(basename "$CONFIG_PATH" .yaml)" \
    policy.pretrained_name_or_path="$MODEL_PATH" \
    logging.output_dir="$OUTPUT_DIR" \
    evaluation.num_episodes="$NUM_EPISODES"

# 检查结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ 评估完成!"
    echo "📁 结果保存在: $OUTPUT_DIR"

    # 显示最新结果文件
    LATEST_RESULT=$(ls -t "$OUTPUT_DIR"/eval_results_*.json 2>/dev/null | head -n1)
    if [[ -n "$LATEST_RESULT" ]]; then
        echo "📊 最新结果文件: $LATEST_RESULT"

        # 提取并显示关键结果
        if command -v jq &> /dev/null; then
            echo ""
            echo "📈 关键指标:"
            echo "   - 成功率: $(jq -r '.stats.success_rate' "$LATEST_RESULT" | awk '{printf "%.1f%%", $1*100}')"
            echo "   - 平均推理时间: $(jq -r '.stats.avg_inference_time' "$LATEST_RESULT" | awk '{printf "%.2f ms", $1*1000}')"
        fi
    fi
else
    echo ""
    echo "❌ 评估失败!"
    exit 1
fi