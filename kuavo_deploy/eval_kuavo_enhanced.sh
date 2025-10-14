#!/bin/bash

# ==============================================================================
# SmolVLA增强版部署脚本（无需重新训练的优化）
# ==============================================================================
#
# 功能：
# 1. 推理后处理：平滑滤波 + 精细操作增益
# 2. 精确Language Instruction
#
# 使用方法：
#   bash kuavo_deploy/eval_kuavo_enhanced.sh
#
# ==============================================================================

echo "========================================================================"
echo "🚀 SmolVLA Enhanced Deployment (Inference Optimization)"
echo "========================================================================"

# 配置文件
CONFIG_FILE="configs/deploy/kuavo_smolvla_sim_env_enhanced.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    echo "Please make sure you have copied the enhanced config file."
    exit 1
fi

# 检查增强版脚本是否存在
EVAL_SCRIPT="kuavo_deploy/examples/eval/eval_smolvla_policy_enhanced.py"
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "❌ Error: Enhanced eval script not found: $EVAL_SCRIPT"
    echo "Please make sure you have copied the enhanced eval script."
    exit 1
fi

# 检查后处理模块是否存在
POSTPROC_MODULE="kuavo_deploy/utils/action_postprocessing.py"
if [ ! -f "$POSTPROC_MODULE" ]; then
    echo "❌ Error: Action postprocessing module not found: $POSTPROC_MODULE"
    echo "Please make sure you have copied the postprocessing module."
    exit 1
fi

echo "✅ All required files found"
echo ""

# 显示配置信息
echo "📋 Configuration:"
echo "   Config file: $CONFIG_FILE"
echo "   Eval script: $EVAL_SCRIPT"
echo ""

# 运行增强版部署
echo "🎯 Starting enhanced deployment..."
echo "========================================================================"

python -m kuavo_deploy.examples.eval.eval_smolvla_policy_enhanced \
    --config-name $(basename $CONFIG_FILE .yaml) \
    --config-path ../../configs/deploy

EXIT_CODE=$?

echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Enhanced deployment completed successfully!"
else
    echo "❌ Enhanced deployment failed with exit code: $EXIT_CODE"
fi
echo "========================================================================"

exit $EXIT_CODE
