#!/bin/bash

# VLA Transformer Policy部署脚本
# 用于在仿真或真实机器人上测试VLA策略

set -e

echo "🤖 VLA Transformer Policy Deployment Script"
echo "==========================================="

# 默认配置文件路径
DEFAULT_CONFIG="configs/deploy/kuavo_vla_sim_env.yaml"

# 检查是否提供了配置文件参数
if [ $# -eq 0 ]; then
    echo "使用默认配置: $DEFAULT_CONFIG"
    CONFIG_FILE=$DEFAULT_CONFIG
else
    CONFIG_FILE=$1
    echo "使用指定配置: $CONFIG_FILE"
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo ""
echo "📋 配置文件: $CONFIG_FILE"
echo ""

# 运行部署脚本
echo "🚀 启动VLA策略部署..."
echo ""

python kuavo_deploy/examples/scripts/script_auto_test.py \
    --task vla_inference \
    --config "$CONFIG_FILE"

echo ""
echo "✅ VLA部署完成"
