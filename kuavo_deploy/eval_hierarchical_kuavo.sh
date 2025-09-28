#!/bin/bash
# 分层架构Kuavo机器人控制脚本
# 专门支持分层人形机器人Diffusion Policy的部署和测试

cleanup() {
    echo "⏹️ 捕获到 Ctrl+C，开始终止分层架构任务"
    if [ -n "$current_pid" ] && kill -0 "$current_pid" 2>/dev/null; then
        echo "⏹️ 捕获到 Ctrl+C，正在终止分层架构任务 (PID: $current_pid)..."
        kill -9 "$current_pid"
        wait "$current_pid" 2>/dev/null
    fi
    exit 130
}

# 捕获 Ctrl+C
trap cleanup SIGINT SIGTERM

echo "🤖 === 分层架构Kuavo机器人控制示例 ==="
echo "支持分层人形机器人Diffusion Policy的部署和仿真测试"
echo ""
echo "📋 控制功能说明:"
echo "  🔄 暂停/恢复: 发送 SIGUSR1 信号 (kill -USR1 <PID>)"
echo "  ⏹️  停止任务: 发送 SIGUSR2 信号 (kill -USR2 <PID>)"
echo "  📊 查看日志: tail -f log/kuavo_deploy/hierarchical_deploy.log"
echo ""

# 获取脚本所在目录
if [ -n "$BASH_SOURCE" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
fi

# 分层架构脚本文件路径
SCRIPT="$SCRIPT_DIR/examples/scripts/script_hierarchical.py"
AUTO_TEST_SCRIPT="$SCRIPT_DIR/examples/scripts/script_hierarchical_auto_test.py"

# 交互式控制器函数
start_interactive_controller() {
    echo ""
    echo "🎮 分层架构交互式控制器已启动"
    echo "任务PID: $current_pid"
    echo ""
    echo "📋 可用命令:"
    echo "  p/pause    - 暂停/恢复任务"
    echo "  s/stop     - 停止任务"
    echo "  l/log      - 查看实时日志"
    echo "  h/help     - 显示帮助"
    echo "  stats      - 显示分层架构性能统计"
    echo ""

    while true; do
        # 检查任务是否还在运行
        if [ -n "$current_pid" ] && kill -0 "$current_pid" 2>/dev/null; then
            echo -n "🟢 分层架构任务运行中 (PID: $current_pid) > "
        else
            echo "🔴 分层架构任务已结束"
            current_pid=""
            break
        fi

        read -r cmd

        case $cmd in
            p|pause)
                if [ -n "$current_pid" ] && kill -0 "$current_pid" 2>/dev/null; then
                    echo "🔄 发送暂停/恢复信号..."
                    kill -USR1 "$current_pid"
                else
                    echo "❌ 任务不在运行"
                fi
                ;;
            s|stop)
                if [ -n "$current_pid" ] && kill -0 "$current_pid" 2>/dev/null; then
                    echo "⏹️  发送停止信号..."
                    kill -USR2 "$current_pid"
                    wait "$current_pid" 2>/dev/null
                    echo "✅ 分层架构任务已停止"
                    current_pid=""
                    break
                else
                    echo "❌ 任务不在运行"
                fi
                ;;
            l|log)
                if [ -f "$LOG_DIR/hierarchical_deploy.log" ]; then
                    echo "📊 显示最新日志 (按 Ctrl+C 返回):"
                    echo "=== 分层架构最新日志 ==="
                    tail -n 20 "$LOG_DIR/hierarchical_deploy.log"
                    echo "=== 日志结束 ==="
                else
                    echo "❌ 分层架构日志文件不存在"
                fi
                ;;
            stats)
                echo "📊 分层架构性能统计:"
                echo "   - 查看详细统计请检查日志文件"
                echo "   - 层激活情况会在推理过程中显示"
                ;;
            h|help)
                echo ""
                echo "📋 可用命令:"
                echo "  p/pause    - 暂停/恢复任务"
                echo "  s/stop     - 停止任务"
                echo "  l/log      - 查看实时日志"
                echo "  h/help     - 显示帮助"
                echo "  stats      - 显示分层架构性能统计"
                echo ""
                ;;
            *)
                echo "❌ 未知命令: $cmd"
                echo "输入 'h' 或 'help' 查看可用命令"
                ;;
        esac
    done
}

# 创建log日志路径
LOG_DIR="$(dirname "$SCRIPT_DIR")/log"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p $LOG_DIR
fi
LOG_DIR="$LOG_DIR/kuavo_deploy"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p $LOG_DIR
fi

# 检查脚本是否存在
if [ ! -f "$SCRIPT" ]; then
    echo "错误: 找不到分层架构脚本文件: $SCRIPT"
    exit 1
fi
if [ ! -f "$AUTO_TEST_SCRIPT" ]; then
    echo "错误: 找不到分层架构自动测试脚本: $AUTO_TEST_SCRIPT"
    exit 1
fi

# 显示帮助信息
echo "🤖 分层架构使用示例:"
echo ""
echo "1. 显示帮助信息:"
echo "python $SCRIPT --help"
echo ""

echo "2. 干运行模式 - 查看将要执行的操作:"
echo "python $SCRIPT --task go --dry_run --config /path/to/hierarchical_config.yaml"
echo ""

echo "3. 到达工作位置 (分层控制):"
echo "python $SCRIPT --task go --config /path/to/hierarchical_config.yaml"
echo ""

echo "4. 从当前位置直接运行分层模型:"
echo "python $SCRIPT --task run --config /path/to/hierarchical_config.yaml"
echo ""

echo "5. 到达工作位置并运行分层模型:"
echo "python $SCRIPT --task go_run --config /path/to/hierarchical_config.yaml"
echo ""

echo "6. 插值至bag的最后一帧状态开始运行分层模型:"
echo "python $SCRIPT --task here_run --config /path/to/hierarchical_config.yaml"
echo ""

echo "7. 回到零位:"
echo "python $SCRIPT --task back_to_zero --config /path/to/hierarchical_config.yaml"
echo ""

echo "8. 分层架构自动测试 (仿真):"
echo "python $AUTO_TEST_SCRIPT --task auto_test --config /path/to/hierarchical_config.yaml"
echo ""

echo "=== 分层架构任务说明 ==="
echo "go          - 先插值到bag第一帧的位置，再回放bag包前往工作位置"
echo "run         - 从当前位置直接运行分层模型"
echo "go_run      - 到达工作位置直接运行分层模型"
echo "here_run    - 插值至bag的最后一帧状态开始运行分层模型"
echo "back_to_zero - 中断模型推理后，倒放bag包回到0位"
echo "auto_test   - 分层架构自动测试，包含性能分析"
echo ""
echo "🔧 分层架构特色:"
echo "  📊 实时层性能监控"
echo "  ⚡ 自适应推理延迟控制"
echo "  🛡️  安全层优先级保障"
echo "  📈 详细性能统计分析"
echo ""

# 交互式选择
echo "请选择要执行的示例: 1. 显示普通测试帮助信息 2. 显示分层架构自动测试帮助信息 3. 进一步选择示例"
echo "1. 执行: python $SCRIPT --help"
echo "2. 执行: python $AUTO_TEST_SCRIPT --help"
echo "3. 进一步选择示例"

echo "请选择要执行的示例 (1-3) 或按 Enter 退出:"
read -r help_choice

case $help_choice in
    1)
        echo "执行: python $SCRIPT --help"
        python "$SCRIPT" --help
        ;;
    2)
        echo "执行: python $AUTO_TEST_SCRIPT --help"
        python "$AUTO_TEST_SCRIPT" --help
        ;;
    3)
        echo "请输入分层架构配置文件路径:"
        read -r config_path

        # 打印model path
        if [ -f "$config_path" ]; then
            echo "📁 分层架构配置文件路径: $config_path"
            echo "🔍 正在解析配置文件..."

            # 使用Python解析YAML并提取model path相关参数
            python3 -c "
import yaml
import sys

try:
    with open('$config_path', 'r') as f:
        config = yaml.safe_load(f)

    task = config.get('task', 'N/A')
    method = config.get('method', 'N/A')
    timestamp = config.get('timestamp', 'N/A')
    epoch = config.get('epoch', 'N/A')
    policy_type = config.get('policy_type', 'N/A')

    model_path = f'outputs/train/{task}/{method}/{timestamp}/epoch{epoch}'

    print(f'📋 分层架构模型配置信息:')
    print(f'   Task: {task}')
    print(f'   Method: {method}')
    print(f'   Timestamp: {timestamp}')
    print(f'   Epoch: {epoch}')
    print(f'   Policy Type: {policy_type}')
    print(f'📂 完整模型路径: {model_path}')

    # 检查模型路径是否存在
    import os
    if os.path.exists(model_path):
        print(f'✅ 分层架构模型路径存在')
    else:
        print(f'❌ 分层架构模型路径不存在')

    # 检查分层架构配置
    if 'hierarchical' in config:
        h_config = config['hierarchical']
        enabled_layers = h_config.get('enabled_layers', [])
        print(f'🏗️  启用的层: {enabled_layers}')
        print(f'⏱️  延迟预算: {h_config.get(\"latency_budget_ms\", \"N/A\")}ms')

except Exception as e:
    print(f'❌ 解析配置文件失败: {e}')
    sys.exit(1)
"
        else
            echo "❌ 配置文件不存在: $config_path"
            exit 1
        fi

        # 初始化进程ID变量
        current_pid=""

        # 进入分层架构选择循环
        while true; do
            echo "🤖 可选择的分层架构任务:"
            echo "1. 先插值到bag第一帧的位置，再回放bag包前往工作位置(干运行模式)"
            echo "执行: python $SCRIPT --task go --dry_run --config $config_path"
            echo "2. 先插值到bag第一帧的位置，再回放bag包前往工作位置"
            echo "执行: python $SCRIPT --task go --config $config_path"
            echo "3. 从当前位置直接运行分层模型"
            echo "执行: python $SCRIPT --task run --config $config_path"
            echo "4. 到达工作位置并直接运行分层模型"
            echo "执行: python $SCRIPT --task go_run --config $config_path"
            echo "5. 插值至bag的最后一帧状态开始运行分层模型"
            echo "执行: python $SCRIPT --task here_run --config $config_path"
            echo "6. 回到零位"
            echo "执行: python $SCRIPT --task back_to_zero --config $config_path"
            echo "7. 先插值到bag第一帧的位置，再回放bag包前往工作位置(启用详细输出)"
            echo "执行: python $SCRIPT --task go --verbose --config $config_path"
            echo "8. 分层架构自动测试，执行eval_episodes次"
            echo "执行: python $AUTO_TEST_SCRIPT --task auto_test --config $config_path"
            echo "9. 退出"
            echo "请选择要执行的任务 (1-9)"
            read -r choice

            case $choice in
                1)
                    echo "🔍 执行: python $SCRIPT --task go --dry_run --config $config_path"
                    python "$SCRIPT" --task go --dry_run --config "$config_path" > $LOG_DIR/hierarchical_deploy.log 2>&1
                    ;;
                2)
                    echo "🤖 执行: python $SCRIPT --task go --config $config_path"
                    echo "分层架构任务将在后台运行，启动交互式控制器..."
                    python "$SCRIPT" --task go --config "$config_path" > $LOG_DIR/hierarchical_deploy.log 2>&1 &
                    current_pid=$!
                    echo "分层架构任务已启动，PID: $current_pid"
                    start_interactive_controller
                    ;;
                3)
                    echo "🚀 执行: python $SCRIPT --task run --config $config_path"
                    echo "分层架构任务将在后台运行，启动交互式控制器..."
                    python "$SCRIPT" --task run --config "$config_path" > $LOG_DIR/hierarchical_deploy.log 2>&1 &
                    current_pid=$!
                    echo "分层架构任务已启动，PID: $current_pid"
                    start_interactive_controller
                    ;;
                4)
                    echo "🎯 执行: python $SCRIPT --task go_run --config $config_path"
                    echo "分层架构任务将在后台运行，启动交互式控制器..."
                    python "$SCRIPT" --task go_run --config "$config_path" > $LOG_DIR/hierarchical_deploy.log 2>&1 &
                    current_pid=$!
                    echo "分层架构任务已启动，PID: $current_pid"
                    start_interactive_controller
                    ;;
                5)
                    echo "📐 执行: python $SCRIPT --task here_run --config $config_path"
                    echo "分层架构任务将在后台运行，启动交互式控制器..."
                    python "$SCRIPT" --task here_run --config "$config_path" > $LOG_DIR/hierarchical_deploy.log 2>&1 &
                    current_pid=$!
                    echo "分层架构任务已启动，PID: $current_pid"
                    start_interactive_controller
                    ;;
                6)
                    echo "🏠 执行: python $SCRIPT --task back_to_zero --config $config_path"
                    echo "分层架构任务将在后台运行，启动交互式控制器..."
                    python "$SCRIPT" --task back_to_zero --config "$config_path" > $LOG_DIR/hierarchical_deploy.log 2>&1 &
                    current_pid=$!
                    echo "分层架构任务已启动，PID: $current_pid"
                    start_interactive_controller
                    ;;
                7)
                    echo "📢 执行: python $SCRIPT --task go --verbose --config $config_path"
                    echo "分层架构任务将在后台运行，启动交互式控制器..."
                    python "$SCRIPT" --task go --verbose --config "$config_path" > $LOG_DIR/hierarchical_deploy.log 2>&1 &
                    current_pid=$!
                    echo "分层架构任务已启动，PID: $current_pid"
                    start_interactive_controller
                    ;;
                8)
                    echo "🧪 执行: python $AUTO_TEST_SCRIPT --task auto_test --config $config_path"
                    echo "分层架构自动测试将在后台运行，启动交互式控制器..."
                    python "$AUTO_TEST_SCRIPT" --task auto_test --config "$config_path" > $LOG_DIR/hierarchical_deploy.log 2>&1 &
                    current_pid=$!
                    echo "分层架构自动测试已启动，PID: $current_pid"
                    start_interactive_controller
                    ;;
                9)
                    echo "👋 退出分层架构控制"
                    exit 0
                    ;;
                *)
                    echo "❌ 无效选择: $choice"
                    ;;
            esac
        done
        ;;
    "")
        echo "👋 退出"
        ;;
    *)
        echo "❌ 无效选择: $help_choice"
        ;;
esac