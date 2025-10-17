#!/bin/bash
# 训练监控快捷脚本

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🤖 训练监控快捷脚本${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "用法: ./monitor.sh [选项]"
    echo ""
    echo "选项:"
    echo "  basic              - 启动基础监控器（默认）"
    echo "  advanced           - 启动高级监控器（终端仪表板）"
    echo "  plot               - 启动高级监控器（图表模式）"
    echo "  gpu                - 启动高级监控器（带GPU监控）"
    echo "  report             - 生成训练报告"
    echo "  install-deps       - 安装所有依赖"
    echo "  help               - 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  ./monitor.sh                    # 快速查看训练状态"
    echo "  ./monitor.sh advanced           # 启动实时仪表板"
    echo "  ./monitor.sh gpu                # 带GPU监控的仪表板"
    echo "  ./monitor.sh plot               # 查看训练曲线"
    echo "  ./monitor.sh report             # 生成训练报告"
    echo ""
}

# 检查Python环境
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3未找到，请先安装Python3${NC}"
        exit 1
    fi
}

# 安装依赖
install_deps() {
    echo -e "${BLUE}📦 安装训练监控依赖...${NC}"
    echo ""
    
    echo -e "${YELLOW}安装必需依赖...${NC}"
    pip3 install tensorboard || {
        echo -e "${RED}❌ 安装失败: tensorboard${NC}"
        exit 1
    }
    
    echo ""
    echo -e "${YELLOW}安装可选依赖（用于高级功能）...${NC}"
    pip3 install rich matplotlib psutil GPUtil
    
    echo ""
    echo -e "${GREEN}✅ 依赖安装完成！${NC}"
    echo ""
    echo "现在可以使用:"
    echo "  ./monitor.sh basic      - 基础监控"
    echo "  ./monitor.sh advanced   - 高级监控"
    echo "  ./monitor.sh gpu        - GPU监控"
    echo ""
}

# 基础监控
monitor_basic() {
    echo -e "${BLUE}🔍 启动基础监控器...${NC}"
    cd "$PROJECT_ROOT"
    python3 kuavo_train/monitor_training.py "$@"
}

# 高级监控（终端）
monitor_advanced() {
    echo -e "${BLUE}🚀 启动高级监控器（终端仪表板）...${NC}"
    cd "$PROJECT_ROOT"
    python3 kuavo_train/monitor_training_advanced.py --mode terminal "$@"
}

# 高级监控（图表）
monitor_plot() {
    echo -e "${BLUE}📊 启动高级监控器（图表模式）...${NC}"
    cd "$PROJECT_ROOT"
    python3 kuavo_train/monitor_training_advanced.py --mode plot "$@"
}

# GPU监控
monitor_gpu() {
    echo -e "${BLUE}🎮 启动GPU监控...${NC}"
    cd "$PROJECT_ROOT"
    python3 kuavo_train/monitor_training_advanced.py --monitor-gpu --mode terminal "$@"
}

# 生成报告
generate_report() {
    echo -e "${BLUE}📝 生成训练报告...${NC}"
    cd "$PROJECT_ROOT"
    python3 kuavo_train/monitor_training.py --save-report --plot "$@"
}

# 主函数
main() {
    check_python
    
    # 如果没有参数，显示帮助
    if [ $# -eq 0 ]; then
        # 默认启动基础监控
        monitor_basic
        exit 0
    fi
    
    # 解析命令
    case "$1" in
        basic)
            shift
            monitor_basic "$@"
            ;;
        advanced)
            shift
            monitor_advanced "$@"
            ;;
        plot)
            shift
            monitor_plot "$@"
            ;;
        gpu)
            shift
            monitor_gpu "$@"
            ;;
        report)
            shift
            generate_report "$@"
            ;;
        install-deps)
            install_deps
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}❌ 未知选项: $1${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"

