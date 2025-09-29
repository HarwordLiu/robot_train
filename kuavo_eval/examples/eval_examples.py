# -*- coding: utf-8 -*-
"""
评估系统使用示例

展示如何使用离线评估系统
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

def example_usage():
    """展示使用示例"""

    print("🤖 Kuavo Offline Evaluation System - Usage Examples")
    print("="*60)

    print("\n📋 1. 快速验证1-epoch模型")
    print("   用于验证刚训练完成的模型是否正常工作")
    print("   python kuavo_eval/scripts/quick_validation.py \\")
    print("     --config configs/eval/offline_hierarchical_eval.yaml \\")
    print("     --checkpoint outputs/train/task_400_episodes/humanoid_hierarchical/run_xxx/epoch1")

    print("\n📊 2. 完整的分层架构模型评估")
    print("   对分层架构模型进行全面评估")
    print("   python kuavo_eval/scripts/run_offline_eval.py \\")
    print("     --config configs/eval/offline_hierarchical_eval.yaml \\")
    print("     --episodes 10 \\")
    print("     --output-dir outputs/evaluation/hierarchical")

    print("\n🔄 3. 传统diffusion模型评估")
    print("   对传统diffusion模型进行评估")
    print("   python kuavo_eval/scripts/run_offline_eval.py \\")
    print("     --config configs/eval/offline_diffusion_eval.yaml \\")
    print("     --episodes 10 \\")
    print("     --output-dir outputs/evaluation/diffusion")

    print("\n⚡ 4. 快速模式对比评估")
    print("   快速对比两种模型的性能")
    print("   python kuavo_eval/scripts/run_offline_eval.py \\")
    print("     --config configs/eval/offline_hierarchical_eval.yaml \\")
    print("     --quick \\")
    print("     --no-plots")

    print("\n🔧 5. 自定义配置文件使用")
    print("   根据需要修改配置文件中的以下关键参数：")
    print("   - model.checkpoint_path: 指向你的模型检查点")
    print("   - test_data.root: 指向你的lerobot数据目录")
    print("   - test_data.episodes_range: 设置测试的episode范围")
    print("   - test_data.max_episodes: 设置最大测试episode数")

    print("\n📁 6. 配置文件模板")
    print("   基础配置: configs/eval/base_eval_config.yaml")
    print("   分层架构: configs/eval/offline_hierarchical_eval.yaml")
    print("   传统diffusion: configs/eval/offline_diffusion_eval.yaml")

    print("\n📈 7. 输出文件说明")
    print("   评估完成后会生成以下文件：")
    print("   - JSON报告: 详细的评估数据")
    print("   - CSV摘要: 表格格式的关键指标")
    print("   - Markdown报告: 人类可读的分析报告")
    print("   - 可视化图表: 性能分析图表")

    print("\n⚠️  8. 常见问题解决")
    print("   - 检查点不存在: 确认模型已训练并保存")
    print("   - 数据路径错误: 确认lerobot数据目录正确")
    print("   - GPU内存不足: 使用 --device cpu 或减少batch_size")
    print("   - 权限问题: 确保脚本有执行权限 chmod +x scripts/*.py")

    print("\n✨ 9. 高级用法")
    print("   - 批量评估: 使用shell脚本循环评估多个checkpoint")
    print("   - 性能对比: 评估不同epoch的模型并对比结果")
    print("   - 自定义指标: 修改配置文件添加特定的评估指标")

    print("\n" + "="*60)
    print("🎯 开始你的第一次评估：")
    print("   1. 确认你有训练好的模型检查点")
    print("   2. 修改配置文件中的路径")
    print("   3. 运行快速验证确认模型可用")
    print("   4. 执行完整评估获得详细分析")
    print("="*60)

if __name__ == "__main__":
    example_usage()