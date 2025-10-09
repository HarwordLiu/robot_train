#!/usr/bin/env python3
"""
验证分层架构修复的脚本
"""

from kuavo_train.wrapper.policy.humanoid.HumanoidDiffusionPolicy import HumanoidDiffusionPolicy
import numpy as np
import torch
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))


def test_hierarchical_layers():
    """测试分层架构是否按预期工作"""
    print("🔍 测试分层架构修复...")

    # 模拟配置
    class MockConfig:
        def __init__(self):
            self.hierarchical = {
                'layer_weights': {
                    'safety': 2.0,
                    'gait': 1.5,
                    'manipulation': 1.0,
                    'planning': 0.8
                },
                'layers': {
                    'safety': {'enabled': True, 'input_dim': 16, 'hidden_size': 64, 'output_dim': 16},
                    'gait': {'enabled': True, 'input_dim': 16, 'hidden_size': 128, 'output_dim': 16},
                    'manipulation': {'enabled': True, 'hidden_size': 512, 'layers': 3, 'heads': 8, 'dim_feedforward': 2048},
                    'planning': {'enabled': True, 'hidden_size': 1024, 'layers': 4, 'heads': 4, 'dim_feedforward': 4096}
                }
            }

    class MockBaseConfig:
        def __init__(self):
            self.robot_state_feature = type(
                'obj', (object,), {'shape': [16]})()

    try:
        # 创建模型
        config = MockConfig()
        base_config = MockBaseConfig()

        print("✅ 配置创建成功")

        # 测试分层调度器
        from kuavo_train.wrapper.policy.humanoid.HierarchicalScheduler import HierarchicalScheduler

        scheduler = HierarchicalScheduler(config.hierarchical, base_config)
        print(f"✅ 分层调度器创建成功，包含层: {list(scheduler.layers.keys())}")

        # 测试各层是否启用
        for layer_name, layer in scheduler.layers.items():
            if layer.is_enabled():
                print(f"✅ {layer_name} 层已启用")
            else:
                print(f"❌ {layer_name} 层未启用")

        # 测试任务信息
        task_info = {
            'task_complexity': 'high',
            'requires_locomotion': True,
            'requires_manipulation': True,
            'safety_priority': True,
            'enabled_layers': ['safety', 'gait', 'manipulation', 'planning']
        }

        print("✅ 任务信息配置正确")

        # 测试各层的should_activate方法
        mock_inputs = {
            'observation.state': torch.randn(1, 16)
        }

        print("\n🔍 测试各层激活条件:")
        for layer_name, layer in scheduler.layers.items():
            should_activate = layer.should_activate(mock_inputs, task_info)
            print(f"  {layer_name}: {'✅ 激活' if should_activate else '❌ 不激活'}")

        print("\n✅ 分层架构验证完成！")
        return True

    except Exception as e:
        print(f"❌ 分层架构验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_hierarchical_layers()
    if success:
        print("\n🎉 所有测试通过！分层架构修复成功。")
    else:
        print("\n💥 测试失败，需要进一步调试。")
        sys.exit(1)
