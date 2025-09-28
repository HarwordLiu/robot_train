# -*- coding: utf-8 -*-
"""
测试分层架构框架的基础功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import time
from kuavo_train.wrapper.policy.humanoid.layers.SafetyReflexLayer import SafetyReflexLayer
from kuavo_train.wrapper.policy.humanoid.layers.GaitControlLayer import GaitControlLayer
from kuavo_train.wrapper.policy.humanoid.layers.ManipulationLayer import ManipulationLayer
from kuavo_train.wrapper.policy.humanoid.layers.GlobalPlanningLayer import GlobalPlanningLayer
from kuavo_train.wrapper.policy.humanoid.HierarchicalScheduler import HierarchicalScheduler


class MockConfig:
    """模拟配置对象 - 匹配实际机器人配置"""
    def __init__(self):
        # 适配only_arm=true配置：双臂14维+手爪2维=16维
        self.robot_state_feature = type('obj', (object,), {'shape': [16]})()
        self.use_hierarchical = True


def test_safety_layer():
    """测试安全反射层"""
    print("🧪 Testing SafetyReflexLayer...")

    config = {
        'input_dim': 16,  # 适配实际机器人配置
        'hidden_size': 64,
        'output_dim': 16,
        'emergency_threshold': 0.8,
        'tilt_threshold_degrees': 15.0,
        'enabled': True
    }

    base_config = MockConfig()
    layer = SafetyReflexLayer(config, base_config)

    # 创建测试输入 - 匹配实际机器人状态维度
    batch_size = 4
    seq_len = 10
    inputs = {
        'observation.state': torch.randn(batch_size, seq_len, 16)  # 双臂+手爪配置
    }

    # 测试前向传播
    start_time = time.time()
    output = layer.forward_with_timing(inputs)
    execution_time = (time.time() - start_time) * 1000

    print("  ✅ Safety layer execution time: {:.2f}ms".format(execution_time))
    print("  ✅ Emergency status: {}".format(output['emergency']))
    print("  ✅ Output keys: {}".format(list(output.keys())))

    # 验证延迟要求
    assert execution_time < 20, "Safety layer too slow: {}ms".format(execution_time)
    assert 'emergency' in output
    assert 'balance_action' in output

    print("  ✅ SafetyReflexLayer test passed!")
    return True


def test_gait_layer():
    """测试步态控制层"""
    print("🧪 Testing GaitControlLayer...")

    config = {
        'gru_hidden': 128,
        'gru_layers': 2,
        'tf_layers': 2,
        'tf_heads': 4,
        'tf_dim': 128,
        'enabled': True
    }

    base_config = MockConfig()
    layer = GaitControlLayer(config, base_config)

    # 创建测试输入 - 匹配实际机器人配置
    batch_size = 4
    seq_len = 15
    inputs = {
        'observation.state': torch.randn(batch_size, seq_len, 16)  # 适配双臂+手爪配置
    }

    # 测试前向传播
    start_time = time.time()
    output = layer.forward_with_timing(inputs)
    execution_time = (time.time() - start_time) * 1000

    print(f"  ✅ Gait layer execution time: {execution_time:.2f}ms")
    print(f"  ✅ Output keys: {list(output.keys())}")

    # 验证
    assert execution_time < 50, f"Gait layer too slow: {execution_time}ms"
    assert 'gait_features' in output
    assert 'action' in output

    print("  ✅ GaitControlLayer test passed!")
    return True


def test_manipulation_layer():
    """测试操作控制层"""
    print("🧪 Testing ManipulationLayer...")

    config = {
        'hidden_size': 512,
        'layers': 3,
        'heads': 8,
        'dim_feedforward': 2048,
        'enabled': True
    }

    base_config = MockConfig()
    layer = ManipulationLayer(config, base_config)

    # 创建测试输入 - 匹配实际机器人配置
    batch_size = 2
    seq_len = 8
    inputs = {
        'observation.state': torch.randn(batch_size, seq_len, 16),  # 适配双臂+手爪配置
        'observation.images': torch.randn(batch_size, seq_len, 1280)  # 模拟视觉特征
    }

    # 测试前向传播
    start_time = time.time()
    output = layer.forward_with_timing(inputs)
    execution_time = (time.time() - start_time) * 1000

    print(f"  ✅ Manipulation layer execution time: {execution_time:.2f}ms")
    print(f"  ✅ Output keys: {list(output.keys())}")

    # 验证
    assert execution_time < 200, f"Manipulation layer too slow: {execution_time}ms"
    assert 'manipulation_features' in output
    assert 'action' in output

    print("  ✅ ManipulationLayer test passed!")
    return True


def test_planning_layer():
    """测试全局规划层"""
    print("🧪 Testing GlobalPlanningLayer...")

    config = {
        'hidden_size': 1024,
        'layers': 4,
        'heads': 16,
        'dim_feedforward': 4096,
        'enabled': True
    }

    base_config = MockConfig()
    layer = GlobalPlanningLayer(config, base_config)

    # 创建测试输入 - 匹配实际机器人配置
    batch_size = 2
    seq_len = 5
    inputs = {
        'observation.state': torch.randn(batch_size, seq_len, 16),  # 适配双臂+手爪配置
        'observation.images': torch.randn(batch_size, seq_len, 1280)
    }

    # 测试前向传播（只有高复杂度任务才激活）
    context = {'task_complexity': 'high'}

    start_time = time.time()
    output = layer.forward_with_timing(inputs, context)
    execution_time = (time.time() - start_time) * 1000

    print(f"  ✅ Planning layer execution time: {execution_time:.2f}ms")
    print(f"  ✅ Output keys: {list(output.keys())}")

    # 验证
    assert execution_time < 1000, f"Planning layer too slow: {execution_time}ms"
    assert 'global_features' in output
    assert 'action' in output

    print("  ✅ GlobalPlanningLayer test passed!")
    return True


def test_hierarchical_scheduler():
    """测试分层调度器"""
    print("🧪 Testing HierarchicalScheduler...")

    # 构建配置
    hierarchical_config = {
        'layers': {
            'safety': {
                'type': 'GRU',
                'input_dim': 16,  # 适配实际机器人配置
                'hidden_size': 64,
                'output_dim': 16,
                'enabled': True
            },
            'gait': {
                'type': 'Hybrid',
                'gru_hidden': 128,
                'gru_layers': 2,
                'tf_layers': 2,
                'tf_heads': 4,
                'enabled': True
            },
            'manipulation': {
                'type': 'Transformer',
                'hidden_size': 512,
                'layers': 3,
                'heads': 8,
                'dim_feedforward': 2048,
                'enabled': True
            },
            'planning': {
                'type': 'Transformer',
                'hidden_size': 1024,
                'layers': 4,
                'heads': 16,
                'dim_feedforward': 4096,
                'enabled': False  # 默认禁用最复杂的层
            }
        },
        'layer_weights': {
            'safety': 2.0,
            'gait': 1.5,
            'manipulation': 1.0,
            'planning': 0.8
        }
    }

    base_config = MockConfig()
    scheduler = HierarchicalScheduler(hierarchical_config, base_config)

    # 创建测试输入 - 匹配实际机器人配置
    batch_size = 2
    seq_len = 10
    batch = {
        'observation.state': torch.randn(batch_size, seq_len, 16)  # 适配双臂+手爪配置
    }

    task_info = {
        'task_complexity': 'medium',
        'requires_locomotion': True,
        'requires_manipulation': True
    }

    # 测试分层前向传播
    start_time = time.time()
    outputs = scheduler(batch, task_info)
    execution_time = (time.time() - start_time) * 1000

    print(f"  ✅ Scheduler execution time: {execution_time:.2f}ms")
    print(f"  ✅ Active layers: {list(outputs.keys())}")

    # 测试推理模式
    start_time = time.time()
    inference_outputs = scheduler.inference_mode(batch, task_info, latency_budget_ms=50.0)
    inference_time = (time.time() - start_time) * 1000

    print(f"  ✅ Inference mode time: {inference_time:.2f}ms")
    print(f"  ✅ Within budget: {inference_outputs.get('_inference_stats', {}).get('within_budget', False)}")

    # 获取性能统计
    stats = scheduler.get_performance_stats()
    print(f"  ✅ Performance stats: {stats}")

    print("  ✅ HierarchicalScheduler test passed!")
    return True


def test_integration():
    """集成测试"""
    print("🧪 Running integration tests...")

    # 运行所有单独的测试
    tests = [
        test_safety_layer,
        test_gait_layer,
        test_manipulation_layer,
        test_planning_layer,
        test_hierarchical_scheduler
    ]

    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ {test_func.__name__} failed: {e}")

    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! Hierarchical framework is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    print("🚀 Starting Hierarchical Framework Tests\n")

    try:
        success = test_integration()

        if success:
            print("\n✅ Framework ready for training!")
            print("📝 Next steps:")
            print("   1. Train with: python kuavo_train/train_policy.py --config-name=humanoid_diffusion_config")
            print("   2. Monitor layer performance and adjust configurations")
            print("   3. Fine-tune layer weights and activation thresholds")
        else:
            print("\n❌ Framework needs fixes before training")

    except Exception as e:
        print(f"\n💥 Test framework error: {e}")
        import traceback
        traceback.print_exc()