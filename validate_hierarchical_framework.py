# -*- coding: utf-8 -*-
"""
分层架构框架完整验证脚本

使用方法：
python validate_hierarchical_framework.py

该脚本将验证：
1. 各个层的基础功能
2. 分层调度器的工作情况
3. 整体集成测试
4. 性能基准测试
"""
import sys
import os
import torch
import time
import traceback

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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


class HierarchicalFrameworkValidator:
    """分层架构框架验证器"""

    def __init__(self):
        self.test_results = {}
        self.performance_stats = {}

    def log(self, message, level="INFO"):
        """日志输出"""
        prefix = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅ ",
            "ERROR": "❌ ",
            "WARNING": "⚠️ ",
            "TEST": "🧪 "
        }
        print(prefix.get(level, "") + message)

    def test_safety_layer(self):
        """测试安全反射层"""
        self.log("Testing SafetyReflexLayer...", "TEST")

        try:
            config = {
                'input_dim': 16,
                'hidden_size': 64,
                'output_dim': 16,
                'emergency_threshold': 0.8,
                'tilt_threshold_degrees': 15.0,
                'enabled': True
            }

            base_config = MockConfig()
            layer = SafetyReflexLayer(config, base_config)

            # 创建测试输入
            batch_size = 4
            seq_len = 10
            inputs = {
                'observation.state': torch.randn(batch_size, seq_len, 16)
            }

            # 测试前向传播
            start_time = time.time()
            output = layer.forward_with_timing(inputs)
            execution_time = (time.time() - start_time) * 1000

            # 验证输出
            required_keys = ['emergency', 'balance_action', 'emergency_action', 'safety_status']
            for key in required_keys:
                assert key in output, "Missing key: {}".format(key)

            # 验证延迟要求
            assert execution_time < 20, "Safety layer too slow: {:.2f}ms".format(execution_time)

            self.performance_stats['safety_layer'] = {
                'execution_time_ms': execution_time,
                'output_keys': list(output.keys()),
                'emergency_detected': bool(torch.any(output['emergency']).item())
            }

            self.log("Safety layer execution time: {:.2f}ms".format(execution_time), "SUCCESS")
            self.log("Emergency status: {}".format(output['emergency']), "SUCCESS")
            self.test_results['safety_layer'] = True

        except Exception as e:
            self.log("Safety layer test failed: {}".format(str(e)), "ERROR")
            self.test_results['safety_layer'] = False

    def test_gait_layer(self):
        """测试步态控制层"""
        self.log("Testing GaitControlLayer...", "TEST")

        try:
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

            # 创建测试输入
            batch_size = 4
            seq_len = 15
            inputs = {
                'observation.state': torch.randn(batch_size, seq_len, 16)
            }

            # 测试前向传播
            start_time = time.time()
            output = layer.forward_with_timing(inputs)
            execution_time = (time.time() - start_time) * 1000

            # 验证输出
            required_keys = ['gait_features', 'planned_gait', 'action']
            for key in required_keys:
                assert key in output, "Missing key: {}".format(key)

            # 验证延迟要求
            assert execution_time < 50, "Gait layer too slow: {:.2f}ms".format(execution_time)

            self.performance_stats['gait_layer'] = {
                'execution_time_ms': execution_time,
                'output_keys': list(output.keys())
            }

            self.log("Gait layer execution time: {:.2f}ms".format(execution_time), "SUCCESS")
            self.test_results['gait_layer'] = True

        except Exception as e:
            self.log("Gait layer test failed: {}".format(str(e)), "ERROR")
            self.test_results['gait_layer'] = False

    def test_manipulation_layer(self):
        """测试操作控制层"""
        self.log("Testing ManipulationLayer...", "TEST")

        try:
            config = {
                'hidden_size': 512,
                'layers': 3,
                'heads': 8,
                'dim_feedforward': 2048,
                'enabled': True
            }

            base_config = MockConfig()
            layer = ManipulationLayer(config, base_config)

            # 创建测试输入
            batch_size = 2
            seq_len = 8
            inputs = {
                'observation.state': torch.randn(batch_size, seq_len, 16)
            }

            # 测试前向传播
            start_time = time.time()
            output = layer.forward_with_timing(inputs)
            execution_time = (time.time() - start_time) * 1000

            # 验证输出
            required_keys = ['manipulation_features', 'action']
            for key in required_keys:
                assert key in output, "Missing key: {}".format(key)

            # 验证延迟要求
            assert execution_time < 200, "Manipulation layer too slow: {:.2f}ms".format(execution_time)

            self.performance_stats['manipulation_layer'] = {
                'execution_time_ms': execution_time,
                'output_keys': list(output.keys())
            }

            self.log("Manipulation layer execution time: {:.2f}ms".format(execution_time), "SUCCESS")
            self.test_results['manipulation_layer'] = True

        except Exception as e:
            self.log("Manipulation layer test failed: {}".format(str(e)), "ERROR")
            self.test_results['manipulation_layer'] = False

    def test_planning_layer(self):
        """测试全局规划层"""
        self.log("Testing GlobalPlanningLayer...", "TEST")

        try:
            config = {
                'hidden_size': 1024,
                'layers': 4,
                'heads': 16,
                'dim_feedforward': 4096,
                'enabled': True
            }

            base_config = MockConfig()
            layer = GlobalPlanningLayer(config, base_config)

            # 创建测试输入
            batch_size = 2
            seq_len = 5
            inputs = {
                'observation.state': torch.randn(batch_size, seq_len, 16)
            }

            # 测试前向传播（高复杂度任务）
            context = {'task_complexity': 'high'}

            start_time = time.time()
            output = layer.forward_with_timing(inputs, context)
            execution_time = (time.time() - start_time) * 1000

            # 验证输出
            required_keys = ['global_features', 'action']
            for key in required_keys:
                assert key in output, "Missing key: {}".format(key)

            # 验证延迟要求
            assert execution_time < 1000, "Planning layer too slow: {:.2f}ms".format(execution_time)

            self.performance_stats['planning_layer'] = {
                'execution_time_ms': execution_time,
                'output_keys': list(output.keys())
            }

            self.log("Planning layer execution time: {:.2f}ms".format(execution_time), "SUCCESS")
            self.test_results['planning_layer'] = True

        except Exception as e:
            self.log("Planning layer test failed: {}".format(str(e)), "ERROR")
            self.test_results['planning_layer'] = False

    def test_hierarchical_scheduler(self):
        """测试分层调度器"""
        self.log("Testing HierarchicalScheduler...", "TEST")

        try:
            # 构建配置
            hierarchical_config = {
                'layers': {
                    'safety': {
                        'type': 'GRU',
                        'input_dim': 16,
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

            # 创建测试输入
            batch_size = 2
            seq_len = 10
            batch = {
                'observation.state': torch.randn(batch_size, seq_len, 16)
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

            # 验证输出
            expected_layers = ['safety', 'gait', 'manipulation']
            for layer_name in expected_layers:
                assert layer_name in outputs, "Missing layer output: {}".format(layer_name)

            # 测试推理模式
            start_time = time.time()
            inference_outputs = scheduler.inference_mode(batch, task_info, latency_budget_ms=50.0)
            inference_time = (time.time() - start_time) * 1000

            # 获取性能统计
            stats = scheduler.get_performance_stats()

            self.performance_stats['scheduler'] = {
                'training_time_ms': execution_time,
                'inference_time_ms': inference_time,
                'active_layers': list(outputs.keys()),
                'within_budget': inference_outputs.get('_inference_stats', {}).get('within_budget', False),
                'performance_stats': stats
            }

            self.log("Scheduler training time: {:.2f}ms".format(execution_time), "SUCCESS")
            self.log("Scheduler inference time: {:.2f}ms".format(inference_time), "SUCCESS")
            self.log("Active layers: {}".format(list(outputs.keys())), "SUCCESS")
            self.test_results['scheduler'] = True

        except Exception as e:
            self.log("Scheduler test failed: {}".format(str(e)), "ERROR")
            self.test_results['scheduler'] = False

    def test_config_loading(self):
        """测试配置文件加载"""
        self.log("Testing configuration loading...", "TEST")

        try:
            import yaml
            config_path = "configs/policy/humanoid_diffusion_config.yaml"

            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                # 验证关键配置项
                assert 'policy' in config, "Missing policy configuration"
                assert 'hierarchical' in config['policy'], "Missing hierarchical configuration"
                assert 'layers' in config['policy']['hierarchical'], "Missing layers configuration"

                layers_config = config['policy']['hierarchical']['layers']
                expected_layers = ['safety', 'gait', 'manipulation', 'planning']

                for layer_name in expected_layers:
                    assert layer_name in layers_config, "Missing layer config: {}".format(layer_name)

                self.log("Configuration file loaded successfully", "SUCCESS")
                self.test_results['config_loading'] = True
            else:
                self.log("Configuration file not found: {}".format(config_path), "WARNING")
                self.test_results['config_loading'] = False

        except Exception as e:
            self.log("Configuration loading test failed: {}".format(str(e)), "ERROR")
            self.test_results['config_loading'] = False

    def benchmark_performance(self):
        """性能基准测试"""
        self.log("Running performance benchmarks...", "TEST")

        try:
            # 不同batch size的性能测试
            batch_sizes = [1, 2, 4, 8]
            seq_lens = [5, 10, 16, 32]

            benchmark_results = {}

            for batch_size in batch_sizes:
                for seq_len in seq_lens:
                    # 测试安全层性能
                    config = {'input_dim': 16, 'hidden_size': 64, 'output_dim': 16, 'enabled': True}
                    base_config = MockConfig()
                    layer = SafetyReflexLayer(config, base_config)

                    inputs = {
                        'observation.state': torch.randn(batch_size, seq_len, 16)
                    }

                    # 预热
                    for _ in range(5):
                        _ = layer.forward(inputs)

                    # 测量
                    times = []
                    for _ in range(20):
                        start_time = time.time()
                        _ = layer.forward(inputs)
                        times.append((time.time() - start_time) * 1000)

                    avg_time = sum(times) / len(times)
                    benchmark_results["safety_b{}_s{}".format(batch_size, seq_len)] = avg_time

            self.performance_stats['benchmarks'] = benchmark_results
            self.log("Performance benchmarks completed", "SUCCESS")
            self.test_results['benchmarks'] = True

        except Exception as e:
            self.log("Benchmark test failed: {}".format(str(e)), "ERROR")
            self.test_results['benchmarks'] = False

    def run_all_tests(self):
        """运行所有测试"""
        self.log("Starting Hierarchical Framework Validation", "INFO")
        self.log("=" * 50, "INFO")

        test_functions = [
            self.test_safety_layer,
            self.test_gait_layer,
            self.test_manipulation_layer,
            self.test_planning_layer,
            self.test_hierarchical_scheduler,
            self.test_config_loading,
            self.benchmark_performance
        ]

        for test_func in test_functions:
            try:
                test_func()
            except Exception as e:
                self.log("Unexpected error in {}: {}".format(test_func.__name__, str(e)), "ERROR")
                traceback.print_exc()

    def generate_report(self):
        """生成测试报告"""
        self.log("=" * 50, "INFO")
        self.log("Test Results Summary", "INFO")
        self.log("=" * 50, "INFO")

        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)

        for test_name, result in self.test_results.items():
            status = "PASSED" if result else "FAILED"
            level = "SUCCESS" if result else "ERROR"
            self.log("{}: {}".format(test_name, status), level)

        self.log("", "INFO")
        self.log("Overall: {}/{} tests passed".format(passed, total), "INFO")

        if passed == total:
            self.log("🎉 All tests passed! Framework is ready for training.", "SUCCESS")
            self.log("", "INFO")
            self.log("Next steps:", "INFO")
            self.log("1. Train with: python kuavo_train/train_policy.py --config-name=humanoid_diffusion_config", "INFO")
            self.log("2. Monitor layer performance during training", "INFO")
            self.log("3. Adjust layer weights and thresholds based on results", "INFO")
        else:
            self.log("❌ Some tests failed. Framework needs fixes before training.", "ERROR")

        # 保存性能统计
        if self.performance_stats:
            self.log("", "INFO")
            self.log("Performance Statistics:", "INFO")
            for component, stats in self.performance_stats.items():
                self.log("  {}: {}".format(component, stats), "INFO")

    def save_results(self, filename="hierarchical_test_results.txt"):
        """保存测试结果到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Hierarchical Framework Test Results\n")
                f.write("=" * 50 + "\n\n")

                f.write("Test Results:\n")
                for test_name, result in self.test_results.items():
                    status = "PASSED" if result else "FAILED"
                    f.write("  {}: {}\n".format(test_name, status))

                f.write("\nPerformance Statistics:\n")
                for component, stats in self.performance_stats.items():
                    f.write("  {}:\n".format(component))
                    if isinstance(stats, dict):
                        for key, value in stats.items():
                            f.write("    {}: {}\n".format(key, value))
                    else:
                        f.write("    {}\n".format(stats))

            self.log("Results saved to {}".format(filename), "SUCCESS")
        except Exception as e:
            self.log("Failed to save results: {}".format(str(e)), "ERROR")


def main():
    """主函数"""
    validator = HierarchicalFrameworkValidator()

    # 运行所有测试
    validator.run_all_tests()

    # 生成报告
    validator.generate_report()

    # 保存结果
    validator.save_results()


if __name__ == "__main__":
    main()