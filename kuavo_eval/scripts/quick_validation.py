#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证脚本

专门用于1-epoch模型的快速验证，提供精简但关键的评估指标
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import lerobot_patches.custom_patches

import argparse
import time
import torch
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf

from kuavo_eval.scripts.run_offline_eval import load_config, create_evaluator

class QuickValidator:
    """快速验证器"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.results = {}

    def run_quick_validation(self) -> Dict[str, Any]:
        """
        运行快速验证

        Returns:
            验证结果字典
        """
        print("⚡ Starting Quick Validation...")
        start_time = time.time()

        # 修改配置为快速模式
        self._setup_quick_mode()

        # 创建评估器
        evaluator = create_evaluator(self.config)

        # 执行最小化评估
        try:
            print("🔄 Running minimal evaluation...")
            results = evaluator.evaluate()

            # 提取关键指标
            key_metrics = self._extract_key_metrics(results)

            # 验证模型健康状况
            health_check = self._check_model_health(key_metrics)

            total_time = time.time() - start_time

            validation_results = {
                'status': 'success',
                'total_time': total_time,
                'key_metrics': key_metrics,
                'health_check': health_check,
                'model_type': self.config.model.type,
                'checkpoint_path': self.config.model.checkpoint_path,
                'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            return validation_results

        except Exception as e:
            error_time = time.time() - start_time
            return {
                'status': 'failed',
                'error': str(e),
                'total_time': error_time,
                'model_type': self.config.model.type,
                'checkpoint_path': self.config.model.checkpoint_path,
                'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

    def _setup_quick_mode(self) -> None:
        """设置快速验证模式"""
        # 减少测试数据量
        self.config.test_data.max_episodes = 2
        self.config.test_data.max_steps_per_episode = 10

        # 禁用复杂分析
        self.config.evaluation.temporal_consistency.enable = False
        self.config.report.include_plots = False
        self.config.report.detailed_analysis = False

        # 只保留核心指标
        self.config.evaluation.action_metrics = ['mse', 'mae']

        # 模型特定快速设置
        if self.config.model.type == 'humanoid_diffusion':
            # 禁用分层架构的复杂分析
            if hasattr(self.config, 'hierarchical_evaluation'):
                self.config.hierarchical_evaluation.layer_consistency_check.enable = False
                self.config.hierarchical_evaluation.layer_weight_analysis.enable = False

        elif self.config.model.type == 'diffusion':
            # 禁用diffusion的复杂分析
            if hasattr(self.config, 'diffusion_evaluation'):
                self.config.diffusion_evaluation.denoising_analysis.enable = False
                self.config.diffusion_evaluation.inference_steps_analysis.enable = False

        print("⚙️  Quick mode configured: 2 episodes, 10 steps each")

    def _extract_key_metrics(self, results) -> Dict[str, Any]:
        """提取关键指标"""
        key_metrics = {}

        # 基础动作指标
        action_metrics = results.action_metrics
        if 'overall_avg_mse' in action_metrics:
            key_metrics['mse'] = action_metrics['overall_avg_mse']
        if 'overall_avg_mae' in action_metrics:
            key_metrics['mae'] = action_metrics['overall_avg_mae']

        # 性能指标
        performance_metrics = results.performance_metrics
        if 'overall_avg_inference_time' in performance_metrics:
            key_metrics['avg_inference_time'] = performance_metrics['overall_avg_inference_time']

        # 模型特定关键指标
        if self.config.model.type == 'humanoid_diffusion':
            # 分层架构关键指标
            if 'budget_compliance_rate' in performance_metrics:
                key_metrics['budget_compliance'] = performance_metrics['budget_compliance_rate']

            # 安全层激活率
            if 'safety_activation_rate' in performance_metrics:
                key_metrics['safety_activation'] = performance_metrics['safety_activation_rate']

        elif self.config.model.type == 'diffusion':
            # Diffusion关键指标
            if 'overall_steps_per_second' in performance_metrics:
                key_metrics['inference_speed'] = performance_metrics['overall_steps_per_second']

        # 总体指标
        key_metrics['total_episodes'] = results.summary.get('total_episodes_evaluated', 0)
        key_metrics['total_steps'] = results.summary.get('total_inference_steps', 0)

        return key_metrics

    def _check_model_health(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """检查模型健康状况"""
        health_check = {
            'overall_status': 'unknown',
            'checks': {}
        }

        checks_passed = 0
        total_checks = 0

        # 检查MSE
        if 'mse' in metrics:
            total_checks += 1
            mse = metrics['mse']
            if mse < 0.1:
                health_check['checks']['mse'] = {'status': 'pass', 'value': mse, 'threshold': 0.1}
                checks_passed += 1
            elif mse < 0.5:
                health_check['checks']['mse'] = {'status': 'warning', 'value': mse, 'threshold': 0.1}
            else:
                health_check['checks']['mse'] = {'status': 'fail', 'value': mse, 'threshold': 0.1}

        # 检查推理时间
        if 'avg_inference_time' in metrics:
            total_checks += 1
            inference_time = metrics['avg_inference_time']
            threshold = 100.0  # 100ms
            if inference_time < threshold:
                health_check['checks']['inference_time'] = {'status': 'pass', 'value': inference_time, 'threshold': threshold}
                checks_passed += 1
            elif inference_time < threshold * 2:
                health_check['checks']['inference_time'] = {'status': 'warning', 'value': inference_time, 'threshold': threshold}
            else:
                health_check['checks']['inference_time'] = {'status': 'fail', 'value': inference_time, 'threshold': threshold}

        # 分层架构特定检查
        if self.config.model.type == 'humanoid_diffusion':
            if 'budget_compliance' in metrics:
                total_checks += 1
                compliance = metrics['budget_compliance']
                if compliance > 0.8:
                    health_check['checks']['budget_compliance'] = {'status': 'pass', 'value': compliance, 'threshold': 0.8}
                    checks_passed += 1
                elif compliance > 0.5:
                    health_check['checks']['budget_compliance'] = {'status': 'warning', 'value': compliance, 'threshold': 0.8}
                else:
                    health_check['checks']['budget_compliance'] = {'status': 'fail', 'value': compliance, 'threshold': 0.8}

            if 'safety_activation' in metrics:
                total_checks += 1
                safety_rate = metrics['safety_activation']
                if safety_rate > 0.8:  # 安全层应该经常激活
                    health_check['checks']['safety_activation'] = {'status': 'pass', 'value': safety_rate, 'threshold': 0.8}
                    checks_passed += 1
                elif safety_rate > 0.5:
                    health_check['checks']['safety_activation'] = {'status': 'warning', 'value': safety_rate, 'threshold': 0.8}
                else:
                    health_check['checks']['safety_activation'] = {'status': 'fail', 'value': safety_rate, 'threshold': 0.8}

        # 计算整体状态
        if total_checks == 0:
            health_check['overall_status'] = 'unknown'
        else:
            pass_rate = checks_passed / total_checks
            if pass_rate >= 0.8:
                health_check['overall_status'] = 'healthy'
            elif pass_rate >= 0.5:
                health_check['overall_status'] = 'warning'
            else:
                health_check['overall_status'] = 'unhealthy'

        health_check['pass_rate'] = pass_rate if total_checks > 0 else 0.0
        health_check['checks_passed'] = checks_passed
        health_check['total_checks'] = total_checks

        return health_check

def print_validation_results(results: Dict[str, Any]) -> None:
    """打印验证结果"""
    print("\n" + "="*50)
    print("⚡ QUICK VALIDATION RESULTS")
    print("="*50)

    if results['status'] == 'success':
        print(f"✅ Status: {results['status'].upper()}")
        print(f"⏱️  Total Time: {results['total_time']:.2f}s")
        print(f"🤖 Model Type: {results['model_type']}")

        # 关键指标
        print("\n📊 Key Metrics:")
        key_metrics = results['key_metrics']
        for metric, value in key_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

        # 健康检查
        print("\n🏥 Health Check:")
        health = results['health_check']
        status_emoji = {
            'healthy': '✅',
            'warning': '⚠️',
            'unhealthy': '❌',
            'unknown': '❓'
        }

        overall_status = health['overall_status']
        print(f"  Overall Status: {status_emoji.get(overall_status, '❓')} {overall_status.upper()}")
        print(f"  Pass Rate: {health['pass_rate']:.1%} ({health['checks_passed']}/{health['total_checks']})")

        # 详细检查结果
        if health['checks']:
            print("\n  Detailed Checks:")
            for check_name, check_result in health['checks'].items():
                status = check_result['status']
                value = check_result['value']
                threshold = check_result['threshold']

                check_emoji = {'pass': '✅', 'warning': '⚠️', 'fail': '❌'}

                if isinstance(value, float):
                    print(f"    {check_emoji.get(status, '❓')} {check_name}: {value:.4f} (threshold: {threshold})")
                else:
                    print(f"    {check_emoji.get(status, '❓')} {check_name}: {value} (threshold: {threshold})")

        # 建议
        print("\n💡 Recommendations:")
        if overall_status == 'healthy':
            print("  🎉 Model appears to be working well!")
            print("  ✨ Ready for full evaluation or deployment testing")
        elif overall_status == 'warning':
            print("  ⚠️  Model shows some performance issues")
            print("  🔧 Consider tuning hyperparameters or training longer")
        elif overall_status == 'unhealthy':
            print("  ❌ Model has significant issues")
            print("  🔨 Check training process and data quality")
            print("  📊 Run full evaluation for detailed analysis")
        else:
            print("  ❓ Unable to determine model health")
            print("  🔍 Run full evaluation for more information")

    else:
        print(f"❌ Status: {results['status'].upper()}")
        print(f"⏱️  Time to Failure: {results['total_time']:.2f}s")
        print(f"🤖 Model Type: {results['model_type']}")
        print(f"❌ Error: {results['error']}")

        print("\n🔧 Troubleshooting:")
        print("  1. Check model checkpoint path exists")
        print("  2. Verify data directory and format")
        print("  3. Ensure sufficient GPU memory")
        print("  4. Check configuration file syntax")

    print("="*50 + "\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Quick validation for 1-epoch models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 快速验证分层架构模型
  python quick_validation.py --config ../../configs/eval/offline_hierarchical_eval.yaml

  # 快速验证并指定检查点
  python quick_validation.py --config ../../configs/eval/offline_diffusion_eval.yaml --checkpoint ./outputs/train/task/method/run_xxx/epoch1

  # 使用CPU进行验证
  python quick_validation.py --config ../../configs/eval/offline_hierarchical_eval.yaml --device cpu
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to evaluation configuration file'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Model checkpoint path (overrides config)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        help='Device to use for validation'
    )

    args = parser.parse_args()

    try:
        # 加载配置
        print(f"📖 Loading configuration: {args.config}")
        config = load_config(args.config)

        # 应用命令行覆盖
        if args.checkpoint:
            config.model.checkpoint_path = args.checkpoint
            print(f"🔄 Using checkpoint: {args.checkpoint}")

        if args.device:
            config.common.device = args.device
            print(f"💻 Using device: {args.device}")

        # 验证检查点存在
        checkpoint_path = Path(config.model.checkpoint_path)
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print("💡 Make sure the model has been trained and saved")
            sys.exit(1)

        # 创建快速验证器
        validator = QuickValidator(config)

        # 运行验证
        results = validator.run_quick_validation()

        # 打印结果
        print_validation_results(results)

        # 设置退出码
        if results['status'] == 'success':
            if results['health_check']['overall_status'] in ['healthy', 'warning']:
                sys.exit(0)
            else:
                sys.exit(2)  # 模型不健康
        else:
            sys.exit(1)  # 验证失败

    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()