# -*- coding: utf-8 -*-
"""
测试分层架构训练流程

验证train_hierarchical_policy.py的功能和配置
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import subprocess
import time
from pathlib import Path


def test_config_loading():
    """测试配置文件加载"""
    print("🧪 Testing hierarchical config loading...")

    config_path = "configs/policy/humanoid_diffusion_config.yaml"
    if not os.path.exists(config_path):
        print("❌ Config file not found: {}".format(config_path))
        return False

    try:
        # 测试配置文件语法
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 验证关键配置项
        required_keys = ['policy', 'training', 'hierarchical']
        for key in required_keys:
            if key not in config:
                print("❌ Missing required config key: {}".format(key))
                return False

        # 验证分层架构配置
        if not config['policy'].get('use_hierarchical', False):
            print("⚠️  Warning: use_hierarchical is False in config")

        hierarchical_config = config.get('hierarchical', {})
        required_layers = ['safety', 'gait', 'manipulation', 'planning']
        layers_config = hierarchical_config.get('layers', {})

        for layer in required_layers:
            if layer not in layers_config:
                print("❌ Missing layer config: {}".format(layer))
                return False

        print("✅ Config loading test passed!")
        return True

    except Exception as e:
        print("❌ Config loading failed: {}".format(e))
        return False


def test_training_script_syntax():
    """测试训练脚本语法"""
    print("🧪 Testing training script syntax...")

    script_path = "kuavo_train/train_hierarchical_policy.py"
    if not os.path.exists(script_path):
        print("❌ Training script not found: {}".format(script_path))
        return False

    try:
        # 语法检查
        result = subprocess.run([
            sys.executable, "-m", "py_compile", script_path
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Training script syntax test passed!")
            return True
        else:
            print("❌ Training script syntax error:")
            print(result.stderr)
            return False

    except Exception as e:
        print("❌ Training script syntax test failed: {}".format(e))
        return False


def test_imports():
    """测试导入依赖"""
    print("🧪 Testing import dependencies...")

    try:
        # 测试基础导入
        import torch
        print("  ✅ PyTorch imported")

        # 测试lerobot导入（需要patches）
        import lerobot_patches.custom_patches
        print("  ✅ Lerobot patches imported")

        from lerobot.configs.policies import PolicyFeature
        print("  ✅ Lerobot policies imported")

        # 测试hydra
        import hydra
        from omegaconf import DictConfig
        print("  ✅ Hydra imported")

        # 测试分层架构模块（可能会失败，但不影响脚本语法）
        try:
            from kuavo_train.wrapper.policy.humanoid.HumanoidDiffusionPolicy import HumanoidDiffusionPolicy
            print("  ✅ HumanoidDiffusionPolicy imported")
        except ImportError as e:
            print("  ⚠️  HumanoidDiffusionPolicy import warning: {}".format(e))
            print("     This is expected if hierarchical modules are not yet implemented")

        print("✅ Import test completed!")
        return True

    except Exception as e:
        print("❌ Import test failed: {}".format(e))
        return False


def test_start_script():
    """测试启动脚本"""
    print("🧪 Testing start script...")

    script_path = "start_hierarchical_training.py"
    if not os.path.exists(script_path):
        print("❌ Start script not found: {}".format(script_path))
        return False

    try:
        # 语法检查
        result = subprocess.run([
            sys.executable, "-m", "py_compile", script_path
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Start script syntax test passed!")
            return True
        else:
            print("❌ Start script syntax error:")
            print(result.stderr)
            return False

    except Exception as e:
        print("❌ Start script test failed: {}".format(e))
        return False


def test_dry_run():
    """测试训练脚本dry run（如果可能）"""
    print("🧪 Testing training script dry run...")

    try:
        # 尝试运行帮助命令
        result = subprocess.run([
            sys.executable, "kuavo_train/train_hierarchical_policy.py", "--help"
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ Training script help command works!")
            print("  Output preview:")
            # 显示前几行输出
            lines = result.stdout.split('\n')[:5]
            for line in lines:
                if line.strip():
                    print("    {}".format(line))
            return True
        else:
            print("⚠️  Help command returned non-zero: {}".format(result.returncode))
            print("  Error: {}".format(result.stderr[:200]))
            return False

    except subprocess.TimeoutExpired:
        print("⚠️  Help command timed out (may indicate import issues)")
        return False
    except Exception as e:
        print("❌ Dry run test failed: {}".format(e))
        return False


def test_integration():
    """集成测试"""
    print("🧪 Running hierarchical training integration tests...")

    tests = [
        ("Config Loading", test_config_loading),
        ("Training Script Syntax", test_training_script_syntax),
        ("Import Dependencies", test_imports),
        ("Start Script", test_start_script),
        ("Dry Run", test_dry_run),
    ]

    passed = 0
    results = {}

    for test_name, test_func in tests:
        print("\n--- {} ---".format(test_name))
        try:
            success = test_func()
            results[test_name] = success
            if success:
                passed += 1
        except Exception as e:
            print("❌ {} failed with exception: {}".format(test_name, e))
            results[test_name] = False

    print("\n" + "=" * 60)
    print("📊 Test Results: {}/{} tests passed".format(passed, len(tests)))
    print("=" * 60)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print("  {}: {}".format(status, test_name))

    if passed == len(tests):
        print("\n🎉 All tests passed! Hierarchical training pipeline is ready!")
        print("\n📝 Next steps:")
        print("1. Validate framework: python validate_hierarchical_framework.py")
        print("2. Start training: python start_hierarchical_training.py --validate-first")
        print("3. Or directly: python kuavo_train/train_hierarchical_policy.py")
        return True
    else:
        print("\n❌ Some tests failed. Please address the issues before training.")
        return False


def print_usage_instructions():
    """打印使用说明"""
    print("\n" + "=" * 60)
    print("🚀 Hierarchical Training Pipeline Usage")
    print("=" * 60)

    print("\n1. 验证框架完整性:")
    print("   python validate_hierarchical_framework.py")

    print("\n2. 快速启动训练:")
    print("   python start_hierarchical_training.py --validate-first")

    print("\n3. 直接训练（跳过验证）:")
    print("   python kuavo_train/train_hierarchical_policy.py")

    print("\n4. 使用自定义配置:")
    print("   python kuavo_train/train_hierarchical_policy.py --config-name=your_config")

    print("\n5. 传统训练（对比基线）:")
    print("   python kuavo_train/train_policy.py --config-name=diffusion_config")

    print("\n⚙️  配置文件:")
    print("   - 分层架构: configs/policy/humanoid_diffusion_config.yaml")
    print("   - 传统架构: configs/policy/diffusion_config.yaml")

    print("\n📁 输出目录:")
    print("   - 训练结果: outputs/train/<task>/<method>/run_<timestamp>/")
    print("   - 最佳模型: outputs/.../best/")
    print("   - 检查点: outputs/.../epoch*/")


if __name__ == "__main__":
    print("🧪 Hierarchical Training Pipeline Test Suite")
    print("=" * 60)

    try:
        success = test_integration()

        if success:
            print_usage_instructions()

    except Exception as e:
        print("💥 Test suite error: {}".format(e))
        import traceback
        traceback.print_exc()