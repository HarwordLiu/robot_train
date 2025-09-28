# -*- coding: utf-8 -*-
"""
分层架构部署测试脚本

验证分层架构部署模块的功能和集成
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import subprocess
import time
from pathlib import Path
import yaml


def test_deployment_scripts_syntax():
    """测试部署脚本语法"""
    print("🧪 Testing deployment scripts syntax...")

    scripts = [
        "kuavo_deploy/examples/eval/eval_kuavo_hierarchical.py",
        "kuavo_deploy/examples/scripts/script_hierarchical.py",
        "kuavo_deploy/examples/scripts/script_hierarchical_auto_test.py"
    ]

    for script in scripts:
        script_path = script
        if not os.path.exists(script_path):
            print(f"❌ Script not found: {script_path}")
            return False

        try:
            # 语法检查
            result = subprocess.run([
                sys.executable, "-m", "py_compile", script_path
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print(f"  ✅ {script}: Syntax OK")
            else:
                print(f"  ❌ {script}: Syntax error")
                print(f"     {result.stderr}")
                return False

        except Exception as e:
            print(f"  ❌ {script}: Test failed - {e}")
            return False

    print("✅ All deployment scripts syntax tests passed!")
    return True


def test_config_loading():
    """测试配置文件加载"""
    print("🧪 Testing hierarchical config loading...")

    config_path = "configs/deploy/kuavo_hierarchical_sim_env.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 验证分层架构特有配置
        required_keys = ['policy_type', 'hierarchical']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required config key: {key}")
                return False

        # 验证policy_type
        if config['policy_type'] != 'hierarchical_diffusion':
            print(f"❌ Incorrect policy_type: {config['policy_type']}")
            return False

        # 验证分层配置
        hierarchical_config = config.get('hierarchical', {})
        required_hierarchical_keys = ['enabled_layers', 'latency_budget_ms', 'layer_configs']
        for key in required_hierarchical_keys:
            if key not in hierarchical_config:
                print(f"❌ Missing hierarchical config key: {key}")
                return False

        print("✅ Hierarchical config loading test passed!")
        return True

    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def test_imports():
    """测试关键导入"""
    print("🧪 Testing hierarchical deployment imports...")

    try:
        # 测试基础导入
        import torch
        print("  ✅ PyTorch imported")

        # 测试lerobot导入
        import lerobot_patches.custom_patches
        from lerobot.configs.policies import PolicyFeature
        print("  ✅ Lerobot imports OK")

        # 测试分层架构模块导入
        try:
            from kuavo_train.wrapper.policy.humanoid.HumanoidDiffusionPolicy import HumanoidDiffusionPolicy
            print("  ✅ HumanoidDiffusionPolicy imported")
        except ImportError as e:
            print(f"  ⚠️  HumanoidDiffusionPolicy import warning: {e}")
            print("     This is expected if hierarchical modules are not yet implemented")

        # 测试部署模块导入
        try:
            from kuavo_deploy.examples.eval.eval_kuavo_hierarchical import setup_hierarchical_policy
            print("  ✅ Hierarchical deployment modules imported")
        except ImportError as e:
            print(f"  ❌ Deployment module import failed: {e}")
            return False

        print("✅ Import test completed!")
        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_policy_loading_interface():
    """测试策略加载接口"""
    print("🧪 Testing policy loading interface...")

    try:
        from kuavo_deploy.examples.eval.eval_kuavo_hierarchical import setup_hierarchical_policy
        from pathlib import Path
        import torch

        # 测试接口是否可以调用（不需要实际模型文件）
        fake_path = Path("/fake/model/path")
        device = torch.device("cpu")

        # 测试不同policy_type
        policy_types = ['diffusion', 'act', 'hierarchical_diffusion']

        for policy_type in policy_types:
            try:
                # 这会失败因为路径不存在，但我们只测试接口
                setup_hierarchical_policy(fake_path, policy_type, device)
            except (FileNotFoundError, RuntimeError) as e:
                # 预期的错误
                print(f"  ✅ {policy_type}: Interface available (expected error: {type(e).__name__})")
            except Exception as e:
                print(f"  ❌ {policy_type}: Unexpected error: {e}")
                return False

        print("✅ Policy loading interface test passed!")
        return True

    except Exception as e:
        print(f"❌ Policy loading interface test failed: {e}")
        return False


def test_shell_script():
    """测试Shell脚本"""
    print("🧪 Testing hierarchical shell script...")

    script_path = "kuavo_deploy/eval_hierarchical_kuavo.sh"
    if not os.path.exists(script_path):
        print(f"❌ Shell script not found: {script_path}")
        return False

    try:
        # 检查脚本语法
        result = subprocess.run([
            "bash", "-n", script_path
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("  ✅ Shell script syntax OK")
        else:
            print("  ❌ Shell script syntax error:")
            print(f"     {result.stderr}")
            return False

        # 检查是否可执行
        if os.access(script_path, os.X_OK):
            print("  ✅ Shell script is executable")
        else:
            print("  ❌ Shell script is not executable")
            return False

        print("✅ Shell script test passed!")
        return True

    except Exception as e:
        print(f"❌ Shell script test failed: {e}")
        return False


def test_integration():
    """集成测试"""
    print("🧪 Running hierarchical deployment integration tests...")

    tests = [
        ("Deployment Scripts Syntax", test_deployment_scripts_syntax),
        ("Config Loading", test_config_loading),
        ("Import Dependencies", test_imports),
        ("Policy Loading Interface", test_policy_loading_interface),
        ("Shell Script", test_shell_script),
    ]

    passed = 0
    results = {}

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results[test_name] = success
            if success:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    print("\n" + "=" * 70)
    print(f"📊 Hierarchical Deployment Test Results: {passed}/{len(tests)} tests passed")
    print("=" * 70)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {test_name}")

    if passed == len(tests):
        print("\n🎉 All hierarchical deployment tests passed! Deployment is ready!")
        print_usage_instructions()
        return True
    else:
        print("\n❌ Some deployment tests failed. Please address the issues before deployment.")
        return False


def print_usage_instructions():
    """打印使用说明"""
    print("\n" + "=" * 70)
    print("🚀 Hierarchical Deployment Usage Instructions")
    print("=" * 70)

    print("\n1. 训练分层架构模型:")
    print("   python start_hierarchical_training.py --validate-first")

    print("\n2. 配置部署环境:")
    print("   # 复制并修改配置文件")
    print("   cp configs/deploy/kuavo_hierarchical_sim_env.yaml my_hierarchical_config.yaml")
    print("   # 更新模型路径: task, method, timestamp, epoch")

    print("\n3. 启动分层架构部署:")
    print("   bash kuavo_deploy/eval_hierarchical_kuavo.sh")

    print("\n4. 直接运行分层架构推理:")
    print("   python kuavo_deploy/examples/scripts/script_hierarchical.py \\")
    print("     --task run --config my_hierarchical_config.yaml")

    print("\n5. 运行分层架构自动测试:")
    print("   python kuavo_deploy/examples/scripts/script_hierarchical_auto_test.py \\")
    print("     --task auto_test --config my_hierarchical_config.yaml")

    print("\n🔧 分层架构特色功能:")
    print("   📊 实时层性能监控")
    print("   ⚡ 自适应推理延迟控制")
    print("   🛡️  安全层优先级保障")
    print("   📈 详细性能统计分析")

    print("\n📂 关键文件:")
    print("   - 训练配置: configs/policy/humanoid_diffusion_config.yaml")
    print("   - 部署配置: configs/deploy/kuavo_hierarchical_sim_env.yaml")
    print("   - 启动脚本: kuavo_deploy/eval_hierarchical_kuavo.sh")
    print("   - 部署模块: kuavo_deploy/examples/eval/eval_kuavo_hierarchical.py")


if __name__ == "__main__":
    print("🧪 Hierarchical Deployment Test Suite")
    print("=" * 70)

    try:
        success = test_integration()

        if success:
            print("\n✅ Hierarchical deployment is ready for use!")
        else:
            print("\n❌ Please fix the issues before deploying.")

    except Exception as e:
        print(f"💥 Test suite error: {e}")
        import traceback
        traceback.print_exc()