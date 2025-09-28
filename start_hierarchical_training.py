# -*- coding: utf-8 -*-
"""
分层架构训练启动脚本

使用方法：
python start_hierarchical_training.py [--validate-first] [--config-name CONFIG_NAME]
"""
import argparse
import subprocess
import sys
import os


def run_validation():
    """运行框架验证"""
    print("🧪 Running framework validation...")
    try:
        result = subprocess.run([sys.executable, "validate_hierarchical_framework.py"],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Validation passed!")
            return True
        else:
            print("❌ Validation failed!")
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print("❌ Validation error:", str(e))
        return False


def start_training(config_name="humanoid_diffusion_config"):
    """启动分层架构训练"""
    print("🚀 Starting hierarchical training...")

    # 构建分层架构专用训练命令
    cmd = [
        sys.executable,
        "kuavo_train/train_hierarchical_policy.py",
        "--config-name={}".format(config_name)
    ]

    print("Hierarchical training command:", " ".join(cmd))

    try:
        # 运行训练
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print("❌ Training error:", str(e))


def main():
    parser = argparse.ArgumentParser(description="Start hierarchical framework training")
    parser.add_argument("--validate-first", action="store_true",
                       help="Run validation before training")
    parser.add_argument("--config-name", default="humanoid_diffusion_config",
                       help="Config file name to use")

    args = parser.parse_args()

    print("🤖 Hierarchical Humanoid Diffusion Policy Training")
    print("=" * 50)

    # 检查配置文件
    config_path = "configs/policy/{}.yaml".format(args.config_name)
    if not os.path.exists(config_path):
        print("❌ Config file not found: {}".format(config_path))
        print("Available configs:")
        config_dir = "configs/policy"
        if os.path.exists(config_dir):
            for f in os.listdir(config_dir):
                if f.endswith('.yaml'):
                    print("  - {}".format(f[:-5]))
        return

    # 可选的验证步骤
    if args.validate_first:
        if not run_validation():
            print("⚠️ Validation failed. Continue training anyway? (y/N)")
            if input().lower() != 'y':
                print("Training cancelled.")
                return

    # 启动训练
    start_training(args.config_name)


if __name__ == "__main__":
    main()