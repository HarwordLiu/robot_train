#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线评估主脚本

支持分层架构和传统diffusion模型的离线评估
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import lerobot_patches.custom_patches

import argparse
import logging
from typing import Optional
from omegaconf import DictConfig, OmegaConf

# 导入评估器
from kuavo_eval.core.hierarchical_evaluator import HierarchicalEvaluator
from kuavo_eval.core.diffusion_evaluator import DiffusionEvaluator
from kuavo_eval.utils.report_generator import EvaluationReportGenerator

def load_config(config_path: str) -> DictConfig:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置对象
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # 加载主配置
    config = OmegaConf.load(config_path)

    # 处理defaults（继承基础配置）
    if 'defaults' in config:
        for default_config in config.defaults:
            if isinstance(default_config, str):
                default_path = config_path.parent / f"{default_config}.yaml"
                if default_path.exists():
                    base_config = OmegaConf.load(default_path)
                    # 合并配置，当前配置优先
                    config = OmegaConf.merge(base_config, config)

    return config

def create_evaluator(config: DictConfig):
    """
    根据配置创建相应的评估器

    Args:
        config: 配置对象

    Returns:
        评估器实例
    """
    model_type = config.model.type

    if model_type == 'humanoid_diffusion':
        print(f"🤖 Creating HierarchicalEvaluator for {model_type}")
        return HierarchicalEvaluator(config)
    elif model_type == 'diffusion':
        print(f"📝 Creating DiffusionEvaluator for {model_type}")
        return DiffusionEvaluator(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def validate_config(config: DictConfig) -> None:
    """
    验证配置的有效性

    Args:
        config: 配置对象
    """
    required_fields = [
        'model.type',
        'model.checkpoint_path',
        'test_data.root',
        'test_data.repo_id',
        'common.device'
    ]

    for field in required_fields:
        if not OmegaConf.select(config, field):
            raise ValueError(f"Missing required config field: {field}")

    # 检查模型路径
    checkpoint_path = Path(config.model.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    # 检查数据路径
    data_root = Path(config.test_data.root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root directory not found: {data_root}")

    print("✅ Configuration validation passed")

def setup_logging(config: DictConfig) -> None:
    """
    设置日志

    Args:
        config: 配置对象
    """
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)

    # 配置根日志器
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 设置特定库的日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

def print_evaluation_info(config: DictConfig) -> None:
    """
    打印评估信息

    Args:
        config: 配置对象
    """
    print("\n" + "="*60)
    print("🚀 KUAVO OFFLINE EVALUATION")
    print("="*60)
    print(f"📋 Model Type: {config.model.type}")
    print(f"📂 Checkpoint: {config.model.checkpoint_path}")
    print(f"📊 Data Root: {config.test_data.root}")
    print(f"🎯 Repository: {config.test_data.repo_id}")
    print(f"📈 Episodes Range: {config.test_data.episodes_range}")
    print(f"🔢 Max Episodes: {config.test_data.max_episodes}")
    print(f"💻 Device: {config.common.device}")
    print("="*60 + "\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Kuavo Offline Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 评估分层架构模型
  python run_offline_eval.py --config ../../configs/eval/offline_hierarchical_eval.yaml

  # 评估传统diffusion模型
  python run_offline_eval.py --config ../../configs/eval/offline_diffusion_eval.yaml

  # 使用自定义输出目录
  python run_offline_eval.py --config ../../configs/eval/offline_hierarchical_eval.yaml --output-dir ./my_results

  # 详细模式
  python run_offline_eval.py --config ../../configs/eval/offline_hierarchical_eval.yaml --verbose
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to evaluation configuration file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results (overrides config)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Model checkpoint path (overrides config)'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        help='Maximum number of episodes to evaluate (overrides config)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        help='Device to use for evaluation (overrides config)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Enable quick validation mode (reduced episodes and steps)'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )

    args = parser.parse_args()

    try:
        # 加载配置
        print(f"📖 Loading configuration from {args.config}")
        config = load_config(args.config)

        # 应用命令行覆盖
        if args.output_dir:
            config.common.output_dir = args.output_dir

        if args.checkpoint:
            config.model.checkpoint_path = args.checkpoint

        if args.episodes:
            config.test_data.max_episodes = args.episodes

        if args.device:
            config.common.device = args.device

        if args.verbose:
            config.logging.level = 'DEBUG'

        if args.quick:
            # 启用快速验证模式
            config.test_data.max_episodes = min(3, config.test_data.max_episodes)
            config.test_data.max_steps_per_episode = min(20, config.test_data.max_steps_per_episode)
            print("⚡ Quick validation mode enabled")

        if args.no_plots:
            config.report.include_plots = False

        # 设置日志
        setup_logging(config)

        # 验证配置
        validate_config(config)

        # 打印评估信息
        print_evaluation_info(config)

        # 创建评估器
        evaluator = create_evaluator(config)

        # 执行评估
        print("🔄 Starting evaluation...")
        results = evaluator.evaluate()

        # 生成报告
        print("📊 Generating evaluation report...")
        report_generator = EvaluationReportGenerator(
            output_dir=config.common.output_dir,
            config=OmegaConf.to_container(config, resolve=True)
        )

        generated_files = report_generator.generate_comprehensive_report(
            results.__dict__,
            model_type=config.model.type
        )

        # 打印结果摘要
        print("\n" + "="*60)
        print("✅ EVALUATION COMPLETED")
        print("="*60)
        print("📈 Results Summary:")
        for key, value in results.summary.items():
            print(f"  {key}: {value}")

        print(f"\n📁 Generated Files:")
        for file_type, file_path in generated_files.items():
            print(f"  {file_type}: {file_path}")

        print("\n🎉 Evaluation completed successfully!")

    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        logging.exception("Detailed error information:")
        sys.exit(1)

if __name__ == "__main__":
    main()