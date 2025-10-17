#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控器测试脚本 - 验证监控器功能是否正常

用法:
    python kuavo_train/test_monitor.py
"""

import sys
from pathlib import Path

def check_module(module_name):
    """检查模块是否可导入"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def main():
    print("=" * 70)
    print("🧪 训练监控器功能测试")
    print("=" * 70)
    print()
    
    # 1. 检查依赖
    print("📦 检查依赖包...")
    print("-" * 70)
    
    deps = {
        'tensorboard': ('TensorBoard', '必需', '用于解析训练事件'),
        'matplotlib': ('Matplotlib', '可选', '用于绘制图表'),
        'rich': ('Rich', '可选', '用于美化终端输出'),
        'psutil': ('psutil', '可选', '用于系统监控'),
        'GPUtil': ('GPUtil', '可选', '用于GPU监控')
    }
    
    results = {}
    for module, (name, level, desc) in deps.items():
        available = check_module(module)
        results[module] = available
        
        status = "✅" if available else "❌"
        level_tag = f"[{level}]"
        print(f"{status} {name:15s} {level_tag:8s} - {desc}")
    
    print()
    
    # 2. 检查监控脚本
    print("📄 检查监控脚本...")
    print("-" * 70)
    
    project_root = Path(__file__).parent.parent
    scripts = {
        'monitor_training.py': '基础监控器',
        'monitor_training_advanced.py': '高级监控器',
        'monitor.sh': '快捷启动脚本',
        'TRAINING_MONITOR_README.md': '使用文档'
    }
    
    all_scripts_exist = True
    for script_name, desc in scripts.items():
        script_path = project_root / 'kuavo_train' / script_name
        exists = script_path.exists()
        all_scripts_exist = all_scripts_exist and exists
        
        status = "✅" if exists else "❌"
        print(f"{status} {script_name:35s} - {desc}")
    
    print()
    
    # 3. 查找训练运行
    print("🔍 查找训练运行...")
    print("-" * 70)
    
    train_dir = project_root / 'outputs' / 'train'
    if train_dir.exists():
        run_dirs = []
        for task_dir in train_dir.iterdir():
            if task_dir.is_dir():
                for method_dir in task_dir.iterdir():
                    if method_dir.is_dir():
                        for run_dir in method_dir.glob("run_*"):
                            if run_dir.is_dir():
                                run_dirs.append(run_dir)
        
        if run_dirs:
            run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            print(f"✅ 找到 {len(run_dirs)} 个训练运行")
            print(f"\n最新的5个运行:")
            for i, run_dir in enumerate(run_dirs[:5], 1):
                rel_path = run_dir.relative_to(project_root)
                print(f"  {i}. {rel_path}")
            
            # 检查最新运行的TensorBoard事件
            latest_run = run_dirs[0]
            tb_events = list(latest_run.glob('events.out.tfevents.*'))
            if tb_events:
                print(f"\n✅ 最新运行包含TensorBoard事件文件")
            else:
                print(f"\n⚠️  最新运行缺少TensorBoard事件文件")
        else:
            print("⚠️  未找到训练运行")
            print("   请先运行训练脚本")
    else:
        print("❌ 训练输出目录不存在")
        print(f"   期望位置: {train_dir}")
    
    print()
    
    # 4. 生成测试报告
    print("📊 测试结果总结")
    print("-" * 70)
    
    # 基础功能可用性
    basic_available = results.get('tensorboard', False)
    advanced_available = results.get('rich', False) and results.get('matplotlib', False)
    gpu_monitoring = results.get('GPUtil', False)
    
    print(f"基础监控器:   {'✅ 可用' if basic_available else '❌ 不可用（需要安装tensorboard）'}")
    print(f"高级监控器:   {'✅ 可用' if advanced_available else '⚠️  部分功能不可用（需要安装rich和matplotlib）'}")
    print(f"GPU监控:      {'✅ 可用' if gpu_monitoring else '⚠️  不可用（需要安装GPUtil）'}")
    
    print()
    
    # 5. 使用建议
    print("💡 使用建议")
    print("-" * 70)
    
    if not basic_available:
        print("🔧 安装基础依赖:")
        print("   pip install tensorboard")
        print()
    
    if not advanced_available:
        print("🔧 安装高级功能依赖:")
        missing = []
        if not results.get('rich'):
            missing.append('rich')
        if not results.get('matplotlib'):
            missing.append('matplotlib')
        print(f"   pip install {' '.join(missing)}")
        print()
    
    if basic_available:
        print("🚀 快速开始:")
        print("   # 方式1: 使用快捷脚本")
        print("   ./kuavo_train/monitor.sh")
        print()
        print("   # 方式2: 直接运行Python脚本")
        print("   python kuavo_train/monitor_training.py")
        print()
    
    if advanced_available:
        print("🎨 启动高级监控:")
        print("   ./kuavo_train/monitor.sh advanced")
        print("   或")
        print("   python kuavo_train/monitor_training_advanced.py")
        print()
    
    print("📖 查看完整文档:")
    print("   cat kuavo_train/TRAINING_MONITOR_README.md")
    print()
    
    # 6. 最终状态
    print("=" * 70)
    if basic_available and all_scripts_exist:
        print("✅ 训练监控器已就绪，可以开始使用！")
    elif basic_available:
        print("⚠️  基础功能可用，但部分脚本缺失")
    else:
        print("❌ 请先安装必需依赖: pip install tensorboard")
    print("=" * 70)

if __name__ == "__main__":
    main()

