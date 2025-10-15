#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多相机深度融合测试脚本

测试SmolVLA多相机深度融合功能的实现效果
"""

from kuavo_deploy.utils.multi_camera_fusion import (
    create_multi_camera_fusion,
    benchmark_multi_camera_fusion
)
from kuavo_deploy.utils.depth_conversion import (
    depth_to_rgb_for_smolvla,
    benchmark_depth_conversion
)
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


def test_depth_conversion():
    """测试深度转换功能"""
    print("🔍 测试深度转换功能")
    print("=" * 50)

    # 创建测试深度图像
    depth_image = np.random.randint(0, 1000, (480, 640), dtype=np.uint16)

    # 测试转换
    rgb_tensor = depth_to_rgb_for_smolvla(
        depth_image,
        target_size=(512, 512),
        depth_range=(0, 1000),
        device='cpu'
    )

    print(f"✅ 深度转换成功")
    print(f"   输入形状: {depth_image.shape}")
    print(f"   输出形状: {rgb_tensor.shape}")
    print(f"   数据类型: {rgb_tensor.dtype}")
    print(f"   数值范围: [{rgb_tensor.min():.3f}, {rgb_tensor.max():.3f}]")

    return rgb_tensor


def test_multi_camera_fusion():
    """测试多相机融合功能"""
    print("\n🔍 测试多相机融合功能")
    print("=" * 50)

    # 创建测试观测数据
    obs = {}

    # RGB相机数据
    for camera in ['head_cam_h', 'wrist_cam_l', 'wrist_cam_r']:
        obs[camera] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 深度相机数据
    for camera in ['depth_h', 'depth_l', 'depth_r']:
        obs[camera] = np.random.randint(0, 1000, (480, 640), dtype=np.uint16)

    # 状态数据
    obs['state'] = np.random.randn(16)

    # 创建融合处理器
    fusion_processor = create_multi_camera_fusion(
        target_size=(512, 512),
        depth_range=(0, 1000),
        device='cpu',
        enable_depth=True
    )

    # 处理观测数据
    processed_obs = fusion_processor.process_observations_simple(obs)

    print(f"✅ 多相机融合成功")
    print(
        f"   输入相机数: {len([k for k in obs.keys() if 'cam' in k or 'depth' in k])}")
    print(
        f"   输出张量数: {len([k for k in processed_obs.keys() if 'observation' in k])}")

    # 显示处理结果
    print(f"\n📊 处理结果:")
    for key, value in processed_obs.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.shape} ({value.dtype})")
        else:
            print(f"   {key}: {type(value)}")

    return processed_obs


def test_performance():
    """测试性能"""
    print("\n🔍 性能测试")
    print("=" * 50)

    # 深度转换性能测试
    benchmark_depth_conversion()

    # 多相机融合性能测试
    benchmark_multi_camera_fusion()


def test_visualization():
    """测试可视化效果"""
    print("\n🔍 可视化测试")
    print("=" * 50)

    # 创建模拟深度图像（模拟传送带场景）
    depth_image = np.zeros((480, 640), dtype=np.uint16)

    # 添加一些深度层次
    depth_image[100:200, 200:400] = 300  # 近距离物体
    depth_image[250:350, 150:500] = 600  # 中距离物体
    depth_image[400:450, 100:600] = 900  # 远距离背景

    # 转换为RGB伪彩色
    rgb_tensor = depth_to_rgb_for_smolvla(
        depth_image,
        target_size=(512, 512),
        depth_range=(0, 1000),
        device='cpu'
    )

    # 转换为numpy用于显示
    rgb_image = rgb_tensor.squeeze(0).permute(1, 2, 0).numpy()

    print(f"✅ 可视化测试完成")
    print(f"   深度图像范围: [{depth_image.min()}, {depth_image.max()}]")
    print(f"   RGB图像范围: [{rgb_image.min():.3f}, {rgb_image.max():.3f}]")

    # 保存图像用于检查
    try:
        import cv2
        cv2.imwrite('/tmp/depth_original.png', depth_image)
        cv2.imwrite('/tmp/depth_rgb.png', (rgb_image * 255).astype(np.uint8))
        print(f"   图像已保存到 /tmp/ 目录")
    except ImportError:
        print(f"   需要安装 opencv-python 来保存图像")

    return rgb_image


def test_smolvla_compatibility():
    """测试SmolVLA兼容性"""
    print("\n🔍 SmolVLA兼容性测试")
    print("=" * 50)

    # 创建模拟观测数据
    obs = {}

    # RGB相机数据
    for camera in ['head_cam_h', 'wrist_cam_l', 'wrist_cam_r']:
        obs[camera] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 深度相机数据
    for camera in ['depth_h', 'depth_l', 'depth_r']:
        obs[camera] = np.random.randint(0, 1000, (480, 640), dtype=np.uint16)

    # 状态数据
    obs['state'] = np.random.randn(16)

    # 创建融合处理器
    fusion_processor = create_multi_camera_fusion(
        target_size=(512, 512),
        depth_range=(0, 1000),
        device='cpu',
        enable_depth=True
    )

    # 处理观测数据
    processed_obs = fusion_processor.process_observations_simple(obs)

    # 添加语言指令（SmolVLA需要）
    processed_obs['task'] = [
        'Pick up the moving object from the conveyor belt']

    print(f"✅ SmolVLA兼容性测试完成")
    print(f"   观测键: {list(processed_obs.keys())}")

    # 验证张量形状
    for key, value in processed_obs.items():
        if isinstance(value, torch.Tensor):
            if len(value.shape) == 4:  # 图像张量
                print(f"   {key}: {value.shape} ✅ (图像张量)")
            elif len(value.shape) == 2:  # 状态张量
                print(f"   {key}: {value.shape} ✅ (状态张量)")
        elif isinstance(value, list):  # 语言指令
            print(f"   {key}: {len(value)} 条指令 ✅ (语言指令)")

    return processed_obs


def main():
    """主测试函数"""
    print("🚀 SmolVLA多相机深度融合测试")
    print("=" * 60)

    try:
        # 1. 测试深度转换
        test_depth_conversion()

        # 2. 测试多相机融合
        test_multi_camera_fusion()

        # 3. 测试性能
        test_performance()

        # 4. 测试可视化
        test_visualization()

        # 5. 测试SmolVLA兼容性
        test_smolvla_compatibility()

        print("\n🎉 所有测试通过！")
        print("✅ SmolVLA多相机深度融合功能已成功实现")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
