"""
深度图像到RGB颜色映射工具

将深度图像转换为RGB伪彩色图像，使SmolVLA能够处理深度信息
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Tuple, List
import time


def depth_to_rgb_opencv(depth_image: np.ndarray,
                        colormap_type: int = cv2.COLORMAP_JET,
                        depth_range: Tuple[float, float] = (0, 1000)) -> np.ndarray:
    """
    使用OpenCV将深度图像转换为RGB伪彩色图像

    Args:
        depth_image: 深度图像 [H, W] 或 [H, W, 1]
        colormap_type: OpenCV颜色映射类型 (cv2.COLORMAP_JET, cv2.COLORMAP_RAINBOW等)
        depth_range: 深度值范围 (min_depth, max_depth)

    Returns:
        rgb_image: RGB伪彩色图像 [H, W, 3]
    """
    # 确保输入是单通道
    if len(depth_image.shape) == 3:
        depth_image = depth_image.squeeze()

    # 裁剪到指定范围
    depth_clipped = np.clip(depth_image, depth_range[0], depth_range[1])

    # 归一化到 [0, 255]
    depth_normalized = cv2.normalize(
        depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # 应用颜色映射
    rgb_image = cv2.applyColorMap(depth_normalized, colormap_type)

    return rgb_image


def depth_to_rgb_torch(depth_tensor: torch.Tensor,
                       colormap_lut: torch.Tensor,
                       depth_range: Tuple[float, float] = (0, 1000)) -> torch.Tensor:
    """
    使用PyTorch将深度张量转换为RGB伪彩色张量

    Args:
        depth_tensor: 深度张量 [B, 1, H, W] 或 [B, H, W]
        colormap_lut: 颜色映射查找表 [256, 3]
        depth_range: 深度值范围 (min_depth, max_depth)

    Returns:
        rgb_tensor: RGB伪彩色张量 [B, 3, H, W]
    """
    # 确保输入维度正确
    if depth_tensor.dim() == 3:
        depth_tensor = depth_tensor.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

    batch_size, channels, height, width = depth_tensor.shape

    # 裁剪到指定范围
    depth_clipped = torch.clamp(depth_tensor, depth_range[0], depth_range[1])

    # 归一化到 [0, 1]
    depth_min, depth_max = depth_range
    depth_normalized = (depth_clipped - depth_min) / (depth_max - depth_min)

    # 映射到颜色索引 [0, 255]
    indices = (depth_normalized * (colormap_lut.shape[0] - 1)).long()
    indices = torch.clamp(indices, 0, colormap_lut.shape[0] - 1)

    # 应用颜色映射
    rgb_tensor = colormap_lut[indices].permute(
        0, 4, 1, 2, 3).squeeze(4)  # [B, 3, H, W]

    return rgb_tensor


def create_jet_colormap_lut(device: str = 'cpu') -> torch.Tensor:
    """
    创建Jet颜色映射查找表

    Args:
        device: 设备 ('cpu' 或 'cuda')

    Returns:
        colormap_lut: Jet颜色映射查找表 [256, 3]
    """
    lut = torch.zeros(256, 3, device=device)

    for i in range(256):
        value = i / 255.0
        r, g, b = jet_colormap(value)
        lut[i] = torch.tensor([r, g, b], device=device) / 255.0

    return lut


def jet_colormap(value: float) -> Tuple[float, float, float]:
    """
    Jet颜色映射函数

    Args:
        value: 归一化值 [0, 1]

    Returns:
        (r, g, b): RGB颜色值 [0, 1]
    """
    # Jet颜色映射的数学定义
    if value < 0.125:
        # 深蓝到蓝
        r = 0
        g = 0
        b = 0.5 + 4 * value
    elif value < 0.375:
        # 蓝到青
        r = 0
        g = 4 * (value - 0.125)
        b = 1
    elif value < 0.625:
        # 青到绿
        r = 0
        g = 1
        b = 1 - 4 * (value - 0.375)
    elif value < 0.875:
        # 绿到黄
        r = 4 * (value - 0.625)
        g = 1
        b = 0
    else:
        # 黄到红
        r = 1
        g = 1 - 4 * (value - 0.875)
        b = 0

    return r, g, b


def depth_to_rgb_for_smolvla(depth_image: Union[np.ndarray, torch.Tensor],
                             target_size: Tuple[int, int] = (512, 512),
                             depth_range: Tuple[float, float] = (0, 1000),
                             device: str = 'cpu',
                             use_padding: bool = True) -> torch.Tensor:
    """
    为SmolVLA将深度图像转换为RGB伪彩色张量

    支持两种处理方式：
    1. use_padding=True: 保持长宽比，用padding填充 (推荐用于高精度任务)
    2. use_padding=False: 直接resize到目标尺寸 (快速处理)

    Args:
        depth_image: 深度图像 [H, W] 或 [H, W, 1]
        target_size: 目标尺寸 (height, width)
        depth_range: 深度值范围 (min_depth, max_depth)
        device: 设备
        use_padding: 是否使用padding方式保持长宽比

    Returns:
        rgb_tensor: RGB张量 [1, 3, H, W]
    """
    # 转换为numpy数组
    if isinstance(depth_image, torch.Tensor):
        depth_np = depth_image.cpu().numpy()
    else:
        depth_np = depth_image

    # 确保是单通道
    if len(depth_np.shape) == 3:
        depth_np = depth_np.squeeze()

    # 转换为RGB伪彩色
    rgb_image = depth_to_rgb_opencv(depth_np, cv2.COLORMAP_JET, depth_range)

    if use_padding:
        # 使用padding方式保持长宽比
        rgb_tensor = _resize_with_padding(rgb_image, target_size, device)
    else:
        # 直接resize到目标尺寸
        if rgb_image.shape[:2] != target_size:
            rgb_image = cv2.resize(
                rgb_image, target_size[::-1], interpolation=cv2.INTER_LINEAR)

        # 转换为PyTorch张量
        rgb_tensor = torch.from_numpy(
            rgb_image).permute(2, 0, 1).float() / 255.0
        rgb_tensor = rgb_tensor.unsqueeze(0)  # 添加batch维度 [1, 3, H, W]
        rgb_tensor = rgb_tensor.to(device)

    return rgb_tensor


def _resize_with_padding(rgb_image: np.ndarray,
                         target_size: Tuple[int, int],
                         device: str) -> torch.Tensor:
    """
    使用padding方式调整图像尺寸，保持长宽比

    Args:
        rgb_image: RGB图像 [H, W, 3]
        target_size: 目标尺寸 (height, width)
        device: 设备

    Returns:
        rgb_tensor: RGB张量 [1, 3, H, W]
    """
    from torchvision.transforms import functional as F
    from torchvision.transforms import InterpolationMode

    # 转换为tensor
    tensor_img = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0

    h, w = tensor_img.shape[-2:]
    target_h, target_w = target_size

    # 计算缩放比例（保持长宽比）
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    tensor_img = F.resize(
        tensor_img, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)

    # Pad到目标尺寸
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    tensor_img = torch.nn.functional.pad(
        tensor_img,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode='constant',
        value=0  # 用0填充（深度=0表示无效区域）
    )

    # 添加batch维度
    return tensor_img.unsqueeze(0).to(device, non_blocking=True)


def benchmark_depth_conversion():
    """性能基准测试"""
    print("🔍 深度转换性能测试")
    print("=" * 50)

    # 测试数据
    depth_image = np.random.randint(0, 1000, (512, 512), dtype=np.uint16)

    # OpenCV实现测试
    times_opencv = []
    for _ in range(100):
        start_time = time.time()
        rgb_image = depth_to_rgb_opencv(depth_image)
        conversion_time = (time.time() - start_time) * 1000
        times_opencv.append(conversion_time)

    print(f"OpenCV实现:")
    print(f"  平均时间: {np.mean(times_opencv):.2f}ms")
    print(f"  标准差: {np.std(times_opencv):.2f}ms")
    print(f"  最大时间: {np.max(times_opencv):.2f}ms")

    # PyTorch实现测试
    depth_tensor = torch.from_numpy(
        depth_image).unsqueeze(0).unsqueeze(0).float()
    colormap_lut = create_jet_colormap_lut()

    times_torch = []
    for _ in range(100):
        start_time = time.time()
        rgb_tensor = depth_to_rgb_torch(depth_tensor, colormap_lut)
        conversion_time = (time.time() - start_time) * 1000
        times_torch.append(conversion_time)

    print(f"\nPyTorch实现:")
    print(f"  平均时间: {np.mean(times_torch):.2f}ms")
    print(f"  标准差: {np.std(times_torch):.2f}ms")
    print(f"  最大时间: {np.max(times_torch):.2f}ms")

    print(
        f"\n推荐使用: {'OpenCV' if np.mean(times_opencv) < np.mean(times_torch) else 'PyTorch'}")


if __name__ == "__main__":
    # 运行性能测试
    benchmark_depth_conversion()
