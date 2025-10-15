"""
多相机深度融合预处理模块

为SmolVLA提供多相机RGB+深度信息的融合预处理功能
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Union
import time
from .depth_conversion import depth_to_rgb_for_smolvla


class MultiCameraDepthFusion:
    """
    多相机深度融合处理器

    支持3个RGB相机 + 3个深度相机的融合处理
    """

    def __init__(self,
                 target_size: Tuple[int, int] = (512, 512),
                 depth_range: Tuple[float, float] = (0, 1000),
                 device: str = 'cpu',
                 enable_depth: bool = True):
        """
        初始化多相机深度融合器

        Args:
            target_size: 目标图像尺寸 (height, width)
            depth_range: 深度值范围 (min_depth, max_depth)
            device: 设备
            enable_depth: 是否启用深度处理
        """
        self.target_size = target_size
        self.depth_range = depth_range
        self.device = device
        self.enable_depth = enable_depth

        # 相机配对映射
        self.camera_pairs = {
            'head_cam_h': 'depth_h',      # 头部RGB + 头部深度
            'wrist_cam_l': 'depth_l',     # 左手RGB + 左手深度
            'wrist_cam_r': 'depth_r',     # 右手RGB + 右手深度
        }

        # RGB相机列表
        self.rgb_cameras = ['head_cam_h', 'wrist_cam_l', 'wrist_cam_r']

        # 深度相机列表
        self.depth_cameras = ['depth_h', 'depth_l', 'depth_r']

        print(f"✅ MultiCameraDepthFusion initialized")
        print(f"   Target size: {target_size}")
        print(f"   Depth range: {depth_range}")
        print(f"   Device: {device}")
        print(f"   Enable depth: {enable_depth}")
        print(f"   Camera pairs: {self.camera_pairs}")

    def img_preprocess_smolvla(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        SmolVLA RGB图像预处理

        Args:
            image: 输入图像 [H, W, 3] 或 [3, H, W]

        Returns:
            processed_tensor: 预处理后的张量 [1, 3, H, W]
        """
        from torchvision.transforms import functional as F
        from torchvision.transforms import InterpolationMode

        # 转换为tensor
        if isinstance(image, np.ndarray):
            tensor_img = torch.from_numpy(
                image).permute(2, 0, 1).float() / 255.0
        else:
            tensor_img = image.float()

        # 确保是3通道
        if tensor_img.shape[0] != 3:
            tensor_img = tensor_img.unsqueeze(0).repeat(3, 1, 1)

        h, w = tensor_img.shape[-2:]
        target_h, target_w = self.target_size

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
            value=0
        )

        # 添加batch维度
        return tensor_img.unsqueeze(0).to(self.device, non_blocking=True)

    def process_single_camera_pair(self,
                                   rgb_key: str,
                                   depth_key: str,
                                   obs: Dict) -> Dict[str, torch.Tensor]:
        """
        处理单个相机对（RGB + 深度）

        Args:
            rgb_key: RGB相机键名
            depth_key: 深度相机键名
            obs: 观测数据

        Returns:
            processed_data: 处理后的数据
        """
        processed_data = {}

        # 处理RGB图像
        if rgb_key in obs:
            rgb_image = obs[rgb_key]
            processed_data[f"observation.{rgb_key}"] = self.img_preprocess_smolvla(
                rgb_image)

        # 处理深度图像
        if self.enable_depth and depth_key in obs:
            depth_image = obs[depth_key]
            depth_rgb = depth_to_rgb_for_smolvla(
                depth_image,
                target_size=self.target_size,
                depth_range=self.depth_range,
                device=self.device
            )
            processed_data[f"observation.{depth_key}"] = depth_rgb

        return processed_data

    def process_observations(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """
        处理多相机观测数据

        Args:
            obs: 原始观测数据

        Returns:
            observation: 处理后的观测数据
        """
        observation = {}

        # 处理所有相机对
        for rgb_key, depth_key in self.camera_pairs.items():
            pair_data = self.process_single_camera_pair(
                rgb_key, depth_key, obs)
            observation.update(pair_data)

        # 处理其他观测数据（如状态信息）
        for key, value in obs.items():
            if key not in self.rgb_cameras and key not in self.depth_cameras:
                if 'state' in key.lower():
                    observation[f"observation.{key}"] = torch.tensor(
                        value, dtype=torch.float32
                    ).unsqueeze(0).to(self.device)

        return observation

    def process_observations_simple(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """
        简单的多相机处理（独立处理每个相机）

        Args:
            obs: 原始观测数据

        Returns:
            observation: 处理后的观测数据
        """
        observation = {}

        # 处理所有RGB相机
        for camera in self.rgb_cameras:
            if camera in obs:
                observation[f"observation.{camera}"] = self.img_preprocess_smolvla(
                    obs[camera])

        # 处理所有深度相机（转换为RGB伪彩色）
        if self.enable_depth:
            for camera in self.depth_cameras:
                if camera in obs:
                    depth_rgb = depth_to_rgb_for_smolvla(
                        obs[camera],
                        target_size=self.target_size,
                        depth_range=self.depth_range,
                        device=self.device
                    )
                    observation[f"observation.{camera}"] = depth_rgb

        # 处理状态信息
        for key, value in obs.items():
            if key not in self.rgb_cameras and key not in self.depth_cameras:
                if 'state' in key.lower():
                    observation[f"observation.{key}"] = torch.tensor(
                        value, dtype=torch.float32
                    ).unsqueeze(0).to(self.device)

        return observation

    def get_processing_stats(self) -> Dict[str, any]:
        """
        获取处理统计信息

        Returns:
            stats: 统计信息
        """
        return {
            'target_size': self.target_size,
            'depth_range': self.depth_range,
            'device': self.device,
            'enable_depth': self.enable_depth,
            'rgb_cameras': self.rgb_cameras,
            'depth_cameras': self.depth_cameras,
            'camera_pairs': self.camera_pairs,
        }


def create_multi_camera_fusion(target_size: Tuple[int, int] = (512, 512),
                               depth_range: Tuple[float, float] = (0, 1000),
                               device: str = 'cpu',
                               enable_depth: bool = True) -> MultiCameraDepthFusion:
    """
    创建多相机深度融合器

    Args:
        target_size: 目标图像尺寸
        depth_range: 深度值范围
        device: 设备
        enable_depth: 是否启用深度处理

    Returns:
        fusion_processor: 深度融合处理器
    """
    return MultiCameraDepthFusion(
        target_size=target_size,
        depth_range=depth_range,
        device=device,
        enable_depth=enable_depth
    )


def benchmark_multi_camera_fusion():
    """多相机融合性能测试"""
    print("🔍 多相机融合性能测试")
    print("=" * 50)

    # 创建测试数据
    obs = {}
    for camera in ['head_cam_h', 'wrist_cam_l', 'wrist_cam_r']:
        obs[camera] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    for camera in ['depth_h', 'depth_l', 'depth_r']:
        obs[camera] = np.random.randint(0, 1000, (480, 640), dtype=np.uint16)

    obs['state'] = np.random.randn(16)

    # 创建融合器
    fusion_processor = create_multi_camera_fusion(device='cpu')

    # 测试处理时间
    times = []
    for _ in range(50):
        start_time = time.time()
        processed_obs = fusion_processor.process_observations_simple(obs)
        processing_time = (time.time() - start_time) * 1000
        times.append(processing_time)

    print(f"多相机融合处理:")
    print(f"  平均时间: {np.mean(times):.2f}ms")
    print(f"  标准差: {np.std(times):.2f}ms")
    print(f"  最大时间: {np.max(times):.2f}ms")

    print(f"\n处理结果:")
    for key, value in processed_obs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    return fusion_processor


if __name__ == "__main__":
    # 运行性能测试
    fusion_processor = benchmark_multi_camera_fusion()

    # 显示统计信息
    stats = fusion_processor.get_processing_stats()
    print(f"\n融合器统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
