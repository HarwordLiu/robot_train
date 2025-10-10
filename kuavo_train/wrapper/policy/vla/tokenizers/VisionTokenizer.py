"""
VisionTokenizer: 将RGB和Depth图像转换为patch tokens
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class VisionTokenizer(nn.Module):
    """
    将视觉输入（RGB + Depth）转换为patch tokens

    设计思路：
    - 使用卷积层将图像分割为patches
    - 每个patch投影到embed_dim维度
    - 添加位置编码
    - 支持多相机输入
    """

    def __init__(self, patch_size: int = 16, embed_dim: int = 512, image_size: int = 224):
        """
        Args:
            patch_size: Patch大小（正方形）
            embed_dim: Token embedding维度
            image_size: 输入图像尺寸（假设正方形）
        """
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.image_size = image_size

        # 计算patch数量
        self.num_patches = (image_size // patch_size) ** 2

        # RGB图像投影层：将3通道图像转为tokens
        self.rgb_projection = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

        # Depth图像投影层：将1通道深度图转为tokens
        self.depth_projection = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

        # 位置编码（可学习）
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

        # 模态类型embedding（区分RGB和Depth）
        self.modality_embedding = nn.Embedding(2, embed_dim)  # 0=RGB, 1=Depth

        # LayerNorm
        self.norm = nn.LayerNorm(embed_dim)

        print(
            f"✅ VisionTokenizer initialized: {self.num_patches} patches per image, {embed_dim}D tokens")

    def forward(self, rgb_images: torch.Tensor, depth_images: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            rgb_images: [B, num_cameras, 3, H, W] 或 [B, 3, H, W]
            depth_images: [B, num_cameras, 1, H, W] 或 [B, 1, H, W] (可选)

        Returns:
            tokens: [B, num_tokens, embed_dim]
                    其中 num_tokens = num_cameras * num_patches (如果有depth则x2)
        """
        device = rgb_images.device

        # 处理输入维度
        if len(rgb_images.shape) == 4:
            # [B, 3, H, W] -> [B, 1, 3, H, W]
            rgb_images = rgb_images.unsqueeze(1)

        batch_size, num_cameras, channels, height, width = rgb_images.shape

        all_tokens = []

        # 处理RGB图像
        # [B, num_cameras, 3, H, W] -> [B*num_cameras, 3, H, W]
        rgb_flat = rgb_images.view(
            batch_size * num_cameras, channels, height, width)

        # 投影到tokens: [B*num_cameras, embed_dim, H/patch_size, W/patch_size]
        rgb_patches = self.rgb_projection(rgb_flat)

        # 重排为token序列: [B*num_cameras, embed_dim, num_patches] -> [B*num_cameras, num_patches, embed_dim]
        rgb_tokens = rgb_patches.flatten(2).transpose(1, 2)

        # 添加位置编码
        rgb_tokens = rgb_tokens + self.positional_encoding

        # 添加模态embedding (RGB=0)
        modality_embed_rgb = self.modality_embedding(
            torch.zeros(1, dtype=torch.long, device=device))
        rgb_tokens = rgb_tokens + modality_embed_rgb

        # 归一化
        rgb_tokens = self.norm(rgb_tokens)

        # 重塑为 [B, num_cameras*num_patches, embed_dim]
        rgb_tokens = rgb_tokens.view(
            batch_size, num_cameras * self.num_patches, self.embed_dim)
        all_tokens.append(rgb_tokens)

        # 处理Depth图像（如果提供）
        if depth_images is not None:
            if len(depth_images.shape) == 4:
                depth_images = depth_images.unsqueeze(1)

            _, num_depth_cameras, depth_channels, _, _ = depth_images.shape
            depth_flat = depth_images.view(
                batch_size * num_depth_cameras, depth_channels, height, width)

            # 投影到tokens
            depth_patches = self.depth_projection(depth_flat)
            depth_tokens = depth_patches.flatten(2).transpose(1, 2)

            # 添加位置编码
            depth_tokens = depth_tokens + self.positional_encoding

            # 添加模态embedding (Depth=1)
            modality_embed_depth = self.modality_embedding(
                torch.ones(1, dtype=torch.long, device=device))
            depth_tokens = depth_tokens + modality_embed_depth

            # 归一化
            depth_tokens = self.norm(depth_tokens)

            # 重塑
            depth_tokens = depth_tokens.view(
                batch_size, num_depth_cameras * self.num_patches, self.embed_dim)
            all_tokens.append(depth_tokens)

        # 拼接所有tokens
        final_tokens = torch.cat(all_tokens, dim=1)

        return final_tokens

    def get_num_tokens(self, num_rgb_cameras: int = 1, num_depth_cameras: int = 0) -> int:
        """计算总token数量"""
        return (num_rgb_cameras + num_depth_cameras) * self.num_patches
