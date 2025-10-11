"""
加载预训练的Patch Embedding权重
支持从DINO, MAE, CLIP等预训练ViT模型加载
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import warnings


def load_pretrained_patch_embedding(
    patch_size: int = 32,
    embed_dim: int = 256,
    model_name: str = 'dino'
) -> Optional[torch.Tensor]:
    """
    从预训练ViT模型加载patch embedding权重

    Args:
        patch_size: 目标patch大小（16或32）
        embed_dim: 目标embedding维度
        model_name: 预训练模型名称 ('dino', 'mae', 'clip')

    Returns:
        weight: [embed_dim, 3, patch_size, patch_size] 卷积权重
                如果加载失败返回None
    """
    try:
        if model_name.lower() == 'dino':
            return _load_from_dino(patch_size, embed_dim)
        elif model_name.lower() == 'mae':
            return _load_from_mae(patch_size, embed_dim)
        elif model_name.lower() == 'clip':
            return _load_from_clip(patch_size, embed_dim)
        else:
            warnings.warn(f"Unknown model_name: {model_name}, using DINO as default")
            return _load_from_dino(patch_size, embed_dim)
    except Exception as e:
        warnings.warn(f"Failed to load pretrained weights: {e}")
        return None


def _load_from_dino(patch_size: int, embed_dim: int) -> Optional[torch.Tensor]:
    """
    从DINO ViT模型加载权重

    DINO默认配置：
    - vit_small: patch=16, dim=384
    - vit_base: patch=16, dim=768
    """
    try:
        # 尝试加载DINO小模型（更轻量）
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)

        # 提取patch_embed的投影层权重
        # shape: [384, 3, 16, 16]
        pretrained_weight = model.patch_embed.proj.weight.data.clone()
        pretrained_dim, in_channels, pretrained_patch_h, pretrained_patch_w = pretrained_weight.shape

        print(f"📦 Loaded DINO pretrained weights: [{pretrained_dim}, {in_channels}, {pretrained_patch_h}, {pretrained_patch_w}]")

        # 1. 调整patch size（如果需要）
        if patch_size != pretrained_patch_h:
            # 使用插值调整patch大小
            # 方法：将卷积核reshape后插值，再reshape回去
            weight = F.interpolate(
                pretrained_weight,
                size=(patch_size, patch_size),
                mode='bilinear',
                align_corners=False
            )
            print(f"   ✓ Interpolated patch size: {pretrained_patch_h}→{patch_size}")
        else:
            weight = pretrained_weight

        # 2. 调整embedding维度（如果需要）
        if embed_dim != pretrained_dim:
            # 方法1：如果目标维度更小，使用PCA降维
            # 方法2：如果目标维度更大，用零填充
            if embed_dim < pretrained_dim:
                # 简单截断（保留前embed_dim维）
                weight = weight[:embed_dim]
                print(f"   ✓ Truncated embed dim: {pretrained_dim}→{embed_dim}")
            else:
                # 零填充
                padding = torch.zeros(embed_dim - pretrained_dim, in_channels, patch_size, patch_size)
                weight = torch.cat([weight, padding], dim=0)
                print(f"   ✓ Padded embed dim: {pretrained_dim}→{embed_dim}")

        print(f"✅ Final weight shape: {list(weight.shape)}")
        return weight

    except Exception as e:
        warnings.warn(f"Failed to load DINO weights: {e}")
        return None


def _load_from_mae(patch_size: int, embed_dim: int) -> Optional[torch.Tensor]:
    """
    从MAE (Masked Autoencoder) 模型加载权重

    MAE默认配置：
    - vit_base: patch=16, dim=768
    """
    try:
        # MAE需要timm库
        import timm

        # 加载预训练MAE模型
        model = timm.create_model('vit_base_patch16_224', pretrained=True)

        pretrained_weight = model.patch_embed.proj.weight.data.clone()
        pretrained_dim, in_channels, pretrained_patch_h, pretrained_patch_w = pretrained_weight.shape

        print(f"📦 Loaded MAE pretrained weights: [{pretrained_dim}, {in_channels}, {pretrained_patch_h}, {pretrained_patch_w}]")

        # 调整尺寸（同DINO）
        if patch_size != pretrained_patch_h:
            weight = F.interpolate(
                pretrained_weight,
                size=(patch_size, patch_size),
                mode='bilinear',
                align_corners=False
            )
            print(f"   ✓ Interpolated patch size: {pretrained_patch_h}→{patch_size}")
        else:
            weight = pretrained_weight

        if embed_dim != pretrained_dim:
            if embed_dim < pretrained_dim:
                weight = weight[:embed_dim]
                print(f"   ✓ Truncated embed dim: {pretrained_dim}→{embed_dim}")
            else:
                padding = torch.zeros(embed_dim - pretrained_dim, in_channels, patch_size, patch_size)
                weight = torch.cat([weight, padding], dim=0)
                print(f"   ✓ Padded embed dim: {pretrained_dim}→{embed_dim}")

        print(f"✅ Final weight shape: {list(weight.shape)}")
        return weight

    except ImportError:
        warnings.warn("timm not installed, cannot load MAE weights. Install with: pip install timm")
        return None
    except Exception as e:
        warnings.warn(f"Failed to load MAE weights: {e}")
        return None


def _load_from_clip(patch_size: int, embed_dim: int) -> Optional[torch.Tensor]:
    """
    从CLIP模型加载权重

    CLIP默认配置：
    - ViT-B/16: patch=16, dim=768
    - ViT-B/32: patch=32, dim=768
    """
    try:
        import clip

        # 根据patch_size选择模型
        if patch_size == 16:
            model_type = "ViT-B/16"
        elif patch_size == 32:
            model_type = "ViT-B/32"
        else:
            warnings.warn(f"CLIP doesn't have pretrained weights for patch_size={patch_size}, using ViT-B/16")
            model_type = "ViT-B/16"

        model, _ = clip.load(model_type, device="cpu")

        # CLIP的patch embedding在visual.conv1
        pretrained_weight = model.visual.conv1.weight.data.clone()
        pretrained_dim, in_channels, pretrained_patch_h, pretrained_patch_w = pretrained_weight.shape

        print(f"📦 Loaded CLIP {model_type} weights: [{pretrained_dim}, {in_channels}, {pretrained_patch_h}, {pretrained_patch_w}]")

        # 调整尺寸
        if patch_size != pretrained_patch_h:
            weight = F.interpolate(
                pretrained_weight,
                size=(patch_size, patch_size),
                mode='bilinear',
                align_corners=False
            )
            print(f"   ✓ Interpolated patch size: {pretrained_patch_h}→{patch_size}")
        else:
            weight = pretrained_weight

        if embed_dim != pretrained_dim:
            if embed_dim < pretrained_dim:
                weight = weight[:embed_dim]
                print(f"   ✓ Truncated embed dim: {pretrained_dim}→{embed_dim}")
            else:
                padding = torch.zeros(embed_dim - pretrained_dim, in_channels, patch_size, patch_size)
                weight = torch.cat([weight, padding], dim=0)
                print(f"   ✓ Padded embed dim: {pretrained_dim}→{embed_dim}")

        print(f"✅ Final weight shape: {list(weight.shape)}")
        return weight

    except ImportError:
        warnings.warn("clip not installed, cannot load CLIP weights. Install with: pip install git+https://github.com/openai/CLIP.git")
        return None
    except Exception as e:
        warnings.warn(f"Failed to load CLIP weights: {e}")
        return None


# 测试函数
if __name__ == "__main__":
    print("Testing pretrained patch embedding loader...")

    # 测试不同配置
    configs = [
        (32, 256, 'dino'),  # 我们的配置
        (16, 384, 'dino'),  # DINO原生
        (32, 768, 'clip'),  # CLIP ViT-B/32
    ]

    for patch_size, embed_dim, model_name in configs:
        print(f"\n{'='*60}")
        print(f"Testing: patch_size={patch_size}, embed_dim={embed_dim}, model={model_name}")
        print(f"{'='*60}")

        weight = load_pretrained_patch_embedding(patch_size, embed_dim, model_name)

        if weight is not None:
            print(f"✅ Success! Weight shape: {weight.shape}")
        else:
            print(f"❌ Failed to load weights")
