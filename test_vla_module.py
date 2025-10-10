#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLA模块测试脚本

验证Token化VLA策略的各个组件是否正常工作
"""

from kuavo_train.wrapper.policy.vla.decoders.DiffusionDecoder import DiffusionDecoder
from kuavo_train.wrapper.policy.vla.tokenizers.ActionTokenizer import ActionTokenizer
from kuavo_train.wrapper.policy.vla.tokenizers.StateTokenizer import StateTokenizer
from kuavo_train.wrapper.policy.vla.tokenizers.VisionTokenizer import VisionTokenizer
import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def test_vision_tokenizer():
    """测试VisionTokenizer"""
    print("\n" + "=" * 70)
    print("测试VisionTokenizer")
    print("=" * 70)

    tokenizer = VisionTokenizer(patch_size=16, embed_dim=512, image_size=224)

    # 模拟输入
    batch_size = 2
    num_cameras = 3
    rgb_images = torch.randn(batch_size, num_cameras, 3, 224, 224)
    depth_images = torch.randn(batch_size, num_cameras, 1, 224, 224)

    # Token化
    tokens = tokenizer(rgb_images, depth_images)

    print(f"✅ 输入: RGB {rgb_images.shape}, Depth {depth_images.shape}")
    print(f"✅ 输出: Tokens {tokens.shape}")
    print(f"✅ 预期: [2, 1176, 512] (3相机RGB + 3相机Depth, 每相机196patches)")

    assert tokens.shape == (
        batch_size, 1176, 512), f"Shape mismatch: {tokens.shape}"
    print("✅ VisionTokenizer测试通过!")

    return True


def test_state_tokenizer():
    """测试StateTokenizer"""
    print("\n" + "=" * 70)
    print("测试StateTokenizer")
    print("=" * 70)

    tokenizer = StateTokenizer(embed_dim=512)

    # 模拟16维状态
    batch_size = 2
    state_dim = 16
    state = torch.randn(batch_size, state_dim)

    # 关节配置（16维手臂配置）
    joint_configs = [
        {'idx': i, 'type': ['shoulder', 'shoulder', 'shoulder', 'elbow', 'wrist', 'wrist', 'wrist', 'gripper',
                            'shoulder', 'shoulder', 'shoulder', 'elbow', 'wrist', 'wrist', 'wrist', 'gripper'][i],
         'side': 0 if i < 8 else 1, 'id': i, 'name': f'joint_{i}'}
        for i in range(state_dim)
    ]

    # Token化
    tokens = tokenizer(state, joint_configs)

    print(f"✅ 输入: State {state.shape}")
    print(f"✅ 输出: Tokens {tokens.shape}")
    print(f"✅ 预期: [2, 16, 512]")

    assert tokens.shape == (batch_size, state_dim,
                            512), f"Shape mismatch: {tokens.shape}"
    print("✅ StateTokenizer测试通过!")

    return True


def test_action_tokenizer():
    """测试ActionTokenizer"""
    print("\n" + "=" * 70)
    print("测试ActionTokenizer")
    print("=" * 70)

    action_dim = 16
    horizon = 8
    tokenizer = ActionTokenizer(
        action_dim=action_dim, horizon=horizon, embed_dim=512)

    # 模拟动作序列
    batch_size = 2
    actions = torch.randn(batch_size, horizon, action_dim)

    # Token化
    action_tokens = tokenizer.tokenize(actions)
    print(f"✅ 输入: Actions {actions.shape}")
    print(f"✅ tokenize输出: Tokens {action_tokens.shape}")
    print(f"✅ 预期: [2, 8, 512]")

    assert action_tokens.shape == (
        batch_size, horizon, 512), f"Shape mismatch: {action_tokens.shape}"

    # 反Token化
    decoded_actions = tokenizer.detokenize(action_tokens)
    print(f"✅ detokenize输出: Actions {decoded_actions.shape}")
    print(f"✅ 预期: [2, 8, 16]")

    assert decoded_actions.shape == (
        batch_size, horizon, action_dim), f"Shape mismatch: {decoded_actions.shape}"
    print("✅ ActionTokenizer测试通过!")

    return True


def test_diffusion_decoder():
    """测试DiffusionDecoder"""
    print("\n" + "=" * 70)
    print("测试DiffusionDecoder")
    print("=" * 70)

    action_dim = 16
    horizon = 8
    decoder = DiffusionDecoder(
        action_dim=action_dim,
        horizon=horizon,
        context_dim=512,
        num_train_timesteps=100
    )

    # 模拟输入
    batch_size = 2
    target_actions = torch.randn(batch_size, horizon, action_dim)
    context_tokens = torch.randn(batch_size, 100, 512)  # 假设100个context tokens

    # 训练模式：计算loss
    loss = decoder.compute_loss(target_actions, context_tokens)
    print(f"✅ 训练loss: {loss.item():.4f}")
    assert loss.ndim == 0, "Loss should be a scalar"

    # 推理模式：采样
    sampled_actions = decoder.sample(context_tokens, num_inference_steps=10)
    print(f"✅ 采样输出: Actions {sampled_actions.shape}")
    print(f"✅ 预期: [2, 8, 16]")

    assert sampled_actions.shape == (
        batch_size, horizon, action_dim), f"Shape mismatch: {sampled_actions.shape}"
    print("✅ DiffusionDecoder测试通过!")

    return True


def test_integration():
    """集成测试：完整的forward流程"""
    print("\n" + "=" * 70)
    print("集成测试：完整Forward流程")
    print("=" * 70)

    # 创建所有组件
    vision_tokenizer = VisionTokenizer(
        patch_size=16, embed_dim=512, image_size=224)
    state_tokenizer = StateTokenizer(embed_dim=512)

    # Transformer Encoder
    import torch.nn as nn
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048, batch_first=True
    )
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    # Diffusion Decoder
    diffusion_decoder = DiffusionDecoder(
        action_dim=16, horizon=8, context_dim=512, num_train_timesteps=100
    )

    # 模拟输入
    batch_size = 2
    rgb_images = torch.randn(batch_size, 1, 3, 224, 224)
    state = torch.randn(batch_size, 16)
    target_actions = torch.randn(batch_size, 8, 16)

    joint_configs = [
        {'idx': i, 'type': ['shoulder', 'elbow', 'wrist', 'gripper'] * 4,
         'side': 0 if i < 8 else 1, 'id': i, 'name': f'joint_{i}'}
        for i in range(16)
    ]

    # Forward流程
    print("1️⃣  Vision tokenization...")
    vision_tokens = vision_tokenizer(rgb_images, None)
    print(f"   Vision tokens: {vision_tokens.shape}")

    print("2️⃣  State tokenization...")
    state_tokens = state_tokenizer(state, joint_configs)
    print(f"   State tokens: {state_tokens.shape}")

    print("3️⃣  Concatenate tokens...")
    all_tokens = torch.cat([vision_tokens, state_tokens], dim=1)
    print(f"   All tokens: {all_tokens.shape}")

    print("4️⃣  Transformer encoding...")
    context_tokens = transformer_encoder(all_tokens)
    print(f"   Context tokens: {context_tokens.shape}")

    print("5️⃣  Diffusion loss computation...")
    loss = diffusion_decoder.compute_loss(target_actions, context_tokens)
    print(f"   Loss: {loss.item():.4f}")

    print("6️⃣  Diffusion sampling...")
    sampled_actions = diffusion_decoder.sample(
        context_tokens, num_inference_steps=5)
    print(f"   Sampled actions: {sampled_actions.shape}")

    print("✅ 集成测试通过!")

    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("🧪 VLA模块测试套件")
    print("=" * 70)

    tests = [
        ("VisionTokenizer", test_vision_tokenizer),
        ("StateTokenizer", test_state_tokenizer),
        ("ActionTokenizer", test_action_tokenizer),
        ("DiffusionDecoder", test_diffusion_decoder),
        ("Integration", test_integration),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name}测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name:20s} {status}")

    all_passed = all(success for _, success in results)

    print("=" * 70)
    if all_passed:
        print("🎉 所有测试通过!")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
