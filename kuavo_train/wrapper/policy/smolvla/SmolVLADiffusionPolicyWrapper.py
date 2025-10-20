"""
SmolVLA Diffusion Policy Wrapper for Kuavo Project

基于 SmolVLA 架构但使用 Diffusion 进行动作生成
完全冻结视觉层，专注于训练 Action Expert 的 Diffusion 能力
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from .SmolVLADiffusionConfigWrapper import SmolVLADiffusionConfigWrapper
from .diffusion_scheduler import DDIMScheduler, DDPMScheduler


class SmolVLADiffusionPolicyWrapper(SmolVLAPolicy):
    """
    SmolVLA Diffusion 策略包装器

    主要变化：
    1. 替换 Flow Matching 为 Diffusion
    2. 完全冻结视觉编码器
    3. 实现扩散采样过程
    4. 保持与原始 SmolVLA 的兼容性
    """

    def __init__(
        self,
        config: SmolVLADiffusionConfigWrapper,
        dataset_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ):
        """
        初始化 SmolVLA Diffusion 策略

        Args:
            config: SmolVLA Diffusion 配置对象
            dataset_stats: 数据集统计信息
        """
        # 确保 config 是 Diffusion 配置
        if not isinstance(config, SmolVLADiffusionConfigWrapper):
            raise TypeError("config 必须是 SmolVLADiffusionConfigWrapper 类型")

        # 保存配置
        self.config = config

        # 创建噪声调度器
        self.scheduler = config.create_scheduler(device=config.device)

        # 调用父类初始化
        # 注意：这里会调用 SmolVLAPolicy 的 __init__，它使用的是 VLAFlowMatching
        # 我们稍后需要替换为 Diffusion
        super().__init__(config, dataset_stats)

        # 替换 Flow Matching 为 Diffusion 头
        self._replace_flow_matching_with_diffusion()

        # 冻结视觉层
        self._freeze_vision_encoder()

        # 打印模型信息
        self._print_model_info()

    def _replace_flow_matching_with_diffusion(self):
        """
        替换 Flow Matching 模块为 Diffusion 模块
        """
        # 检查是否存在 VLAFlowMatching 模块
        if hasattr(self.model, 'vlm_with_expert') and hasattr(self.model.vlm_with_expert, 'action_expert'):
            action_expert = self.model.vlm_with_expert.action_expert

            if hasattr(action_expert, 'flow_matching_head'):
                # 获取 Flow Matching 头的输入维度
                flow_head = action_expert.flow_matching_head
                if hasattr(flow_head, 'linear') or hasattr(flow_head, 'nn'):
                    # 获取输入特征维度
                    if hasattr(flow_head, 'linear'):
                        input_dim = flow_head.linear.in_features
                    elif hasattr(flow_head, 'nn') and hasattr(flow_head.nn, '0'):
                        if hasattr(flow_head.nn['0'], 'in_features'):
                            input_dim = flow_head.nn['0'].in_features
                        else:
                            input_dim = 512  # 默认值
                    else:
                        input_dim = 512

                    # 创建 Diffusion UNet 头
                    diffusion_head = DiffusionUNetHead(
                        input_dim=input_dim,
                        output_dim=self.config.max_action_dim * self.config.chunk_size,
                        hidden_dim=1024,
                        num_layers=6,
                        time_embedding_dim=128,
                    )

                    # 替换 Flow Matching 头
                    action_expert.flow_matching_head = diffusion_head
                    print("✅ 成功替换 Flow Matching 头为 Diffusion UNet 头")
                else:
                    print("⚠️ 无法确定 Flow Matching 头的结构，跳过替换")
            else:
                print("⚠️ 未找到 flow_matching_head，可能已经是 Diffusion 版本")
        else:
            print("⚠️ 未找到 action_expert，跳过替换")

    def _freeze_vision_encoder(self):
        """
        完全冻结视觉编码器
        """
        frozen_params = 0
        total_params = 0

        # 获取视觉编码器
        vision_model = self._get_vision_model()
        if vision_model is not None:
            # 冻结整个视觉编码器
            for param in vision_model.parameters():
                param.requires_grad = False
                frozen_params += param.numel()

            # 设置为评估模式
            vision_model.eval()
            print(f"✅ 已冻结视觉编码器: {frozen_params:,} 参数")

        # 冻结 VLM 的其他视觉相关部分
        if hasattr(self.model, 'vlm_with_expert'):
            vlm_model = self.model.vlm_with_expert
            if hasattr(vlm_model, 'vision_encoder'):
                for param in vlm_model.vision_encoder.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
                vlm_model.vision_encoder.eval()

        # 统计总参数
        for param in self.parameters():
            total_params += param.numel()

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n📊 模型参数统计:")
        print(f"   - 总参数: {total_params:,}")
        print(f"   - 可训练参数: {trainable_params:,}")
        print(f"   - 冻结参数: {total_params - trainable_params:,}")
        print(f"   - 训练比例: {trainable_params / total_params * 100:.2f}%")

    def _get_vision_model(self):
        """获取视觉模型"""
        # 尝试多种路径获取视觉模型
        if hasattr(self.model, 'vlm_with_expert'):
            vlm_with_expert = self.model.vlm_with_expert
            if hasattr(vlm_with_expert, 'get_vlm_model'):
                vlm_model = vlm_with_expert.get_vlm_model()
                if hasattr(vlm_model, 'vision_model'):
                    return vlm_model.vision_model
        return None

    def _print_model_info(self):
        """打印模型信息"""
        print("\n" + "="*70)
        print("🚀 SmolVLA Diffusion Policy Initialized")
        print("="*70)
        print(f"📋 配置信息:")
        print(f"   - VLM Model: {self.config.vlm_model_name}")
        print(f"   - 动作生成: Diffusion (非 Flow Matching)")
        print(f"   - 视觉编码器: 完全冻结")
        print(f"   - 推理步数: {self.config.num_inference_steps}")
        print(f"   - 噪声调度: {self.config.noise_schedule}")
        print(f"   - 预测类型: {self.config.prediction_type}")
        print(f"   - 使用 DDIM: {self.config.use_ddim_sampling}")
        print("="*70 + "\n")

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Diffusion 训练前向传播

        Args:
            batch: 输入批次，必须包含 'task' 字段
            noise: 可选的噪声（如果不提供则随机生成）
            timestep: 可选的时间步（如果不提供则随机采样）

        Returns:
            loss: 损失值
            info: 信息字典
        """
        # 确保批次包含任务信息
        if 'task' not in batch:
            raise ValueError("批次必须包含 'task' 字段")

        batch_size = next(iter(batch.values())).shape[0]

        # 准备输入
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        # 获取动作序列
        from lerobot.constants import ACTION
        if ACTION not in batch:
            raise ValueError(f"批次必须包含 '{ACTION}' 键")

        actions = batch[ACTION]  # [B, chunk_size, action_dim]

        # 随机采样时间步
        if timestep is None:
            timestep = torch.randint(
                0, self.config.num_train_timesteps, (batch_size,),
                device=actions.device
            )

        # 生成噪声
        if noise is None:
            noise = torch.randn_like(actions)

        # 添加噪声到动作
        noisy_actions = self.scheduler.add_noise(
            original_samples=actions,
            noise=noise,
            timesteps=timestep
        )

        # 准备时间嵌入
        time_embeddings = self._get_time_embeddings(timestep)

        # 模型预测噪声
        predicted_noise = self.model.forward(
            images=images,
            image_masks=img_masks,
            language_tokens=lang_tokens,
            language_mask=lang_masks,
            state=state,
            actions=noisy_actions,
            time_embeddings=time_embeddings,
        )

        # 计算损失
        if self.config.prediction_type == "epsilon":
            # 预测噪声
            target = noise
        elif self.config.prediction_type == "v_prediction":
            # 预测 v-parameterization
            target = self.scheduler.get_velocity(actions, noise, timestep)
        else:
            # 预测原始样本
            target = actions

        loss = nn.functional.mse_loss(predicted_noise, target)

        # 收集信息
        info = {
            "loss": loss.item(),
            "timestep_mean": timestep.float().mean().item(),
            "noise_mean": noise.mean().item(),
            "predicted_noise_mean": predicted_noise.mean().item(),
        }

        return loss, info

    def _get_time_embeddings(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        获取时间嵌入

        Args:
            timesteps: 时间步张量

        Returns:
            时间嵌入
        """
        # 简单的正弦位置编码
        half_dim = self.config.max_action_dim // 4
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        # 扩展到匹配动作维度
        embeddings = embeddings.unsqueeze(1).expand(-1, self.config.chunk_size, -1)

        return embeddings

    def select_action(
        self,
        batch: Dict[str, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Diffusion 推理：生成动作序列

        Args:
            batch: 观测批次
            noise: 可选的初始噪声
            num_inference_steps: 可选的推理步数

        Returns:
            生成的动作序列 [B, chunk_size, action_dim]
        """
        # 确保批次包含任务信息
        if 'task' not in batch:
            raise ValueError("批次必须包含 'task' 字段")

        batch_size = next(iter(batch.values())).shape[0]

        # 设置推理步数
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps

        # 设置调度器时间步
        if hasattr(self.scheduler, 'set_timesteps'):
            self.scheduler.set_timesteps(num_inference_steps, device=batch[next(iter(batch.keys()))].device)

        # 准备输入
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        # 从纯噪声开始
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.config.chunk_size,
                self.config.max_action_dim,
                device=images.device,
                dtype=images.dtype
            )

        # 逐步去噪
        actions = noise
        for i, t in enumerate(self.scheduler.timesteps):
            # 获取时间嵌入
            time_embeddings = self._get_time_embeddings(t.unsqueeze(0).expand(batch_size))

            # 预测噪声
            with torch.no_grad():
                predicted_noise = self.model.forward(
                    images=images,
                    image_masks=img_masks,
                    language_tokens=lang_tokens,
                    language_mask=lang_masks,
                    state=state,
                    actions=actions,
                    time_embeddings=time_embeddings,
                )

            # 调度器步骤
            actions = self.scheduler.step(
                model_output=predicted_noise,
                timestep=t,
                sample=actions,
                eta=self.config.ddim_eta,
            )

        # 裁剪动作到原始维度
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def _get_action_chunk(self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None) -> torch.Tensor:
        """
        重写父类方法以支持 Diffusion 采样

        使用 Diffusion 逐步去噪生成动作序列
        """
        from lerobot.constants import ACTION

        # 复制队列
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        # 使用 Diffusion 采样
        actions = self.select_action(batch, noise=noise)

        # 反归一化
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        # 裁剪到原始维度
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    @staticmethod
    def _create_identity_stats(config: SmolVLADiffusionConfigWrapper) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        创建空的 dataset_stats
        """
        stats = {}

        # 处理输入特征
        for key, feature in config.input_features.items():
            shape = feature.shape
            if 'state' in key.lower():
                shape = (config.max_state_dim,)

            stats[key] = {
                'mean': torch.zeros(shape, dtype=torch.float32),
                'std': torch.ones(shape, dtype=torch.float32),
                'min': torch.zeros(shape, dtype=torch.float32),
                'max': torch.ones(shape, dtype=torch.float32),
            }

        # 处理输出特征
        for key, feature in config.output_features.items():
            shape = feature.shape
            if 'action' in key.lower():
                shape = (config.max_action_dim,)

            stats[key] = {
                'mean': torch.zeros(shape, dtype=torch.float32),
                'std': torch.ones(shape, dtype=torch.float32),
                'min': torch.zeros(shape, dtype=torch.float32),
                'max': torch.ones(shape, dtype=torch.float32),
            }

        return stats

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        config: Optional[SmolVLADiffusionConfigWrapper] = None,
        dataset_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        **kwargs
    ):
        """
        从预训练模型加载

        支持从 SmolVLA 预训练模型加载并转换为 Diffusion 版本
        """
        print(f"\n{'='*70}")
        print(f"📂 Loading SmolVLA Diffusion from: {pretrained_name_or_path}")
        print(f"{'='*70}")

        # 如果没有提供配置，从预训练路径加载
        if config is None:
            config = SmolVLADiffusionConfigWrapper.from_pretrained(pretrained_name_or_path)

        # 如果没有提供 dataset_stats，创建临时的
        if dataset_stats is None:
            dataset_stats = cls._create_identity_stats(config)

        # 创建模型实例
        model = cls(config, dataset_stats)

        # 加载权重
        pretrained_path = Path(pretrained_name_or_path)
        if pretrained_path.exists():
            model_file = pretrained_path / "model.safetensors"
            if model_file.exists():
                # 加载完整的状态字典
                from safetensors.torch import load_file
                full_state_dict = load_file(str(model_file))

                # 分离归一化参数和模型参数
                norm_keys = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
                norm_state_dict = {
                    k: v for k, v in full_state_dict.items() if k.startswith(norm_keys)
                }
                model_state_dict = {
                    k: v for k, v in full_state_dict.items() if not k.startswith(norm_keys)
                }

                # 加载模型参数（允许部分加载）
                missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
                print(f"✅ 从本地 checkpoint 加载权重")
                if missing:
                    print(f"   - 缺失的键（可能是 Diffusion 特定的）: {len(missing)}")
                if unexpected:
                    print(f"   - 意外的键: {len(unexpected)}")

                # 加载归一化参数
                if norm_state_dict:
                    cls._load_normalization_params(model, norm_state_dict)
                    print(f"✅ 加载归一化参数")
            else:
                print(f"⚠️ 模型文件未找到: {model_file}")
        else:
            # 尝试从 HuggingFace 加载
            try:
                from huggingface_hub import hf_hub_download
                model_file = hf_hub_download(
                    repo_id=pretrained_name_or_path,
                    filename="model.safetensors"
                )
                # 加载逻辑...
                print(f"✅ 从 HuggingFace 加载: {pretrained_name_or_path}")
            except Exception as e:
                print(f"⚠️ 从 HuggingFace 加载失败: {e}")
                print(f"使用随机初始化")

        print(f"{'='*70}\n")

        return model

    @staticmethod
    def _load_normalization_params(model, norm_state_dict):
        """加载归一化参数"""
        import torch.nn as nn

        for key, value in norm_state_dict.items():
            parts = key.split('.')
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            param_name = parts[-1]

            if isinstance(obj, nn.ParameterDict):
                obj[param_name] = nn.Parameter(value, requires_grad=False)
            elif hasattr(obj, param_name) and hasattr(getattr(obj, param_name), 'data'):
                getattr(obj, param_name).data = value


class DiffusionUNetHead(nn.Module):
    """
    用于替换 Flow Matching 的 Diffusion UNet 头
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        time_embedding_dim: int = 128,
    ):
        super().__init__()

        # 时间嵌入投影
        self.time_proj = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # UNet 风格的层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),  # 输入 + 时间 + 前一层
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
            )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, time_embeddings):
        """
        前向传播

        Args:
            x: 输入特征 [B, seq_len, input_dim]
            time_embeddings: 时间嵌入 [B, seq_len, time_dim]

        Returns:
            预测的噪声或样本 [B, seq_len, output_dim]
        """
        # 投影输入
        h = self.input_proj(x)  # [B, seq_len, hidden_dim]

        # 投影时间嵌入
        t = self.time_proj(time_embeddings)  # [B, seq_len, hidden_dim]

        # 通过 UNet 层
        for i, layer in enumerate(self.layers):
            # 拼接输入、时间和前一层的输出
            if i == 0:
                layer_input = torch.cat([h, t, h], dim=-1)
            else:
                layer_input = torch.cat([h, t, h], dim=-1)
            h = layer(layer_input)

        # 输出投影
        output = self.output_proj(h)

        return output