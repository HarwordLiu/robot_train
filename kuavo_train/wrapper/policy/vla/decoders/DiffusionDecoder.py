"""
DiffusionDecoder: 在token空间进行diffusion处理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from ..tokenizers.ActionTokenizer import ActionTokenizer


class DiffusionDecoder(nn.Module):
    """
    在token空间进行diffusion的解码器

    核心思路：
    - 在512维token空间做diffusion，而非16维动作空间
    - 使用Transformer Decoder作为去噪网络
    - 训练：真实动作tokenize后加噪声，学习去噪
    - 推理：从随机噪声tokens迭代去噪，最后detokenize成动作
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        context_dim: int = 512,
        num_train_timesteps: int = 100,
        num_denoiser_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        noise_scheduler_type: str = "DDPM",
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0
    ):
        """
        Args:
            action_dim: 动作维度（从配置读取）
            horizon: 动作序列长度
            context_dim: Token维度（应与vision和state tokens一致）
            num_train_timesteps: 训练时的diffusion步数
            num_denoiser_layers: Transformer decoder层数
            num_heads: 注意力头数
            dim_feedforward: FFN维度
            noise_scheduler_type: 噪声调度器类型
            beta_schedule: Beta调度策略
            beta_start: Beta起始值
            beta_end: Beta结束值
            prediction_type: 预测类型（epsilon或sample）
            clip_sample: 是否裁剪采样
            clip_sample_range: 裁剪范围
        """
        super().__init__()

        self.action_dim = action_dim
        self.horizon = horizon
        self.context_dim = context_dim
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        # ActionTokenizer（核心组件：双向转换）
        self.action_tokenizer = ActionTokenizer(
            action_dim=action_dim,
            horizon=horizon,
            embed_dim=context_dim
        )

        # Transformer Decoder（去噪网络）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=context_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN架构，更稳定
        )
        self.denoiser = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_denoiser_layers
        )

        # 时间步条件编码（用于告诉网络当前噪声水平）
        self.time_mlp = nn.Sequential(
            nn.Linear(context_dim, context_dim * 4),
            nn.SiLU(),
            nn.Linear(context_dim * 4, context_dim)
        )

        # 噪声调度器
        from lerobot.policies.diffusion.modeling_diffusion import _make_noise_scheduler
        self.noise_scheduler = _make_noise_scheduler(
            noise_scheduler_type=noise_scheduler_type,
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range
        )

        print(
            f"✅ DiffusionDecoder initialized: {action_dim}D actions, {horizon} horizon, {num_train_timesteps} train steps")

    def _get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """
        创建sinusoidal时间步embedding

        Args:
            timesteps: [B] 时间步
            embedding_dim: embedding维度

        Returns:
            embeddings: [B, embedding_dim]
        """
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(
            half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if embedding_dim % 2 == 1:  # 奇数维度，补零
            emb = F.pad(emb, (0, 1))

        return emb

    def compute_loss(
        self,
        target_actions: torch.Tensor,
        context_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        训练forward：计算diffusion loss

        Args:
            target_actions: [B, horizon, action_dim] 目标动作序列
            context_tokens: [B, num_context_tokens, context_dim] 上下文tokens（vision+state）
            mask: [B, horizon] 可选的mask（用于变长序列）

        Returns:
            loss: 标量tensor
        """
        batch_size = target_actions.shape[0]
        device = target_actions.device

        # 1. 动作 → action tokens
        action_tokens = self.action_tokenizer.tokenize(target_actions)
        # [B, horizon, context_dim]

        # 2. 采样随机时间步
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long
        )

        # 3. 采样噪声
        noise = torch.randn_like(action_tokens)

        # 4. 添加噪声到action tokens
        noisy_action_tokens = self.noise_scheduler.add_noise(
            action_tokens, noise, timesteps
        )

        # 5. 时间步embedding
        time_emb = self._get_timestep_embedding(timesteps, self.context_dim)
        time_emb = self.time_mlp(time_emb)  # [B, context_dim]

        # 将时间embedding添加到context中
        time_emb_expanded = time_emb.unsqueeze(1)  # [B, 1, context_dim]
        enhanced_context = torch.cat(
            [time_emb_expanded, context_tokens], dim=1)
        # [B, 1+num_context_tokens, context_dim]

        # 6. Transformer去噪预测
        predicted_tokens = self.denoiser(
            tgt=noisy_action_tokens,      # query: noisy action tokens
            memory=enhanced_context        # key & value: context + time
        )
        # [B, horizon, context_dim]

        # 7. 计算loss
        if self.prediction_type == "epsilon":
            # 预测噪声
            target = noise
        elif self.prediction_type == "sample":
            # 预测原始样本
            target = action_tokens
        else:
            raise ValueError(
                f"Unknown prediction_type: {self.prediction_type}")

        # MSE loss
        if mask is not None:
            # 应用mask
            mask_expanded = mask.unsqueeze(-1)  # [B, horizon, 1]
            loss = F.mse_loss(predicted_tokens * mask_expanded,
                              target * mask_expanded, reduction='sum')
            loss = loss / mask_expanded.sum()
        else:
            loss = F.mse_loss(predicted_tokens, target)

        return loss

    @torch.no_grad()
    def sample(
        self,
        context_tokens: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        推理forward：DDPM采样生成动作

        Args:
            context_tokens: [B, num_context_tokens, context_dim] 上下文tokens
            num_inference_steps: 推理步数（None则使用训练步数）
            generator: 随机数生成器

        Returns:
            actions: [B, horizon, action_dim] 预测的动作序列
        """
        batch_size = context_tokens.shape[0]
        device = context_tokens.device

        if num_inference_steps is None:
            num_inference_steps = self.num_train_timesteps

        # 1. 从纯噪声开始
        action_tokens = torch.randn(
            batch_size,
            self.horizon,
            self.context_dim,
            device=device,
            generator=generator
        )

        # 2. 设置推理步数
        self.noise_scheduler.set_timesteps(num_inference_steps)

        # 3. DDPM采样循环
        for t in self.noise_scheduler.timesteps:
            # 时间步embedding
            timesteps_batch = torch.full(
                (batch_size,), t, device=device, dtype=torch.long)
            time_emb = self._get_timestep_embedding(
                timesteps_batch, self.context_dim)
            time_emb = self.time_mlp(time_emb)

            # 增强context
            time_emb_expanded = time_emb.unsqueeze(1)
            enhanced_context = torch.cat(
                [time_emb_expanded, context_tokens], dim=1)

            # Transformer去噪
            denoised_tokens = self.denoiser(
                tgt=action_tokens,
                memory=enhanced_context
            )

            # 去噪一步
            action_tokens = self.noise_scheduler.step(
                denoised_tokens, t, action_tokens
            ).prev_sample

        # 4. tokens → 动作
        actions = self.action_tokenizer.detokenize(action_tokens)

        return actions

    def forward(
        self,
        target_actions: torch.Tensor,
        context_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """训练时调用compute_loss"""
        return self.compute_loss(target_actions, context_tokens, mask)
