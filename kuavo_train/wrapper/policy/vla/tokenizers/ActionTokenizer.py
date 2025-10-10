"""
ActionTokenizer: 实现动作空间和token空间的双向转换
"""
import torch
import torch.nn as nn
from typing import Optional


class ActionTokenizer(nn.Module):
    """
    动作Token化器：实现动作↔tokens的双向转换

    核心作用：
    1. tokenize(): 训练时将真实动作转为tokens（用于加噪声学习）
    2. detokenize(): 推理时将去噪后的tokens转回动作

    设计思路：
    - 每个时间步独立编码（因为diffusion在时间维度处理）
    - 使用时间步embedding区分不同时刻
    - 编码器和解码器分离，支持在token空间做diffusion
    """

    def __init__(self, action_dim: int, horizon: int, embed_dim: int = 512):
        """
        Args:
            action_dim: 动作维度（从配置读取，支持16/36等）
            horizon: 动作序列长度
            embed_dim: Token embedding维度
        """
        super().__init__()

        self.action_dim = action_dim
        self.horizon = horizon
        self.embed_dim = embed_dim

        # 编码器：动作 → tokens（训练时使用）
        # 所有时间步共享同一个编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # 解码器：tokens → 动作（推理时使用）
        # 所有时间步共享解码器
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, action_dim)
        )

        # 时间步embedding
        self.time_embedding = nn.Embedding(horizon, embed_dim)

        print(
            f"✅ ActionTokenizer initialized: {action_dim}D actions, {horizon} timesteps, {embed_dim}D tokens")

    def tokenize(self, actions: torch.Tensor) -> torch.Tensor:
        """
        训练时：动作序列 → token序列

        Args:
            actions: [B, horizon, action_dim] 目标动作序列

        Returns:
            tokens: [B, horizon, embed_dim] action tokens
        """
        batch_size, horizon, action_dim = actions.shape
        device = actions.device

        assert horizon == self.horizon, f"Expected horizon {self.horizon}, got {horizon}"
        assert action_dim == self.action_dim, f"Expected action_dim {self.action_dim}, got {action_dim}"

        # 使用共享编码器编码所有时间步
        tokens = self.action_encoder(actions)  # [B, horizon, embed_dim]

        # 添加时间步embedding（广播方式）
        time_ids = torch.arange(horizon, device=device)  # [horizon]
        time_embeds = self.time_embedding(time_ids)  # [horizon, embed_dim]
        tokens = tokens + time_embeds.unsqueeze(0)  # [B, horizon, embed_dim] + [1, horizon, embed_dim]

        return tokens

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        推理时：token序列 → 动作序列

        Args:
            tokens: [B, horizon, embed_dim] 去噪后的action tokens

        Returns:
            actions: [B, horizon, action_dim] 预测的动作序列
        """
        batch_size, horizon, embed_dim = tokens.shape

        assert horizon == self.horizon, f"Expected horizon {self.horizon}, got {horizon}"
        assert embed_dim == self.embed_dim, f"Expected embed_dim {self.embed_dim}, got {embed_dim}"

        # 解码所有tokens（共享解码器）
        actions = self.action_decoder(tokens)  # [B, horizon, action_dim]

        return actions

    def expand_action_dim(self, new_action_dim: int, freeze_old_weights: bool = True):
        """
        扩展动作维度（例如从16维到36维）

        策略：
        - 编码器：重新初始化（因为输入维度变了）
        - 解码器：前N维复用权重，新维度随机初始化

        Args:
            new_action_dim: 新的动作维度
            freeze_old_weights: 是否冻结旧权重
        """
        if new_action_dim == self.action_dim:
            print(
                f"⚠️  Action dimension already {new_action_dim}, no expansion needed")
            return

        print(
            f"🔧 Expanding ActionTokenizer: {self.action_dim}D → {new_action_dim}D")

        old_action_dim = self.action_dim

        # 1. 扩展解码器
        # Linear(embed_dim, old_action_dim)
        old_decoder_final_layer = self.action_decoder[-1]
        # [old_action_dim, embed_dim]
        old_weight = old_decoder_final_layer.weight.data
        old_bias = old_decoder_final_layer.bias.data      # [old_action_dim]

        # 创建新的解码器最后一层
        new_final_layer = nn.Linear(self.embed_dim, new_action_dim)

        # 复用前old_action_dim维的权重
        new_final_layer.weight.data[:old_action_dim] = old_weight
        new_final_layer.bias.data[:old_action_dim] = old_bias

        # 新维度随机初始化（已经由PyTorch默认完成）

        # 替换解码器最后一层
        self.action_decoder[-1] = new_final_layer

        # 2. 重新初始化编码器（因为输入维度变了）
        self.action_encoder = nn.Sequential(
            nn.Linear(new_action_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )

        # 3. 可选：冻结解码器的旧权重
        if freeze_old_weights:
            # 部分冻结解码器（只冻结前old_action_dim维的输出权重）
            # 注意：PyTorch不支持部分权重冻结，这里只是标记
            print(f"💡 Consider fine-tuning with small learning rate for old dimensions")

        self.action_dim = new_action_dim

        print(f"✅ ActionTokenizer expanded to {new_action_dim}D")
