"""
StateTokenizer: 将机器人状态转换为tokens
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any


class StateTokenizer(nn.Module):
    """
    将机器人关节状态转换为tokens

    设计思路：
    - 每个关节独立token化
    - 同类型关节共享type embedding
    - 通过side和id embedding区分不同关节
    - 支持16维/36维等任意维度配置
    """

    def __init__(self, embed_dim: int = 512, max_joints: int = 50):
        """
        Args:
            embed_dim: Token embedding维度
            max_joints: 支持的最大关节数量
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.max_joints = max_joints

        # 关节类型embeddings（同类型关节共享权重）
        # 每种类型都是一个小型MLP，将1维关节值映射到embed_dim
        self.joint_type_embeddings = nn.ModuleDict({
            'shoulder': nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            ),
            'elbow': nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            ),
            'wrist': nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            ),
            'gripper': nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            ),
            'hip': nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            ),
            'knee': nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            ),
            'ankle': nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            ),
        })

        # 侧边embedding: 0=left, 1=right, 2=center
        self.side_embedding = nn.Embedding(3, embed_dim)

        # 关节ID embedding: 每个关节有唯一ID
        self.joint_id_embedding = nn.Embedding(max_joints, embed_dim)

        # 最终归一化层
        self.norm = nn.LayerNorm(embed_dim)

        print(
            f"✅ StateTokenizer initialized: {embed_dim}D tokens, max {max_joints} joints")

    def forward(self, state: torch.Tensor, joint_configs: List[Dict[str, Any]]) -> torch.Tensor:
        """
        将状态转换为tokens

        Args:
            state: [B, state_dim] 机器人关节状态
            joint_configs: 关节配置列表，每个元素是一个字典:
                {
                    'idx': int,        # 在state中的索引
                    'type': str,       # 关节类型 (shoulder, elbow等)
                    'side': int,       # 侧边 (0=left, 1=right, 2=center)
                    'id': int,         # 关节唯一ID
                    'name': str        # 关节名称（可选，用于调试）
                }

        Returns:
            tokens: [B, num_joints, embed_dim]
        """
        batch_size = state.shape[0]
        device = state.device

        tokens = []

        for joint_info in joint_configs:
            idx = joint_info['idx']
            joint_type = joint_info['type']
            side = joint_info['side']
            joint_id = joint_info['id']

            # 提取关节值 [B, 1]
            joint_value = state[:, idx:idx+1]

            # 基础embedding（根据关节类型）
            if joint_type not in self.joint_type_embeddings:
                raise ValueError(
                    f"Unknown joint type: {joint_type}. Available types: {list(self.joint_type_embeddings.keys())}")

            token = self.joint_type_embeddings[joint_type](
                joint_value)  # [B, embed_dim]

            # 添加侧边embedding
            side_tensor = torch.tensor([side], dtype=torch.long, device=device)
            token = token + self.side_embedding(side_tensor)  # [B, embed_dim]

            # 添加关节ID embedding
            id_tensor = torch.tensor(
                [joint_id], dtype=torch.long, device=device)
            # [B, embed_dim]
            token = token + self.joint_id_embedding(id_tensor)

            tokens.append(token)

        # 堆叠所有关节tokens [B, num_joints, embed_dim]
        tokens = torch.stack(tokens, dim=1)

        # 归一化
        tokens = self.norm(tokens)

        return tokens

    def get_num_tokens(self, joint_configs: List[Dict[str, Any]]) -> int:
        """计算状态token数量"""
        return len(joint_configs)
