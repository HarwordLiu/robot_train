"""
SmolVLA Policy Wrapper for Kuavo Project

SmolVLA的Kuavo项目包装器，继承lerobot的SmolVLAPolicy，
添加Kuavo特定的功能和兼容性处理。
"""

import torch
from typing import Dict, Any, Optional
from pathlib import Path

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


class SmolVLAPolicyWrapper(SmolVLAPolicy):
    """
    Kuavo项目的SmolVLA策略包装器

    直接继承lerobot的SmolVLAPolicy，添加：
    1. Kuavo项目的初始化日志
    2. 兼容Kuavo数据格式
    3. 支持多任务顺序训练

    Usage:
        # 训练模式
        policy = SmolVLAPolicyWrapper(config, dataset_stats)
        loss, info = policy.forward(batch)

        # 推理模式
        action = policy.select_action(batch)
    """

    def __init__(
        self,
        config: SmolVLAConfig,
        dataset_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ):
        """
        初始化SmolVLA策略

        Args:
            config: SmolVLAConfig配置对象
            dataset_stats: 数据集统计信息（用于归一化）
        """
        # 调用父类SmolVLAPolicy的初始化
        super().__init__(config, dataset_stats)

        # Kuavo项目特定日志
        print("\n" + "="*70)
        print("🤖 SmolVLA Policy Initialized for Kuavo Project")
        print("="*70)
        print(f"VLM Backbone: {config.vlm_model_name}")
        print(f"Action Dimension: {config.max_action_dim} (Kuavo Dual-Arm)")
        print(f"Chunk Size: {config.chunk_size}")
        print(f"Action Steps per Inference: {config.n_action_steps}")
        print(f"Freeze Vision Encoder: {config.freeze_vision_encoder}")
        print(f"Train Expert Only: {config.train_expert_only}")

        # 打印模型参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)
        print(f"\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Frozen: {total_params - trainable_params:,}")
        print("="*70 + "\n")

    def prepare_batch_with_language(
        self,
        batch: Dict[str, torch.Tensor],
        language_instruction: str
    ) -> Dict[str, torch.Tensor]:
        """
        为batch添加language instruction

        SmolVLA需要language instruction作为任务条件，
        这个方法确保每个batch都包含正确的language field

        Args:
            batch: 输入batch
            language_instruction: 任务的language instruction

        Returns:
            包含language字段的batch
        """
        if 'task' not in batch:
            # 为batch中的每个样本添加相同的language instruction
            batch_size = next(iter(batch.values())).shape[0]
            batch['task'] = [language_instruction] * batch_size

        return batch

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        训练forward

        Args:
            batch: 输入batch，必须包含'task'字段
            noise: 可选的噪声（Flow Matching使用）
            time: 可选的时间步（Flow Matching使用）

        Returns:
            loss: 标量tensor
            info: 信息字典
        """
        # 确保batch包含task字段
        if 'task' not in batch:
            raise ValueError(
                "Batch must contain 'task' field for SmolVLA. "
                "Use prepare_batch_with_language() to add language instruction."
            )

        # 调用父类forward
        return super().forward(batch, noise, time)

    def select_action(
        self,
        batch: Dict[str, torch.Tensor],
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        推理forward：生成动作

        Args:
            batch: 观测batch，必须包含'task'字段
            noise: 可选的噪声（用于测试）

        Returns:
            action: [B, action_dim] 单步动作
        """
        # 确保batch包含task字段
        if 'task' not in batch:
            raise ValueError(
                "Batch must contain 'task' field for SmolVLA inference. "
                "Provide language instruction to specify which task to execute."
            )

        # 调用父类select_action
        return super().select_action(batch, noise)

    @staticmethod
    def _create_identity_stats(config: SmolVLAConfig) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        创建"空"的dataset_stats，使归一化成为恒等变换
        
        对于每个feature：
        - mean = 0（减去0不改变数据）
        - std = 1（除以1不改变数据）
        
        注意：对于 state 和 action，使用 max_state_dim 和 max_action_dim（32维）
        而不是实际的维度（16维），以匹配训练时的填充维度。
        
        Args:
            config: SmolVLA配置对象
            
        Returns:
            包含所有features的identity stats字典
        """
        stats = {}
        
        # 处理input features（observations）
        for key, feature in config.input_features.items():
            shape = feature.shape
            
            # 对于state，使用max_state_dim而不是实际维度
            if 'state' in key.lower():
                shape = (config.max_state_dim,)
            
            stats[key] = {
                'mean': torch.zeros(shape, dtype=torch.float32),
                'std': torch.ones(shape, dtype=torch.float32),
                'min': torch.zeros(shape, dtype=torch.float32),
                'max': torch.ones(shape, dtype=torch.float32),
            }
        
        # 处理output features（actions）
        for key, feature in config.output_features.items():
            shape = feature.shape
            
            # 对于action，使用max_action_dim而不是实际维度
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
        config: Optional[SmolVLAConfig] = None,
        dataset_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        **kwargs
    ):
        """
        从预训练模型加载

        Args:
            pretrained_name_or_path:
                - HuggingFace模型ID（如'lerobot/smolvla_base'）
                - 本地路径（如'outputs/train/.../best'）
            config: 可选的配置对象（如果提供，会override预训练配置）
            dataset_stats: 数据集统计信息

        Returns:
            加载的SmolVLAPolicyWrapper实例
        """
        print(f"\n{'='*70}")
        print(f"📂 Loading SmolVLA from: {pretrained_name_or_path}")
        print(f"{'='*70}")

        # 如果没有提供config，从预训练路径加载
        if config is None:
            from .SmolVLAConfigWrapper import SmolVLAConfigWrapper
            config = SmolVLAConfigWrapper.from_pretrained(
                pretrained_name_or_path)

        # 如果没有提供dataset_stats，创建临时的identity stats用于初始化
        # 真实的归一化参数会从checkpoint中加载
        if dataset_stats is None:
            print("⚠️  No dataset_stats provided. Will load normalization params from checkpoint.")
            dataset_stats = cls._create_identity_stats(config)

        # 创建模型实例
        model = cls(config, dataset_stats)

        # 加载权重
        pretrained_path = Path(pretrained_name_or_path)
        if pretrained_path.exists():
            # 本地checkpoint
            model_file = pretrained_path / "model.safetensors"
            if model_file.exists():
                # 加载完整的 state_dict（包括归一化参数）
                from safetensors.torch import load_file
                full_state_dict = load_file(str(model_file))
                
                # 分离归一化参数和模型参数
                norm_keys = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
                norm_state_dict = {k: v for k, v in full_state_dict.items() if k.startswith(norm_keys)}
                model_state_dict = {k: v for k, v in full_state_dict.items() if not k.startswith(norm_keys)}
                
                # 先加载模型参数（不包括归一化）
                missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
                print(f"✅ Loaded model weights from local checkpoint")
                
                # 再加载归一化参数（如果存在）
                if norm_state_dict:
                    model.load_state_dict(norm_state_dict, strict=False)
                    print(f"✅ Loaded normalization parameters from checkpoint")
                    print(f"   - {len([k for k in norm_state_dict.keys() if 'normalize_inputs' in k])} input norm params")
                    print(f"   - {len([k for k in norm_state_dict.keys() if 'normalize_targets' in k])} target norm params")
                    print(f"   - {len([k for k in norm_state_dict.keys() if 'unnormalize_outputs' in k])} unnorm params")
                else:
                    print(f"⚠️  No normalization parameters found in checkpoint. Using identity normalization.")
            else:
                print(f"⚠️  Model file not found: {model_file}")
        else:
            # HuggingFace模型
            try:
                from huggingface_hub import hf_hub_download
                model_file = hf_hub_download(
                    repo_id=pretrained_name_or_path,
                    filename="model.safetensors"
                )
                from lerobot.policies.smolvla.modeling_smolvla import load_smolvla
                model = load_smolvla(
                    model,
                    model_file,
                    device='cpu',
                    checkpoint_keys_mapping="model._orig_mod.//model."
                )
                print(
                    f"✅ Loaded weights from HuggingFace: {pretrained_name_or_path}")
            except Exception as e:
                print(f"⚠️  Failed to load from HuggingFace: {e}")
                print(f"Using random initialization")

        print(f"{'='*70}\n")
        return model

    def save_pretrained(self, save_directory: Path) -> None:
        """
        保存模型

        Args:
            save_directory: 保存目录路径

        注意：依赖 SmolVLAConfigWrapper 已将所有 OmegaConf 对象转换为原生 Python 对象，
        因此可以直接使用 lerobot 的标准保存方式。
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving SmolVLA model to {save_directory}")

        # 保存配置（使用 lerobot 标准方式）
        self.config._save_pretrained(save_directory)

        # 保存模型权重
        from safetensors.torch import save_file
        model_file = save_directory / "model.safetensors"
        save_file(self.state_dict(), str(model_file))

        print(f"✅ Model saved successfully")
        print(f"   Config: {save_directory / 'config.json'}")
        print(f"   Weights: {model_file}")
