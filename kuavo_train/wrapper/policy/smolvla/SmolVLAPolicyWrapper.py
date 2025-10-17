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

        # 🆕 应用灵活的视觉层冻结策略
        self._apply_flexible_vision_freezing()

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

    def _get_action_chunk(self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None) -> torch.Tensor:
        """
        重写父类方法以修复维度不匹配问题

        正确的顺序：
        1. 模型预测（输出32D归一化的动作）
        2. 用32D参数反归一化
        3. 裁剪到16D（Kuavo实际维度）

        父类的实现顺序错误（先裁剪再反归一化），导致维度不匹配。
        """
        from lerobot.constants import ACTION

        # Copy queues so that we don't modify the originals
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        # 准备输入
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        # 模型采样（输出32D归一化的动作）
        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise)

        # 先在32D空间反归一化（使用32D的mean/std）
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        # 然后裁剪到原始维度（16D）
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    @staticmethod
    def _create_identity_stats(config: SmolVLAConfig) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        创建"空"的dataset_stats，使归一化成为恒等变换

        对于每个feature：
        - mean = 0（减去0不改变数据）
        - std = 1（除以1不改变数据）

        注意：对于 state 和 action，使用 max_state_dim 和 max_action_dim（32维）
        而不是实际的维度（16维），以匹配训练时的填充维度。

        对于深度特征的shape不匹配问题，会在加载checkpoint时通过broadcasting自动解决。

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
        apply_freezing: bool = False,  # 🆕 默认不应用冻结（推理模式）
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
            apply_freezing: 是否应用视觉层冻结策略
                - True: 应用冻结（训练时使用）
                - False: 不应用冻结（推理时使用，默认）

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
            print(
                "⚠️  No dataset_stats provided. Will load normalization params from checkpoint.")
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
                norm_keys = ("normalize_inputs",
                             "normalize_targets", "unnormalize_outputs")
                norm_state_dict = {
                    k: v for k, v in full_state_dict.items() if k.startswith(norm_keys)}
                model_state_dict = {
                    k: v for k, v in full_state_dict.items() if not k.startswith(norm_keys)}

                # 先加载模型参数（不包括归一化）
                missing, unexpected = model.load_state_dict(
                    model_state_dict, strict=False)
                print(f"✅ Loaded model weights from local checkpoint")

                # 再加载归一化参数（如果存在）
                if norm_state_dict:
                    # 修复深度特征归一化参数的shape不匹配问题
                    # checkpoint中深度特征的归一化参数是(1,1,1)，但模型初始化时创建的是(1,480,640)
                    # 我们需要保持(1,1,1)以便在forward时自动broadcast到任意分辨率

                    import torch.nn as nn

                    # 直接访问并替换归一化模块中的参数
                    for key, value in norm_state_dict.items():
                        # 通过名称访问嵌套的参数
                        # 例如: normalize_inputs.buffer_observation_depth_h.mean
                        parts = key.split('.')
                        obj = model

                        # 导航到目标对象（例如ParameterDict）
                        for part in parts[:-1]:
                            obj = getattr(obj, part)

                        # 获取最后一个属性名（例如'mean'）
                        param_name = parts[-1]

                        # 如果是ParameterDict，直接替换其中的Parameter
                        if isinstance(obj, nn.ParameterDict):
                            current_param = obj[param_name]
                            checkpoint_shape = value.shape
                            current_shape = current_param.shape

                            if checkpoint_shape != current_shape:
                                print(
                                    f"🔧 Keeping compact shape for {key}: {checkpoint_shape} (model had {current_shape})")

                            # 创建新的Parameter对象，保持checkpoint的shape
                            obj[param_name] = nn.Parameter(
                                value, requires_grad=False)
                        else:
                            # 其他情况，尝试直接赋值
                            if hasattr(obj, param_name):
                                current_param = getattr(obj, param_name)
                                if hasattr(current_param, 'data'):
                                    current_param.data = value

                    print(f"✅ Loaded normalization parameters from checkpoint")
                    print(
                        f"   - {len([k for k in norm_state_dict.keys() if 'normalize_inputs' in k])} input norm params")
                    print(
                        f"   - {len([k for k in norm_state_dict.keys() if 'normalize_targets' in k])} target norm params")
                    print(
                        f"   - {len([k for k in norm_state_dict.keys() if 'unnormalize_outputs' in k])} unnorm params")
                else:
                    print(
                        f"⚠️  No normalization parameters found in checkpoint. Using identity normalization.")
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

        # 🆕 在加载权重后重新应用灵活冻结策略（仅在训练模式下）
        # 因为有些层可能在权重加载后才完全初始化
        if apply_freezing and (config.unfreeze_vision_layers is not None or
                               config.freeze_vision_layers is not None or
                               config.freeze_vision_ratio is not None):
            print("\n🔧 重新应用灵活视觉层冻结策略（在权重加载后）...")
            model._apply_flexible_vision_freezing()
        elif not apply_freezing:
            print("\n💡 推理模式：跳过视觉层冻结策略应用（所有层正常工作）")

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

    def _apply_flexible_vision_freezing(self):
        """
        应用灵活的视觉层冻结策略

        支持三种配置方式（优先级从高到低）：
        1. unfreeze_vision_layers: 指定要解冻的层索引（支持负数索引）
        2. freeze_vision_layers: 指定要冻结的层索引
        3. freeze_vision_ratio: 按比例冻结前N%的层

        如果没有配置以上任何选项，使用默认的 freeze_vision_encoder 行为。
        """
        config = self.config

        # 如果没有配置灵活冻结策略，使用默认行为
        if (config.unfreeze_vision_layers is None and
            config.freeze_vision_layers is None and
                config.freeze_vision_ratio is None):
            return

        # 获取 vision_model（SmolVLM的视觉编码器）
        # 完整路径: self.model.vlm_with_expert.vlm.model.vision_model
        try:
            vision_model = None

            # 路径1: 通过 model.vlm_with_expert
            if hasattr(self, 'model') and hasattr(self.model, 'vlm_with_expert'):
                vlm_with_expert = self.model.vlm_with_expert
                if hasattr(vlm_with_expert, 'get_vlm_model'):
                    vlm_model = vlm_with_expert.get_vlm_model()
                    if hasattr(vlm_model, 'vision_model'):
                        vision_model = vlm_model.vision_model

            # 路径2: 直接通过 model 的 get_vlm_model
            if vision_model is None and hasattr(self, 'model') and hasattr(self.model, 'get_vlm_model'):
                vlm_model = self.model.get_vlm_model()
                if hasattr(vlm_model, 'vision_model'):
                    vision_model = vlm_model.vision_model

            # 路径3: 如果 self 本身是 VLAFlowMatching
            if vision_model is None and hasattr(self, 'vlm_with_expert'):
                if hasattr(self.vlm_with_expert, 'get_vlm_model'):
                    vlm_model = self.vlm_with_expert.get_vlm_model()
                    if hasattr(vlm_model, 'vision_model'):
                        vision_model = vlm_model.vision_model

            if vision_model is None:
                print("⚠️  无法找到 vision_model，跳过灵活冻结策略")
                print(f"   DEBUG: self 类型: {type(self).__name__}")
                if hasattr(self, 'model'):
                    print(
                        f"   DEBUG: self.model 类型: {type(self.model).__name__}")
                    if hasattr(self.model, 'vlm_with_expert'):
                        print(f"   DEBUG: self.model.vlm_with_expert 存在")
                return

        except Exception as e:
            print(f"⚠️  访问 vision_model 时出错: {e}")
            import traceback
            traceback.print_exc()
            return

        # 获取视觉编码器的所有层
        vision_layers = vision_model.encoder.layers
        total_layers = len(vision_layers)

        print(f"\n{'='*70}")
        print(f"🔧 应用灵活视觉层冻结策略")
        print(f"{'='*70}")

        # 打印调试信息
        print(f"📊 Vision Model 信息:")
        print(f"   - Vision Model 类型: {type(vision_model).__name__}")
        print(f"   - 是否有 encoder: {hasattr(vision_model, 'encoder')}")
        if hasattr(vision_model, 'config'):
            print(f"   - Config: {type(vision_model.config).__name__}")

        print(f"\nVision Encoder 总层数: {total_layers}")

        # 确定要冻结/解冻的层
        frozen_layers = set()
        unfrozen_layers = set()

        # 优先级1: unfreeze_vision_layers
        if config.unfreeze_vision_layers is not None:
            print(f"\n策略: 解冻指定层 {config.unfreeze_vision_layers}")

            # 默认所有层都冻结
            frozen_layers = set(range(total_layers))

            # 解冻指定的层（支持负数索引）
            for idx in config.unfreeze_vision_layers:
                if idx < 0:
                    actual_idx = total_layers + idx
                else:
                    actual_idx = idx

                if 0 <= actual_idx < total_layers:
                    frozen_layers.discard(actual_idx)
                    unfrozen_layers.add(actual_idx)
                else:
                    print(
                        f"⚠️  警告: 层索引 {idx} (实际: {actual_idx}) 超出范围 [0, {total_layers-1}]")

        # 优先级2: freeze_vision_layers
        elif config.freeze_vision_layers is not None:
            print(f"\n策略: 冻结指定层 {config.freeze_vision_layers}")

            # 默认所有层都解冻
            unfrozen_layers = set(range(total_layers))

            # 冻结指定的层
            for idx in config.freeze_vision_layers:
                if 0 <= idx < total_layers:
                    frozen_layers.add(idx)
                    unfrozen_layers.discard(idx)
                else:
                    print(f"⚠️  警告: 层索引 {idx} 超出范围 [0, {total_layers-1}]")

        # 优先级3: freeze_vision_ratio
        elif config.freeze_vision_ratio is not None:
            ratio = config.freeze_vision_ratio
            if not 0.0 <= ratio <= 1.0:
                print(
                    f"⚠️  警告: freeze_vision_ratio={ratio} 不在 [0.0, 1.0] 范围内，使用默认行为")
                return

            num_frozen = int(total_layers * ratio)
            print(f"\n策略: 按比例冻结前 {ratio:.1%} 的层 (前 {num_frozen} 层)")

            frozen_layers = set(range(num_frozen))
            unfrozen_layers = set(range(num_frozen, total_layers))

        # 应用冻结策略
        for layer_idx in range(total_layers):
            layer = vision_layers[layer_idx]

            if layer_idx in frozen_layers:
                # 冻结层
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                # 解冻层
                for param in layer.parameters():
                    param.requires_grad = True

        # 打印结果摘要
        print(f"\n✅ 冻结策略应用完成:")
        print(f"   🔒 冻结层数: {len(frozen_layers)} / {total_layers}")
        print(f"   🔓 解冻层数: {len(unfrozen_layers)} / {total_layers}")

        if frozen_layers:
            frozen_list = sorted(list(frozen_layers))
            if len(frozen_list) <= 10:
                print(f"   🔒 冻结层索引: {frozen_list}")
            else:
                print(f"   🔒 冻结层索引: [{frozen_list[0]}...{frozen_list[-1]}]")

        if unfrozen_layers:
            unfrozen_list = sorted(list(unfrozen_layers))
            if len(unfrozen_list) <= 10:
                print(f"   🔓 解冻层索引: {unfrozen_list}")
            else:
                print(
                    f"   🔓 解冻层索引: [{unfrozen_list[0]}...{unfrozen_list[-1]}]")

        print(f"{'='*70}\n")
