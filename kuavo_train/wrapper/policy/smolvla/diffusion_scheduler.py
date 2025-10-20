"""
Diffusion 噪声调度器实现
为 SmolVLA Diffusion 提供噪声调度和采样功能
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple
from enum import Enum


class SchedulerType(Enum):
    """调度器类型"""
    DDPM = "ddpm"
    DDIM = "ddim"


class PredictionType(Enum):
    """预测类型"""
    EPSILON = "epsilon"
    V_PREDICTION = "v_prediction"
    SAMPLE = "sample"


class NoiseScheduler:
    """
    Diffusion 噪声调度器

    支持多种噪声调度和采样策略：
    - Linear noise schedule
    - Cosine noise schedule
    - DDPM sampling
    - DDIM sampling (加速推理)
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: Optional[int] = None,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
    ):
        """
        初始化噪声调度器

        Args:
            num_train_timesteps: 训练时的总时间步数
            num_inference_steps: 推理时的步数（None则使用训练步数）
            beta_start: 起始 beta 值
            beta_end: 结束 beta 值
            beta_schedule: beta 调度类型 ("linear", "cosine", "sqrt_linear")
            prediction_type: 预测类型 ("epsilon", "v_prediction", "sample")
            clip_sample: 是否裁剪样本
            clip_sample_range: 裁剪范围
            set_alpha_to_one: 是否将最终 alpha 设为 1
            steps_offset: 步数偏移
        """
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps or num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.set_alpha_to_one = set_alpha_to_one
        self.steps_offset = steps_offset

        # 初始化调度器
        self._init_scheduler()

    def _init_scheduler(self):
        """初始化调度器参数"""
        # 创建 beta 调度
        if self.beta_schedule == "linear":
            self.betas = torch.linspace(
                self.beta_start,
                self.beta_end,
                self.num_train_timesteps,
                dtype=torch.float32
            )
        elif self.beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule()
        elif self.beta_schedule == "sqrt_linear":
            self.betas = torch.linspace(
                self.beta_start ** 0.5,
                self.beta_end ** 0.5,
                self.num_train_timesteps,
                dtype=torch.float32
            ) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

        # 计算 alpha 相关参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 标准 DPDM 参数
        self.one = torch.tensor(1.0)

        # 创建用于采样的时间步
        self.timesteps = torch.arange(
            self.num_train_timesteps - self.steps_offset,
            -1,
            -1,
            dtype=torch.long
        )

        # 设置推理时间步
        if self.num_inference_steps < self.num_train_timesteps:
            self._set_timesteps()

    def _cosine_beta_schedule(self, timesteps: int = 1000, s: float = 0.008):
        """
        余弦 beta 调度

        参考: https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _set_timesteps(self):
        """设置推理时间步（用于加速推理）"""
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        self.timesteps = (
            torch.arange(0, self.num_inference_steps) * step_ratio
        ).round().long().to(torch.device("cpu"))

        if self.steps_offset:
            self.timesteps += self.steps_offset

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        缩放模型输入（DDIM 需要）

        Args:
            sample: 输入样本
            timestep: 当前时间步

        Returns:
            缩放后的样本
        """
        return sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        向原始样本添加噪声

        Args:
            original_samples: 原始样本（干净的动作）
            noise: 要添加的噪声
            timesteps: 时间步

        Returns:
            加噪后的样本
        """
        # 确保时间步在设备上
        timesteps = timesteps.to(original_samples.device)

        # 计算 sqrt(alpha_cumprod) 和 sqrt(1-alpha_cumprod)
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 添加噪声
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple]:
        """
        执行一步去噪

        Args:
            model_output: 模型输出（预测的噪声或样本）
            timestep: 当前时间步
            sample: 当前样本（噪声样本）
            eta: DDIM 的随机性参数（0为确定性）
            use_clipped_model_output: 是否使用裁剪的模型输出
            generator: 随机数生成器
            variance_noise: 方差噪声（DDIM 需要）
            return_dict: 是否返回字典

        Returns:
            去噪后的样本
        """
        # 1. 预测之前的样本值 (x_0) 和 原始噪声值 (eps)
        if self.prediction_type == "epsilon":
            pred_epsilon = model_output
            pred_original_sample = self._predict_x0_from_eps(
                sample=sample,
                timestep=timestep,
                eps=pred_epsilon
            )
        elif self.prediction_type == "v_prediction":
            pred_original_sample, pred_epsilon = self._predict_x0_and_eps_from_v(
                sample=sample,
                timestep=timestep,
                v_pred=model_output
            )
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = self._predict_eps_from_x0(
                sample=sample,
                timestep=timestep,
                x0=pred_original_sample
            )
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")

        # 2. 计算前一个时间步的噪声
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # 3. 计算系数
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.one

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 4. 计算预测的原始样本方差
        if prev_timestep == 0:
            variance = torch.tensor(0.0)
        else:
            variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        # 5. 采样
        pred_sample_direction = (1 - alpha_prod_t_prev - variance) ** 0.5 * pred_epsilon
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5)

        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample + pred_sample_direction
        )

        # 6. 添加噪声（DDIM）
        if eta > 0:
            if variance_noise is None and generator is not None:
                variance_noise = torch.randn(
                    sample.shape,
                    generator=generator,
                    device=sample.device,
                    dtype=sample.dtype
                )

            variance = eta * variance ** 0.5

            if variance_noise is not None:
                pred_prev_sample = pred_prev_sample + variance * variance_noise

        # 7. 裁剪（可选）
        if self.clip_sample:
            pred_prev_sample = pred_prev_sample.clamp(
                -self.clip_sample_range,
                self.clip_sample_range
            )

        return pred_prev_sample

    def _predict_x0_from_eps(self, sample: torch.Tensor, timestep: int, eps: torch.Tensor) -> torch.Tensor:
        """从噪声预测原始样本"""
        alpha_prod_t = self.alphas_cumprod[timestep]
        sqrt_one_minus_alpha_prod = (1 - alpha_prod_t) ** 0.5
        sqrt_alpha_prod = alpha_prod_t ** 0.5

        return (sample - sqrt_one_minus_alpha_prod * eps) / sqrt_alpha_prod

    def _predict_eps_from_x0(self, sample: torch.Tensor, timestep: int, x0: torch.Tensor) -> torch.Tensor:
        """从原始样本预测噪声"""
        alpha_prod_t = self.alphas_cumprod[timestep]
        sqrt_one_minus_alpha_prod = (1 - alpha_prod_t) ** 0.5
        sqrt_alpha_prod = alpha_prod_t ** 0.5

        return (sample - sqrt_alpha_prod * x0) / sqrt_one_minus_alpha_prod

    def _predict_x0_and_eps_from_v(self, sample: torch.Tensor, timestep: int, v_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 v 参数化预测原始样本和噪声"""
        alpha_prod_t = self.alphas_cumprod[timestep]
        sqrt_alpha_prod = alpha_prod_t ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alpha_prod_t) ** 0.5

        pred_original_sample = sqrt_alpha_prod * sample - sqrt_one_minus_alpha_prod * v_pred
        pred_epsilon = sqrt_alpha_prod * v_pred + sqrt_one_minus_alpha_prod * sample

        return pred_original_sample, pred_epsilon

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timestep: int) -> torch.Tensor:
        """计算 v 参数化"""
        alpha_prod_t = self.alphas_cumprod[timestep]
        sqrt_alpha_prod = alpha_prod_t ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alpha_prod_t) ** 0.5

        return sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample


class DDPMScheduler(NoiseScheduler):
    """DDPM 调度器"""

    def __init__(self, **kwargs):
        kwargs.setdefault("set_alpha_to_one", True)
        super().__init__(**kwargs)


class DDIMScheduler(NoiseScheduler):
    """DDIM 调度器（加速推理）"""

    def __init__(self, **kwargs):
        kwargs.setdefault("set_alpha_to_one", False)
        super().__init__(**kwargs)

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置 DDIM 推理时间步

        Args:
            num_inference_steps: 推理步数
            device: 设备
        """
        self.num_inference_steps = num_inference_steps

        # 创建均匀间隔的时间步
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (
            torch.arange(0, self.num_inference_steps) * step_ratio
        ).round().long()

        # 加上偏移
        timesteps += self.steps_offset

        self.timesteps = timesteps.to(device)

        # 创建用于插值的 alpha
        self._step_index = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator=None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Tuple]:
        """
        DDIM 采样步骤

        Args:
            model_output: 模型输出
            timestep: 当前时间步
            sample: 当前样本
            eta: 随机性参数（0为确定性采样）
            generator: 随机数生成器
            return_dict: 是否返回字典

        Returns:
            去噪后的样本
        """
        return super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            eta=eta,
            generator=generator,
            return_dict=return_dict,
        )