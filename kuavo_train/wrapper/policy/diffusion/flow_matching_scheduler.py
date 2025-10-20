"""
Flow Matching Scheduler for Robot Control
基于最优传输的流匹配调度器，可替代传统 Diffusion 调度器

参考:
- Flow Matching for Generative Modeling (Lipman et al., 2023)
- Rectified Flow (Liu et al., 2022)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple
from enum import Enum


class FlowMatchingType(Enum):
    """Flow Matching 类型"""
    CONDITIONAL = "conditional"  # 条件流匹配（默认）
    OPTIMAL_TRANSPORT = "optimal_transport"  # 最优传输流匹配
    RECTIFIED = "rectified"  # 修正流


class FlowMatchingScheduler:
    """
    Flow Matching 调度器

    相比 DDPM/DDIM:
    - 使用 ODE 而非 SDE
    - 训练时间步为连续 [0, 1] 而非离散 [0, T]
    - 预测速度场 v_t 而非噪声 ε
    - 推理步数可以大幅减少（10-20步）

    训练过程:
    1. 采样 t ~ U[0, 1]
    2. 计算插值: x_t = (1-t)·x_0 + t·x_1
    3. 计算目标速度: v_t = x_1 - x_0
    4. 训练模型预测 v_t

    推理过程:
    1. 从 x_0 ~ N(0, I) 开始
    2. 使用 ODE 求解器（Euler/RK4）积分
    3. 得到 x_1（生成的动作）
    """

    def __init__(
        self,
        num_inference_steps: int = 10,
        flow_matching_type: str = "conditional",
        sigma: float = 0.0,  # 噪声水平（0表示确定性流）
        use_ode_solver: str = "euler",  # "euler" 或 "rk4"
        device: Union[str, torch.device] = "cpu",
    ):
        """
        初始化 Flow Matching 调度器

        Args:
            num_inference_steps: 推理步数（通常10-20步即可）
            flow_matching_type: 流匹配类型
            sigma: 噪声水平（用于随机流匹配）
            use_ode_solver: ODE 求解器类型
            device: 设备
        """
        self.num_inference_steps = num_inference_steps
        self.flow_matching_type = FlowMatchingType(flow_matching_type)
        self.sigma = sigma
        self.use_ode_solver = use_ode_solver
        self.device = device

        # 设置时间步
        self.timesteps = None
        self._init_timesteps()

        print(f"✅ FlowMatchingScheduler 已初始化:")
        print(f"   - 推理步数: {num_inference_steps}")
        print(f"   - 类型: {flow_matching_type}")
        print(f"   - ODE求解器: {use_ode_solver}")
        print(f"   - 噪声水平: {sigma}")

    def _init_timesteps(self):
        """初始化时间步（从 0 到 1）"""
        self.timesteps = torch.linspace(
            0, 1, self.num_inference_steps + 1,
            device=self.device
        )

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置推理时间步

        Args:
            num_inference_steps: 推理步数
            device: 设备
        """
        self.num_inference_steps = num_inference_steps
        if device is not None:
            self.device = device
        self._init_timesteps()

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        训练时添加噪声（线性插值）

        Flow Matching: x_t = (1-t)·x_0 + t·x_1
        其中 x_0 是噪声，x_1 是目标样本

        Args:
            original_samples: 原始样本（目标动作）x_1 [B, T, D]
            noise: 噪声样本 x_0 [B, T, D]
            timesteps: 时间步 t ∈ [0, 1] [B]

        Returns:
            插值后的样本 x_t
        """
        # 确保 timesteps 在 [0, 1] 范围内
        timesteps = timesteps.to(original_samples.device)

        # 扩展维度以匹配样本形状 [B] -> [B, 1, 1]
        t_expanded = timesteps.view(-1, 1, 1)

        # 线性插值: x_t = (1-t)·noise + t·original
        noisy_samples = (1 - t_expanded) * noise + \
            t_expanded * original_samples

        # 可选：添加小量噪声（随机流匹配）
        if self.sigma > 0:
            additional_noise = torch.randn_like(noisy_samples) * self.sigma
            noisy_samples = noisy_samples + additional_noise

        return noisy_samples

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        计算目标速度场 v_t

        对于条件流匹配: v_t = x_1 - x_0

        Args:
            sample: 目标样本 x_1
            noise: 初始噪声 x_0
            timesteps: 时间步（在 Flow Matching 中不影响速度）

        Returns:
            目标速度场 v_t
        """
        # Flow Matching 的速度场是常数（从 x_0 指向 x_1）
        return sample - noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple]:
        """
        执行一步 ODE 求解（推理时使用）

        使用 Euler 方法: x_{t+dt} = x_t + v_t * dt
        或 RK4 方法获得更高精度

        Args:
            model_output: 模型预测的速度场 v_t
            timestep: 当前时间步
            sample: 当前样本 x_t
            return_dict: 是否返回字典

        Returns:
            下一时间步的样本 x_{t+dt}
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item()

        # 计算时间步长
        dt = 1.0 / self.num_inference_steps

        if self.use_ode_solver == "euler":
            # Euler 方法: x_{t+dt} = x_t + v_t * dt
            prev_sample = sample + model_output * dt

        elif self.use_ode_solver == "rk4":
            # RK4 方法（更精确但需要4次前向传播）
            # 注意：这需要额外的模型调用，这里仅做演示
            # 实际使用时需要在外部实现
            prev_sample = sample + model_output * dt

        else:
            raise ValueError(f"Unknown ODE solver: {self.use_ode_solver}")

        if return_dict:
            return {"prev_sample": prev_sample}
        return prev_sample

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Optional[Union[int, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        缩放模型输入（Flow Matching 不需要缩放）

        Args:
            sample: 输入样本
            timestep: 时间步（未使用）

        Returns:
            原样返回
        """
        return sample

    def __len__(self):
        return self.num_inference_steps


class OptimalTransportFlowScheduler(FlowMatchingScheduler):
    """
    最优传输流匹配调度器

    使用最优传输理论构建更优的传输路径
    可以进一步提升采样效率
    """

    def __init__(self, **kwargs):
        super().__init__(flow_matching_type="optimal_transport", **kwargs)
        print("⚠️ 最优传输流匹配需要额外的配对算法")
        print("   当前使用简化版本（等价于条件流匹配）")

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用最优传输路径添加噪声

        理论上应该使用 Sinkhorn 算法等计算最优配对
        这里使用简化版本
        """
        # 简化版本：直接使用线性插值
        # 完整实现需要：
        # 1. 计算 cost matrix
        # 2. 使用 Sinkhorn 算法求解最优传输计划
        # 3. 根据传输计划进行配对
        return super().add_noise(original_samples, noise, timesteps)


class RectifiedFlowScheduler(FlowMatchingScheduler):
    """
    修正流调度器

    通过多次修正（rectification）使流更加直接
    可以用更少的步数达到相同质量
    """

    def __init__(self, num_rectifications: int = 1, **kwargs):
        super().__init__(flow_matching_type="rectified", **kwargs)
        self.num_rectifications = num_rectifications
        print(f"✅ 修正流调度器（修正次数: {num_rectifications}）")

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        修正流的速度场

        经过修正后，轨迹更接近直线
        """
        # 基础版本仍然使用 v_t = x_1 - x_0
        # 完整实现需要通过多次训练进行修正
        return super().get_velocity(sample, noise, timesteps)


def create_flow_matching_scheduler(
    scheduler_type: str = "conditional",
    num_inference_steps: int = 10,
    **kwargs
) -> FlowMatchingScheduler:
    """
    工厂函数：创建 Flow Matching 调度器

    Args:
        scheduler_type: 调度器类型
        num_inference_steps: 推理步数
        **kwargs: 其他参数

    Returns:
        Flow Matching 调度器实例
    """
    if scheduler_type == "conditional":
        return FlowMatchingScheduler(
            num_inference_steps=num_inference_steps,
            **kwargs
        )
    elif scheduler_type == "optimal_transport":
        return OptimalTransportFlowScheduler(
            num_inference_steps=num_inference_steps,
            **kwargs
        )
    elif scheduler_type == "rectified":
        return RectifiedFlowScheduler(
            num_inference_steps=num_inference_steps,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ============= 辅助函数 =============

def compare_schedulers_info():
    """打印 Flow Matching 和 Diffusion 的对比信息"""
    print("\n" + "="*70)
    print("📊 Flow Matching vs Diffusion 对比")
    print("="*70)

    info = """
    | 特性                | Diffusion (DDPM)    | Flow Matching        |
    |---------------------|---------------------|----------------------|
    | 理论基础            | SDE                 | ODE                  |
    | 训练时间步          | [0, T] (离散)       | [0, 1] (连续)        |
    | 推理步数            | 50-1000步           | 10-20步 ⚡           |
    | 预测目标            | 噪声 ε              | 速度场 v_t           |
    | 采样确定性          | 随机(DDPM)          | 确定性 ✅            |
    | 推理速度            | 慢                  | 快(3-10倍) 🚀        |
    | 训练复杂度          | 需要噪声调度        | 简单线性插值         |
    | 适用场景            | 高质量生成          | 实时控制 ⭐          |

    推荐使用 Flow Matching 的场景:
    ✅ 机器人实时控制
    ✅ 高频率动作生成
    ✅ 资源受限设备
    ✅ 延迟敏感应用
    """
    print(info)
    print("="*70 + "\n")


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试 FlowMatchingScheduler")

    # 创建调度器
    scheduler = FlowMatchingScheduler(num_inference_steps=10)

    # 模拟训练数据
    batch_size, horizon, action_dim = 8, 16, 14
    target_actions = torch.randn(batch_size, horizon, action_dim)
    noise = torch.randn_like(target_actions)
    timesteps = torch.rand(batch_size)  # [0, 1]

    # 添加噪声
    noisy_actions = scheduler.add_noise(target_actions, noise, timesteps)
    print(f"✅ 添加噪声: {noisy_actions.shape}")

    # 计算速度场
    velocity = scheduler.get_velocity(target_actions, noise, timesteps)
    print(f"✅ 计算速度场: {velocity.shape}")

    # 模拟推理步骤
    current_sample = noise
    for i, t in enumerate(scheduler.timesteps[:-1]):
        # 假设模型预测的速度场
        pred_velocity = velocity  # 实际应该是模型输出
        current_sample = scheduler.step(pred_velocity, t, current_sample)
        if i % 3 == 0:
            print(
                f"  Step {i}: t={t:.2f}, sample_mean={current_sample.mean():.4f}")

    print(f"\n✅ 推理完成: {current_sample.shape}")
    print(f"📊 目标均值: {target_actions.mean():.4f}")
    print(f"📊 生成均值: {current_sample.mean():.4f}")

    # 打印对比信息
    compare_schedulers_info()
