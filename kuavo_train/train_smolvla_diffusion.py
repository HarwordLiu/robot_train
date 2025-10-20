"""
SmolVLA Diffusion 训练脚本

基于 SmolVLA 架构但使用 Diffusion 进行动作生成训练
完全冻结视觉层，专注于训练 Action Expert 的 Diffusion 能力
"""

import os
import sys
import torch
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import logging
from datetime import datetime
import json

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 导入必要模块
from lerobot.common.policies import Policy
from lerobot.common.utils import init_logging, set_seed
from leroto import lerobot_datasets

# 导入项目模块
from kuavo_train.wrapper.policy.smolvla import SmolVLADiffusionPolicyWrapper
from kuavo_train.trainer.trainer import Trainer
from kuavo_train.datasets.dataset_utils import make_dataset


class DiffusionTrainer(Trainer):
    """
    Diffusion 专用训练器
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_diffusion = True
        self.cfg = cfg

    def train_epoch(self, epoch):
        """
        训练一个 epoch

        Args:
            epoch: 当前 epoch
        """
        self.policy.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.dataloader):
            # 准备 batch
            batch = self.prepare_batch(batch)

            # 前向传播
            loss, info = self.policy.forward(batch)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if hasattr(self.cfg.policy, 'optimizer_grad_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.cfg.policy.optimizer_grad_clip_norm
                )

            self.optimizer.step()
            self.scheduler.step()

            # 统计
            total_loss += loss.item()
            num_batches += 1

            # 日志
            if batch_idx % self.cfg.training.log_freq == 0:
                step = epoch * len(self.dataloader) + batch_idx
                self.log_training_info(step, loss.item(), info)

        # 返回平均损失
        return total_loss / num_batches

    def prepare_batch(self, batch):
        """
        准备批次数据
        """
        # 移动数据到设备
        device = self.cfg.training.device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # 确保有语言指令
        if 'task' not in batch:
            # 使用默认语言指令
            batch['task'] = [self.cfg.task.get('language_instruction', 'Complete the task')] * len(batch[next(iter(batch))])

        return batch

    def log_training_info(self, step, loss, info):
        """
        记录训练信息
        """
        if self.use_wandb:
            import wandb
            log_dict = {
                'train/loss': loss,
                'train/step': step,
            }

            # 添加 Diffusion 特定信息
            if 'timestep_mean' in info:
                log_dict['train/timestep_mean'] = info['timestep_mean']
            if 'noise_mean' in info:
                log_dict['train/noise_mean'] = info['noise_mean']
            if 'predicted_noise_mean' in info:
                log_dict['train/predicted_noise_mean'] = info['predicted_noise_mean']

            # 添加学习率
            if hasattr(self, 'scheduler'):
                log_dict['train/lr'] = self.scheduler.get_last_lr()[0]

            wandb.log(log_dict)

        # 控制台输出
        if step % (self.cfg.training.log_freq * 10) == 0:
            print(f"Step {step}: Loss = {loss:.6f}")
            if 'timestep_mean' in info:
                print(f"  - Avg Timestep: {info['timestep_mean']:.2f}")

    def validate(self, epoch):
        """
        验证模型
        """
        if not hasattr(self, 'eval_dataloader') or self.eval_dataloader is None:
            return None

        self.policy.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self.prepare_batch(batch)

                # 前向传播
                loss, info = self.policy.forward(batch)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else None

        # 记录验证损失
        if avg_loss is not None and self.use_wandb:
            import wandb
            wandb.log({
                'val/loss': avg_loss,
                'val/epoch': epoch,
            })

        return avg_loss


def setup_logging(cfg):
    """设置日志"""
    log_dir = Path(cfg.training.output_directory) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def setup_wandb(cfg):
    """设置 wandb"""
    if not cfg.get('use_wandb', False):
        return None

    import wandb

    wandb.init(
        project="smolvla-diffusion",
        name=f"{cfg.task}_{cfg.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=cfg.training.output_directory
    )

    return wandb


def make_policy(cfg, dataset_stats):
    """
    创建策略模型
    """
    # 根据配置决定是从预训练模型加载还是从头训练
    if hasattr(cfg.training, 'resume_from') and cfg.training.resume_from == 'pretrained':
        # 从预训练模型加载
        policy = SmolVLADiffusionPolicyWrapper.from_pretrained(
            pretrained_name_or_path=cfg.training.pretrained_path,
            config=cfg.policy,
            dataset_stats=dataset_stats
        )
    else:
        # 创建新模型
        policy = SmolVLADiffusionPolicyWrapper(cfg.policy, dataset_stats)

    return policy


def make_optimizer(cfg, policy):
    """
    创建优化器
    """
    # 获取可训练参数
    trainable_params = [p for p in policy.parameters() if p.requires_grad]

    # 创建优化器
    if cfg.policy.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=cfg.policy.optimizer_lr,
            betas=cfg.policy.optimizer_betas,
            eps=cfg.policy.optimizer_eps,
            weight_decay=cfg.policy.optimizer_weight_decay
        )
    elif cfg.policy.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.policy.optimizer_lr,
            betas=cfg.policy.optimizer_betas,
            eps=cfg.policy.optimizer_eps,
            weight_decay=cfg.policy.optimizer_weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器类型: {cfg.policy.optimizer_type}")

    return optimizer


def make_scheduler(cfg, optimizer):
    """
    创建学习率调度器
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.policy.scheduler_decay_steps,
        eta_min=cfg.policy.scheduler_decay_lr
    )

    return scheduler


def evaluate_policy(policy, cfg, epoch):
    """
    评估策略
    """
    # 这里可以添加评估逻辑
    # 例如在模拟环境中运行策略
    print(f"\n📊 Epoch {epoch} 评估:")
    print("   - 策略评估功能待实现")
    return {}


@hydra.main(
    version_base=None,
    config_path="../configs/policy",
    config_name="smolvla_diffusion_config"
)
def main(cfg):
    """
    主训练函数
    """
    # 设置设备
    device = torch.device(cfg.training.device if torch.cuda.is_available() else 'cpu')
    cfg.training.device = str(device)

    print(f"\n{'='*70}")
    print("🚀 开始 SmolVLA Diffusion 训练")
    print(f"{'='*70}")
    print(f"📋 配置信息:")
    print(f"   - 任务: {cfg.task}")
    print(f"   - 方法: {cfg.method}")
    print(f"   - 设备: {device}")
    print(f"   - 批大小: {cfg.training.batch_size}")
    print(f"   - 学习率: {cfg.policy.optimizer_lr}")
    print(f"   - 推理步数: {cfg.policy.num_inference_steps}")
    print(f"   - 视觉编码器冻结: {cfg.policy.freeze_vision_encoder}")
    print(f"{'='*70}\n")

    # 设置随机种子
    set_seed(cfg.training.seed)

    # 设置日志
    logger = setup_logging(cfg)

    # 设置 wandb
    wandb_run = setup_wandb(cfg)
    cfg.use_wandb = wandb_run is not None

    # 创建数据集
    print("📦 创建数据集...")
    train_dataset, eval_dataset = make_dataset(cfg)
    print(f"   - 训练样本数: {len(train_dataset)}")
    if eval_dataset:
        print(f"   - 验证样本数: {len(eval_dataset)}")

    # 创建数据加载器
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        drop_last=cfg.training.drop_last,
        prefetch_factor=cfg.training.prefetch_factor,
        persistent_workers=cfg.training.persistent_workers
    )

    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            drop_last=False
        )

    # 获取数据集统计信息
    if hasattr(train_dataset, 'stats') and train_dataset.stats is not None:
        dataset_stats = train_dataset.stats
    else:
        # 创建空的统计信息
        dataset_stats = SmolVLADiffusionPolicyWrapper._create_identity_stats(cfg.policy)

    # 创建策略
    print("🧠 创建策略模型...")
    policy = make_policy(cfg, dataset_stats)
    policy.to(device)

    # 创建优化器
    print("⚙️ 创建优化器...")
    optimizer = make_optimizer(cfg, policy)

    # 创建学习率调度器
    scheduler = make_scheduler(cfg, optimizer)

    # 创建训练器
    trainer = DiffusionTrainer(cfg)
    trainer.policy = policy
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.dataloader = train_dataloader
    trainer.eval_dataloader = eval_dataloader
    trainer.use_wandb = cfg.use_wandb

    # 保存配置
    output_dir = Path(cfg.training.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_file = output_dir / "config.yaml"
    with open(config_file, 'w') as f:
        OmegaConf.save(cfg, f)

    print(f"💾 配置已保存到: {config_file}")

    # 开始训练
    print(f"\n🏃 开始训练 (共 {cfg.training.max_epoch} epochs)...")
    best_loss = float('inf')

    for epoch in range(cfg.training.max_epoch):
        print(f"\nEpoch {epoch + 1}/{cfg.training.max_epoch}")
        print("-" * 50)

        # 训练一个 epoch
        avg_loss = trainer.train_epoch(epoch)
        print(f"训练损失: {avg_loss:.6f}")

        # 验证
        if eval_dataloader is not None and epoch % cfg.training.validation_freq_epoch == 0:
            val_loss = trainer.validate(epoch)
            if val_loss:
                print(f"验证损失: {val_loss:.6f}")

        # 保存检查点
        if epoch % cfg.training.save_freq_epoch == 0:
            checkpoint_dir = output_dir / f"checkpoint_epoch_{epoch}"
            checkpoint_dir.mkdir(exist_ok=True)

            # 保存模型
            policy.save_pretrained(checkpoint_dir)

            # 保存训练状态
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_dir / "training_state.pt")

            print(f"✅ 检查点已保存: {checkpoint_dir}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dir = output_dir / "best"
            best_dir.mkdir(exist_ok=True)
            policy.save_pretrained(best_dir)
            print(f"🏆 最佳模型已更新 (损失: {best_loss:.6f})")

        # 评估策略（可选）
        if epoch % 10 == 0:
            evaluate_policy(policy, cfg, epoch)

    print(f"\n✅ 训练完成!")
    print(f"   - 最佳损失: {best_loss:.6f}")
    print(f"   - 输出目录: {output_dir}")

    # 关闭 wandb
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()