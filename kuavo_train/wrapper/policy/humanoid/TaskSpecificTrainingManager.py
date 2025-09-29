# -*- coding: utf-8 -*-
"""
任务特定训练管理器

管理分层架构在多任务场景下的训练，支持：
- 任务特定的课程学习策略
- 渐进式任务数据集成
- 防止灾难性遗忘的权重调整
- 任务条件层激活控制
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from omegaconf import DictConfig


@dataclass
class TaskInfo:
    """任务信息配置"""
    task_id: int
    name: str
    complexity_level: int  # 1-4, 1最简单
    required_layers: List[str]
    primary_capabilities: List[str]
    data_available: bool = False
    episode_count: int = 0


class TaskSpecificTrainingManager:
    """任务特定训练管理器"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = logging.getLogger("TaskTrainingManager")

        # 预定义任务配置
        self.task_definitions = {
            1: TaskInfo(
                task_id=1,
                name="dynamic_grasping",
                complexity_level=2,  # 中等复杂度，需要快速反应
                required_layers=["safety", "manipulation"],
                primary_capabilities=["object_detection",
                                      "trajectory_tracking", "grasp_control"]
            ),
            2: TaskInfo(
                task_id=2,
                name="package_weighing",
                complexity_level=3,  # 高复杂度，需要双臂协调
                required_layers=["safety", "gait", "manipulation"],
                primary_capabilities=["dual_arm_coordination",
                                      "weight_estimation", "balance_control"]
            ),
            3: TaskInfo(
                task_id=3,
                name="precise_placement",
                complexity_level=3,  # 高复杂度，需要精确控制
                required_layers=["safety", "manipulation", "planning"],
                primary_capabilities=["spatial_reasoning",
                                      "orientation_control", "precision_placement"]
            ),
            4: TaskInfo(
                task_id=4,
                name="full_process_sorting",
                complexity_level=4,  # 最高复杂度，全身协调
                required_layers=["safety", "gait", "manipulation", "planning"],
                primary_capabilities=["whole_body_coordination",
                                      "sequence_planning", "multi_modal_control"]
            )
        }

        # 当前可用任务
        self.available_tasks = []
        self.current_training_phase = 1

        # 任务特定权重配置
        self.task_layer_weights = {
            1: {"safety": 2.0, "gait": 0.5, "manipulation": 2.0, "planning": 0.8},
            2: {"safety": 2.0, "gait": 1.8, "manipulation": 1.5, "planning": 1.0},
            3: {"safety": 2.0, "gait": 0.8, "manipulation": 1.8, "planning": 2.0},
            4: {"safety": 2.0, "gait": 1.5, "manipulation": 1.5, "planning": 1.5}
        }

        # 任务特定课程学习阶段（从配置文件读取）
        self.task_curriculum_stages = self._build_task_curriculum_stages(
            config)

    def _build_task_curriculum_stages(self, config: DictConfig):
        """从配置文件构建课程学习阶段"""
        # 默认配置
        default_stages = {
            1: {  # 动态抓取 - 快速反应导向
                "stage1": {"name": "safety_reflex", "layers": ["safety"], "epochs": 30},
                "stage2": {"name": "basic_manipulation", "layers": ["safety", "manipulation"], "epochs": 70},
                "stage3": {"name": "full_grasping", "layers": ["safety", "gait", "manipulation"], "epochs": 100}
            },
            2: {  # 称重 - 平衡协调导向
                "stage1": {"name": "safety_base", "layers": ["safety"], "epochs": 25},
                "stage2": {"name": "gait_control", "layers": ["safety", "gait"], "epochs": 50},
                "stage3": {"name": "dual_arm_coord", "layers": ["safety", "gait", "manipulation"], "epochs": 75},
                "stage4": {"name": "full_weighing", "layers": ["safety", "gait", "manipulation", "planning"], "epochs": 100}
            },
            3: {  # 摆放 - 精确控制导向
                "stage1": {"name": "safety_base", "layers": ["safety"], "epochs": 20},
                "stage2": {"name": "precise_manipulation", "layers": ["safety", "manipulation"], "epochs": 60},
                "stage3": {"name": "spatial_planning", "layers": ["safety", "manipulation", "planning"], "epochs": 80},
                "stage4": {"name": "full_placement", "layers": ["safety", "gait", "manipulation", "planning"], "epochs": 100}
            },
            4: {  # 分拣 - 全能力导向
                "stage1": {"name": "safety_foundation", "layers": ["safety"], "epochs": 20},
                "stage2": {"name": "movement_control", "layers": ["safety", "gait"], "epochs": 40},
                "stage3": {"name": "manipulation_skills", "layers": ["safety", "gait", "manipulation"], "epochs": 60},
                "stage4": {"name": "integrated_planning", "layers": ["safety", "gait", "manipulation", "planning"], "epochs": 80},
                "stage5": {"name": "full_sorting", "layers": ["safety", "gait", "manipulation", "planning"], "epochs": 100}
            }
        }

        # 尝试从配置文件覆盖epochs设置
        try:
            if hasattr(config, 'policy') and hasattr(config.policy, 'hierarchical'):
                hierarchical_config = config.policy.hierarchical
                if hasattr(hierarchical_config, 'curriculum_learning'):
                    curriculum_config = hierarchical_config.curriculum_learning

                    # 优先使用universal_stages配置
                    if hasattr(curriculum_config, 'universal_stages'):
                        universal_stages = curriculum_config.universal_stages

                        # 覆盖任务1的课程学习阶段
                        default_stages[1] = {
                            "stage1": {
                                "name": "safety_reflex",
                                "layers": ["safety"],
                                "epochs": universal_stages.get("stage1", {}).get("epochs", 1)
                            },
                            "stage2": {
                                "name": "basic_manipulation",
                                "layers": ["safety", "manipulation"],
                                "epochs": universal_stages.get("stage2", {}).get("epochs", 1)
                            },
                            "stage3": {
                                "name": "full_grasping",
                                "layers": ["safety", "gait", "manipulation"],
                                "epochs": universal_stages.get("stage3", {}).get("epochs", 1)
                            },
                            "stage4": {
                                "name": "full_grasping",
                                "layers": ["safety", "gait", "manipulation", "planning"],
                                "epochs": universal_stages.get("stage4", {}).get("epochs", 1)
                            }
                        }

                        print("📝 从配置文件读取课程学习epochs:")
                        for stage_name, stage_config in default_stages[1].items():
                            print(
                                f"   {stage_name}: {stage_config['epochs']} epochs")

        except Exception as e:
            print(f"⚠️  读取配置文件epochs失败: {e}, 使用默认设置")

        return default_stages

    def register_available_task(self, task_id: int, episode_count: int, data_path: str):
        """注册可用任务数据"""
        if task_id in self.task_definitions:
            task_info = self.task_definitions[task_id]
            task_info.data_available = True
            task_info.episode_count = episode_count

            if task_id not in self.available_tasks:
                self.available_tasks.append(task_id)
                self.available_tasks.sort()

            self.logger.info(
                f"✅ 注册任务{task_id}数据: {task_info.name}, {episode_count}个episodes")

            # 更新训练阶段
            self.current_training_phase = max(self.available_tasks)
            return True
        return False

    def get_current_curriculum_stages(self) -> Dict[str, Any]:
        """获取当前可用任务的课程学习阶段"""
        if not self.available_tasks:
            return {}

        # 如果只有一个任务，使用该任务的专用课程
        if len(self.available_tasks) == 1:
            task_id = self.available_tasks[0]
            return self.task_curriculum_stages[task_id]

        # 多任务情况下，构建混合课程学习
        return self._build_progressive_curriculum()

    def _build_progressive_curriculum(self) -> Dict[str, Any]:
        """构建渐进式多任务课程学习"""
        stages = {}
        stage_counter = 1

        # 按任务复杂度排序
        sorted_tasks = sorted(self.available_tasks,
                              key=lambda x: self.task_definitions[x].complexity_level)

        # 为每个任务创建专门的适应阶段
        for task_id in sorted_tasks:
            task_info = self.task_definitions[task_id]
            task_stages = self.task_curriculum_stages[task_id]

            # 为每个任务添加适应阶段
            for stage_name, stage_config in task_stages.items():
                adapted_stage_name = f"stage{stage_counter}_{task_info.name}"
                stages[adapted_stage_name] = {
                    **stage_config,
                    "name": adapted_stage_name,
                    "target_task": task_id,
                    "task_weight": 1.0 / len(self.available_tasks),
                    "epochs": max(10, stage_config["epochs"] // len(self.available_tasks))
                }
                stage_counter += 1

        # 添加最终融合阶段
        stages[f"stage{stage_counter}_integration"] = {
            "name": "multi_task_integration",
            "layers": ["safety", "gait", "manipulation", "planning"],
            "epochs": 50,
            "target_task": "all",
            "task_weight": "balanced"
        }

        return stages

    def get_task_specific_layer_weights(self, task_id: Optional[int] = None) -> Dict[str, float]:
        """获取任务特定的层权重"""
        if task_id is None or task_id not in self.task_layer_weights:
            # 返回平衡权重
            return {"safety": 2.0, "gait": 1.5, "manipulation": 1.0, "planning": 0.8}

        return self.task_layer_weights[task_id]

    def get_task_data_sampling_strategy(self) -> Dict[str, Any]:
        """获取任务数据采样策略"""
        if len(self.available_tasks) <= 1:
            return {"strategy": "single_task", "task_weights": {}}

        # 多任务采样策略
        sampling_weights = {}
        total_episodes = sum(self.task_definitions[tid].episode_count
                             for tid in self.available_tasks)

        for task_id in self.available_tasks:
            task_info = self.task_definitions[task_id]

            # 基于复杂度和数据量的权重计算
            complexity_factor = task_info.complexity_level / 4.0
            data_factor = task_info.episode_count / total_episodes

            # 平衡复杂度和数据可用性
            sampling_weights[task_id] = 0.6 * \
                complexity_factor + 0.4 * data_factor

        # 归一化权重
        total_weight = sum(sampling_weights.values())
        sampling_weights = {k: v/total_weight for k,
                            v in sampling_weights.items()}

        return {
            "strategy": "weighted_sampling",
            "task_weights": sampling_weights,
            "anti_forgetting": True,
            "rehearsal_ratio": 0.2  # 20%的数据用于防遗忘
        }

    def should_activate_layer_for_task(self, layer_name: str, task_id: int,
                                       current_stage: Optional[str] = None) -> bool:
        """判断特定任务是否应该激活特定层"""
        task_info = self.task_definitions.get(task_id)
        if not task_info:
            return True  # 默认激活

        # 检查层是否在任务的必需层列表中
        if layer_name not in task_info.required_layers:
            return False

        # 如果指定了当前阶段，检查该阶段是否包含此层
        if current_stage:
            current_curriculum = self.get_current_curriculum_stages()
            stage_config = current_curriculum.get(current_stage, {})
            return layer_name in stage_config.get("layers", [])

        return True

    def get_task_specific_loss_weights(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """获取批次的任务特定损失权重"""
        # 如果batch中包含任务标识，使用特定权重
        if "task_id" in batch:
            task_ids = batch["task_id"].cpu().numpy() if isinstance(
                batch["task_id"], torch.Tensor) else batch["task_id"]

            # 计算批次中各任务的平均权重
            batch_weights = {"safety": 0, "gait": 0,
                             "manipulation": 0, "planning": 0}

            for task_id in np.unique(task_ids):
                if task_id in self.task_layer_weights:
                    task_weight = (task_ids == task_id).sum() / len(task_ids)
                    task_layer_weights = self.task_layer_weights[task_id]

                    for layer, weight in task_layer_weights.items():
                        batch_weights[layer] += weight * task_weight

            return batch_weights

        # 默认使用当前训练阶段的主要任务权重
        if self.available_tasks:
            primary_task = max(self.available_tasks)  # 使用最新的任务作为主要任务
            return self.task_layer_weights.get(primary_task,
                                               {"safety": 2.0, "gait": 1.5, "manipulation": 1.0, "planning": 0.8})

        return {"safety": 2.0, "gait": 1.5, "manipulation": 1.0, "planning": 0.8}

    def prepare_training_config_for_phase(self, phase: int) -> Dict[str, Any]:
        """为特定训练阶段准备配置"""
        phase_config = {
            "phase": phase,
            "available_tasks": self.available_tasks[:phase],  # 只使用前N个任务
            "curriculum_stages": {},
            "layer_weights": {},
            "data_sampling": {}
        }

        # 构建该阶段的课程学习配置
        if phase == 1:
            # 单任务训练阶段
            phase_config["curriculum_stages"] = self.task_curriculum_stages[1]
            phase_config["layer_weights"] = self.task_layer_weights[1]
            phase_config["data_sampling"] = {
                "strategy": "single_task", "primary_task": 1}

        else:
            # 多任务渐进训练阶段
            phase_config["curriculum_stages"] = self._build_progressive_curriculum()
            phase_config["layer_weights"] = self._compute_weighted_layer_config(
                self.available_tasks[:phase])
            phase_config["data_sampling"] = self.get_task_data_sampling_strategy()

        return phase_config

    def _compute_weighted_layer_config(self, task_list: List[int]) -> Dict[str, float]:
        """计算多任务的加权层配置"""
        if len(task_list) == 1:
            return self.task_layer_weights[task_list[0]]

        # 基于任务复杂度的加权平均
        weights = {}
        total_complexity = sum(
            self.task_definitions[tid].complexity_level for tid in task_list)

        for layer in ["safety", "gait", "manipulation", "planning"]:
            weighted_sum = 0
            for task_id in task_list:
                task_complexity = self.task_definitions[task_id].complexity_level
                task_weight = task_complexity / total_complexity
                layer_weight = self.task_layer_weights[task_id][layer]
                weighted_sum += task_weight * layer_weight

            weights[layer] = weighted_sum

        return weights

    def should_enable_anti_forgetting(self) -> bool:
        """判断是否应该启用防遗忘机制"""
        return len(self.available_tasks) > 1

    def get_rehearsal_data_ratio(self) -> float:
        """获取旧任务数据重放比例"""
        if len(self.available_tasks) <= 1:
            return 0.0

        # 根据任务数量调整重放比例
        return min(0.3, 0.1 * len(self.available_tasks))

    def validate_training_readiness(self) -> Tuple[bool, List[str]]:
        """验证训练准备状态"""
        issues = []

        if not self.available_tasks:
            issues.append("没有可用的任务数据")

        for task_id in self.available_tasks:
            task_info = self.task_definitions[task_id]
            if not task_info.data_available:
                issues.append(f"任务{task_id}数据未正确注册")
            if task_info.episode_count == 0:
                issues.append(f"任务{task_id}没有有效的episode数据")

        return len(issues) == 0, issues

    def save_training_state(self, output_dir: Path, epoch: int, performance_stats: Dict[str, Any]):
        """保存任务特定训练状态"""
        state = {
            "training_phase": self.current_training_phase,
            "available_tasks": self.available_tasks,
            "task_definitions": {tid: {
                "name": info.name,
                "complexity_level": info.complexity_level,
                "required_layers": info.required_layers,
                "data_available": info.data_available,
                "episode_count": info.episode_count
            } for tid, info in self.task_definitions.items()},
            "current_config": self.prepare_training_config_for_phase(self.current_training_phase),
            "epoch": epoch,
            "performance_stats": performance_stats,
            "timestamp": torch.get_rng_state().tolist()  # 保存随机状态
        }

        state_file = output_dir / "task_training_state.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        self.logger.info(f"💾 任务训练状态已保存到: {state_file}")

    def load_training_state(self, state_file: Path) -> bool:
        """加载任务特定训练状态"""
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            self.current_training_phase = state.get("training_phase", 1)
            self.available_tasks = state.get("available_tasks", [])

            # 恢复任务信息
            saved_tasks = state.get("task_definitions", {})
            for task_id_str, task_data in saved_tasks.items():
                task_id = int(task_id_str)
                if task_id in self.task_definitions:
                    self.task_definitions[task_id].data_available = task_data["data_available"]
                    self.task_definitions[task_id].episode_count = task_data["episode_count"]

            self.logger.info(f"✅ 任务训练状态已从 {state_file} 恢复")
            return True

        except Exception as e:
            self.logger.error(f"❌ 加载训练状态失败: {e}")
            return False

    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练总结信息"""
        return {
            "current_phase": self.current_training_phase,
            "available_tasks": [
                {
                    "id": tid,
                    "name": self.task_definitions[tid].name,
                    "complexity": self.task_definitions[tid].complexity_level,
                    "episodes": self.task_definitions[tid].episode_count,
                    "required_layers": self.task_definitions[tid].required_layers
                }
                for tid in self.available_tasks
            ],
            "total_episodes": sum(self.task_definitions[tid].episode_count for tid in self.available_tasks),
            "curriculum_stages": len(self.get_current_curriculum_stages()),
            "anti_forgetting_enabled": self.should_enable_anti_forgetting(),
            "rehearsal_ratio": self.get_rehearsal_data_ratio()
        }

    def print_training_plan(self):
        """打印训练计划"""
        summary = self.get_training_summary()

        print("🎯 任务特定训练计划")
        print("=" * 60)
        print(f"当前训练阶段: {summary['current_phase']}")
        print(f"可用任务数量: {len(summary['available_tasks'])}")
        print(f"总episode数: {summary['total_episodes']}")
        print(f"课程学习阶段: {summary['curriculum_stages']}")
        print(f"防遗忘机制: {'启用' if summary['anti_forgetting_enabled'] else '禁用'}")

        print("\n📋 任务详情:")
        for task in summary['available_tasks']:
            print(
                f"  任务{task['id']}: {task['name']} (复杂度: {task['complexity']}/4)")
            print(
                f"    Episodes: {task['episodes']}, 需要层: {task['required_layers']}")

        if summary['anti_forgetting_enabled']:
            print(f"\n🔄 重放比例: {summary['rehearsal_ratio']:.1%}")

        print()
