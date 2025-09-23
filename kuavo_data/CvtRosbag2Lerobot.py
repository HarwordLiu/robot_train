"""
Script to convert Kuavo rosbag data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""
import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
import dataclasses
from pathlib import Path
import shutil
import hydra
from omegaconf import DictConfig
from typing import Literal
import sys
import os
from rich.logging import RichHandler
import logging
import resource

logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)


from pympler import asizeof
import matplotlib.pyplot as plt

def get_attr_sizes(obj, prefix=""):
    """递归获取对象每个属性及嵌套属性的内存占用"""
    sizes = {}
    for attr in dir(obj):
        if attr.startswith("__"):
            continue
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        key = f"{prefix}.{attr}" if prefix else attr
        size = asizeof.asizeof(value)
        sizes[key] = size
        # 如果是自定义类实例，递归获取
        if hasattr(value, "__dict__"):
            sizes.update(get_attr_sizes(value, prefix=key))
    return sizes

def visualize_memory(attr_sizes, top_n=20):
    """可视化内存占用"""
    # 按大小排序，取前 top_n
    sorted_attrs = sorted(attr_sizes.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels, sizes = zip(*sorted_attrs)
    sizes_kb = [s / 1024 /1024 for s in sizes]

    plt.figure(figsize=(12, 6))
    plt.barh(labels[::-1], sizes_kb[::-1])
    plt.xlabel("Memory (MB)")
    plt.title(f"Top {top_n} attributes by memory usage")
    plt.tight_layout()
    plt.show()




log_print = logging.getLogger("rich")

try:
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    log_print.warning("import lerobot.common.xxx will be deprecated in lerobot v2.0, please use lerobot.xxx instead in the future.")
except Exception as import_error:
    try:
        import lerobot
    except Exception as lerobot_error:
        log_print.error("Error: lerobot package not found. Please change to 'third_party/lerobot' and install it using 'pip install -e .'.")
        sys.exit(1)
    log_print.info("Error: "+ str(import_error))
    log_print.info("Import lerobot.common.xxx is deprecated in lerobot v2.0, try to use import lerobot.xxx instead ...")
    try:
        from lerobot.datasets.lerobot_dataset import LEROBOT_HOME
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        log_print.info("import lerobot.datasets.lerobot_dataset ok!")
    except Exception as import_failed:
        log_print.info("Error:"+str(import_failed))
        if "LEROBOT_HOME" in str(import_failed):
            try:
                from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
                log_print.info("import lerobot.datasets.lerobot_dataset HF_LEROBOT_HOME,  LeRobotDataset ok!")
            except Exception as e:
                log_print.error(str(e))
                sys.exit(1)


import numpy as np
import torch
import tqdm
import json

import common.kuavo_dataset as kuavo
import glob


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None

DEFAULT_DATASET_CONFIG = DatasetConfig()


def clear_episode_range(dataset_path: Path, start_idx: int, end_idx: int):
    """
    清除指定范围内的episode数据文件和元数据

    Args:
        dataset_path: 数据集目录路径
        start_idx: 起始episode索引
        end_idx: 结束episode索引
    """
    if not dataset_path.exists():
        log_print.info(f"Dataset path {dataset_path} does not exist, skipping range clearing")
        return

    log_print.info(f"Clearing episodes {start_idx} to {end_idx} from {dataset_path}")

    # 清除数据文件 (parquet格式: data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet)
    data_dir = dataset_path / "data"
    if data_dir.exists():
        for ep_idx in range(start_idx, end_idx + 1):
            # 计算chunk编号 (假设每个chunk包含1000个episode)
            episode_chunk = ep_idx // 1000
            episode_file = data_dir / f"chunk-{episode_chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
            if episode_file.exists():
                episode_file.unlink()
                log_print.info(f"Removed episode file: {episode_file}")

            # 清除对应的视频文件
            videos_dir = dataset_path / "videos"
            if videos_dir.exists():
                episode_video_pattern = f"episode_{ep_idx:06d}_*.mp4"
                for video_file in videos_dir.glob(f"**/{episode_video_pattern}"):
                    video_file.unlink()
                    log_print.info(f"Removed video file: {video_file}")

            # 清除对应的图像文件 (格式: images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png)
            images_dir = dataset_path / "images"
            if images_dir.exists():
                episode_dir_pattern = f"episode_{ep_idx:06d}"
                # 遍历所有camera/image_key目录
                for image_key_dir in images_dir.iterdir():
                    if image_key_dir.is_dir():
                        episode_image_dir = image_key_dir / episode_dir_pattern
                        if episode_image_dir.exists():
                            # 删除该episode的所有图像文件
                            for image_file in episode_image_dir.glob("frame_*.png"):
                                image_file.unlink()
                                log_print.info(f"Removed image file: {image_file}")
                            # 删除空的episode目录
                            if not any(episode_image_dir.iterdir()):
                                episode_image_dir.rmdir()
                                log_print.info(f"Removed empty episode directory: {episode_image_dir}")

    # 清除空的chunk目录
    if data_dir.exists():
        for chunk_dir in data_dir.glob("chunk-*"):
            if chunk_dir.is_dir() and not any(chunk_dir.iterdir()):
                chunk_dir.rmdir()
                log_print.info(f"Removed empty chunk directory: {chunk_dir}")

    # 更新meta文件
    _update_meta_files(dataset_path, start_idx, end_idx)

    log_print.info(f"Finished clearing episodes {start_idx} to {end_idx}")


def _update_meta_files(dataset_path: Path, start_idx: int, end_idx: int):
    """
    更新meta目录中的元数据文件，移除指定范围的episode记录

    Args:
        dataset_path: 数据集目录路径
        start_idx: 起始episode索引
        end_idx: 结束episode索引
    """
    meta_dir = dataset_path / "meta"
    if not meta_dir.exists():
        log_print.info("Meta directory does not exist, skipping meta file updates")
        return

    # 更新 episodes.jsonl
    episodes_file = meta_dir / "episodes.jsonl"
    if episodes_file.exists():
        log_print.info("Updating episodes.jsonl")
        lines = []
        with open(episodes_file, 'r') as f:
            for line in f:
                episode_data = json.loads(line.strip())
                # 保留不在删除范围内的episode记录
                if not (start_idx <= episode_data.get('episode_index', -1) <= end_idx):
                    lines.append(line.strip())

        # 重写文件
        with open(episodes_file, 'w') as f:
            for line in lines:
                f.write(line + '\n')
        log_print.info(f"Updated episodes.jsonl, removed {end_idx - start_idx + 1} episode records")

    # 更新 episodes_stats.jsonl
    episodes_stats_file = meta_dir / "episodes_stats.jsonl"
    if episodes_stats_file.exists():
        log_print.info("Updating episodes_stats.jsonl")
        lines = []
        with open(episodes_stats_file, 'r') as f:
            for line in f:
                stats_data = json.loads(line.strip())
                # 保留不在删除范围内的episode统计记录
                if not (start_idx <= stats_data.get('episode_index', -1) <= end_idx):
                    lines.append(line.strip())

        # 重写文件
        with open(episodes_stats_file, 'w') as f:
            for line in lines:
                f.write(line + '\n')
        log_print.info(f"Updated episodes_stats.jsonl, removed {end_idx - start_idx + 1} episode stats records")

    # 更新 info.json 中的总episode数和帧数
    info_file = meta_dir / "info.json"
    if info_file.exists():
        log_print.info("Updating info.json")
        try:
            with open(info_file, 'r') as f:
                info_data = json.load(f)
        except json.JSONDecodeError as e:
            log_print.error(f"Error reading info.json: {e}")
            log_print.error("info.json file appears to be corrupted. Skipping info.json update.")
            return

        # 计算被删除的episodes的总帧数和任务数
        removed_frames = 0
        removed_tasks = set()
        episodes_stats_file = meta_dir / "episodes_stats.jsonl"
        if episodes_stats_file.exists():
            with open(episodes_stats_file, 'r') as f:
                for line in f:
                    stats_data = json.loads(line.strip())
                    episode_idx = stats_data.get('episode_index', -1)
                    if start_idx <= episode_idx <= end_idx:
                        removed_frames += stats_data.get('length', 0)
                        # 收集被删除episodes的任务类型
                        task = stats_data.get('task', None)
                        if task:
                            removed_tasks.add(task)

        # 计算删除后剩余的任务数
        remaining_tasks = set()
        if episodes_stats_file.exists():
            with open(episodes_stats_file, 'r') as f:
                for line in f:
                    stats_data = json.loads(line.strip())
                    episode_idx = stats_data.get('episode_index', -1)
                    if not (start_idx <= episode_idx <= end_idx):  # 保留的episodes
                        task = stats_data.get('task', None)
                        if task:
                            remaining_tasks.add(task)

        removed_count = end_idx - start_idx + 1

        # 更新total_episodes
        if 'total_episodes' in info_data:
            info_data['total_episodes'] = max(0, info_data['total_episodes'] - removed_count)
            log_print.info(f"Updated total_episodes: reduced by {removed_count}")

        # 更新total_frames
        if 'total_frames' in info_data:
            info_data['total_frames'] = max(0, info_data['total_frames'] - removed_frames)
            log_print.info(f"Updated total_frames: reduced by {removed_frames}")

        # 更新total_tasks
        if 'total_tasks' in info_data:
            info_data['total_tasks'] = len(remaining_tasks)
            log_print.info(f"Updated total_tasks: now {info_data['total_tasks']}")

        # 更新total_chunks (基于剩余的episodes计算)
        if 'total_episodes' in info_data and 'chunks_size' in info_data:
            chunks_size = info_data.get('chunks_size', 1000)
            total_episodes = info_data['total_episodes']
            info_data['total_chunks'] = (total_episodes + chunks_size - 1) // chunks_size if total_episodes > 0 else 0
            log_print.info(f"Updated total_chunks: now {info_data['total_chunks']}")

        # 更新total_videos (如果使用视频)
        if 'total_videos' in info_data and 'video_path' in info_data and info_data['video_path']:
            # 对于kuavo数据集，每个frame都有对应的视频帧，所以total_videos等于total_frames
            info_data['total_videos'] = info_data.get('total_frames', 0)
            log_print.info(f"Updated total_videos: now {info_data['total_videos']}")

        try:
            with open(info_file, 'w') as f:
                json.dump(info_data, f, indent=2)
            log_print.info("Updated info.json")
        except Exception as e:
            log_print.error(f"Error writing info.json: {e}")
            log_print.error("Failed to update info.json")


def clear_all_dataset(dataset_path: Path):
    """
    完全清除数据集

    Args:
        dataset_path: 数据集目录路径
    """
    if dataset_path.exists():
        log_print.info(f"Clearing entire dataset: {dataset_path}")
        shutil.rmtree(dataset_path)
        log_print.info("Dataset cleared successfully")


def get_cameras(bag_data: dict) -> list[str]:
    """
    /cam_l/color/image_raw/compressed                    : sensor_msgs/CompressedImage                
    /cam_r/color/image_raw/compressed                    : sensor_msgs/CompressedImage                
    /zedm/zed_node/left/image_rect_color/compressed      : sensor_msgs/CompressedImage                
    /zedm/zed_node/right/image_rect_color/compressed     : sensor_msgs/CompressedImage 
    """
    cameras = []

    for k in kuavo.DEFAULT_CAMERA_NAMES:
        cameras.append(k)
    return cameras

def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "image",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
    clear_dataset: bool = True,
) -> LeRobotDataset:
    
    # 根据config的参数决定是否为半身和末端的关节类型
    motors = DEFAULT_JOINT_NAMES_LIST
    # TODO: auto detect cameras
    cameras = kuavo.DEFAULT_CAMERA_NAMES

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": {
                "motors": motors
            }
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": {
                "motors": motors
            }
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        if 'depth' in cam:
            features[f"observation.{cam}"] = {
                "dtype": "uint16", 
                "shape": (1, kuavo.RESIZE_H, kuavo.RESIZE_W),  # Attention: for datasets.features "image" and "video", it must be c,h,w style! 
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }
        else:
            features[f"observation.images.{cam}"] = {
                "dtype": mode,
                "shape": (3, kuavo.RESIZE_H, kuavo.RESIZE_W),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            }

    if clear_dataset and Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=kuavo.TRAIN_HZ,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
        root=root,
    )

def load_raw_images_per_camera(bag_data: dict) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in get_cameras(bag_data):
        imgs_per_cam[camera] = np.array([msg['data'] for msg in bag_data[camera]])
        # print(f"camera {camera} image", imgs_per_cam[camera].shape)
    
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

    bag_reader = kuavo.KuavoRosbagReader()
    bag_data = bag_reader.process_rosbag(ep_path)
    
    state = np.array([msg['data'] for msg in bag_data['observation.state']], dtype=np.float32)
    action = np.array([msg['data'] for msg in bag_data['action']], dtype=np.float32)
    action_kuavo_arm_traj = np.array([msg['data'] for msg in bag_data['action.kuavo_arm_traj']], dtype=np.float32)
    claw_state = np.array([msg['data'] for msg in bag_data['observation.claw']], dtype=np.float64)
    claw_action= np.array([msg['data'] for msg in bag_data['action.claw']], dtype=np.float64)
    qiangnao_state = np.array([msg['data'] for msg in bag_data['observation.qiangnao']], dtype=np.float64)
    qiangnao_action= np.array([msg['data'] for msg in bag_data['action.qiangnao']], dtype=np.float64)
    rq2f85_state = np.array([msg['data'] for msg in bag_data['observation.rq2f85']], dtype=np.float64)
    rq2f85_action= np.array([msg['data'] for msg in bag_data['action.rq2f85']], dtype=np.float64)
    # print("eef_type shape: ",claw_action.shape,qiangnao_action.shape, rq2f85_action.shape)
    action[:, 12:26] = action_kuavo_arm_traj    

    velocity = None
    effort = None
    
    imgs_per_cam = load_raw_images_per_camera(bag_data)
    
    return imgs_per_cam, state, action, velocity, effort ,claw_state ,claw_action,qiangnao_state,qiangnao_action, rq2f85_state, rq2f85_action


def diagnose_frame_data(data):
    for k, v in data.items():
        print(f"Field: {k}")
        print(f"  Shape    : {v.shape}")
        print(f"  Dtype    : {v.dtype}")
        print(f"  Type     : {type(v).__name__}")
        print("-" * 40)


def populate_dataset(
    dataset: LeRobotDataset,
    bag_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
    episode_start_idx: int = 0,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(bag_files))
    failed_bags = []
    for i, ep_idx in enumerate(tqdm.tqdm(episodes)):
        ep_path = bag_files[ep_idx]
        # 计算实际的episode索引
        actual_episode_idx = episode_start_idx + i
        from termcolor import colored
        print(colored(f"Processing {ep_path} -> Episode {actual_episode_idx}", "yellow", attrs=["bold"]))
        # 默认读取所有的数据如果话题不存在相应的数值应该是一个空的数据
        try:
            imgs_per_cam, state, action, velocity, effort ,claw_state, claw_action,qiangnao_state,qiangnao_action, rq2f85_state, rq2f85_action = load_raw_episode_data(ep_path)
        except Exception as e:
            print(f"❌ Error processing {ep_path}: {e}")
            failed_bags.append(str(ep_path))
            continue
        # 对手部进行二值化处理
        if kuavo.IS_BINARY:
            qiangnao_state = np.where(qiangnao_state > 50, 1, 0)
            qiangnao_action = np.where(qiangnao_action > 50, 1, 0)
            claw_state = np.where(claw_state > 50, 1, 0)
            claw_action = np.where(claw_action > 50, 1, 0)
            rq2f85_state = np.where(rq2f85_state > 0.4, 1, 0)
            rq2f85_action = np.where(rq2f85_action > 70, 1, 0)
        else:
            # 进行数据归一化处理
            claw_state = claw_state / 100
            claw_action = claw_action / 100
            qiangnao_state = qiangnao_state / 100
            qiangnao_action = qiangnao_action / 100
            rq2f85_state = rq2f85_state / 0.8
            rq2f85_action = rq2f85_action / 140
        print("eef_type shape: ",claw_action.shape,qiangnao_action.shape, rq2f85_action.shape)
        if len(claw_action)==0 and len(qiangnao_action) == 0:
            claw_action = rq2f85_action
            claw_state = rq2f85_state
        ########################
        # delta 处理
        ########################
        # =====================
        # 为了解决零点问题，将每帧与第一帧相减
        if kuavo.RELATIVE_START:
            # 每个state, action与他们的第一帧相减
            state = state - state[0]
            action = action - action[0]
            
        # ===只处理delta action
        if kuavo.DELTA_ACTION:
            # delta_action = action[1:] - state[:-1]
            # trim = lambda x: x[1:] if (x is not None) and (len(x) > 0) else x
            # state, action, velocity, effort, claw_state, claw_action, qiangnao_state, qiangnao_action = \
            #     map(
            #         trim, 
            #         [state, action, velocity, effort, claw_state, claw_action, qiangnao_state, qiangnao_action]
            #         )
            # for camera, img_array in imgs_per_cam.items():
            #     imgs_per_cam[camera] = img_array[1:]
            # action = delta_action

            # delta_action = np.concatenate(([action[0]-state[0]], action[1:] - action[:-1]), axis=0)
            # action = delta_action

            delta_action = action-state
            action = delta_action

        # 为当前episode创建新的buffer并指定episode索引
        dataset.episode_buffer = dataset.create_episode_buffer(episode_index=actual_episode_idx)

        num_frames = state.shape[0]
        for i in range(num_frames):
            if kuavo.ONLY_HALF_UP_BODY:
                if kuavo.USE_LEJU_CLAW:
                    # 使用lejuclaw进行上半身关节数据转换
                    if kuavo.CONTROL_HAND_SIDE == "left" or kuavo.CONTROL_HAND_SIDE == "both":
                        output_state = state[i, kuavo.SLICE_ROBOT[0][0]:kuavo.SLICE_ROBOT[0][-1]]
                        output_state = np.concatenate((output_state, claw_state[i, kuavo.SLICE_CLAW[0][0]:kuavo.SLICE_CLAW[0][-1]].astype(np.float32)), axis=0)
                        output_action = action[i, kuavo.SLICE_ROBOT[0][0]:kuavo.SLICE_ROBOT[0][-1]]
                        output_action = np.concatenate((output_action, claw_action[i, kuavo.SLICE_CLAW[0][0]:kuavo.SLICE_CLAW[0][-1]].astype(np.float32)), axis=0)
                    if kuavo.CONTROL_HAND_SIDE == "right" or kuavo.CONTROL_HAND_SIDE == "both":
                        if kuavo.CONTROL_HAND_SIDE == "both":
                            output_state = np.concatenate((output_state, state[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]), axis=0)
                            output_state = np.concatenate((output_state, claw_state[i, kuavo.SLICE_CLAW[1][0]:kuavo.SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)
                            output_action = np.concatenate((output_action, action[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]), axis=0)
                            output_action = np.concatenate((output_action, claw_action[i, kuavo.SLICE_CLAW[1][0]:kuavo.SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)
                        else:
                            output_state = state[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]
                            output_state = np.concatenate((output_state, claw_state[i, kuavo.SLICE_CLAW[1][0]:kuavo.SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)
                            output_action = action[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]
                            output_action = np.concatenate((output_action, claw_action[i, kuavo.SLICE_CLAW[1][0]:kuavo.SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)

                elif kuavo.USE_QIANGNAO:
                    # 类型: kuavo_sdk/robotHandPosition
                    # left_hand_position (list of float): 左手位置，包含6个元素，每个元素的取值范围为[0, 100], 0 为张开，100 为闭合。
                    # right_hand_position (list of float): 右手位置，包含6个元素，每个元素的取值范围为[0, 100], 0 为张开，100 为闭合。
                    # 构造qiangnao类型的output_state的数据结构的长度应该为26
                    if kuavo.CONTROL_HAND_SIDE == "left" or kuavo.CONTROL_HAND_SIDE == "both":
                        output_state = state[i, kuavo.SLICE_ROBOT[0][0]:kuavo.SLICE_ROBOT[0][-1]]
                        output_state = np.concatenate((output_state, qiangnao_state[i, kuavo.SLICE_DEX[0][0]:kuavo.SLICE_DEX[0][-1]].astype(np.float32)), axis=0)

                        output_action = action[i, kuavo.SLICE_ROBOT[0][0]:kuavo.SLICE_ROBOT[0][-1]]
                        output_action = np.concatenate((output_action, qiangnao_action[i, kuavo.SLICE_DEX[0][0]:kuavo.SLICE_DEX[0][-1]].astype(np.float32)), axis=0)
                    if kuavo.CONTROL_HAND_SIDE == "right" or kuavo.CONTROL_HAND_SIDE == "both":
                        if kuavo.CONTROL_HAND_SIDE == "both":
                            output_state = np.concatenate((output_state, state[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]), axis=0)
                            output_state = np.concatenate((output_state, qiangnao_state[i, kuavo.SLICE_DEX[1][0]:kuavo.SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                            output_action = np.concatenate((output_action, action[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]), axis=0)
                            output_action = np.concatenate((output_action, qiangnao_action[i, kuavo.SLICE_DEX[1][0]:kuavo.SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                        else:
                            output_state = state[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]
                            output_state = np.concatenate((output_state, qiangnao_state[i, kuavo.SLICE_DEX[1][0]:kuavo.SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                            output_action = action[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]
                            output_action = np.concatenate((output_action, qiangnao_action[i, kuavo.SLICE_DEX[1][0]:kuavo.SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                    # output_action = np.concatenate((output_action, action[i, 26:28]), axis=0)
            else:
                if kuavo.USE_LEJU_CLAW:
                    # 使用lejuclaw进行全身关节数据转换
                    # 原始的数据是28个关节的数据对应原始的state和action数据的长度为28
                    # 数据顺序:
                    # 前 12 个数据为下肢电机数据:
                    #     0~5 为左下肢数据 (l_leg_roll, l_leg_yaw, l_leg_pitch, l_knee, l_foot_pitch, l_foot_roll)
                    #     6~11 为右下肢数据 (r_leg_roll, r_leg_yaw, r_leg_pitch, r_knee, r_foot_pitch, r_foot_roll)
                    # 接着 14 个数据为手臂电机数据:
                    #     12~18 左臂电机数据 ("l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch", "l_hand_yaw", "l_hand_pitch", "l_hand_roll")
                    #     19~25 为右臂电机数据 ("r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch", "r_hand_yaw", "r_hand_pitch", "r_hand_roll")
                    # 最后 2 个为头部电机数据: head_yaw 和 head_pitch
                    
                    # TODO：构造目标切片
                    output_state = state[i, 0:19]
                    output_state = np.insert(output_state, 19, claw_state[i, 0].astype(np.float32))
                    output_state = np.concatenate((output_state, state[i, 19:26]), axis=0)
                    output_state = np.insert(output_state, 19, claw_state[i, 1].astype(np.float32))
                    output_state = np.concatenate((output_state, state[i, 26:28]), axis=0)

                    output_action = action[i, 0:19]
                    output_action = np.insert(output_action, 19, claw_action[i, 0].astype(np.float32))
                    output_action = np.concatenate((output_action, action[i, 19:26]), axis=0)
                    output_action = np.insert(output_action, 19, claw_action[i, 1].astype(np.float32))
                    output_action = np.concatenate((output_action, action[i, 26:28]), axis=0)

                elif kuavo.USE_QIANGNAO:
                    output_state = state[i, 0:19]
                    output_state = np.concatenate((output_state, qiangnao_state[i, 0:6].astype(np.float32)), axis=0)
                    output_state = np.concatenate((output_state, state[i, 19:26]), axis=0)
                    output_state = np.concatenate((output_state, qiangnao_state[i, 6:12].astype(np.float32)), axis=0)
                    output_state = np.concatenate((output_state, state[i, 26:28]), axis=0)

                    output_action = action[i, 0:19]
                    output_action = np.concatenate((output_action, qiangnao_action[i, 0:6].astype(np.float32)),axis=0)
                    output_action = np.concatenate((output_action, action[i, 19:26]), axis=0)
                    output_action = np.concatenate((output_action, qiangnao_action[i, 6:12].astype(np.float32)), axis=0)
                    output_action = np.concatenate((output_action, action[i, 26:28]), axis=0)  
            frame = {
                "observation.state": torch.from_numpy(output_state).type(torch.float32),
                "action": torch.from_numpy(output_action).type(torch.float32),
            }
            
            for camera, img_array in imgs_per_cam.items():
                if "depth" in camera:
                    # frame[f"observation.{camera}"] = img_array[i]
                    min_depth, max_dpeth = kuavo.DEPTH_RANGE[0], kuavo.DEPTH_RANGE[1]
                    frame[f"observation.{camera}"] = np.clip(img_array[i], min_depth, max_dpeth)
                    print("[info]: Clip depth in range %d ~ %d"%(min_depth, max_dpeth))
                else:
                    frame[f"observation.images.{camera}"] = img_array[i]
            
            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            # frame["task"] = task
            # diagnose_frame_data(frame)
            dataset.add_frame(frame, task=task)
        # dataset.save_episode(task="Pick the black workpiece from the white conveyor belt on your left and place it onto the white box in front of you",)
        # raise ValueError("stop!")


        # usage = resource.getrusage(resource.RUSAGE_SELF)
        # print(f"~~~~~~~~~~~~~~Before Memory usage: {usage.ru_maxrss / 1024:.2f} MB")
        # print(dataset.episode_buffer)
        dataset.save_episode()
        # usage = resource.getrusage(resource.RUSAGE_SELF)
        # print(f"~~~~~~~~~~~~~~After Memory usage: {usage.ru_maxrss / 1024:.2f} MB")
        # print(dataset.episode_buffer)
        # sizes = get_attr_sizes(dataset)
        # for k, v in sizes.items():
        #     print(f"{k}: {v/1024/1024:.2f} MB")
        # visualize_memory(sizes, top_n=10)
        # dataset.hf_dataset = None  # reduce memory usage in data convert
        # del dataset.hf_dataset
        dataset.hf_dataset = dataset.create_hf_dataset()  # reset to reduce memory usage in data convert

    # 将失败的bag文件写入error.txt
    if failed_bags:
        with open("error.txt", "w") as f:
            for bag in failed_bags:
                f.write(bag + "\n")
        print(f"❌ {len(failed_bags)} failed bags written to error.txt")

    return dataset
            


def port_kuavo_rosbag(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: str,
    n: int | None = None,
    episode_range: list[int] | None = None,
    clear_dataset: bool = True,
):
    # 根据配置决定是否清理数据
    dataset_path = Path(root)

    if episode_range is not None:
        # 清理指定范围的episode数据
        start_idx, end_idx = episode_range
        clear_episode_range(dataset_path, start_idx, end_idx)
    elif clear_dataset:
        # 完全清理数据集
        clear_all_dataset(dataset_path)

    bag_reader = kuavo.KuavoRosbagReader()
    bag_files = bag_reader.list_bag_files(raw_dir)

    # 处理episode范围或bag文件数量限制
    episode_start_idx = 0
    if episode_range is not None:
        start_idx, end_idx = episode_range
        episode_start_idx = start_idx
        # 选择对应范围的bag文件
        num_episodes = end_idx - start_idx + 1
        if len(bag_files) < num_episodes:
            log_print.warning(f"Warning: Requested {num_episodes} episodes, but only {len(bag_files)} bag files are available.")
            num_episodes = len(bag_files)
        bag_files = bag_files[:num_episodes]
        episodes = list(range(num_episodes))
    elif isinstance(n, int) and n > 0:
        num_available_bags = len(bag_files)
        if n > num_available_bags:
            log_print.warning(f"Warning: Requested {n} bags, but only {num_available_bags} are available. Using all available bags.")
            n = num_available_bags

        # random sample num_of_bag files
        select_idx = np.random.choice(num_available_bags, n, replace=False)
        bag_files = [bag_files[i] for i in select_idx]
        episodes = None
    else:
        episodes = None
    
    # 根据是否需要完全清理来决定创建还是加载数据集
    if episode_range is None:
        # 完全清理模式：创建新数据集
        dataset = create_empty_dataset(
            repo_id,
            robot_type="kuavo4pro",
            mode=mode,
            has_effort=False,
            has_velocity=False,
            dataset_config=dataset_config,
            root=root,
            clear_dataset=True,
        )
    else:
        # 范围清理模式：尝试加载现有数据集，如果不存在则创建新的
        dataset_path = Path(root) / repo_id
        if dataset_path.exists():
            log_print.info(f"Loading existing dataset from {dataset_path}")
            dataset = LeRobotDataset(repo_id, root=root)
        else:
            log_print.info(f"Creating new dataset at {dataset_path}")
            dataset = create_empty_dataset(
                repo_id,
                robot_type="kuavo4pro",
                mode=mode,
                has_effort=False,
                has_velocity=False,
                dataset_config=dataset_config,
                root=root,
                clear_dataset=False,
            )
    dataset = populate_dataset(
        dataset,
        bag_files,
        task=task,
        episodes=episodes,
        episode_start_idx=episode_start_idx,
    )
    # dataset.consolidate()
    
@hydra.main(config_path="../configs/data/", config_name="KuavoRosbag2Lerobot", version_base=None)
def main(cfg: DictConfig):

    global DEFAULT_JOINT_NAMES_LIST
    kuavo.init_parameters(cfg)

    n = cfg.rosbag.num_used
    raw_dir = cfg.rosbag.rosbag_dir
    version = cfg.rosbag.lerobot_dir
    episode_range = cfg.rosbag.episode_range

    task_name = os.path.basename(raw_dir)
    repo_id = f'lerobot/{task_name}'
    lerobot_dir = os.path.join(raw_dir,"../",version,"lerobot")

    # 根据episode_range决定清理策略
    clear_dataset = episode_range is None
    
    half_arm = len(kuavo.DEFAULT_ARM_JOINT_NAMES) // 2
    half_claw = len(kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES) // 2
    half_dexhand = len(kuavo.DEFAULT_DEXHAND_JOINT_NAMES) // 2
    UP_START_INDEX = 12
    if kuavo.ONLY_HALF_UP_BODY:
        if kuavo.USE_LEJU_CLAW:
            DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw] \
                                    + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
            arm_slice = [
                (kuavo.SLICE_ROBOT[0][0] - UP_START_INDEX, kuavo.SLICE_ROBOT[0][-1] - UP_START_INDEX),(kuavo.SLICE_CLAW[0][0] + half_arm, kuavo.SLICE_CLAW[0][-1] + half_arm), 
                (kuavo.SLICE_ROBOT[1][0] - UP_START_INDEX + half_claw, kuavo.SLICE_ROBOT[1][-1] - UP_START_INDEX + half_claw), (kuavo.SLICE_CLAW[1][0] + half_arm * 2, kuavo.SLICE_CLAW[1][-1] + half_arm * 2)
                ]
        elif kuavo.USE_QIANGNAO:  
            DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand] \
                                    + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[half_dexhand:]               
            arm_slice = [
                (kuavo.SLICE_ROBOT[0][0] - UP_START_INDEX, kuavo.SLICE_ROBOT[0][-1] - UP_START_INDEX),(kuavo.SLICE_DEX[0][0] + half_arm, kuavo.SLICE_DEX[0][-1] + half_arm), 
                (kuavo.SLICE_ROBOT[1][0] - UP_START_INDEX + half_dexhand, kuavo.SLICE_ROBOT[1][-1] - UP_START_INDEX + half_dexhand), (kuavo.SLICE_DEX[1][0] + half_arm * 2, kuavo.SLICE_DEX[1][-1] + half_arm * 2)
                ]
        DEFAULT_JOINT_NAMES_LIST = [DEFAULT_ARM_JOINT_NAMES[k] for l, r in arm_slice for k in range(l, r)]  
    else:
        if kuavo.USE_LEJU_CLAW:
            DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw] \
                                    + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
        elif kuavo.USE_QIANGNAO:
            DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand] \
                                    + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[half_dexhand:]             
        DEFAULT_JOINT_NAMES_LIST = kuavo.DEFAULT_LEG_JOINT_NAMES + DEFAULT_ARM_JOINT_NAMES + kuavo.DEFAULT_HEAD_JOINT_NAMES

    port_kuavo_rosbag(
        raw_dir,
        repo_id,
        root=lerobot_dir,
        n=n,
        task=kuavo.TASK_DESCRIPTION,
        episode_range=episode_range,
        clear_dataset=clear_dataset
    )

if __name__ == "__main__":
    
    main()
    

    