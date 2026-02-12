#!/usr/bin/env python

# Copyright (c) 2025, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from isaaclab_arena_gr00t.config.dataset_config import Gr00tDatasetConfig
from isaaclab_arena_gr00t.data_utils.image_conversion import resize_frames_with_padding
from isaaclab_arena_gr00t.data_utils.io_utils import (
    create_config_from_yaml,
    load_config_from_yaml,
    load_json,
    load_robot_joints_config_from_yaml,
)
from isaaclab_arena_gr00t.data_utils.joints_conversion import remap_sim_joints_to_policy_joints
from isaaclab_arena_gr00t.data_utils.robot_eef_pose import EefPose
from isaaclab_arena_gr00t.data_utils.robot_joints import JointsAbsPosition
from lerobot.datasets.lerobot_dataset import LeRobotDataset


LOGGER = logging.getLogger(__name__)


def _episode_sort_key(episode_name: str) -> tuple[int, str]:
    """Sort episodes numerically by the trailing index in demo_* names."""
    parts = episode_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (int(parts[1]), episode_name)
    return (1_000_000_000, episode_name)


def _build_policy_joint_names(config: Gr00tDatasetConfig) -> list[str]:
    policy_joints_config = load_robot_joints_config_from_yaml(config.policy_joints_config_path)
    policy_joints_names = []
    for joint_group in policy_joints_config.keys():
        for joint_name in policy_joints_config[joint_group]:
            policy_joints_names.append(joint_name)
    return policy_joints_names


def _build_video_feature(video_shape: tuple[int, int, int], fps: int, vcodec: str) -> dict[str, Any]:
    height, width, channels = video_shape
    return {
        "dtype": "video",
        "shape": [channels, height, width],
        "names": ["channel", "height", "width"],
        "video_info": {
            "video.width": width,
            "video.height": height,
            "video.fps": fps,
            "video.codec": vcodec,
            "video.pix_fmt": "yuv420p",
            "video.channels": channels,
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def _infer_features(
    example_episode: dict[str, Any],
    video_shapes: dict[str, tuple[int, int, int]],
    config: Gr00tDatasetConfig,
    vcodec: str,
) -> dict[str, dict]:
    policy_joints_names = _build_policy_joint_names(config)
    features: dict[str, dict] = {}

    for key, value in example_episode.items():
        if key in video_shapes:
            features[key] = _build_video_feature(video_shapes[key], config.fps, vcodec)
            continue

        if isinstance(value, list):
            dtype = "string"
            shape = (1,)
        else:
            array = np.asarray(value)
            if array.dtype.kind in {"U", "S", "O"}:
                dtype = "string"
                shape = (1,)
            else:
                dtype = array.dtype.name
                shape = tuple(array.shape[1:]) if array.ndim > 1 else (1,)

        feature = {"dtype": dtype, "shape": shape}

        if key in (config.lerobot_keys["state"], config.lerobot_keys["action"]):
            feature["names"] = [name for name in policy_joints_names]

        features[key] = feature

    return features


def _resolve_video_mappings(
    trajectory: h5py.Group, config: Gr00tDatasetConfig, policy_modality_config: dict[str, Any]
) -> list[tuple[str, str]]:
    camera_obs = trajectory.get("camera_obs", None)
    if camera_obs is None:
        raise KeyError("HDF5 trajectory is missing 'camera_obs' group")

    camera_keys = set(camera_obs.keys())
    mappings: list[tuple[str, str]] = []

    video_config = policy_modality_config.get("video", {})
    if video_config:
        for video_name, video_entry in video_config.items():
            lerobot_key = video_entry.get("original_key", video_name)

            candidate_names = []
            if video_name:
                candidate_names.append(f"{video_name}_rgb")
            if isinstance(lerobot_key, str) and lerobot_key:
                lerobot_suffix = lerobot_key.rsplit(".", 1)[-1]
                candidate_names.append(f"{lerobot_suffix}_rgb")

            sim_cam_name = next((name for name in candidate_names if name in camera_keys), None)
            if sim_cam_name is None:
                raise ValueError(
                    f"No matching camera found for '{lerobot_key}'. Available cameras: {sorted(camera_keys)}"
                )
            mappings.append((lerobot_key, sim_cam_name))
        return mappings

    if config.pov_cam_name_sim in camera_keys:
        return [(config.lerobot_keys["video"], config.pov_cam_name_sim)]

    raise ValueError(
        f"No video mapping found. Expected '{config.pov_cam_name_sim}' in camera_obs. "
        f"Available cameras: {sorted(camera_keys)}"
    )


def _extract_teleop_command(trajectory: h5py.Group, teleop_key: str, config: Gr00tDatasetConfig) -> np.ndarray:
    assert "action" in trajectory.keys()
    assert teleop_key in config.hdf5_keys
    teleop_command = trajectory["action"][config.hdf5_keys[teleop_key]][:-1]
    return np.asarray(teleop_command)


def _prepare_episode_data(
    trajectory: h5py.Group,
    config: Gr00tDatasetConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    data: dict[str, Any] = {}

    policy_modality_config = load_json(config.modality_template_path)
    policy_joints_config = load_robot_joints_config_from_yaml(config.policy_joints_config_path)
    action_joints_config = load_robot_joints_config_from_yaml(config.action_joints_config_path)
    state_joints_config = load_robot_joints_config_from_yaml(config.state_joints_config_path)

    for key in ["state", "action"]:
        lerobot_key_name = config.lerobot_keys[key]
        if key == "state":
            joints = trajectory["obs"][config.hdf5_keys[key]][:-1]
            input_joints_config = state_joints_config
        else:
            joints = trajectory[config.hdf5_keys[key]][:-1]
            input_joints_config = action_joints_config

        joints = JointsAbsPosition.from_array(joints, input_joints_config, device="cpu")
        remapped_joints = remap_sim_joints_to_policy_joints(joints, policy_joints_config)

        ordered_joints = []
        for joint_group in policy_modality_config[key].keys():
            if joint_group in {
                "left_wrist_pose",
                "right_wrist_pose",
                "base_height_command",
                "navigate_command",
                "torso_orientation_rpy_command",
            }:
                continue
            num_joints = (
                policy_modality_config[key][joint_group]["end"]
                - policy_modality_config[key][joint_group]["start"]
            )

            if joint_group not in remapped_joints.keys():
                remapped_joints[joint_group] = np.zeros(
                    (joints.get_joints_pos().shape[0], num_joints), dtype=np.float64
                )
            else:
                assert remapped_joints[joint_group].shape[1] == num_joints
            ordered_joints.append(remapped_joints[joint_group])

        data[lerobot_key_name] = np.concatenate(ordered_joints, axis=1)

    length = data[config.lerobot_keys["action"]].shape[0]
    assert length == data[config.lerobot_keys["state"]].shape[0]

    for key in ["obs", "action"]:
        if key not in trajectory.keys():
            continue
        eef_pose = {}
        for side in ["left", "right"]:
            if f"{side}_eef_pos" in config.hdf5_keys and f"{side}_eef_quat" in config.hdf5_keys:
                side_eef_pos = trajectory[key][config.hdf5_keys[f"{side}_eef_pos"]]
                side_eef_quat = trajectory[key][config.hdf5_keys[f"{side}_eef_quat"]]
                side_eef_pose = EefPose.from_array(side_eef_pos[:-1], side_eef_quat[:-1], device="cpu")
                eef_pose[side] = side_eef_pose.get_eef_pose()
        if "left" in eef_pose and "right" in eef_pose:
            eef_pose = np.concatenate(
                [eef_pose["left"].numpy(), eef_pose["right"].numpy()], axis=1
            ).astype(np.float64)
            assert eef_pose.shape == (length, 14), f"{eef_pose.shape} != ({length}, 14)"
            lerobot_key_name = config.lerobot_keys[f"{key}_eef_pose"]
            data[lerobot_key_name] = eef_pose

    teleop_command_keys = [
        "teleop_base_height_command",
        "teleop_navigate_command",
        "teleop_torso_orientation_rpy_command",
    ]
    for teleop_key in teleop_command_keys:
        if teleop_key in config.hdf5_keys:
            data[config.lerobot_keys[teleop_key]] = _extract_teleop_command(
                trajectory, teleop_key, config
            )

    annotation_key = config.lerobot_keys["annotation"][0]
    data[annotation_key] = [config.language_instruction] * length

    reward = np.zeros((length, 1), dtype=np.float64)
    reward[-1] = 1.0
    done = np.zeros((length, 1), dtype=bool)
    done[-1] = True
    data["next.reward"] = reward
    data["next.done"] = done
    data["observation.img_state_delta"] = np.zeros((length, 1), dtype=np.float64)

    frames_by_key: dict[str, np.ndarray] = {}
    for lerobot_key, sim_cam_name in _resolve_video_mappings(trajectory, config, policy_modality_config):
        frames = np.array(trajectory["camera_obs"][sim_cam_name])
        frames = frames[:-1]

        if config.target_image_size != config.original_image_size:
            frames = resize_frames_with_padding(
                frames, target_image_size=config.target_image_size, bgr_conversion=False, pad_img=True
            )

        if len(frames) != length:
            raise ValueError(f"Video length {len(frames)} does not match data length {length}")

        frames_by_key[lerobot_key] = frames

    return data, frames_by_key


def _ensure_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        response = input(f"Output path {output_dir} exists. Remove it? (y/n): ")
        if response.lower() != "y":
            raise RuntimeError("Aborting to avoid overwriting existing dataset.")
        import shutil

        shutil.rmtree(output_dir)


def convert_hdf5_to_lerobot_v30(
    config: Gr00tDatasetConfig,
    repo_id: str,
    output_dir: Path,
    vcodec: str,
    image_writer_threads: int,
    image_writer_processes: int,
    batch_encoding_size: int,
) -> None:
    _ensure_output_dir(output_dir)

    LOGGER.info("Loading HDF5 file: %s", config.hdf5_file_path)
    hdf5_handler = h5py.File(config.hdf5_file_path, "r")
    hdf5_data = hdf5_handler["data"]
    trajectory_ids = sorted(hdf5_data.keys(), key=_episode_sort_key)

    dataset = None
    try:
        for trajectory_id in tqdm(trajectory_ids, desc="episodes"):
            trajectory = hdf5_data[trajectory_id]
            episode_data, frames_by_key = _prepare_episode_data(trajectory, config)

            if dataset is None:
                example_episode = dict(episode_data)
                video_shapes = {}
                for video_key, frames in frames_by_key.items():
                    video_shapes[video_key] = (frames.shape[1], frames.shape[2], frames.shape[3])
                    example_episode[video_key] = frames[0]
                features = _infer_features(example_episode, video_shapes, config, vcodec)

                dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=config.fps,
                    features=features,
                    root=output_dir,
                    robot_type=config.robot_type,
                    use_videos=True,
                    image_writer_processes=image_writer_processes,
                    image_writer_threads=image_writer_threads,
                    batch_encoding_size=batch_encoding_size,
                    vcodec=vcodec,
                )

            lengths = {frames.shape[0] for frames in frames_by_key.values()}
            if len(lengths) != 1:
                raise ValueError(f"Mismatched video lengths: {lengths}")
            length = lengths.pop()

            for t in range(length):
                frame = {}
                for key, values in episode_data.items():
                    if isinstance(values, list):
                        frame[key] = values[t]
                    else:
                        frame[key] = values[t]

                for video_key, frames in frames_by_key.items():
                    frame[video_key] = frames[t]
                frame["task"] = config.language_instruction
                dataset.add_frame(frame)

            dataset.save_episode()

        if dataset is not None:
            dataset.finalize()

    finally:
        hdf5_handler.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Convert HDF5 to LeRobot v3.0 format")
    parser.add_argument("--yaml_file", required=True, help="Path to YAML configuration file")
    parser.add_argument(
        "--data_root",
        default=None,
        help="Override data_root from YAML (directory containing the HDF5 file)",
    )
    parser.add_argument(
        "--hdf5_name",
        default=None,
        help="Override hdf5_name from YAML (HDF5 filename)",
    )
    parser.add_argument(
        "--repo_id",
        default=None,
        help="Dataset repo id used for metadata (defaults to HDF5 filename without extension)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for v3.0 dataset (defaults to config.lerobot_data_dir)",
    )
    parser.add_argument(
        "--vcodec",
        default="h264",
        help="Video codec to use for encoding (h264, hevc, libsvtav1)",
    )
    parser.add_argument(
        "--image_writer_threads",
        type=int,
        default=0,
        help="Number of threads for async image writing (0 disables async writing)",
    )
    parser.add_argument(
        "--image_writer_processes",
        type=int,
        default=0,
        help="Number of processes for async image writing (0 disables multiprocessing)",
    )
    parser.add_argument(
        "--batch_encoding_size",
        type=int,
        default=1,
        help="Number of episodes to batch before encoding videos",
    )

    args = parser.parse_args()

    if args.data_root is None and args.hdf5_name is None:
        config = create_config_from_yaml(args.yaml_file, Gr00tDatasetConfig)
    else:
        config_data = load_config_from_yaml(args.yaml_file, Gr00tDatasetConfig)
        if args.data_root is not None:
            config_data["data_root"] = Path(args.data_root)
        if args.hdf5_name is not None:
            config_data["hdf5_name"] = args.hdf5_name
        config = Gr00tDatasetConfig(**config_data)

    repo_id = args.repo_id
    if repo_id is None:
        repo_id = Path(config.hdf5_name).stem

    output_dir = Path(args.output_dir) if args.output_dir else config.lerobot_data_dir

    convert_hdf5_to_lerobot_v30(
        config=config,
        repo_id=repo_id,
        output_dir=output_dir,
        vcodec=args.vcodec,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        batch_encoding_size=args.batch_encoding_size,
    )


if __name__ == "__main__":
    main()
