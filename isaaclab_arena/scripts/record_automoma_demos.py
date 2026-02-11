# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to replay automoma planner trajectories and record them to HDF5.

This script loads pre-computed joint trajectories from a .pt file, replays them
in the Isaac Lab simulator, and saves the resulting demonstrations (observations,
actions, camera images) into an HDF5 file suitable for conversion to LeRobot
format and policy training.

Two replay modes are supported:

- **Drive mode** (default): Joint position targets are sent as actions to the robot.
  Physics simulation handles contacts and friction, so the result may differ from
  the planner's idealized trajectory.

- **Set-state mode** (``--set_state``): Robot joints and object articulation state
  are directly teleported each step. No physics simulation for the manipulation
  itself — produces pixel-perfect replays of the planned trajectory.

Usage::

    # Drive mode (physics-based replay)
    python isaaclab_arena/scripts/record_automoma_demos.py \\
        --device cpu --enable_cameras \\
        --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \\
        --dataset_file $DATASET_DIR/summit_franka_open_door.hdf5 \\
        --num_episodes 10 \\
        summit_franka_open_door \\
        --object_name microwave_7221 \\
        --scene_name scene_0_seed_0

    # Set-state mode (teleport joints, no physics)
    python isaaclab_arena/scripts/record_automoma_demos.py \\
        --device cpu --enable_cameras --set_state \\
        --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \\
        --dataset_file $DATASET_DIR/summit_franka_open_door_setstate.hdf5 \\
        --num_episodes 50 \\
        summit_franka_open_door \\
        --object_name microwave_7221 \\
        --scene_name scene_0_seed_0
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)

# ---- CLI arguments ----
parser = get_isaaclab_arena_cli_parser()

parser.add_argument(
    "--traj_file",
    type=str,
    required=True,
    help="Path to the .pt trajectory data file from the automoma planner.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    required=True,
    help="Output HDF5 file path for recorded demonstrations.",
)
parser.add_argument(
    "--set_state",
    action="store_true",
    default=False,
    help=(
        "If set, directly teleport robot joints and object state each step "
        "(bypasses physics). Default: use physics-based drive mode."
    ),
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=1,
    help="Number of episodes to replay and record. Default: 1.",
)
parser.add_argument(
    "--start_episode",
    type=int,
    default=0,
    help="Starting episode index in the trajectory file. Default: 0.",
)
parser.add_argument(
    "--only_successful",
    action="store_true",
    default=True,
    help="Only replay episodes marked as successful. Default: True.",
)

add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import tqdm

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

from isaaclab_arena.policy.replay_automoma_trajectory_policy import ReplayAutomomaTrajectoryPolicy


def main():
    # ---- Setup output ----
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # ---- Build environment ----
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    # Configure the recorder
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    # Don't time-out — we control episode length via the trajectory
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    import gymnasium as gym
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    # ---- Create replay policy ----
    policy = ReplayAutomomaTrajectoryPolicy(
        traj_file=args_cli.traj_file,
        episode_index=args_cli.start_episode,
        set_state=args_cli.set_state,
        device=args_cli.device,
        only_successful=args_cli.only_successful,
    )

    num_episodes = min(args_cli.num_episodes, policy.n_episodes - args_cli.start_episode)
    print(f"\n{'=' * 60}")
    print(f"Recording {num_episodes} episodes to {args_cli.dataset_file}")
    print(f"Mode: {'set_state (teleport)' if args_cli.set_state else 'drive (physics)'}")
    print(f"Steps per episode: {policy.n_steps}")
    print(f"{'=' * 60}\n")

    recorded_count = 0

    for ep_idx in range(num_episodes):
        actual_ep = args_cli.start_episode + ep_idx
        policy.episode_index = actual_ep
        policy.reset()

        # Reset environment
        obs, _ = env.reset()

        print(f"[Episode {ep_idx + 1}/{num_episodes}] (traj index {actual_ep})")

        for step in tqdm.tqdm(range(policy.n_steps), desc=f"  Episode {ep_idx + 1}", leave=False):
            with torch.inference_mode():
                action = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(action)

        # Mark episode and export
        if hasattr(env, "recorder_manager"):
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            recorded_count += 1

    print(f"\n{'=' * 60}")
    print(f"Recording complete: {recorded_count} episodes saved to {args_cli.dataset_file}")
    print(f"{'=' * 60}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
