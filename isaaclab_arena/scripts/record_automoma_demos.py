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

Additional options:

- ``--interpolated X``: Linearly interpolate between trajectory keyframes by factor
  X for smoother motion.  E.g. ``--interpolated 4`` inserts 3 frames between each
  pair of original keyframes.

- ``--mobile_base_relative``: Store base actions as relative deltas (Δx, Δy, Δθ)
  instead of absolute positions. Arm / gripper remain absolute.

Usage::

    # Set-state mode with interpolation and mobile-base-relative
    python isaaclab_arena/scripts/record_automoma_demos.py \\
        --enable_cameras --set_state \\
        --interpolated 4 --mobile_base_relative \\
        --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \\
        --dataset_file data/automoma/summit_franka_open_microwave_7221_setstate.hdf5 \\
        --num_episodes 10 \\
        summit_franka_open_door \\
        --object_name microwave_7221 \\
        --scene_name scene_0_seed_0 \\
        --object_center

    # Drive mode (physics-based replay)
    python isaaclab_arena/scripts/record_automoma_demos.py \\
        --enable_cameras \\
        --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \\
        --dataset_file data/automoma/summit_franka_open_microwave_7221_drive.hdf5 \\
        --num_episodes 10 \\
        summit_franka_open_door \\
        --object_name microwave_7221 \\
        --scene_name scene_0_seed_0 \\
        --object_center
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
parser.add_argument(
    "--interpolated",
    type=int,
    default=1,
    help=(
        "Interpolation factor for smoothing trajectories.  1 = no interpolation. "
        "4 = insert 3 intermediate frames between each pair of keyframes."
    ),
)
parser.add_argument(
    "--mobile_base_relative",
    action="store_true",
    default=False,
    help=(
        "If set, store base actions as relative deltas (Δx, Δy, Δθ) instead of "
        "absolute positions.  Arm and gripper remain absolute.  During evaluation, "
        "the policy output is integrated with the current base state."
    ),
)
parser.add_argument(
    "--disable_collision",
    action="store_true",
    default=False,
    help=(
        "If set, disable ALL collisions in the entire simulation stage. "
        "This prevents any physics-based contact responses, which is useful "
        "for set-state recording where planner trajectories may cause "
        "interpenetration."
    ),
)

add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import h5py
import numpy as np
import torch
import tqdm

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, ObservationTermCfg as ObsTerm, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.embodiments.summit_franka.summit_franka import SummitFrankaJointSpaceActionsCfg
from isaaclab_arena.policy.replay_automoma_trajectory_policy import ReplayAutomomaTrajectoryPolicy
from isaaclab_arena.utils.sim_utils import (
    deactivate_prims_by_name,
    disable_all_collisions,
    set_lighting_mode,
    sync_cameras_after_reset,
)


class PreStepFlatCameraObservationsRecorder(RecorderTerm):
    """Recorder term that records the camera observations in each step."""

    def record_pre_step(self):
        return "camera_obs", self._env.obs_buf["camera_obs"]


@configclass
class PreStepFlatCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the camera observation recorder term."""

    class_type: type[RecorderTerm] = PreStepFlatCameraObservationsRecorder


@configclass
class ArenaEnvRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Recorder configuration for actions, states, and camera observations."""

    record_pre_step_flat_camera_observations = PreStepFlatCameraObservationsRecorderCfg()


def _postprocess_mobile_base_relative(dataset_file: str, base_dof: int = 3) -> None:
    """Post-process an HDF5 file to convert base actions to relative deltas.

    For each demo:
      - ``actions[:, :base_dof]``  ← absolute base targets
      - ``obs/joint_pos[:, :base_dof]`` ← current base state at each step

    After conversion:
      - ``actions[:, :base_dof]``  ← Δbase = target − current
      - ``processed_actions[:, :base_dof]`` ← same delta

    Args:
        dataset_file: Path to the HDF5 file to post-process.
        base_dof: Number of base DOFs (default 3).
    """
    print(f"\n[PostProcess] Converting base actions to relative (base_dof={base_dof}) ...")
    with h5py.File(dataset_file, "r+") as f:
        if "data" not in f:
            print("[PostProcess] Warning: no 'data' group in HDF5, skipping.")
            return
        for demo_key in sorted(f["data"].keys()):
            demo = f["data"][demo_key]
            if "actions" not in demo or "obs" not in demo:
                continue
            if "joint_pos" not in demo["obs"]:
                continue

            actions = demo["actions"][:]  # (T, D)
            processed = demo["processed_actions"][:]  # (T, D)
            joint_pos = demo["obs"]["joint_pos"][:]  # (T, D)

            T = actions.shape[0]
            if T == 0 or actions.shape[-1] < base_dof or joint_pos.shape[-1] < base_dof:
                continue

            # delta[t] = target_base[t] - current_base[t]
            delta_base = actions[:, :base_dof] - joint_pos[:, :base_dof]

            actions[:, :base_dof] = delta_base
            processed[:, :base_dof] = delta_base

            demo["actions"][...] = actions
            demo["processed_actions"][...] = processed

    print("[PostProcess] Done.")


def main():
    collisionless_replay = args_cli.set_state or args_cli.disable_collision

    # ---- Setup output ----
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # ---- Build environment ----
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    # ---- Override action config for joint-space recording ----
    # Instead of IK-based actions, use direct 12-DOF joint position actions.
    # This makes the recorded `actions` and `processed_actions` both 12-DOF
    # (3 base + 7 arm + 2 gripper), matching the conversion pipeline's expectations.
    env_cfg.actions = SummitFrankaJointSpaceActionsCfg()

    # ---- Override observations to use absolute joint positions ----
    # The conversion pipeline expects absolute joint positions, not relative-to-default.
    env_cfg.observations.policy.joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos)
    env_cfg.observations.policy.joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel)

    # Configure the recorder
    if args_cli.enable_cameras:
        env_cfg.recorders = ArenaEnvRecorderManagerCfg()
    else:
        env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    # Don't time-out — we control episode length via the trajectory
    env_cfg.terminations.time_out = None
    # Also disable the success termination — in drive mode with interpolation,
    # the door may open past the threshold mid-trajectory, which would trigger
    # an auto-reset and snap the robot back to default pose.
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    env_cfg.observations.policy.concatenate_terms = False

    import gymnasium as gym

    env = gym.make(env_name, cfg=env_cfg).unwrapped

    # ---- Post-creation scene fixes ----
    # 1) Deactivate duplicate object prims baked into the background scene USD
    object_name = getattr(args_cli, "object_name", None)
    if object_name:
        deactivate_prims_by_name(
            object_name,
            exclude_paths=(),
            required_path_substrings=("/scene/",),
        )

    # 2) Set lighting to grey mode (mode 2)
    set_lighting_mode(2)

    # 3) Optionally disable ALL collisions in the simulation
    if collisionless_replay:
        disable_all_collisions()

    # ---- Create replay policy ----
    policy = ReplayAutomomaTrajectoryPolicy(
        traj_file=args_cli.traj_file,
        episode_index=args_cli.start_episode,
        set_state=args_cli.set_state,
        device=args_cli.device if hasattr(args_cli, "device") else "cpu",
        only_successful=args_cli.only_successful,
        interpolation_factor=args_cli.interpolated,
    )

    num_episodes = min(args_cli.num_episodes, policy.n_episodes - args_cli.start_episode)
    print(f"\n{'=' * 60}")
    print(f"Recording {num_episodes} episodes to {args_cli.dataset_file}")
    print(f"Mode: {'set_state (teleport)' if args_cli.set_state else 'drive (physics)'}")
    print(f"Collision mode: {'disabled' if collisionless_replay else 'enabled'}")
    print(f"Steps per episode: {policy.n_steps} (raw={policy.n_raw_steps}, interp={args_cli.interpolated}x)")
    print(f"Mobile base relative: {args_cli.mobile_base_relative}")
    print(f"{'=' * 60}\n")

    recorded_count = 0

    # ---- Initial reset ----
    obs, _ = env.reset()
    if collisionless_replay:
        disable_all_collisions()
    # Fix first-frame camera lag: force a render + recompute observations
    obs = sync_cameras_after_reset(env)

    for ep_idx in range(num_episodes):
        actual_ep = args_cli.start_episode + ep_idx
        policy.episode_index = actual_ep
        policy.reset()

        # In drive mode, teleport to trajectory start pose before physics replay
        if not args_cli.set_state:
            policy.set_initial_state(env)
            obs = sync_cameras_after_reset(env)

        print(f"[Episode {ep_idx + 1}/{num_episodes}] (traj index {actual_ep})")

        for step in tqdm.tqdm(range(policy.n_steps), desc=f"  Episode {ep_idx + 1}", leave=False):
            with torch.no_grad():
                action = policy.get_action(env, obs)
            obs, _, terminated, truncated, _ = env.step(action)

        # Mark episode as successful
        env.recorder_manager.set_success_to_episodes(
            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
        )
        recorded_count += 1

        # Reset for next episode (this exports the current one)
        if ep_idx < num_episodes - 1:
            obs, _ = env.reset()
            if collisionless_replay:
                disable_all_collisions()
            obs = sync_cameras_after_reset(env)
        else:
            # Export the last episode
            env.recorder_manager.record_pre_reset(
                torch.tensor([0], device=env.device)
            )

    print(f"\n{'=' * 60}")
    print(f"Recording complete: {recorded_count} episodes saved to {args_cli.dataset_file}")
    print(f"{'=' * 60}")

    env.close()

    # ---- Post-processing: mobile base relative ----
    if args_cli.mobile_base_relative and os.path.exists(args_cli.dataset_file):
        _postprocess_mobile_base_relative(args_cli.dataset_file, base_dof=3)


if __name__ == "__main__":
    main()
    simulation_app.close()
