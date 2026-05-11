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
  are applied by a custom IsaacLab action term that writes joint state during
  ``env.step(action)``. The policy still only supplies the next planned target.

Additional options:

- ``--interpolated X`` / ``--interpolation_type TYPE``: Interpolate between
  trajectory keyframes by factor X for smoother motion.  E.g. ``--interpolated 4``
  inserts 3 frames between each pair of original keyframes.

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
from isaaclab_arena.utils.automoma_record_debug import add_record_debug_args, make_record_debugger

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
        "If set, replay robot and object joint targets through a set-state "
        "IsaacLab action term. Default: use physics-based drive mode."
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
    "--interpolation_type",
    type=str,
    default="linear",
    choices=("none", "linear", "cubic", "smoothstep", "smootherstep", "minjerk"),
    help="Interpolation curve to use with --interpolated.",
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
parser.add_argument(
    "--init_steps",
    type=int,
    default=1,
    help=(
        "Number of Isaac Sim steps to hold the trajectory start state before "
        "recording each episode."
    ),
)
parser.add_argument(
    "--decimation",
    type=int,
    default=None,
    help=(
        "Override IsaacLab env decimation for trajectory replay. "
        "Default: use the environment config value."
    ),
)
parser.add_argument(
    "--validate_record_success",
    action="store_true",
    default=False,
    help=(
        "After each replayed trajectory, evaluate the final state with the same "
        "door-open-and-final-engaged rule used by eval. Failed episodes are "
        "removed from the output HDF5."
    ),
)
parser.add_argument(
    "--record_eval_handle_distance_threshold",
    type=float,
    default=0.1,
    help="Final handle-distance threshold for record success validation. Matches eval default: 0.1.",
)

add_record_debug_args(parser)
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import math

import h5py
import torch
import tqdm

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, ObservationTermCfg as ObsTerm, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.embodiments.summit_franka.summit_franka import (
    SummitFrankaAutomomaSetStateActionsCfg,
    SummitFrankaJointSpaceActionsCfg,
)
from isaaclab_arena.metrics.handle_proximity_rate import (
    compute_open_while_engaged,
    get_cached_handle_proximity_diagnostics,
)
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


def _sorted_demo_keys(data_group) -> list[str]:
    def demo_index(key: str) -> int:
        try:
            return int(key.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 10**12

    return sorted(data_group.keys(), key=demo_index)


def _resolve_openable_object(env_cfg):
    arena_env = getattr(env_cfg, "isaaclab_arena_env", None)
    task = getattr(arena_env, "task", None)
    return getattr(task, "openable_object", None)


def _compact_demo_keys(data_group) -> None:
    for new_index, old_key in enumerate(_sorted_demo_keys(data_group)):
        new_key = f"demo_{new_index}"
        if old_key != new_key:
            data_group.move(old_key, new_key)


def _scalar_from_tensor(value, env_index: int = 0):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().cpu().flatten()[env_index].item()
    return value


def _bool_from_tensor(value, env_index: int = 0) -> bool | None:
    value = _scalar_from_tensor(value, env_index=env_index)
    if value is None:
        return None
    return bool(value)


def _float_from_tensor(value, env_index: int = 0) -> float | None:
    value = _scalar_from_tensor(value, env_index=env_index)
    if value is None:
        return None
    return float(value)


def _safe_hdf5_attr_value(value):
    if value is None:
        return float("nan")
    if isinstance(value, float) and not math.isfinite(value):
        return value
    return value


def _evaluate_record_success(env, openable_object, args_cli) -> dict[str, bool | float | None]:
    compute_open_while_engaged(
        env=env,
        openable_object=openable_object,
        proximity_threshold=args_cli.proximity_threshold,
        proximity_window_steps=args_cli.proximity_window_steps,
        proximity_required_steps=args_cli.proximity_required_steps,
        use_fingertips=not args_cli.disable_fingertip_proximity,
        openness_threshold=args_cli.openness_threshold,
        debug_visualize_handle=args_cli.debug_visualize_handle,
        debug_marker_scale=args_cli.debug_marker_scale,
    )
    diagnostics = get_cached_handle_proximity_diagnostics(env, openable_object)
    if diagnostics is None:
        raise RuntimeError("Record success validation could not read handle proximity diagnostics.")

    door_open = _bool_from_tensor(diagnostics.get("door_open"))
    openness = _float_from_tensor(diagnostics.get("openness"))
    handle_distance = _float_from_tensor(diagnostics.get("handle_distance"))
    final_engaged = handle_distance is not None and handle_distance <= args_cli.record_eval_handle_distance_threshold
    success = bool(door_open and final_engaged)
    return {
        "success": success,
        "final_door_open": bool(door_open) if door_open is not None else False,
        "final_engaged": bool(final_engaged),
        "final_door_openness": openness,
        "final_openness": openness,
        "final_handle_distance": handle_distance,
    }


def _postprocess_record_success(
    dataset_file: str,
    episode_results: list[dict[str, bool | float | int | None]],
    *,
    validate_record_success: bool,
    handle_distance_threshold: float,
) -> tuple[int, int]:
    """Write record success metadata and optionally drop failed demos."""
    if not os.path.exists(dataset_file):
        print(f"[RecordSuccess] Warning: dataset file does not exist: {dataset_file}")
        return 0, 0

    with h5py.File(dataset_file, "r+") as f:
        if "data" not in f:
            print("[RecordSuccess] Warning: no 'data' group in HDF5, skipping.")
            return 0, 0

        data = f["data"]
        demo_keys = _sorted_demo_keys(data)
        if len(demo_keys) != len(episode_results):
            print(
                "[RecordSuccess] Warning: HDF5 demo count "
                f"({len(demo_keys)}) does not match replayed episode count ({len(episode_results)}). "
                "Updating the overlapping prefix only."
            )

        successful = 0
        failed_keys = []
        for demo_key, result in zip(demo_keys, episode_results):
            demo = data[demo_key]
            success = bool(result["success"])
            if success:
                successful += 1
            elif validate_record_success:
                failed_keys.append(demo_key)

            demo.attrs["success"] = success
            demo.attrs["record_success_checked"] = bool(validate_record_success)
            demo.attrs["traj_index"] = int(result["traj_index"])
            if validate_record_success:
                demo.attrs["final_door_open"] = bool(result["final_door_open"])
                demo.attrs["final_engaged"] = bool(result["final_engaged"])
                demo.attrs["final_door_openness"] = _safe_hdf5_attr_value(result["final_door_openness"])
                demo.attrs["final_openness"] = _safe_hdf5_attr_value(result["final_openness"])
                demo.attrs["final_handle_distance"] = _safe_hdf5_attr_value(result["final_handle_distance"])

        for demo_key in failed_keys:
            del data[demo_key]

        if failed_keys:
            _compact_demo_keys(data)

        remaining_keys = _sorted_demo_keys(data)
        total_samples = sum(int(data[key].attrs.get("num_samples", 0)) for key in remaining_keys)
        data.attrs["total"] = total_samples
        data.attrs["record_success_checked"] = bool(validate_record_success)
        data.attrs["record_success_attempted_count"] = len(episode_results)
        data.attrs["record_success_success_count"] = successful
        data.attrs["record_success_saved_count"] = len(remaining_keys)
        data.attrs["record_success_removed_count"] = len(failed_keys)
        if validate_record_success:
            data.attrs["record_success_rule"] = (
                "final_door_open && final_handle_distance <= "
                f"{handle_distance_threshold}"
            )

        return successful, len(remaining_keys)


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
    if args_cli.init_steps < 1:
        raise ValueError("--init_steps must be >= 1.")

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
    if args_cli.decimation is not None:
        if args_cli.decimation < 1:
            raise ValueError("--decimation must be >= 1.")
        env_cfg.decimation = args_cli.decimation

    record_debugger = make_record_debugger(args_cli)

    # ---- Override action config for trajectory recording ----
    # Drive mode records 12D robot joint targets. Set-state mode records a
    # robot+object target and applies it through IsaacLab's action manager.
    object_name = getattr(args_cli, "object_name", None)
    if args_cli.set_state:
        env_cfg.actions = SummitFrankaAutomomaSetStateActionsCfg(object_asset_name=object_name)
    else:
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
    openable_object = _resolve_openable_object(env_cfg)
    if args_cli.validate_record_success and openable_object is None:
        raise RuntimeError(
            "--validate_record_success requires an OpenDoor task with an openable_object, "
            "but none was found in the environment config."
        )

    # ---- Post-creation scene fixes ----
    # 1) Deactivate duplicate object prims baked into the background scene USD
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
        interpolation_type=args_cli.interpolation_type,
    )

    record_debugger.setup(env, env_cfg, policy)
    init_steps = args_cli.init_steps

    num_episodes = min(args_cli.num_episodes, policy.n_episodes - args_cli.start_episode)
    print(f"\n{'=' * 60}")
    print(f"Recording {num_episodes} episodes to {args_cli.dataset_file}")
    print(f"Mode: {'set_state (robot+object state action)' if args_cli.set_state else 'drive (physics)'}")
    print(f"Collision mode: {'disabled' if collisionless_replay else 'enabled'}")
    print(
        f"Steps per episode: {policy.n_steps} "
        f"(raw={policy.n_raw_steps}, interp={args_cli.interpolated}x, type={args_cli.interpolation_type})"
    )
    print(f"Mobile base relative: {args_cli.mobile_base_relative}")
    print(f"Initial state-write Isaac Sim steps: {init_steps}")
    if args_cli.validate_record_success:
        print(
            "Record success validation: enabled "
            f"(openness_threshold={args_cli.openness_threshold}, "
            f"handle_distance_threshold={args_cli.record_eval_handle_distance_threshold}, "
            f"use_fingertips={not args_cli.disable_fingertip_proximity})"
        )
    else:
        print("Record success validation: disabled (saving all replayed episodes)")
    print(f"{'=' * 60}\n")

    recorded_count = 0
    episode_results = []

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

        # Align the first recorded observation with the trajectory start pose.
        policy.set_initial_state(env, init_steps=init_steps, render=True)
        obs = sync_cameras_after_reset(env)

        print(f"[Episode {ep_idx + 1}/{num_episodes}] (traj index {actual_ep})")
        record_debugger.begin_episode(ep_idx, actual_ep)
        record_debugger.after_initial_state(env, policy)

        for step in tqdm.tqdm(range(policy.n_steps), desc=f"  Episode {ep_idx + 1}", leave=False):
            with torch.no_grad():
                action = policy.get_action(env, obs)
            debug_step_state = record_debugger.before_step(env, policy, action, step)
            obs, _, terminated, truncated, _ = env.step(action)
            record_debugger.after_step(env, debug_step_state, step)

        record_debugger.end_episode()

        if args_cli.validate_record_success:
            success_result = _evaluate_record_success(env, openable_object, args_cli)
            print(
                f"[RecordSuccess] traj={actual_ep} success={success_result['success']} "
                f"door_open={success_result['final_door_open']} "
                f"final_engaged={success_result['final_engaged']} "
                f"openness={success_result['final_door_openness']} "
                f"handle_distance={success_result['final_handle_distance']}",
                flush=True,
            )
        else:
            success_result = {
                "success": True,
                "final_door_open": None,
                "final_engaged": None,
                "final_door_openness": None,
                "final_openness": None,
                "final_handle_distance": None,
            }
        success_result["traj_index"] = actual_ep
        episode_results.append(success_result)

        # The recorder's pre-reset hook recomputes success from termination terms,
        # which are disabled above to avoid mid-trajectory resets. We still set the
        # value here for in-memory consistency, then fix HDF5 attrs after export.
        env.recorder_manager.set_success_to_episodes(
            [0], torch.tensor([[bool(success_result["success"])]], dtype=torch.bool, device=env.device)
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
    record_debugger.finish(args_cli.dataset_file)

    env.close()

    successful_count, saved_count = _postprocess_record_success(
        args_cli.dataset_file,
        episode_results,
        validate_record_success=args_cli.validate_record_success,
        handle_distance_threshold=args_cli.record_eval_handle_distance_threshold,
    )
    if args_cli.validate_record_success:
        attempted_count = len(episode_results)
        success_rate = successful_count / attempted_count if attempted_count else 0.0
        print(
            f"[RecordSuccess] attempted={attempted_count} successful={successful_count} "
            f"saved={saved_count} success_rate={success_rate:.2%}",
            flush=True,
        )

    # ---- Post-processing: mobile base relative ----
    if args_cli.mobile_base_relative and os.path.exists(args_cli.dataset_file):
        _postprocess_mobile_base_relative(args_cli.dataset_file, base_dof=3)


if __name__ == "__main__":
    main()
    simulation_app.close()
