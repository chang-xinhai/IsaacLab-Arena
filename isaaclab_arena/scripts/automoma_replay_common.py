# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for AutoMoMa trajectory replay scripts.

This module intentionally contains the common environment setup and replay loop
used by both HDF5 recording and metrics-only replay. The CLI helpers are safe
before launching Isaac Sim; the environment setup functions import Isaac Lab
modules lazily and should be called after the app has been launched.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import torch
import tqdm


@dataclass
class AutomomaReplayContext:
    """Runtime objects shared by AutoMoMa replay entrypoints."""

    env: Any
    env_cfg: Any
    policy: Any
    openable_object: Any
    collisionless_replay: bool
    episode_indices: list[int]
    recorder_mode: str
    sync_cameras: bool


def add_automoma_replay_args(parser: Any) -> None:
    """Register CLI flags shared by record and replay entrypoints."""

    parser.add_argument(
        "--traj_file",
        type=str,
        required=True,
        help="Path to the .pt trajectory data file from the automoma planner.",
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
        help="Number of episodes to replay. Default: 1.",
    )
    parser.add_argument(
        "--start_episode",
        type=int,
        default=0,
        help="Starting episode index in the trajectory file. Default: 0.",
    )
    parser.add_argument(
        "--only_successful",
        dest="only_successful",
        action="store_true",
        default=True,
        help="Replay only episodes marked successful by the planner. Default: True.",
    )
    parser.add_argument(
        "--all_episodes",
        dest="only_successful",
        action="store_false",
        help="Replay all episodes in the trajectory file, including planner failures.",
    )
    parser.add_argument(
        "--interpolated",
        type=int,
        default=5,
        help=(
            "Interpolation factor for smoothing trajectories. 1 = no interpolation. "
            "5 inserts 4 intermediate frames between each pair of keyframes. Default: 5."
        ),
    )
    parser.add_argument(
        "--interpolation_type",
        type=str,
        default="cubic",
        choices=("none", "linear", "cubic", "smoothstep", "smootherstep", "minjerk"),
        help="Interpolation curve to use with --interpolated. Default: cubic.",
    )
    parser.add_argument(
        "--mobile_base_relative",
        action="store_true",
        default=False,
        help=(
            "If set, store recorded base actions as relative deltas (dx, dy, dtheta) "
            "instead of absolute positions. Arm and gripper remain absolute."
        ),
    )
    parser.add_argument(
        "--disable_collision",
        action="store_true",
        default=False,
        help=(
            "If set, disable ALL collisions in the entire simulation stage. "
            "This prevents physics-based contact responses, which is useful for "
            "set-state replay where planner trajectories may cause interpenetration."
        ),
    )
    parser.add_argument(
        "--init_steps",
        type=int,
        default=5,
        help="Number of Isaac Sim steps to hold the trajectory start state before replay. Default: 5.",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=1,
        help="Override IsaacLab env decimation for trajectory replay. Default: 1.",
    )
    parser.add_argument(
        "--record_eval_handle_distance_threshold",
        "--success_handle_distance_threshold",
        dest="record_eval_handle_distance_threshold",
        type=float,
        default=0.1,
        help="Final handle-distance threshold for replay success validation. Matches eval default: 0.1.",
    )
    parser.add_argument(
        "--robot_object_static_friction",
        type=float,
        default=None,
        help=(
            "Override static contact friction on all robot and target-object rigid shapes. "
            "If omitted, AUTOMOMA_ROBOT_OBJECT_STATIC_FRICTION is used when set; otherwise defaults to 1.0."
        ),
    )
    parser.add_argument(
        "--robot_object_dynamic_friction",
        type=float,
        default=None,
        help=(
            "Override dynamic contact friction on all robot and target-object rigid shapes. "
            "If omitted, AUTOMOMA_ROBOT_OBJECT_DYNAMIC_FRICTION is used when set; otherwise defaults to 1.0."
        ),
    )


def add_episode_selection_args(parser: Any) -> None:
    """Register explicit raw trajectory index selection flags."""

    parser.add_argument(
        "--episode_indices",
        type=str,
        default=None,
        help="Comma-separated raw trajectory indices to replay instead of a contiguous range.",
    )
    parser.add_argument(
        "--episode_indices_file",
        type=str,
        default=None,
        help="Text file containing one raw trajectory index per line, or comma-separated indices.",
    )


def parse_episode_indices(args_cli: Any) -> list[int] | None:
    """Parse explicit raw trajectory indices from CLI args."""

    values: list[int] = []
    raw_items: list[str] = []
    if getattr(args_cli, "episode_indices", None):
        raw_items.append(args_cli.episode_indices)
    if getattr(args_cli, "episode_indices_file", None):
        with open(args_cli.episode_indices_file, "r", encoding="utf-8") as f:
            raw_items.append(f.read())
    for raw in raw_items:
        for item in raw.replace(",", "\n").splitlines():
            item = item.strip()
            if not item or item.startswith("#"):
                continue
            values.append(int(item))
    if not values:
        return None
    for value in values:
        if value < 0:
            raise ValueError(f"Episode indices must be non-negative, got {value}")
    return values


def resolve_openable_object(env_cfg: Any) -> Any:
    """Return the OpenDoor task object if the environment exposes one."""

    arena_env = getattr(env_cfg, "isaaclab_arena_env", None)
    task = getattr(arena_env, "task", None)
    return getattr(task, "openable_object", None)


def scalar_from_tensor(value: Any, env_index: int = 0) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().cpu().flatten()[env_index].item()
    return value


def bool_from_tensor(value: Any, env_index: int = 0) -> bool | None:
    value = scalar_from_tensor(value, env_index=env_index)
    if value is None:
        return None
    return bool(value)


def float_from_tensor(value: Any, env_index: int = 0) -> float | None:
    value = scalar_from_tensor(value, env_index=env_index)
    if value is None:
        return None
    return float(value)


def safe_hdf5_attr_value(value: Any) -> Any:
    if value is None:
        return float("nan")
    if isinstance(value, float) and not math.isfinite(value):
        return value
    return value


def _evaluate_openness_only_success(openable_object: Any, env: Any, args_cli: Any, error: Exception) -> dict[str, Any]:
    openness = float_from_tensor(openable_object.get_openness(env))
    door_open = openness is not None and openness >= args_cli.openness_threshold
    return {
        "success": False,
        "final_door_open": bool(door_open),
        "final_engaged": False,
        "final_door_openness": openness,
        "final_openness": openness,
        "final_handle_distance": None,
        "error": str(error),
    }


def evaluate_replay_success(env: Any, openable_object: Any, args_cli: Any) -> dict[str, Any]:
    """Evaluate the current open-door state with the eval success rule."""

    from isaaclab_arena.metrics.handle_proximity_rate import (
        compute_open_while_engaged,
        get_cached_handle_proximity_diagnostics,
    )

    try:
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
    except ValueError as exc:
        return _evaluate_openness_only_success(openable_object, env, args_cli, exc)

    diagnostics = get_cached_handle_proximity_diagnostics(env, openable_object)
    if diagnostics is None:
        raise RuntimeError("Replay success validation could not read handle proximity diagnostics.")

    door_open = bool_from_tensor(diagnostics.get("door_open"))
    openness = float_from_tensor(diagnostics.get("openness"))
    handle_distance = float_from_tensor(diagnostics.get("handle_distance"))
    final_engaged = handle_distance is not None and handle_distance <= args_cli.record_eval_handle_distance_threshold
    success = bool(door_open and final_engaged)
    return {
        "success": success,
        "final_door_open": bool(door_open) if door_open is not None else False,
        "final_engaged": bool(final_engaged),
        "final_door_openness": openness,
        "final_openness": openness,
        "final_handle_distance": handle_distance,
        "error": "",
    }


def _resolve_optional_float(cli_value: float | None, env_name: str) -> float | None:
    if cli_value is not None:
        return float(cli_value)
    env_value = os.environ.get(env_name)
    if env_value in (None, ""):
        return None
    return float(env_value)


def _make_arena_env_recorder_manager_cfg():
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import RecorderTerm, RecorderTermCfg
    from isaaclab.utils import configclass

    class PreStepFlatCameraObservationsRecorder(RecorderTerm):
        """Recorder term that records camera observations at each step."""

        def record_pre_step(self):
            return "camera_obs", self._env.obs_buf["camera_obs"]

    @configclass
    class PreStepFlatCameraObservationsRecorderCfg(RecorderTermCfg):
        class_type: type[RecorderTerm] = PreStepFlatCameraObservationsRecorder

    @configclass
    class ArenaEnvRecorderManagerCfg(ActionStateRecorderManagerCfg):
        record_pre_step_flat_camera_observations = PreStepFlatCameraObservationsRecorderCfg()

    return ArenaEnvRecorderManagerCfg


def _configure_actions_observations_and_recorders(
    env_cfg: Any,
    args_cli: Any,
    *,
    recorder_mode: str,
    dataset_file: str | None,
) -> None:
    import isaaclab.envs.mdp as mdp_isaac_lab
    from isaaclab.envs.manager_based_env_cfg import DefaultEmptyRecorderManagerCfg
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode, ObservationTermCfg as ObsTerm
    from isaaclab_arena.embodiments.summit_franka.summit_franka import (
        SummitFrankaAutomomaSetStateActionsCfg,
        SummitFrankaJointSpaceActionsCfg,
    )

    object_name = getattr(args_cli, "object_name", None)
    if args_cli.set_state:
        env_cfg.actions = SummitFrankaAutomomaSetStateActionsCfg(object_asset_name=object_name)
    else:
        env_cfg.actions = SummitFrankaJointSpaceActionsCfg()

    # The conversion pipeline expects absolute joint positions, not positions relative to defaults.
    env_cfg.observations.policy.joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos)
    env_cfg.observations.policy.joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel)

    if recorder_mode == "none":
        env_cfg.recorders = DefaultEmptyRecorderManagerCfg()
    elif recorder_mode == "hdf5":
        if dataset_file is None:
            raise ValueError("dataset_file is required when recorder_mode='hdf5'.")
        if args_cli.enable_cameras:
            env_cfg.recorders = _make_arena_env_recorder_manager_cfg()()
        else:
            env_cfg.recorders = ActionStateRecorderManagerCfg()
        output_dir = os.path.dirname(dataset_file)
        output_file_name = os.path.splitext(os.path.basename(dataset_file))[0]
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    else:
        raise ValueError(f"Unsupported recorder_mode: {recorder_mode}")

    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    env_cfg.observations.policy.concatenate_terms = False


def build_automoma_replay_context(
    args_cli: Any,
    *,
    recorder_mode: str,
    dataset_file: str | None = None,
    explicit_episode_indices: list[int] | None = None,
    require_openable_object: bool = False,
) -> AutomomaReplayContext:
    """Build a configured IsaacLab environment and replay policy."""

    if args_cli.init_steps < 1:
        raise ValueError("--init_steps must be >= 1.")
    if args_cli.num_episodes < 1:
        raise ValueError("--num_episodes must be >= 1.")
    if args_cli.start_episode < 0:
        raise ValueError("--start_episode must be >= 0.")
    if args_cli.decimation is not None and args_cli.decimation < 1:
        raise ValueError("--decimation must be >= 1.")

    from isaaclab_arena.examples.example_environments.cli import get_arena_builder_from_cli
    from isaaclab_arena.policy.replay_automoma_trajectory_policy import ReplayAutomomaTrajectoryPolicy
    from isaaclab_arena.utils.sim_utils import (
        deactivate_prims_by_name,
        disable_all_collisions,
        set_lighting_mode,
        set_robot_object_material_friction,
    )
    import gymnasium as gym

    if recorder_mode == "hdf5":
        if dataset_file is None:
            raise ValueError("--dataset_file is required for HDF5 recording.")
        output_dir = os.path.dirname(dataset_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

    collisionless_replay = args_cli.set_state or args_cli.disable_collision

    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()
    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation

    _configure_actions_observations_and_recorders(
        env_cfg,
        args_cli,
        recorder_mode=recorder_mode,
        dataset_file=dataset_file,
    )

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    openable_object = resolve_openable_object(env_cfg)
    if require_openable_object and openable_object is None:
        raise RuntimeError("This replay mode requires an OpenDoor task with an openable_object, but none was found.")

    object_name = getattr(args_cli, "object_name", None)
    if object_name:
        deactivate_prims_by_name(object_name, exclude_paths=(), required_path_substrings=("/scene/",))

    set_lighting_mode(2)

    static_friction = _resolve_optional_float(
        args_cli.robot_object_static_friction,
        "AUTOMOMA_ROBOT_OBJECT_STATIC_FRICTION",
    )
    dynamic_friction = _resolve_optional_float(
        args_cli.robot_object_dynamic_friction,
        "AUTOMOMA_ROBOT_OBJECT_DYNAMIC_FRICTION",
    )
    if static_friction is None and dynamic_friction is None:
        static_friction = 1.0
        dynamic_friction = 1.0
    if static_friction is not None or dynamic_friction is not None:
        set_robot_object_material_friction(
            env,
            object_name=object_name,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
        )

    if collisionless_replay:
        disable_all_collisions()

    only_successful = bool(args_cli.only_successful)
    if explicit_episode_indices is not None:
        # Explicit selections are raw trajectory-file indices. Do not filter first,
        # otherwise the selected indices and metrics rows refer to the wrong demos.
        only_successful = False

    initial_episode_index = args_cli.start_episode
    if explicit_episode_indices is not None:
        initial_episode_index = 0

    policy = ReplayAutomomaTrajectoryPolicy(
        traj_file=args_cli.traj_file,
        episode_index=initial_episode_index,
        set_state=args_cli.set_state,
        device=args_cli.device if hasattr(args_cli, "device") else "cpu",
        only_successful=only_successful,
        interpolation_factor=args_cli.interpolated,
        interpolation_type=args_cli.interpolation_type,
    )

    if explicit_episode_indices is None:
        episode_indices = list(
            range(
                args_cli.start_episode,
                min(args_cli.start_episode + args_cli.num_episodes, policy.n_episodes),
            )
        )
    else:
        episode_indices = explicit_episode_indices
        bad_indices = [idx for idx in episode_indices if idx >= policy.n_episodes]
        if bad_indices:
            raise ValueError(
                f"Episode indices out of range for {args_cli.traj_file}: "
                f"{bad_indices[:10]} (max {policy.n_episodes - 1})"
            )

    sync_cameras = bool(args_cli.enable_cameras and recorder_mode == "hdf5")
    return AutomomaReplayContext(
        env=env,
        env_cfg=env_cfg,
        policy=policy,
        openable_object=openable_object,
        collisionless_replay=collisionless_replay,
        episode_indices=episode_indices,
        recorder_mode=recorder_mode,
        sync_cameras=sync_cameras,
    )


def print_replay_header(ctx: AutomomaReplayContext, args_cli: Any, *, title: str, output_path: str | None = None) -> None:
    print(f"\n{'=' * 60}")
    if output_path:
        print(f"{title}: {len(ctx.episode_indices)} episodes -> {output_path}")
    else:
        print(f"{title}: {len(ctx.episode_indices)} episodes")
    print(f"Mode: {'set_state (robot+object state action)' if args_cli.set_state else 'drive (physics)'}")
    print(f"Collision mode: {'disabled' if ctx.collisionless_replay else 'enabled'}")
    print(
        f"Steps per episode: {ctx.policy.n_steps} "
        f"(raw={ctx.policy.n_raw_steps}, interp={args_cli.interpolated}x, type={args_cli.interpolation_type})"
    )
    print(f"Mobile base relative: {args_cli.mobile_base_relative}")
    print(f"Initial state-write Isaac Sim steps: {args_cli.init_steps}")
    print(f"{'=' * 60}\n")


def run_automoma_replay(
    ctx: AutomomaReplayContext,
    args_cli: Any,
    *,
    record_debugger: Any,
    on_step: Callable[[AutomomaReplayContext, int, int, int], None] | None = None,
    after_episode: Callable[[AutomomaReplayContext, int, int], None] | None = None,
    after_last_episode: Callable[[AutomomaReplayContext], None] | None = None,
) -> int:
    """Replay selected episodes and call optional hooks around the shared loop."""

    from isaaclab_arena.utils.sim_utils import disable_all_collisions, sync_cameras_after_reset

    env = ctx.env
    policy = ctx.policy
    obs, _ = env.reset()
    if ctx.collisionless_replay:
        disable_all_collisions()
    if ctx.sync_cameras:
        obs = sync_cameras_after_reset(env)

    replayed_count = 0
    num_episodes = len(ctx.episode_indices)
    for ep_idx, actual_ep in enumerate(ctx.episode_indices):
        policy.episode_index = actual_ep
        policy.reset()

        policy.set_initial_state(env, init_steps=args_cli.init_steps, render=True)
        if ctx.sync_cameras:
            obs = sync_cameras_after_reset(env)

        print(f"[Episode {ep_idx + 1}/{num_episodes}] (traj index {actual_ep})")
        record_debugger.begin_episode(ep_idx, actual_ep)
        record_debugger.after_initial_state(env, policy)

        for step in tqdm.tqdm(range(policy.n_steps), desc=f"  Episode {ep_idx + 1}", leave=False):
            with torch.no_grad():
                action = policy.get_action(env, obs)
            debug_step_state = record_debugger.before_step(env, policy, action, step)
            obs, _, _terminated, _truncated, _info = env.step(action)
            record_debugger.after_step(env, debug_step_state, step)
            if on_step is not None:
                on_step(ctx, ep_idx, actual_ep, step)

        record_debugger.end_episode()
        if after_episode is not None:
            after_episode(ctx, ep_idx, actual_ep)
        replayed_count += 1

        if ep_idx < num_episodes - 1:
            obs, _ = env.reset()
            if ctx.collisionless_replay:
                disable_all_collisions()
            if ctx.sync_cameras:
                obs = sync_cameras_after_reset(env)
        elif after_last_episode is not None:
            after_last_episode(ctx)

    return replayed_count
