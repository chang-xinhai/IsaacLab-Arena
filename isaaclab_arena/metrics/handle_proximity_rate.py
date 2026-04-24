# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import torch
import isaaclab.sim as sim_utils

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import normalize, quat_from_angle_axis

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.metrics.metric_base import MetricBase


_HANDLE_DEBUG_CACHE_PREFIX = "_handle_proximity_debug_cache_"
_HANDLE_NEAR_HISTORY_PREFIX = "_open_door_near_history_"
_HANDLE_DEBUG_MARKERS_PREFIX = "_handle_proximity_debug_markers_"
_HANDLE_DEBUG_MARKERS_WARNED_PREFIX = "_handle_proximity_debug_markers_warned_"
_STABILITY_PREV_OPENNESS_PREFIX = "_open_door_prev_openness_"
_STABILITY_PREV_JOINT_POS_PREFIX = "_open_door_prev_joint_pos_"
_STABILITY_COUNTER_PREFIX = "_open_door_stability_counter_"


class HandleDistanceRecorder(RecorderTerm):
    """Records the minimum robot-to-handle distance at each step."""

    name = "handle_distance"

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.openable_object = cfg.openable_object
        self.use_fingertips = cfg.use_fingertips

    def record_post_step(self):
        diagnostics = compute_handle_proximity_geometry(
            env=self._env,
            openable_object=self.openable_object,
            use_fingertips=self.use_fingertips,
        )
        return self.name, diagnostics["handle_distance"]


@configclass
class HandleDistanceRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = HandleDistanceRecorder
    openable_object: Openable | None = None
    use_fingertips: bool = True


class HandleDiagnosticsRecorder(RecorderTerm):
    """Records per-step geometry and gating diagnostics for handle engagement."""

    name = "handle_diagnostics"

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.openable_object = cfg.openable_object
        self.use_fingertips = cfg.use_fingertips
        self.proximity_threshold = cfg.proximity_threshold

    def record_post_step(self):
        diagnostics = get_cached_handle_proximity_diagnostics(self._env, self.openable_object)
        if diagnostics is None:
            diagnostics = compute_handle_proximity_geometry(
                env=self._env,
                openable_object=self.openable_object,
                use_fingertips=self.use_fingertips,
            )
            diagnostics["proximity_near"] = diagnostics["handle_distance"] <= self.proximity_threshold
            diagnostics["engaged"] = torch.zeros_like(diagnostics["proximity_near"])
            diagnostics["door_open"] = self.openable_object.is_open(self._env)
            diagnostics["openness"] = self.openable_object.get_openness(self._env)

        return self.name, {
            "handle_world_position": diagnostics["handle_world_position"],
            "handle_link_world_position": diagnostics["handle_link_world_position"],
            "closest_robot_world_position": diagnostics["closest_robot_world_position"],
            "ee_world_position": diagnostics["ee_world_position"],
            "fingertip_world_positions": diagnostics["fingertip_world_positions"],
            "handle_distance": diagnostics["handle_distance"],
            "proximity_near": diagnostics["proximity_near"],
            "engaged": diagnostics["engaged"],
            "door_open": diagnostics["door_open"],
            "openness": diagnostics["openness"],
        }


@configclass
class HandleDiagnosticsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = HandleDiagnosticsRecorder
    openable_object: Openable | None = None
    use_fingertips: bool = True
    proximity_threshold: float = 0.12


class HandleProximityRateMetric(MetricBase):
    """Proportion of episodes that reach the handle neighborhood at least once."""

    name = "handle_proximity_rate"
    recorder_term_name = HandleDistanceRecorder.name

    def __init__(self, openable_object: Openable, proximity_threshold: float, use_fingertips: bool = True):
        super().__init__()
        self.openable_object = openable_object
        self.proximity_threshold = proximity_threshold
        self.use_fingertips = use_fingertips

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return HandleDistanceRecorderCfg(
            openable_object=self.openable_object,
            use_fingertips=self.use_fingertips,
        )

    def compute_metric_from_recording(self, recorded_metric_data: list[np.ndarray]) -> float:
        if len(recorded_metric_data) == 0:
            return 0.0
        reached_handle = []
        for episode_data in recorded_metric_data:
            reached_handle.append(np.any(episode_data <= self.proximity_threshold))
        return float(np.mean(reached_handle))


def _get_cache_attr(openable_object: Openable) -> str:
    return f"{_HANDLE_DEBUG_CACHE_PREFIX}{openable_object.name}"


def _get_history_attr(openable_object: Openable) -> str:
    return f"{_HANDLE_NEAR_HISTORY_PREFIX}{openable_object.name}"


def _get_markers_attr(openable_object: Openable) -> str:
    return f"{_HANDLE_DEBUG_MARKERS_PREFIX}{openable_object.name}"


def _get_markers_warned_attr(openable_object: Openable) -> str:
    return f"{_HANDLE_DEBUG_MARKERS_WARNED_PREFIX}{openable_object.name}"


def _get_prev_openness_attr(openable_object: Openable) -> str:
    return f"{_STABILITY_PREV_OPENNESS_PREFIX}{openable_object.name}"


def _get_prev_joint_pos_attr(openable_object: Openable) -> str:
    return f"{_STABILITY_PREV_JOINT_POS_PREFIX}{openable_object.name}"


def _get_stability_counter_attr(openable_object: Openable) -> str:
    return f"{_STABILITY_COUNTER_PREFIX}{openable_object.name}"


def get_cached_handle_proximity_diagnostics(env: ManagerBasedRLEnv, openable_object: Openable) -> dict[str, torch.Tensor] | None:
    return getattr(env, _get_cache_attr(openable_object), None)


def _set_cached_handle_proximity_diagnostics(
    env: ManagerBasedRLEnv,
    openable_object: Openable,
    diagnostics: dict[str, torch.Tensor],
) -> None:
    setattr(env, _get_cache_attr(openable_object), diagnostics)


def _get_handle_link_world_position(env: ManagerBasedRLEnv, openable_object: Openable) -> torch.Tensor:
    asset = env.scene[openable_object.name]
    body_index = asset.body_names.index(openable_object.handle_link_name)
    return asset.data.body_pos_w[:, body_index, :]


def compute_handle_proximity_geometry(
    env: ManagerBasedRLEnv,
    openable_object: Openable,
    use_fingertips: bool = True,
) -> dict[str, torch.Tensor]:
    handle_pos = openable_object.get_handle_position(env)
    handle_link_pos = _get_handle_link_world_position(env, openable_object)
    ee_frame_data = env.scene["ee_frame"].data
    ee_pos = ee_frame_data.target_pos_w[..., 0, :]

    robot_points = ee_pos.unsqueeze(1)
    if use_fingertips and ee_frame_data.target_pos_w.shape[1] > 1:
        fingertip_pos = ee_frame_data.target_pos_w[..., 1:, :]
        robot_points = torch.cat((robot_points, fingertip_pos), dim=1)
    else:
        fingertip_pos = ee_pos.new_empty((env.num_envs, 0, 3))

    distances = torch.linalg.norm(handle_pos.unsqueeze(1) - robot_points, dim=-1)
    closest_indices = distances.argmin(dim=1)
    closest_robot_pos = robot_points[torch.arange(env.num_envs, device=env.device), closest_indices]
    min_distance = distances[torch.arange(env.num_envs, device=env.device), closest_indices]

    return {
        "handle_world_position": handle_pos,
        "handle_link_world_position": handle_link_pos,
        "ee_world_position": ee_pos,
        "fingertip_world_positions": fingertip_pos,
        "closest_robot_world_position": closest_robot_pos,
        "handle_distance": min_distance,
        "closest_robot_index": closest_indices,
    }


def compute_handle_proximity_distance(
    env: ManagerBasedRLEnv,
    openable_object: Openable,
    use_fingertips: bool = True,
) -> torch.Tensor:
    return compute_handle_proximity_geometry(
        env=env,
        openable_object=openable_object,
        use_fingertips=use_fingertips,
    )["handle_distance"]


def has_recent_consecutive_proximity(
    near_history: torch.Tensor,
    valid_lengths: torch.Tensor,
    required_steps: int,
) -> torch.Tensor:
    valid_mask = torch.arange(near_history.shape[1], device=near_history.device).unsqueeze(0) < valid_lengths.unsqueeze(1)
    masked_history = near_history & valid_mask
    if required_steps <= 1:
        return torch.any(masked_history, dim=1)

    batch_size, window = masked_history.shape
    if required_steps > window:
        return torch.zeros(batch_size, dtype=torch.bool, device=near_history.device)

    streak = torch.zeros(batch_size, dtype=torch.int64, device=near_history.device)
    best = torch.zeros(batch_size, dtype=torch.int64, device=near_history.device)
    for step in range(window):
        streak = torch.where(masked_history[:, step], streak + 1, torch.zeros_like(streak))
        best = torch.maximum(best, streak)
    return best >= required_steps


def _update_proximity_history(
    env: ManagerBasedRLEnv,
    openable_object: Openable,
    proximity_window_steps: int,
    near: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history_attr = _get_history_attr(openable_object)
    history = getattr(env, history_attr, None)
    if history is None or history.shape != (env.num_envs, proximity_window_steps):
        history = torch.zeros((env.num_envs, proximity_window_steps), dtype=torch.bool, device=env.device)
        setattr(env, history_attr, history)

    step_count = torch.clamp(env.episode_length_buf, min=1)
    new_episode_mask = step_count <= 1
    if torch.any(new_episode_mask):
        history[new_episode_mask] = False

    write_index = (step_count - 1) % proximity_window_steps
    history[torch.arange(env.num_envs, device=env.device), write_index] = near

    recent_history = torch.roll(history, shifts=1, dims=1)
    valid_lengths = torch.minimum(step_count, torch.full_like(step_count, proximity_window_steps))
    return history, recent_history, valid_lengths


def _make_marker_cfg(openable_object: Openable) -> VisualizationMarkersCfg:
    return VisualizationMarkersCfg(
        prim_path=f"/World/Visuals/{openable_object.name}_handle_debug",
        markers={
            "handle": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "closest_robot": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "ee": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
            "fingertip": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ),
            "handle_link": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
            ),
            "connecting_line": sim_utils.CylinderCfg(
                radius=0.004,
                height=1.0,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
        },
    )


def _compute_connecting_line_pose(
    start_pos: torch.Tensor,
    end_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direction = end_pos - start_pos
    length = torch.norm(direction, dim=-1)
    midpoint = (start_pos + end_pos) / 2

    default_direction = torch.tensor([0.0, 0.0, 1.0], dtype=start_pos.dtype, device=start_pos.device).expand(
        start_pos.shape[0], -1
    )
    safe_direction = torch.where(
        length.unsqueeze(-1) > 1e-8,
        normalize(direction),
        default_direction,
    )
    rotation_axis = torch.linalg.cross(default_direction, safe_direction)
    rotation_axis_norm = torch.norm(rotation_axis, dim=-1)
    safe_axis = torch.where(
        rotation_axis_norm.unsqueeze(-1) > 1e-6,
        normalize(rotation_axis),
        torch.tensor([1.0, 0.0, 0.0], dtype=start_pos.dtype, device=start_pos.device).expand(start_pos.shape[0], -1),
    )
    cos_angle = torch.sum(default_direction * safe_direction, dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)
    orientation = quat_from_angle_axis(angle, safe_axis)
    return midpoint, orientation, length


def _maybe_visualize_handle_debug(
    env: ManagerBasedRLEnv,
    openable_object: Openable,
    diagnostics: dict[str, torch.Tensor],
    marker_scale: float,
) -> None:
    markers_attr = _get_markers_attr(openable_object)
    warned_attr = _get_markers_warned_attr(openable_object)
    markers = getattr(env, markers_attr, None)
    if markers is None:
        try:
            markers = VisualizationMarkers(_make_marker_cfg(openable_object))
            setattr(env, markers_attr, markers)
        except Exception as exc:
            if not getattr(env, warned_attr, False):
                logging.warning(f"[HandleDebug] Failed to create debug markers: {exc}")
                setattr(env, warned_attr, True)
            return

    try:
        handle_pos = diagnostics["handle_world_position"][0].unsqueeze(0)
        closest_robot_pos = diagnostics["closest_robot_world_position"][0].unsqueeze(0)
        line_pos, line_quat, line_length = _compute_connecting_line_pose(closest_robot_pos, handle_pos)

        translations = [handle_pos.squeeze(0), closest_robot_pos.squeeze(0), diagnostics["ee_world_position"][0]]
        marker_indices = [0, 1, 2]
        orientations = [
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=handle_pos.dtype, device=handle_pos.device),
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=handle_pos.dtype, device=handle_pos.device),
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=handle_pos.dtype, device=handle_pos.device),
        ]
        scales = [
            torch.full((3,), float(marker_scale), dtype=handle_pos.dtype, device=handle_pos.device),
            torch.full((3,), float(marker_scale), dtype=handle_pos.dtype, device=handle_pos.device),
            torch.full((3,), float(marker_scale), dtype=handle_pos.dtype, device=handle_pos.device),
        ]

        fingertip_pos = diagnostics["fingertip_world_positions"][0]
        if fingertip_pos.numel() > 0:
            for idx in range(fingertip_pos.shape[0]):
                translations.append(fingertip_pos[idx])
                marker_indices.append(3)
                orientations.append(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=handle_pos.dtype, device=handle_pos.device))
                scales.append(torch.full((3,), float(marker_scale), dtype=handle_pos.dtype, device=handle_pos.device))

        translations.append(diagnostics["handle_link_world_position"][0])
        marker_indices.append(4)
        orientations.append(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=handle_pos.dtype, device=handle_pos.device))
        scales.append(torch.full((3,), float(marker_scale), dtype=handle_pos.dtype, device=handle_pos.device))

        translations.append(line_pos.squeeze(0))
        marker_indices.append(5)
        orientations.append(line_quat.squeeze(0))
        scales.append(
            torch.tensor(
                [max(0.35 * float(marker_scale), 0.05), max(0.35 * float(marker_scale), 0.05), max(line_length.item(), 1e-4)],
                dtype=handle_pos.dtype,
                device=handle_pos.device,
            )
        )

        translations_tensor = torch.stack(translations, dim=0)
        orientations_tensor = torch.stack(orientations, dim=0)
        scales_tensor = torch.stack(scales, dim=0)

        markers.set_visibility(True)
        markers.visualize(
            translations=translations_tensor,
            orientations=orientations_tensor,
            marker_indices=torch.tensor(marker_indices, dtype=torch.int32, device=translations_tensor.device),
            scales=scales_tensor,
        )
    except Exception as exc:
        if not getattr(env, warned_attr, False):
            logging.warning(f"[HandleDebug] Failed to update debug markers: {exc}")
            setattr(env, warned_attr, True)


def compute_open_while_engaged(
    env: ManagerBasedRLEnv,
    openable_object: Openable,
    proximity_threshold: float,
    proximity_window_steps: int,
    proximity_required_steps: int,
    use_fingertips: bool,
    openness_threshold: float | None = None,
    debug_visualize_handle: bool = False,
    debug_marker_scale: float = 1.0,
) -> torch.Tensor:
    door_open = openable_object.is_open(env, threshold=openness_threshold)
    openness = openable_object.get_openness(env)
    if not openable_object.has_handle_reference():
        diagnostics = {
            "handle_world_position": torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device),
            "handle_link_world_position": torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device),
            "ee_world_position": torch.zeros((env.num_envs, 3), dtype=torch.float32, device=torch.device(env.device)),
            "fingertip_world_positions": torch.zeros((env.num_envs, 0, 3), dtype=torch.float32, device=env.device),
            "closest_robot_world_position": torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device),
            "handle_distance": torch.full((env.num_envs,), float("inf"), dtype=torch.float32, device=env.device),
            "proximity_near": torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
            "engaged": torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
            "door_open": door_open,
            "openness": openness,
        }
        _set_cached_handle_proximity_diagnostics(env, openable_object, diagnostics)
        return door_open

    diagnostics = compute_handle_proximity_geometry(env, openable_object, use_fingertips=use_fingertips)
    final_engaged = diagnostics["handle_distance"] <= 0.1

    diagnostics["proximity_near"] = diagnostics["handle_distance"] <= proximity_threshold
    diagnostics["engaged"] = final_engaged
    diagnostics["door_open"] = door_open
    diagnostics["openness"] = openness
    _set_cached_handle_proximity_diagnostics(env, openable_object, diagnostics)

    if debug_visualize_handle:
        _maybe_visualize_handle_debug(
            env=env,
            openable_object=openable_object,
            diagnostics=diagnostics,
            marker_scale=debug_marker_scale,
        )

    return door_open & final_engaged


def compute_stable_open_and_joints(
    env: ManagerBasedRLEnv,
    openable_object: Openable,
    stability_window_steps: int,
    openness_stability_epsilon: float,
    joint_stability_epsilon: float,
    use_fingertips: bool,
    proximity_threshold: float,
    openness_threshold: float | None = None,
    debug_visualize_handle: bool = False,
    debug_marker_scale: float = 1.0,
) -> torch.Tensor:
    openness = openable_object.get_openness(env)
    robot_joint_pos = env.scene["robot"].data.joint_pos
    step_count = torch.clamp(env.episode_length_buf, min=1)
    new_episode_mask = step_count <= 1

    prev_openness_attr = _get_prev_openness_attr(openable_object)
    prev_joint_pos_attr = _get_prev_joint_pos_attr(openable_object)
    counter_attr = _get_stability_counter_attr(openable_object)

    prev_openness = getattr(env, prev_openness_attr, None)
    if prev_openness is None or prev_openness.shape != openness.shape:
        prev_openness = openness.clone()
        setattr(env, prev_openness_attr, prev_openness)

    prev_joint_pos = getattr(env, prev_joint_pos_attr, None)
    if prev_joint_pos is None or prev_joint_pos.shape != robot_joint_pos.shape:
        prev_joint_pos = robot_joint_pos.clone()
        setattr(env, prev_joint_pos_attr, prev_joint_pos)

    stable_counter = getattr(env, counter_attr, None)
    if stable_counter is None or stable_counter.shape != step_count.shape:
        stable_counter = torch.zeros_like(step_count)
        setattr(env, counter_attr, stable_counter)

    if torch.any(new_episode_mask):
        prev_openness[new_episode_mask] = openness[new_episode_mask]
        prev_joint_pos[new_episode_mask] = robot_joint_pos[new_episode_mask]
        stable_counter[new_episode_mask] = 0

    openness_stable = torch.abs(openness - prev_openness) <= openness_stability_epsilon
    joint_stable = torch.max(torch.abs(robot_joint_pos - prev_joint_pos), dim=1).values <= joint_stability_epsilon
    stable_now = openness_stable & joint_stable
    stable_counter[:] = torch.where(stable_now, stable_counter + 1, torch.zeros_like(stable_counter))
    stable_counter[new_episode_mask] = 0

    prev_openness[:] = openness
    prev_joint_pos[:] = robot_joint_pos

    if openable_object.has_handle_reference():
        diagnostics = compute_handle_proximity_geometry(env, openable_object, use_fingertips=use_fingertips)
        diagnostics["proximity_near"] = diagnostics["handle_distance"] <= proximity_threshold
        diagnostics["engaged"] = diagnostics["handle_distance"] <= 0.1
        diagnostics["door_open"] = openable_object.is_open(env, threshold=openness_threshold)
        diagnostics["openness"] = openness
        _set_cached_handle_proximity_diagnostics(env, openable_object, diagnostics)

        if debug_visualize_handle:
            _maybe_visualize_handle_debug(
                env=env,
                openable_object=openable_object,
                diagnostics=diagnostics,
                marker_scale=debug_marker_scale,
            )

    return stable_counter >= stability_window_steps
