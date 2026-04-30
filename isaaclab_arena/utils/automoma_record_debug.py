"""Debug helpers for AutoMoMa demonstration recording.

This module keeps verbose diagnostics out of ``record_automoma_demos.py`` while
still letting the record script opt into joint, handle, and replay-configuration
debugging from the CLI.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def add_record_debug_args(parser: Any) -> None:
    """Register optional record/debug arguments on the record CLI parser."""
    parser.add_argument(
        "--debug_joint_tracking",
        action="store_true",
        default=False,
        help=(
            "Print robot joint target vs. simulator joint state diagnostics during "
            "AutoMoMa trajectory replay."
        ),
    )
    parser.add_argument(
        "--debug_joint_tracking_steps",
        type=int,
        default=5,
        help="Print every step below this step index when --debug_joint_tracking is set.",
    )
    parser.add_argument(
        "--debug_joint_tracking_interval",
        type=int,
        default=10,
        help="Also print every N steps when --debug_joint_tracking is set. Use 0 to disable interval prints.",
    )
    parser.add_argument(
        "--debug_joint_tracking_topk",
        type=int,
        default=4,
        help="Number of largest per-joint errors to print in the joint tracking diagnostic.",
    )
    parser.add_argument(
        "--debug_joint_tracking_tolerance",
        type=float,
        default=0.05,
        help="Tracking error tolerance used by the debug summary.",
    )
    parser.add_argument(
        "--debug_joint_tracking_fk_link",
        type=str,
        default="ee_link",
        help="Robot link used for FK pose-error diagnostics under --debug_joint_tracking.",
    )
    parser.add_argument(
        "--debug_joint_tracking_no_fk",
        action="store_true",
        default=False,
        help="Disable FK pose-error diagnostics under --debug_joint_tracking.",
    )
    parser.add_argument(
        "--debug_handle_tracking",
        action="store_true",
        default=False,
        help="Print handle distance and object openness diagnostics during open-door replay.",
    )
    parser.add_argument(
        "--debug_handle_tracking_steps",
        type=int,
        default=5,
        help="Print every step below this step index when --debug_handle_tracking is set.",
    )
    parser.add_argument(
        "--debug_handle_tracking_interval",
        type=int,
        default=10,
        help="Also print every N steps when --debug_handle_tracking is set. Use 0 to disable interval prints.",
    )
    parser.add_argument(
        "--record_init_steps",
        type=int,
        default=1,
        help=(
            "Number of Isaac Sim steps to hold the trajectory start state before "
            "recording each episode."
        ),
    )
    parser.add_argument(
        "--record_init_render",
        action="store_true",
        default=False,
        help="Render during the initial start-state hold steps.",
    )
    parser.add_argument(
        "--record_decimation",
        type=int,
        default=None,
        help=(
            "Override env_cfg.decimation for replay/record debugging. Higher values "
            "hold each action target for more physics steps."
        ),
    )
    parser.add_argument(
        "--record_render_interval",
        type=int,
        default=None,
        help=(
            "Override env_cfg.sim.render_interval for replay/record debugging. "
            "If omitted with --record_decimation, it defaults to the decimation value."
        ),
    )
    parser.add_argument(
        "--record_actuator_stiffness_scale",
        type=float,
        default=1.0,
        help="Scale robot actuator stiffness before creating the replay environment.",
    )
    parser.add_argument(
        "--record_actuator_damping_scale",
        type=float,
        default=1.0,
        help="Scale robot actuator damping before creating the replay environment.",
    )
    parser.add_argument(
        "--record_actuator_effort_scale",
        type=float,
        default=1.0,
        help="Scale robot actuator effort limits before creating the replay environment.",
    )
    parser.add_argument(
        "--record_camera_width",
        type=int,
        default=None,
        help="Override enabled camera width for replay/record debugging.",
    )
    parser.add_argument(
        "--record_camera_height",
        type=int,
        default=None,
        help="Override enabled camera height for replay/record debugging.",
    )


@dataclass(frozen=True)
class RecordDebugConfig:
    joint_tracking: bool
    joint_tracking_steps: int
    joint_tracking_interval: int
    joint_tracking_topk: int
    joint_tracking_tolerance: float
    joint_tracking_fk_link: str
    joint_tracking_no_fk: bool
    handle_tracking: bool
    handle_tracking_steps: int
    handle_tracking_interval: int
    record_init_steps: int
    record_init_render: bool
    record_decimation: int | None
    record_render_interval: int | None
    record_actuator_stiffness_scale: float
    record_actuator_damping_scale: float
    record_actuator_effort_scale: float
    record_camera_width: int | None
    record_camera_height: int | None

    @classmethod
    def from_args(cls, args: Any) -> "RecordDebugConfig":
        return cls(
            joint_tracking=bool(args.debug_joint_tracking),
            joint_tracking_steps=args.debug_joint_tracking_steps,
            joint_tracking_interval=args.debug_joint_tracking_interval,
            joint_tracking_topk=args.debug_joint_tracking_topk,
            joint_tracking_tolerance=args.debug_joint_tracking_tolerance,
            joint_tracking_fk_link=args.debug_joint_tracking_fk_link,
            joint_tracking_no_fk=bool(args.debug_joint_tracking_no_fk),
            handle_tracking=bool(args.debug_handle_tracking),
            handle_tracking_steps=args.debug_handle_tracking_steps,
            handle_tracking_interval=args.debug_handle_tracking_interval,
            record_init_steps=args.record_init_steps,
            record_init_render=bool(args.record_init_render),
            record_decimation=args.record_decimation,
            record_render_interval=args.record_render_interval,
            record_actuator_stiffness_scale=args.record_actuator_stiffness_scale,
            record_actuator_damping_scale=args.record_actuator_damping_scale,
            record_actuator_effort_scale=args.record_actuator_effort_scale,
            record_camera_width=args.record_camera_width,
            record_camera_height=args.record_camera_height,
        )

    @property
    def tracking_enabled(self) -> bool:
        return self.joint_tracking or self.handle_tracking


def make_record_debugger(args: Any) -> "RecordDebugHooks":
    return RecordDebugHooks(RecordDebugConfig.from_args(args))


def _make_unique_output_path(path: str | Path) -> Path:
    """Return path if unused, otherwise append a timestamp suffix without overwriting."""
    path = Path(path)
    if not path.exists():
        return path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.stem}-{timestamp}{path.suffix}")
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{timestamp}-{counter:02d}{path.suffix}")
        counter += 1
    return candidate


def _format_joint_vector(values: Any) -> str:
    values = values.detach().cpu().flatten().tolist()
    return "[" + ", ".join(f"{value:+.4f}" for value in values) + "]"


def _format_pose_vector(values: Any) -> str:
    if hasattr(values, "detach"):
        values = values.detach().cpu().flatten().tolist()
    else:
        import numpy as np

        values = np.asarray(values).reshape(-1).tolist()
    return "[" + ", ".join(f"{value:+.4f}" for value in values) + "]"


class _PinocchioFkDebugger:
    """Small FK helper for comparing planned and simulated joint-space poses."""

    def __init__(self, joint_names: list[str], link_name: str):
        import pinocchio as pin

        robot_root = Path(
            os.environ.get(
                "AUTOMOMA_ROBOT_ROOT",
                str(Path(__file__).resolve().parents[4] / "assets" / "robot"),
            )
        )
        urdf_path = robot_root / "summit_franka" / "summit_franka.urdf"
        if not urdf_path.exists():
            raise FileNotFoundError(f"Summit Franka URDF not found: {urdf_path}")

        self.pin = pin
        self.urdf_path = urdf_path
        self.link_name = link_name
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(link_name)
        if self.frame_id >= len(self.model.frames):
            raise ValueError(f"FK link '{link_name}' was not found in {urdf_path}")

        self.joint_q_indices = []
        missing_joint_names = []
        for joint_name in joint_names:
            joint_id = self.model.getJointId(joint_name)
            if joint_id >= len(self.model.joints):
                missing_joint_names.append(joint_name)
                continue
            joint_model = self.model.joints[joint_id]
            if joint_model.nq != 1:
                raise ValueError(
                    f"FK debug only supports scalar joints; {joint_name} has nq={joint_model.nq}."
                )
            self.joint_q_indices.append(joint_model.idx_q)

        if missing_joint_names:
            raise ValueError(
                f"FK model is missing action joints: {missing_joint_names}. "
                f"Known joints: {list(self.model.names)}"
            )
        if len(self.joint_q_indices) != len(joint_names):
            raise ValueError(
                f"FK joint mapping has {len(self.joint_q_indices)} joints for "
                f"{len(joint_names)} action joints."
            )

    def pose(self, joint_values: Any) -> tuple[Any, Any]:
        import numpy as np
        import torch

        q = np.zeros(self.model.nq, dtype=np.float64)
        values = joint_values.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
        for source_index, q_index in enumerate(self.joint_q_indices):
            q[q_index] = values[source_index]

        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.frame_id]
        position = torch.from_numpy(np.array(placement.translation, copy=True)).float()
        rotation = torch.from_numpy(np.array(placement.rotation, copy=True)).float()
        return position, rotation


def _scale_numeric_or_mapping(value: Any, scale: float) -> Any:
    if value is None:
        return value
    if isinstance(value, dict):
        return {key: _scale_numeric_or_mapping(item, scale) for key, item in value.items()}
    if isinstance(value, (float, int)):
        return float(value) * scale
    return value


def _scale_robot_actuator_cfgs(env_cfg: Any, config: RecordDebugConfig) -> None:
    scale_fields = (
        ("stiffness", config.record_actuator_stiffness_scale, "--record_actuator_stiffness_scale"),
        ("damping", config.record_actuator_damping_scale, "--record_actuator_damping_scale"),
        ("effort_limit_sim", config.record_actuator_effort_scale, "--record_actuator_effort_scale"),
    )
    if all(scale == 1.0 for _, scale, _ in scale_fields):
        return

    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    actuator_cfgs = getattr(robot_cfg, "actuators", None)
    if not actuator_cfgs:
        print("[RecordActuatorScale] Warning: no robot actuator configs found.", flush=True)
        return

    print("[RecordActuatorScale] Applying robot actuator debug scales:", flush=True)
    for actuator_name, actuator_cfg in actuator_cfgs.items():
        changed_parts = []
        for field_name, scale, arg_name in scale_fields:
            if scale <= 0.0:
                raise ValueError(f"{arg_name} must be > 0.")
            if scale == 1.0 or not hasattr(actuator_cfg, field_name):
                continue
            original_value = getattr(actuator_cfg, field_name)
            setattr(actuator_cfg, field_name, _scale_numeric_or_mapping(original_value, scale))
            changed_parts.append(f"{field_name}x{scale:g}")
        if changed_parts:
            print(f"  {actuator_name}: {', '.join(changed_parts)}", flush=True)


def _override_camera_resolution(env_cfg: Any, config: RecordDebugConfig, enable_cameras: bool) -> None:
    if config.record_camera_width is None and config.record_camera_height is None:
        return
    if not enable_cameras:
        print(
            "[RecordCameraResolution] Warning: --enable_cameras is false; no camera resolution override applied.",
            flush=True,
        )
        return

    width = config.record_camera_width
    height = config.record_camera_height
    if width is not None and width <= 0:
        raise ValueError("--record_camera_width must be > 0.")
    if height is not None and height <= 0:
        raise ValueError("--record_camera_height must be > 0.")

    changed_parts = []
    scene_cfg = getattr(env_cfg, "scene", None)
    if scene_cfg is None:
        print("[RecordCameraResolution] Warning: env_cfg has no scene config.", flush=True)
        return
    for camera_name, camera_cfg in vars(scene_cfg).items():
        if not hasattr(camera_cfg, "width") or not hasattr(camera_cfg, "height"):
            continue
        old_width = getattr(camera_cfg, "width")
        old_height = getattr(camera_cfg, "height")
        if width is not None:
            camera_cfg.width = width
        if height is not None:
            camera_cfg.height = height
        changed_parts.append(f"{camera_name}: {old_width}x{old_height}->{camera_cfg.width}x{camera_cfg.height}")

    if changed_parts:
        print("[RecordCameraResolution] Applying camera resolution overrides:", flush=True)
        for part in changed_parts:
            print(f"  {part}", flush=True)
    else:
        print("[RecordCameraResolution] Warning: no camera configs found to override.", flush=True)


def _resolve_robot_action_term(env: Any) -> tuple[str, Any]:
    """Return the robot action term used for the AutoMoMa replay action."""
    robot = env.scene["robot"]
    for term_name in env.action_manager.active_terms:
        term = env.action_manager.get_term(term_name)
        if getattr(term, "_asset", None) is robot and hasattr(term, "_joint_ids"):
            return term_name, term
    raise RuntimeError(
        "Could not resolve a robot joint action term for joint tracking diagnostics. "
        f"Active terms: {list(env.action_manager.active_terms)}"
    )


def _joint_ids_to_list(joint_ids: Any, num_joints: int) -> list[int]:
    if isinstance(joint_ids, slice):
        return list(range(num_joints))[joint_ids]
    if hasattr(joint_ids, "detach"):
        return [int(value) for value in joint_ids.detach().cpu().flatten().tolist()]
    return [int(value) for value in joint_ids]


def _read_robot_joint_pos_for_term(env: Any, term: Any) -> Any:
    robot = env.scene["robot"]
    return robot.data.joint_pos[:, term._joint_ids].clone()


def _term_robot_action_dim(term: Any) -> int:
    return int(getattr(term, "_num_robot_joints", term.action_dim))


def _action_robot_target(action: Any, term: Any) -> Any:
    return action[:, : _term_robot_action_dim(term)].clone()


def _fk_tracking_stats(fk_debugger: _PinocchioFkDebugger | None, target_row: Any, actual_row: Any) -> dict | None:
    if fk_debugger is None:
        return None

    import torch

    target_pos, target_rot = fk_debugger.pose(target_row)
    actual_pos, actual_rot = fk_debugger.pose(actual_row)
    pos_error = actual_pos - target_pos
    pos_distance = float(torch.linalg.norm(pos_error).item())
    rel_rot = target_rot.transpose(0, 1).matmul(actual_rot)
    cos_angle = torch.clamp((torch.trace(rel_rot) - 1.0) * 0.5, -1.0, 1.0)
    rot_distance = float(torch.arccos(cos_angle).item())

    return {
        "link_name": fk_debugger.link_name,
        "target_pos": target_pos,
        "actual_pos": actual_pos,
        "pos_error": pos_error,
        "pos_distance": pos_distance,
        "rot_distance": rot_distance,
    }


def _object_joint_tracking_stats(env: Any, openable_object: Any, target_obj: Any | None) -> dict[str, Any] | None:
    """Return actual/target door joint state for plotting with joint tracking curves."""
    if openable_object is None:
        return None

    asset = env.scene[openable_object.name]
    joint_name = openable_object.openable_joint_name
    try:
        joint_index = asset.data.joint_names.index(joint_name)
    except ValueError as exc:
        raise ValueError(
            f"Openable joint {joint_name!r} not found on {openable_object.name}; "
            f"available joints: {asset.data.joint_names}"
        ) from exc

    joint_pos = float(asset.data.joint_pos[0, joint_index].detach().cpu().item())
    joint_limits = asset.data.joint_pos_limits[0, joint_index, :].detach().cpu()
    joint_min = float(joint_limits[0].item())
    joint_max = float(joint_limits[1].item())
    openness = float(openable_object.get_openness(env)[0].detach().cpu().item())

    target_joint_pos = None
    target_openness = None
    if target_obj is not None and target_obj.numel() > 0:
        target_joint_pos = float(target_obj[0, 0].detach().cpu().item())
        if joint_max != joint_min:
            target_openness = (target_joint_pos - joint_min) / (joint_max - joint_min)
            if joint_min < 0.0:
                target_openness = 1.0 - target_openness

    return {
        "object_name": openable_object.name,
        "joint_name": joint_name,
        "joint_pos": joint_pos,
        "target_joint_pos": target_joint_pos,
        "joint_error": None if target_joint_pos is None else joint_pos - target_joint_pos,
        "openness": openness,
        "target_openness": target_openness,
        "open_error": None if target_openness is None else openness - target_openness,
        "joint_min": joint_min,
        "joint_max": joint_max,
    }


def _print_joint_tracking(
    *,
    episode_index: int,
    traj_index: int,
    step: int,
    phase: str,
    joint_names: list[str],
    target: Any,
    actual: Any,
    tolerance: float,
    topk: int,
    fk_debugger: _PinocchioFkDebugger | None = None,
    object_stats: dict[str, Any] | None = None,
) -> dict:
    import torch

    target_row = target[0].detach().float().cpu()
    actual_row = actual[0].detach().float().cpu()
    error = actual_row - target_row
    abs_error = error.abs()
    max_error = float(abs_error.max().item())
    mean_error = float(abs_error.mean().item())
    over_tol = int((abs_error > tolerance).sum().item())
    fk_stats = _fk_tracking_stats(fk_debugger, target_row, actual_row)

    print(
        f"[JointTracking][ep={episode_index} traj={traj_index} step={step:04d} {phase}] "
        f"max_abs={max_error:.6f} mean_abs={mean_error:.6f} "
        f"joints_over_{tolerance:g}={over_tol}/{len(joint_names)}",
        flush=True,
    )
    print(f"  target: {_format_joint_vector(target_row)}", flush=True)
    print(f"  actual: {_format_joint_vector(actual_row)}", flush=True)
    print(f"  error : {_format_joint_vector(error)}", flush=True)

    topk = min(max(topk, 0), len(joint_names))
    if topk > 0:
        values, indices = torch.topk(abs_error, k=topk)
        top_parts = []
        for value, index in zip(values.tolist(), indices.tolist()):
            top_parts.append(
                f"{joint_names[index]} target={target_row[index]:+.4f} "
                f"actual={actual_row[index]:+.4f} err={error[index]:+.4f} abs={value:.4f}"
            )
        print("  top   : " + " | ".join(top_parts), flush=True)

    if fk_stats is not None:
        print(
            f"  fk    : link={fk_stats['link_name']} "
            f"pos_dist={fk_stats['pos_distance']:.6f}m "
            f"rot_dist={fk_stats['rot_distance']:.6f}rad",
            flush=True,
        )
        print(f"  fk_tgt: {_format_pose_vector(fk_stats['target_pos'])}", flush=True)
        print(f"  fk_act: {_format_pose_vector(fk_stats['actual_pos'])}", flush=True)

    if object_stats is not None:
        target_part = ""
        if object_stats.get("target_joint_pos") is not None:
            target_part = (
                f" target_angle={object_stats['target_joint_pos']:+.6f}rad"
                f" angle_err={object_stats['joint_error']:+.6f}rad"
            )
            if object_stats.get("target_openness") is not None:
                target_part += (
                    f" target_open={object_stats['target_openness']:.6f}"
                    f" open_err={object_stats['open_error']:+.6f}"
                )
        print(
            f"  object: {object_stats['object_name']}.{object_stats['joint_name']} "
            f"angle={object_stats['joint_pos']:+.6f}rad openness={object_stats['openness']:.6f}"
            f"{target_part}",
            flush=True,
        )

    record = {
        "episode": episode_index,
        "traj": traj_index,
        "step": step,
        "phase": phase,
        "max": max_error,
        "mean": mean_error,
        "over_tol": over_tol,
        "target": target_row,
        "actual": actual_row,
        "error": error,
        "abs_error": abs_error,
    }
    if fk_stats is not None:
        record["fk"] = fk_stats
    if object_stats is not None:
        record["object"] = object_stats
    return record


def _joint_tracking_stats(
    *,
    episode_index: int,
    traj_index: int,
    step: int,
    phase: str,
    target: Any,
    actual: Any,
    tolerance: float,
    fk_debugger: _PinocchioFkDebugger | None = None,
    object_stats: dict[str, Any] | None = None,
) -> dict:
    target_row = target[0].detach().float().cpu()
    actual_row = actual[0].detach().float().cpu()
    error = actual_row - target_row
    abs_error = error.abs()
    fk_stats = _fk_tracking_stats(fk_debugger, target_row, actual_row)
    record = {
        "episode": episode_index,
        "traj": traj_index,
        "step": step,
        "phase": phase,
        "max": float(abs_error.max().item()),
        "mean": float(abs_error.mean().item()),
        "over_tol": int((abs_error > tolerance).sum().item()),
        "target": target_row,
        "actual": actual_row,
        "error": error,
        "abs_error": abs_error,
    }
    if fk_stats is not None:
        record["fk"] = fk_stats
    if object_stats is not None:
        record["object"] = object_stats
    return record


def _summarize_joint_tracking(
    records: list[dict],
    tolerance: float,
    topk: int,
    label: str = "summary",
    joint_names: list[str] | None = None,
) -> None:
    if not records:
        return

    import numpy as np
    import torch

    max_error = max(record["max"] for record in records)
    mean_error = sum(record["mean"] for record in records) / len(records)
    over_tol_steps = sum(1 for record in records if record["over_tol"] > 0)
    print(
        f"[JointTracking][{label}] samples={len(records)} max_abs={max_error:.6f} "
        f"mean_abs={mean_error:.6f} steps_over_{tolerance:g}={over_tol_steps}/{len(records)}",
        flush=True,
    )

    if joint_names is None:
        return

    abs_errors = torch.stack([record["abs_error"] for record in records], dim=0)
    flat_worst_index = int(torch.argmax(abs_errors).item())
    worst_record_index = flat_worst_index // abs_errors.shape[1]
    worst_joint_index = flat_worst_index % abs_errors.shape[1]
    worst_record = records[worst_record_index]
    print(
        f"[JointTracking][{label}][worst] ep={worst_record['episode']} "
        f"traj={worst_record['traj']} step={worst_record['step']:04d} "
        f"joint={joint_names[worst_joint_index]} "
        f"target={worst_record['target'][worst_joint_index]:+.6f} "
        f"actual={worst_record['actual'][worst_joint_index]:+.6f} "
        f"err={worst_record['error'][worst_joint_index]:+.6f}",
        flush=True,
    )

    per_joint_max, _ = torch.max(abs_errors, dim=0)
    per_joint_mean = torch.mean(abs_errors, dim=0)
    topk = min(max(topk, 0), len(joint_names))
    if topk > 0:
        values, indices = torch.topk(per_joint_max, k=topk)
        top_parts = []
        for value, index in zip(values.tolist(), indices.tolist()):
            top_parts.append(f"{joint_names[index]} max={value:.6f} mean={per_joint_mean[index]:.6f}")
        print(f"[JointTracking][{label}][per_joint] " + " | ".join(top_parts), flush=True)

    fk_records = [record for record in records if "fk" in record]
    if not fk_records:
        return

    pos_distances = [record["fk"]["pos_distance"] for record in fk_records]
    rot_distances = [record["fk"]["rot_distance"] for record in fk_records]
    worst_pos_index = int(np.argmax(pos_distances))
    worst_rot_index = int(np.argmax(rot_distances))
    worst_pos_record = fk_records[worst_pos_index]
    worst_rot_record = fk_records[worst_rot_index]
    print(
        f"[JointTracking][{label}][fk] samples={len(fk_records)} "
        f"pos_max={max(pos_distances):.6f}m pos_mean={np.mean(pos_distances):.6f}m "
        f"pos_final={pos_distances[-1]:.6f}m "
        f"rot_max={max(rot_distances):.6f}rad rot_mean={np.mean(rot_distances):.6f}rad "
        f"rot_final={rot_distances[-1]:.6f}rad",
        flush=True,
    )
    print(
        f"[JointTracking][{label}][fk_worst_pos] ep={worst_pos_record['episode']} "
        f"traj={worst_pos_record['traj']} step={worst_pos_record['step']:04d} "
        f"pos_dist={worst_pos_record['fk']['pos_distance']:.6f}m "
        f"pos_err={_format_pose_vector(worst_pos_record['fk']['pos_error'])}",
        flush=True,
    )
    print(
        f"[JointTracking][{label}][fk_worst_rot] ep={worst_rot_record['episode']} "
        f"traj={worst_rot_record['traj']} step={worst_rot_record['step']:04d} "
        f"rot_dist={worst_rot_record['fk']['rot_distance']:.6f}rad",
        flush=True,
    )


def _safe_csv_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def _write_joint_tracking_artifacts(
    records: list[dict],
    joint_names: list[str] | None,
    dataset_file: str,
    tolerance: float,
) -> tuple[Path | None, Path | None]:
    if not records or joint_names is None:
        return None, None

    import numpy as np
    import torch

    dataset_path = Path(dataset_file)
    output_dir = dataset_path.parent / "debug_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{dataset_path.stem}-joint-tracking"
    csv_path = _make_unique_output_path(output_dir / f"{base_name}.csv")
    png_path = _make_unique_output_path(output_dir / f"{base_name}.png")

    joint_columns = [_safe_csv_name(name) for name in joint_names]
    fieldnames = [
        "sample",
        "episode",
        "traj",
        "step",
        "phase",
        "base_max_abs",
        "base_xy_norm",
        "all_joint_max_abs",
        "all_joint_mean_abs",
        "fk_eef_pos_distance_m",
        "fk_eef_rot_distance_rad",
        "object_name",
        "object_joint_name",
        "object_joint_pos_rad",
        "object_joint_target_rad",
        "object_joint_error_rad",
        "object_openness",
        "object_target_openness",
        "object_open_error",
    ]
    for name in joint_columns:
        fieldnames.extend((f"err_{name}", f"abs_{name}"))

    plot_x = []
    base_curve = []
    all_joint_curve = []
    fk_curve = []
    object_joint_curve = []
    object_target_curve = []

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample, record in enumerate(records):
            error = record["error"]
            abs_error = record["abs_error"]
            base_abs = abs_error[: min(3, abs_error.numel())]
            base_error = error[: min(3, error.numel())]
            base_max_abs = float(base_abs.max().item()) if base_abs.numel() else 0.0
            if base_error.numel() >= 2:
                base_xy_norm = float(torch.linalg.norm(base_error[:2]).item())
            else:
                base_xy_norm = base_max_abs
            fk = record.get("fk")
            fk_pos = float(fk["pos_distance"]) if fk is not None else float("nan")
            fk_rot = float(fk["rot_distance"]) if fk is not None else float("nan")
            object_stats = record.get("object")
            if object_stats is None:
                object_name = ""
                object_joint_name = ""
                object_joint_pos = float("nan")
                object_joint_target = float("nan")
                object_joint_error = float("nan")
                object_openness = float("nan")
                object_target_openness = float("nan")
                object_open_error = float("nan")
            else:
                object_name = object_stats["object_name"]
                object_joint_name = object_stats["joint_name"]
                object_joint_pos = float(object_stats["joint_pos"])
                object_joint_target = (
                    float(object_stats["target_joint_pos"])
                    if object_stats.get("target_joint_pos") is not None
                    else float("nan")
                )
                object_joint_error = (
                    float(object_stats["joint_error"])
                    if object_stats.get("joint_error") is not None
                    else float("nan")
                )
                object_openness = float(object_stats["openness"])
                object_target_openness = (
                    float(object_stats["target_openness"])
                    if object_stats.get("target_openness") is not None
                    else float("nan")
                )
                object_open_error = (
                    float(object_stats["open_error"])
                    if object_stats.get("open_error") is not None
                    else float("nan")
                )

            row = {
                "sample": sample,
                "episode": record["episode"],
                "traj": record["traj"],
                "step": record["step"],
                "phase": record["phase"],
                "base_max_abs": base_max_abs,
                "base_xy_norm": base_xy_norm,
                "all_joint_max_abs": record["max"],
                "all_joint_mean_abs": record["mean"],
                "fk_eef_pos_distance_m": fk_pos,
                "fk_eef_rot_distance_rad": fk_rot,
                "object_name": object_name,
                "object_joint_name": object_joint_name,
                "object_joint_pos_rad": object_joint_pos,
                "object_joint_target_rad": object_joint_target,
                "object_joint_error_rad": object_joint_error,
                "object_openness": object_openness,
                "object_target_openness": object_target_openness,
                "object_open_error": object_open_error,
            }
            for joint_name, column_name, err_value, abs_value in zip(
                joint_names,
                joint_columns,
                error.tolist(),
                abs_error.tolist(),
            ):
                row[f"err_{column_name}"] = err_value
                row[f"abs_{column_name}"] = abs_value
            writer.writerow(row)

            plot_x.append(sample)
            base_curve.append(base_max_abs)
            all_joint_curve.append(record["max"])
            fk_curve.append(fk_pos)
            object_joint_curve.append(object_joint_pos)
            object_target_curve.append(object_joint_target)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        ax.plot(plot_x, base_curve, linewidth=1.4, label="base max abs error (m/rad)")
        ax.plot(plot_x, all_joint_curve, linewidth=1.4, label="all-joint max abs error (m/rad)")
        ax.plot(plot_x, fk_curve, linewidth=1.4, label="fk_eef position error (m)")
        ax.axhline(tolerance, color="red", linestyle="--", linewidth=0.9, alpha=0.55)
        ax.set_title(f"Joint tracking error: {dataset_path.stem}")
        ax.set_xlabel("recorded tracking sample")
        ax.set_ylabel("error magnitude")
        ax.grid(True, alpha=0.25)

        object_values = np.asarray(object_joint_curve, dtype=np.float64)
        object_targets = np.asarray(object_target_curve, dtype=np.float64)
        if np.isfinite(object_values).any():
            ax2 = ax.twinx()
            ax2.plot(plot_x, object_joint_curve, color="black", linewidth=1.2, label="door joint angle (rad)")
            if np.isfinite(object_targets).any():
                ax2.plot(
                    plot_x,
                    object_target_curve,
                    color="black",
                    linestyle=":",
                    linewidth=1.0,
                    label="door target angle (rad)",
                )
            ax2.set_ylabel("door joint angle (rad)")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right")
        else:
            ax.legend(loc="upper right")
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"[JointTracking][curve] Warning: failed to write plot {png_path}: {exc}", flush=True)
        png_path = None

    print(f"[JointTracking][curve] csv={csv_path}", flush=True)
    if png_path is not None:
        print(f"[JointTracking][curve] png={png_path}", flush=True)
    return csv_path, png_path


def _resolve_openable_object(env_cfg: Any) -> Any:
    arena_env = getattr(env_cfg, "isaaclab_arena_env", None)
    task = getattr(arena_env, "task", None)
    return getattr(task, "openable_object", None)


def _handle_tracking_stats(env: Any, openable_object: Any, target_obj: Any | None) -> dict[str, float]:
    from isaaclab_arena.metrics.handle_proximity_rate import compute_handle_proximity_geometry

    diagnostics = compute_handle_proximity_geometry(env, openable_object)
    handle_distance = float(diagnostics["handle_distance"][0].detach().cpu().item())
    openness = float(openable_object.get_openness(env)[0].detach().cpu().item())
    target = None
    if target_obj is not None and target_obj.numel() > 0:
        target = float(target_obj[0, 0].detach().cpu().item())
    return {
        "handle_distance": handle_distance,
        "openness": openness,
        "target_openness": target,
    }


def _print_handle_tracking(
    *,
    episode_index: int,
    traj_index: int,
    step: int,
    stats: dict[str, float],
) -> None:
    target_part = ""
    target = stats.get("target_openness")
    if target is not None:
        target_part = f" target_open={target:.6f} open_err={stats['openness'] - target:+.6f}"
    print(
        f"[HandleTracking][ep={episode_index} traj={traj_index} step={step:04d}] "
        f"handle_distance={stats['handle_distance']:.6f} openness={stats['openness']:.6f}"
        f"{target_part}",
        flush=True,
    )


def _summarize_handle_tracking(records: list[dict[str, float]], label: str = "summary") -> None:
    if not records:
        return
    distances = [record["handle_distance"] for record in records]
    openness = [record["openness"] for record in records]
    max_open = max(openness)
    final_open = openness[-1]
    print(
        f"[HandleTracking][{label}] samples={len(records)} "
        f"distance_min={min(distances):.6f} distance_final={distances[-1]:.6f} "
        f"open_max={max_open:.6f} open_final={final_open:.6f} "
        f"open_drop={max_open - final_open:.6f}",
        flush=True,
    )


def _read_actuator_velocity_limits(env: Any, term: Any) -> Any | None:
    import torch

    robot = env.scene["robot"]
    term_joint_ids = _joint_ids_to_list(term._joint_ids, robot.num_joints)
    limit_by_joint_id = {}

    for actuator in getattr(robot, "actuators", {}).values():
        actuator_joint_ids = _joint_ids_to_list(actuator.joint_indices, robot.num_joints)
        actuator_limits = getattr(actuator, "velocity_limit_sim", None)
        if actuator_limits is None:
            actuator_limits = getattr(actuator, "velocity_limit", None)
        if actuator_limits is None:
            continue
        if isinstance(actuator_limits, (float, int)):
            actuator_limits = torch.full(
                (len(actuator_joint_ids),),
                float(actuator_limits),
                dtype=torch.float32,
                device=robot.data.joint_pos.device,
            )
        else:
            actuator_limits = actuator_limits.detach()
            if actuator_limits.dim() == 2:
                actuator_limits = actuator_limits[0]
            actuator_limits = actuator_limits.flatten()
        for local_index, joint_id in enumerate(actuator_joint_ids):
            if local_index < actuator_limits.numel():
                limit_by_joint_id[joint_id] = float(actuator_limits[local_index].item())

    if not limit_by_joint_id:
        return None
    if any(joint_id not in limit_by_joint_id for joint_id in term_joint_ids):
        return None

    return torch.tensor(
        [limit_by_joint_id[joint_id] for joint_id in term_joint_ids],
        dtype=torch.float32,
    )


def _print_drive_velocity_feasibility(policy: Any, env: Any, term: Any) -> None:
    """Print a quick check for whether one-step drive tracking is kinematically feasible."""
    if policy.n_steps <= 1:
        return

    import torch

    joint_names = list(getattr(term, "_joint_names", []))
    step_dt = float(getattr(env, "step_dt", getattr(env, "physics_dt", 0.0)))
    if step_dt <= 0.0:
        print(
            "[JointTracking][velocity] Warning: could not determine env step_dt; skipping feasibility check.",
            flush=True,
        )
        return

    traj = policy._traj_robot.detach().float()
    robot_action_dim = _term_robot_action_dim(term)
    per_step_delta = (traj[:, 1:, :robot_action_dim] - traj[:, :-1, :robot_action_dim]).abs()
    required_velocity = per_step_delta / step_dt
    max_required_velocity = required_velocity.amax(dim=(0, 1)).detach().cpu()

    velocity_limits = _read_actuator_velocity_limits(env, term)

    print(f"[JointTracking][velocity] env_step_dt={step_dt:.6f}s", flush=True)
    if velocity_limits is None:
        print(
            "[JointTracking][velocity] Could not read robot velocity limits; "
            "printing required trajectory velocity only.",
            flush=True,
        )
        for joint_name, required in zip(joint_names, max_required_velocity.tolist()):
            print(f"  {joint_name}: max_required={required:.4f}", flush=True)
        return

    velocity_limits = velocity_limits.cpu()
    ratios = max_required_velocity / torch.clamp(velocity_limits.abs(), min=1e-9)
    worst_ratio, worst_index = torch.max(ratios, dim=0)
    infeasible = ratios > 1.0
    print(
        f"[JointTracking][velocity] worst_required/limit={float(worst_ratio):.3f} "
        f"at {joint_names[int(worst_index)]}",
        flush=True,
    )
    if bool(infeasible.any()):
        suggested_interpolation = max(1, int(torch.ceil(worst_ratio).item()))
        print(
            "[JointTracking][velocity] WARNING: at least one planned per-step joint delta "
            "exceeds the robot velocity limit for drive mode. The simulator state should "
            "not be expected to equal the .pt target after one env.step().",
            flush=True,
        )
        print(
            f"[JointTracking][velocity] Suggested minimum --interpolated factor from "
            f"velocity limits: {suggested_interpolation}",
            flush=True,
        )

    for joint_name, required, limit, ratio in zip(
        joint_names,
        max_required_velocity.tolist(),
        velocity_limits.tolist(),
        ratios.tolist(),
    ):
        marker = " !" if ratio > 1.0 else ""
        print(
            f"  {joint_name}: max_required={required:.4f} limit={limit:.4f} ratio={ratio:.3f}{marker}",
            flush=True,
        )


class RecordDebugHooks:
    """Optional debug hooks used by ``record_automoma_demos.py``."""

    def __init__(self, config: RecordDebugConfig):
        self.config = config
        self.joint_tracking_term = None
        self.joint_tracking_joint_names: list[str] | None = None
        self.joint_tracking_fk = None
        self.openable_object = None
        self.joint_tracking_records: list[dict] = []
        self.episode_joint_tracking_records: list[dict] = []
        self.handle_tracking_records: list[dict[str, float]] = []
        self.episode_handle_tracking_records: list[dict[str, float]] = []
        self.episode_index = 0
        self.traj_index = 0

    @property
    def tracking_enabled(self) -> bool:
        return self.config.tracking_enabled

    @property
    def init_steps(self) -> int:
        return self.config.record_init_steps

    @property
    def render_initial_state(self) -> bool:
        return self.config.record_init_render

    def configure_env(self, env_cfg: Any, enable_cameras: bool) -> None:
        if self.config.record_init_steps < 1:
            raise ValueError("--record_init_steps must be >= 1.")

        if self.config.record_decimation is not None:
            if self.config.record_decimation <= 0:
                raise ValueError("--record_decimation must be > 0.")
            env_cfg.decimation = self.config.record_decimation
            if self.config.record_render_interval is None:
                env_cfg.sim.render_interval = self.config.record_decimation
        if self.config.record_render_interval is not None:
            if self.config.record_render_interval <= 0:
                raise ValueError("--record_render_interval must be > 0.")
            env_cfg.sim.render_interval = self.config.record_render_interval
        _scale_robot_actuator_cfgs(env_cfg, self.config)
        _override_camera_resolution(env_cfg, self.config, enable_cameras)

    def setup(self, env: Any, env_cfg: Any, policy: Any) -> None:
        if not self.tracking_enabled:
            return

        self.openable_object = _resolve_openable_object(env_cfg)

        if self.config.joint_tracking:
            term_name, self.joint_tracking_term = _resolve_robot_action_term(env)
            self.joint_tracking_joint_names = list(getattr(self.joint_tracking_term, "_joint_names", []))
            print(f"[JointTracking] action_term={term_name}", flush=True)
            print(f"[JointTracking] joint_names={self.joint_tracking_joint_names}", flush=True)
            if not self.config.joint_tracking_no_fk:
                try:
                    self.joint_tracking_fk = _PinocchioFkDebugger(
                        self.joint_tracking_joint_names,
                        self.config.joint_tracking_fk_link,
                    )
                    print(
                        f"[JointTracking][fk] link={self.joint_tracking_fk.link_name} "
                        f"urdf={self.joint_tracking_fk.urdf_path}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[JointTracking][fk] Warning: could not initialize FK diagnostics: {exc}",
                        flush=True,
                    )
            if not policy.set_state:
                _print_drive_velocity_feasibility(policy, env, self.joint_tracking_term)
            if self.openable_object is None:
                print(
                    "[JointTracking][object] Warning: could not resolve openable object; "
                    "door angle will not be written to the debug curve.",
                    flush=True,
                )
            else:
                print(
                    f"[JointTracking][object] object={self.openable_object.name} "
                    f"joint={self.openable_object.openable_joint_name}",
                    flush=True,
                )

        if self.config.handle_tracking:
            if self.openable_object is None:
                raise RuntimeError("Could not resolve the openable object for handle tracking diagnostics.")
            print(
                f"[HandleTracking] object={self.openable_object.name} "
                f"handle_link={self.openable_object.handle_link_name}",
                flush=True,
            )

    def begin_episode(self, episode_index: int, traj_index: int) -> None:
        self.episode_index = episode_index
        self.traj_index = traj_index
        self.episode_joint_tracking_records = []
        self.episode_handle_tracking_records = []

    def after_initial_state(self, env: Any, policy: Any) -> None:
        if not self.config.joint_tracking:
            return

        start_target = policy.get_start_robot_joints().unsqueeze(0).to(env.device)
        start_actual = _read_robot_joint_pos_for_term(env, self.joint_tracking_term)
        start_obj_target = policy.get_start_obj_joints().unsqueeze(0).to(env.device)
        object_stats = _object_joint_tracking_stats(env, self.openable_object, start_obj_target)
        _print_joint_tracking(
            episode_index=self.episode_index,
            traj_index=self.traj_index,
            step=0,
            phase="initial_after_set_state",
            joint_names=self.joint_tracking_joint_names or [],
            target=start_target,
            actual=start_actual,
            tolerance=self.config.joint_tracking_tolerance,
            topk=self.config.joint_tracking_topk,
            fk_debugger=self.joint_tracking_fk,
            object_stats=object_stats,
        )

    def before_step(self, env: Any, policy: Any, action: Any, step: int) -> dict[str, Any]:
        if not self.tracking_enabled:
            return {}

        state: dict[str, Any] = {}
        if self.config.joint_tracking:
            state["robot_target"] = _action_robot_target(action, self.joint_tracking_term)

        target_obj = None
        if self.config.joint_tracking or self.config.handle_tracking:
            if policy.set_state:
                target_obj = action[:, policy._n_robot_joints :]
            else:
                target_obj = policy._traj_obj[policy.episode_index, step].unsqueeze(0).to(env.device)
        state["target_obj"] = target_obj
        return state

    def after_step(self, env: Any, step_state: dict[str, Any], step: int) -> None:
        if not self.tracking_enabled:
            return

        target_obj = step_state.get("target_obj")
        if self.config.joint_tracking:
            robot_actual = _read_robot_joint_pos_for_term(env, self.joint_tracking_term)
            object_stats = _object_joint_tracking_stats(env, self.openable_object, target_obj)
            if self._should_print_joint_tracking(step):
                stats = _print_joint_tracking(
                    episode_index=self.episode_index,
                    traj_index=self.traj_index,
                    step=step,
                    phase="post_step",
                    joint_names=self.joint_tracking_joint_names or [],
                    target=step_state["robot_target"],
                    actual=robot_actual,
                    tolerance=self.config.joint_tracking_tolerance,
                    topk=self.config.joint_tracking_topk,
                    fk_debugger=self.joint_tracking_fk,
                    object_stats=object_stats,
                )
            else:
                stats = _joint_tracking_stats(
                    episode_index=self.episode_index,
                    traj_index=self.traj_index,
                    step=step,
                    phase="post_step",
                    target=step_state["robot_target"],
                    actual=robot_actual,
                    tolerance=self.config.joint_tracking_tolerance,
                    fk_debugger=self.joint_tracking_fk,
                    object_stats=object_stats,
                )
            self.joint_tracking_records.append(stats)
            self.episode_joint_tracking_records.append(stats)

        if self.config.handle_tracking:
            handle_stats = _handle_tracking_stats(env, self.openable_object, target_obj)
            if self._should_print_handle_tracking(step):
                _print_handle_tracking(
                    episode_index=self.episode_index,
                    traj_index=self.traj_index,
                    step=step,
                    stats=handle_stats,
                )
            self.handle_tracking_records.append(handle_stats)
            self.episode_handle_tracking_records.append(handle_stats)

    def end_episode(self) -> None:
        if self.config.joint_tracking:
            _summarize_joint_tracking(
                self.episode_joint_tracking_records,
                self.config.joint_tracking_tolerance,
                self.config.joint_tracking_topk,
                label=f"episode_summary ep={self.episode_index} traj={self.traj_index}",
                joint_names=self.joint_tracking_joint_names,
            )
        if self.config.handle_tracking:
            _summarize_handle_tracking(
                self.episode_handle_tracking_records,
                label=f"episode_summary ep={self.episode_index} traj={self.traj_index}",
            )

    def finish(self, dataset_file: str) -> None:
        if self.config.joint_tracking:
            _summarize_joint_tracking(
                self.joint_tracking_records,
                self.config.joint_tracking_tolerance,
                self.config.joint_tracking_topk,
                joint_names=self.joint_tracking_joint_names,
            )
            _write_joint_tracking_artifacts(
                self.joint_tracking_records,
                self.joint_tracking_joint_names,
                dataset_file,
                self.config.joint_tracking_tolerance,
            )
        if self.config.handle_tracking:
            _summarize_handle_tracking(self.handle_tracking_records)

    def _should_print_joint_tracking(self, step: int) -> bool:
        if step < self.config.joint_tracking_steps:
            return True
        interval = self.config.joint_tracking_interval
        return interval > 0 and step % interval == 0

    def _should_print_handle_tracking(self, step: int) -> bool:
        if step < self.config.handle_tracking_steps:
            return True
        interval = self.config.handle_tracking_interval
        return interval > 0 and step % interval == 0
