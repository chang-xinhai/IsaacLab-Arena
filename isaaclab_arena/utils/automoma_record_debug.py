"""Debug helpers for AutoMoMa demonstration recording.

This module keeps verbose diagnostics out of ``record_automoma_demos.py`` while
letting the record script opt into joint and handle tracking from the CLI.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def add_record_debug_args(parser: Any) -> None:
    """Register optional tracking diagnostics on the record CLI parser."""
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help=(
            "Enable artifact-only AutoMoMa record diagnostics. This records joint and "
            "handle tracking data and writes CSV/PNG curves at the end without per-step prints."
        ),
    )
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


@dataclass(frozen=True)
class RecordDebugConfig:
    artifact_only: bool
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

    @classmethod
    def from_args(cls, args: Any) -> "RecordDebugConfig":
        artifact_only = bool(args.debug)
        return cls(
            artifact_only=artifact_only,
            joint_tracking=artifact_only or bool(args.debug_joint_tracking),
            joint_tracking_steps=0 if artifact_only else args.debug_joint_tracking_steps,
            joint_tracking_interval=0 if artifact_only else args.debug_joint_tracking_interval,
            joint_tracking_topk=args.debug_joint_tracking_topk,
            joint_tracking_tolerance=args.debug_joint_tracking_tolerance,
            joint_tracking_fk_link=args.debug_joint_tracking_fk_link,
            joint_tracking_no_fk=bool(args.debug_joint_tracking_no_fk),
            handle_tracking=artifact_only or bool(args.debug_handle_tracking),
            handle_tracking_steps=0 if artifact_only else args.debug_handle_tracking_steps,
            handle_tracking_interval=0 if artifact_only else args.debug_handle_tracking_interval,
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


def _rotation_matrix_to_quat_wxyz(rot: Any) -> Any:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    import torch

    m = rot.detach().float().cpu()
    trace = float(torch.trace(m).item())
    if trace > 0.0:
        s = max(trace + 1.0, 1e-12) ** 0.5 * 2.0
        quat = torch.tensor(
            [
                0.25 * s,
                float((m[2, 1] - m[1, 2]).item()) / s,
                float((m[0, 2] - m[2, 0]).item()) / s,
                float((m[1, 0] - m[0, 1]).item()) / s,
            ],
            dtype=torch.float32,
        )
    elif float(m[0, 0].item()) > float(m[1, 1].item()) and float(m[0, 0].item()) > float(m[2, 2].item()):
        s = max(1.0 + float(m[0, 0].item()) - float(m[1, 1].item()) - float(m[2, 2].item()), 1e-12) ** 0.5 * 2.0
        quat = torch.tensor(
            [
                float((m[2, 1] - m[1, 2]).item()) / s,
                0.25 * s,
                float((m[0, 1] + m[1, 0]).item()) / s,
                float((m[0, 2] + m[2, 0]).item()) / s,
            ],
            dtype=torch.float32,
        )
    elif float(m[1, 1].item()) > float(m[2, 2].item()):
        s = max(1.0 + float(m[1, 1].item()) - float(m[0, 0].item()) - float(m[2, 2].item()), 1e-12) ** 0.5 * 2.0
        quat = torch.tensor(
            [
                float((m[0, 2] - m[2, 0]).item()) / s,
                float((m[0, 1] + m[1, 0]).item()) / s,
                0.25 * s,
                float((m[1, 2] + m[2, 1]).item()) / s,
            ],
            dtype=torch.float32,
        )
    else:
        s = max(1.0 + float(m[2, 2].item()) - float(m[0, 0].item()) - float(m[1, 1].item()), 1e-12) ** 0.5 * 2.0
        quat = torch.tensor(
            [
                float((m[1, 0] - m[0, 1]).item()) / s,
                float((m[0, 2] + m[2, 0]).item()) / s,
                float((m[1, 2] + m[2, 1]).item()) / s,
                0.25 * s,
            ],
            dtype=torch.float32,
        )
    return quat / torch.clamp(torch.linalg.norm(quat), min=1e-9)


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
    target_quat = _rotation_matrix_to_quat_wxyz(target_rot)
    actual_quat = _rotation_matrix_to_quat_wxyz(actual_rot)

    return {
        "link_name": fk_debugger.link_name,
        "target_pos": target_pos,
        "actual_pos": actual_pos,
        "target_quat": target_quat,
        "actual_quat": actual_quat,
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


def _wrap_joint_names(joint_names: list[str], names_per_line: int = 3) -> str:
    lines = []
    for index in range(0, len(joint_names), names_per_line):
        lines.append(", ".join(joint_names[index : index + names_per_line]))
    return "\n".join(lines)


def _find_over_threshold_peaks(values: list[float], threshold: float) -> list[int]:
    peaks = []
    start = None
    for index, value in enumerate(values):
        if value > threshold:
            if start is None:
                start = index
            continue
        if start is not None:
            segment = range(start, index)
            peaks.append(max(segment, key=lambda item: values[item]))
            start = None
    if start is not None:
        segment = range(start, len(values))
        peaks.append(max(segment, key=lambda item: values[item]))
    return peaks


def _annotate_all_joint_error_peaks(
    ax: Any,
    plot_x: list[int],
    all_joint_curve: list[float],
    over_threshold_joints: list[list[str]],
    threshold: float,
) -> None:
    peak_indices = _find_over_threshold_peaks(all_joint_curve, threshold)
    if not peak_indices:
        return

    for annotation_index, peak_index in enumerate(peak_indices):
        joints = over_threshold_joints[peak_index]
        if not joints:
            continue
        x = plot_x[peak_index]
        y = all_joint_curve[peak_index]
        label = f">{threshold:g}: " + _wrap_joint_names(joints)
        y_offset = 16 + (annotation_index % 3) * 10
        ax.scatter([x], [y], color="tab:orange", s=24, zorder=5)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="tab:orange",
            arrowprops={
                "arrowstyle": "-",
                "color": "tab:orange",
                "alpha": 0.7,
                "linewidth": 0.8,
            },
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "tab:orange",
                "alpha": 0.78,
                "linewidth": 0.6,
            },
        )


def _trace_arrays(records: list[dict]) -> dict[str, Any]:
    import numpy as np

    steps = np.asarray([int(record["step"]) for record in records], dtype=np.int64)
    targets = np.asarray([record["target"].tolist() for record in records], dtype=np.float64)
    actuals = np.asarray([record["actual"].tolist() for record in records], dtype=np.float64)
    abs_errors = np.abs(actuals - targets)
    fk_target_pos = []
    fk_actual_pos = []
    fk_pos_gap = []
    fk_rot_gap = []
    for record in records:
        fk = record.get("fk")
        if fk is None:
            fk_target_pos.append([float("nan")] * 3)
            fk_actual_pos.append([float("nan")] * 3)
            fk_pos_gap.append(float("nan"))
            fk_rot_gap.append(float("nan"))
        else:
            fk_target_pos.append([float(value) for value in fk["target_pos"].detach().cpu().tolist()])
            fk_actual_pos.append([float(value) for value in fk["actual_pos"].detach().cpu().tolist()])
            fk_pos_gap.append(float(fk["pos_distance"]))
            fk_rot_gap.append(float(fk["rot_distance"]))
    return {
        "steps": steps,
        "targets": targets,
        "actuals": actuals,
        "abs_errors": abs_errors,
        "fk_target_pos": np.asarray(fk_target_pos, dtype=np.float64),
        "fk_actual_pos": np.asarray(fk_actual_pos, dtype=np.float64),
        "fk_pos_gap": np.asarray(fk_pos_gap, dtype=np.float64),
        "fk_rot_gap": np.asarray(fk_rot_gap, dtype=np.float64),
    }


def _safe_nanmax(values: Any) -> float:
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmax(values))


def _safe_nanlast(values: Any) -> float:
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite[-1])


def _joint_index_groups(joint_names: list[str]) -> dict[str, list[int]]:
    base = [index for index, name in enumerate(joint_names) if name.startswith("base_")]
    gripper = [
        index
        for index, name in enumerate(joint_names)
        if "finger" in name.lower() or "gripper" in name.lower()
    ]
    nongripper = [index for index in range(len(joint_names)) if index not in gripper]
    arm = [index for index in nongripper if index not in base]
    return {
        "base": base,
        "arm": arm,
        "gripper": gripper,
        "nongripper": nongripper,
    }


def _write_episode_trace_plot(
    records: list[dict],
    joint_names: list[str],
    output_path: Path,
    *,
    gripper_only_effective_steps: int = 20,
) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = _trace_arrays(records)
    steps = arrays["steps"]
    targets = arrays["targets"]
    actuals = arrays["actuals"]
    abs_errors = arrays["abs_errors"]
    target_pos = arrays["fk_target_pos"]
    actual_pos = arrays["fk_actual_pos"]
    pos_gap = arrays["fk_pos_gap"]
    rot_gap = arrays["fk_rot_gap"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_pose, ax_joint) = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)

    colors = ("tab:blue", "tab:green", "tab:red")
    labels = ("x", "y", "z")
    if target_pos.shape == actual_pos.shape and np.isfinite(target_pos).any() and np.isfinite(actual_pos).any():
        for axis, color, label in zip(range(3), colors, labels):
            ax_pose.plot(steps, target_pos[:, axis], color=color, linewidth=1.4, label=f"target {label}")
            ax_pose.plot(steps, actual_pos[:, axis], color=color, linestyle="--", linewidth=1.2, label=f"real {label}")
            ax_pose.fill_between(
                steps,
                target_pos[:, axis],
                actual_pos[:, axis],
                color=color,
                alpha=0.10,
                linewidth=0,
            )
    ax_pose.axvspan(
        steps[0],
        min(steps[-1], steps[0] + gripper_only_effective_steps - 1),
        color="tab:gray",
        alpha=0.08,
        label=f"first {gripper_only_effective_steps} steps",
    )
    ax_pose.set_title("EEF FK target vs real")
    ax_pose.set_xlabel("trace step")
    ax_pose.set_ylabel("world position (m)")
    ax_pose.grid(True, alpha=0.25)
    ax_pose.legend(loc="best", fontsize=8, ncol=2)
    ax_pose.text(
        0.01,
        0.98,
        f"max pos={_safe_nanmax(pos_gap):.4f} m\nmax rot={_safe_nanmax(rot_gap):.4f} rad",
        transform=ax_pose.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.78},
    )

    for joint_index, joint_name in enumerate(joint_names):
        ax_joint.plot(steps, abs_errors[:, joint_index], linewidth=1.0, label=joint_name)
    ax_joint.axvspan(
        steps[0],
        min(steps[-1], steps[0] + gripper_only_effective_steps - 1),
        color="tab:gray",
        alpha=0.08,
    )
    ax_joint.set_title("Per-joint absolute target-real gap")
    ax_joint.set_xlabel("trace step")
    ax_joint.set_ylabel("absolute joint gap")
    ax_joint.grid(True, alpha=0.25)
    ax_joint.legend(loc="best", fontsize=8, ncol=2)

    episode = records[0]["episode"]
    traj = records[0]["traj"]
    fig.suptitle(
        f"episode {episode} | traj={traj} | max_pos={_safe_nanmax(pos_gap):.4f}m "
        f"| max_joint={_safe_nanmax(abs_errors):.4f}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _write_episode_trace_artifacts(
    records: list[dict],
    joint_names: list[str],
    output_dir: Path,
    base_name: str,
) -> tuple[Path | None, list[Path]]:
    if not records:
        return None, []

    import csv
    import numpy as np

    by_episode: dict[tuple[int, int], list[dict]] = {}
    for record in records:
        by_episode.setdefault((int(record["episode"]), int(record["traj"])), []).append(record)

    summary_path = _make_unique_output_path(output_dir / f"{base_name}-episode-summary.csv")
    per_episode_dir = output_dir / f"{base_name}-episodes"
    groups = _joint_index_groups(joint_names)

    def span_max(values: Any) -> float:
        return _safe_nanmax(np.ptp(values, axis=0)) if values.size else float("nan")

    fieldnames = [
        "episode",
        "traj",
        "num_steps",
        "first20_steps",
        "max_joint_abs_gap",
        "max_base_abs_gap",
        "max_arm_abs_gap",
        "max_gripper_abs_gap",
        "first20_max_base_abs_gap",
        "first20_max_arm_abs_gap",
        "first20_max_gripper_abs_gap",
        "first20_max_nongripper_abs_gap",
        "first20_target_base_span",
        "first20_actual_base_span",
        "first20_target_arm_span",
        "first20_actual_arm_span",
        "first20_target_gripper_span",
        "first20_actual_gripper_span",
        "first20_target_nongripper_span",
        "first20_actual_nongripper_span",
        "max_eef_pos_gap_m",
        "final_eef_pos_gap_m",
        "max_eef_rot_gap_rad",
        "final_eef_rot_gap_rad",
        "plot",
    ]
    plot_paths: list[Path] = []
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (episode, traj), raw_episode_records in sorted(by_episode.items()):
            episode_records = sorted(raw_episode_records, key=lambda record: int(record["step"]))
            arrays = _trace_arrays(episode_records)
            abs_errors = arrays["abs_errors"]
            targets = arrays["targets"]
            actuals = arrays["actuals"]
            first_n = min(20, abs_errors.shape[0])
            first_slice = slice(0, first_n)
            nongripper = groups["nongripper"]
            base = groups["base"]
            arm = groups["arm"]
            gripper = groups["gripper"]
            first_base_abs = abs_errors[first_slice][:, base] if base else np.empty((first_n, 0))
            first_arm_abs = abs_errors[first_slice][:, arm] if arm else np.empty((first_n, 0))
            first_gripper_abs = abs_errors[first_slice][:, gripper] if gripper else np.empty((first_n, 0))
            first_base_targets = targets[first_slice][:, base] if base else np.empty((first_n, 0))
            first_base_actuals = actuals[first_slice][:, base] if base else np.empty((first_n, 0))
            first_arm_targets = targets[first_slice][:, arm] if arm else np.empty((first_n, 0))
            first_arm_actuals = actuals[first_slice][:, arm] if arm else np.empty((first_n, 0))
            first_gripper_targets = targets[first_slice][:, gripper] if gripper else np.empty((first_n, 0))
            first_gripper_actuals = actuals[first_slice][:, gripper] if gripper else np.empty((first_n, 0))
            first_targets = targets[first_slice][:, nongripper] if nongripper else np.empty((first_n, 0))
            first_actuals = actuals[first_slice][:, nongripper] if nongripper else np.empty((first_n, 0))
            first_nongripper_abs = (
                abs_errors[first_slice][:, nongripper] if nongripper else np.empty((first_n, 0))
            )
            plot_path = per_episode_dir / f"episode_{episode:03d}_traj_{traj:06d}_joint_eef_gap.png"
            _write_episode_trace_plot(episode_records, joint_names, plot_path)
            plot_paths.append(plot_path)
            writer.writerow({
                "episode": episode,
                "traj": traj,
                "num_steps": len(episode_records),
                "first20_steps": first_n,
                "max_joint_abs_gap": _safe_nanmax(abs_errors),
                "max_base_abs_gap": _safe_nanmax(abs_errors[:, base]) if base else float("nan"),
                "max_arm_abs_gap": _safe_nanmax(abs_errors[:, arm]) if arm else float("nan"),
                "max_gripper_abs_gap": _safe_nanmax(abs_errors[:, gripper]) if gripper else float("nan"),
                "first20_max_base_abs_gap": _safe_nanmax(first_base_abs),
                "first20_max_arm_abs_gap": _safe_nanmax(first_arm_abs),
                "first20_max_gripper_abs_gap": _safe_nanmax(first_gripper_abs),
                "first20_max_nongripper_abs_gap": _safe_nanmax(first_nongripper_abs),
                "first20_target_base_span": span_max(first_base_targets),
                "first20_actual_base_span": span_max(first_base_actuals),
                "first20_target_arm_span": span_max(first_arm_targets),
                "first20_actual_arm_span": span_max(first_arm_actuals),
                "first20_target_gripper_span": span_max(first_gripper_targets),
                "first20_actual_gripper_span": span_max(first_gripper_actuals),
                "first20_target_nongripper_span": span_max(first_targets),
                "first20_actual_nongripper_span": span_max(first_actuals),
                "max_eef_pos_gap_m": _safe_nanmax(arrays["fk_pos_gap"]),
                "final_eef_pos_gap_m": _safe_nanlast(arrays["fk_pos_gap"]),
                "max_eef_rot_gap_rad": _safe_nanmax(arrays["fk_rot_gap"]),
                "final_eef_rot_gap_rad": _safe_nanlast(arrays["fk_rot_gap"]),
                "plot": str(plot_path),
            })
    return summary_path, plot_paths


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
    joint_values_png_path = _make_unique_output_path(output_dir / f"{base_name}-joint-values.png")
    eef_pose_png_path = _make_unique_output_path(output_dir / f"{base_name}-eef-pose.png")

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
        "fk_eef_target_x",
        "fk_eef_target_y",
        "fk_eef_target_z",
        "fk_eef_actual_x",
        "fk_eef_actual_y",
        "fk_eef_actual_z",
        "fk_eef_error_x",
        "fk_eef_error_y",
        "fk_eef_error_z",
        "fk_eef_target_qw",
        "fk_eef_target_qx",
        "fk_eef_target_qy",
        "fk_eef_target_qz",
        "fk_eef_actual_qw",
        "fk_eef_actual_qx",
        "fk_eef_actual_qy",
        "fk_eef_actual_qz",
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
        fieldnames.extend((f"target_{name}", f"actual_{name}", f"err_{name}", f"abs_{name}"))

    plot_x = []
    base_curve = []
    all_joint_curve = []
    fk_curve = []
    object_joint_curve = []
    object_target_curve = []
    joint_target_curves = []
    joint_actual_curves = []
    fk_target_pos_curve = []
    fk_actual_pos_curve = []
    fk_rot_curve = []
    over_threshold_joints = []

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
            if fk is None:
                fk_target_pos = [float("nan")] * 3
                fk_actual_pos = [float("nan")] * 3
                fk_pos_error = [float("nan")] * 3
                fk_target_quat = [float("nan")] * 4
                fk_actual_quat = [float("nan")] * 4
            else:
                fk_target_pos = [float(value) for value in fk["target_pos"].detach().cpu().tolist()]
                fk_actual_pos = [float(value) for value in fk["actual_pos"].detach().cpu().tolist()]
                fk_pos_error = [float(value) for value in fk["pos_error"].detach().cpu().tolist()]
                fk_target_quat = [float(value) for value in fk["target_quat"].detach().cpu().tolist()]
                fk_actual_quat = [float(value) for value in fk["actual_quat"].detach().cpu().tolist()]
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
                "fk_eef_target_x": fk_target_pos[0],
                "fk_eef_target_y": fk_target_pos[1],
                "fk_eef_target_z": fk_target_pos[2],
                "fk_eef_actual_x": fk_actual_pos[0],
                "fk_eef_actual_y": fk_actual_pos[1],
                "fk_eef_actual_z": fk_actual_pos[2],
                "fk_eef_error_x": fk_pos_error[0],
                "fk_eef_error_y": fk_pos_error[1],
                "fk_eef_error_z": fk_pos_error[2],
                "fk_eef_target_qw": fk_target_quat[0],
                "fk_eef_target_qx": fk_target_quat[1],
                "fk_eef_target_qy": fk_target_quat[2],
                "fk_eef_target_qz": fk_target_quat[3],
                "fk_eef_actual_qw": fk_actual_quat[0],
                "fk_eef_actual_qx": fk_actual_quat[1],
                "fk_eef_actual_qy": fk_actual_quat[2],
                "fk_eef_actual_qz": fk_actual_quat[3],
                "object_name": object_name,
                "object_joint_name": object_joint_name,
                "object_joint_pos_rad": object_joint_pos,
                "object_joint_target_rad": object_joint_target,
                "object_joint_error_rad": object_joint_error,
                "object_openness": object_openness,
                "object_target_openness": object_target_openness,
                "object_open_error": object_open_error,
            }
            for column_name, target_value, actual_value, err_value, abs_value in zip(
                joint_columns,
                record["target"].tolist(),
                record["actual"].tolist(),
                error.tolist(),
                abs_error.tolist(),
            ):
                row[f"target_{column_name}"] = target_value
                row[f"actual_{column_name}"] = actual_value
                row[f"err_{column_name}"] = err_value
                row[f"abs_{column_name}"] = abs_value
            writer.writerow(row)

            plot_x.append(sample)
            base_curve.append(base_max_abs)
            all_joint_curve.append(record["max"])
            fk_curve.append(fk_pos)
            object_joint_curve.append(object_joint_pos)
            object_target_curve.append(object_joint_target)
            joint_target_curves.append(record["target"].tolist())
            joint_actual_curves.append(record["actual"].tolist())
            fk_target_pos_curve.append(fk_target_pos)
            fk_actual_pos_curve.append(fk_actual_pos)
            fk_rot_curve.append(fk_rot)
            over_threshold_joints.append(
                [
                    joint_name
                    for joint_name, abs_value in zip(joint_names, abs_error.tolist())
                    if abs_value > tolerance
                ]
            )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        ax.plot(plot_x, base_curve, linewidth=1.4, label="base max abs error (m/rad)")
        ax.plot(plot_x, all_joint_curve, linewidth=1.4, label="all-joint max abs error (m/rad)")
        ax.plot(plot_x, fk_curve, linewidth=1.4, label="fk_eef position error (m)")
        ax.axhline(tolerance, color="red", linestyle="--", linewidth=0.9, alpha=0.55)
        finite_error_values = [
            value
            for curve in (base_curve, all_joint_curve, fk_curve)
            for value in curve
            if np.isfinite(value)
        ]
        if finite_error_values:
            ax.set_ylim(top=max(max(finite_error_values), tolerance) * 1.35)
        _annotate_all_joint_error_peaks(ax, plot_x, all_joint_curve, over_threshold_joints, tolerance)
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

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        targets = np.asarray(joint_target_curves, dtype=np.float64)
        actuals = np.asarray(joint_actual_curves, dtype=np.float64)
        if targets.ndim == 2 and actuals.ndim == 2 and targets.shape == actuals.shape:
            cols = 3
            rows = int(np.ceil(len(joint_names) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(15, max(3, rows * 2.4)), sharex=True, constrained_layout=True)
            axes = np.asarray(axes).reshape(-1)
            for joint_index, joint_name in enumerate(joint_names):
                ax = axes[joint_index]
                ax.plot(plot_x, targets[:, joint_index], linestyle=":", linewidth=1.0, label="target")
                ax.plot(plot_x, actuals[:, joint_index], linewidth=1.0, label="actual")
                ax.set_title(joint_name, fontsize=9)
                ax.grid(True, alpha=0.25)
            for ax in axes[len(joint_names):]:
                ax.axis("off")
            axes[0].legend(loc="best", fontsize=8)
            fig.suptitle(f"Joint target vs actual: {dataset_path.stem}")
            fig.supxlabel("recorded tracking sample")
            fig.savefig(joint_values_png_path, dpi=180)
            plt.close(fig)
        else:
            joint_values_png_path = None
    except Exception as exc:
        print(
            f"[JointTracking][curve] Warning: failed to write joint-value plot {joint_values_png_path}: {exc}",
            flush=True,
        )
        joint_values_png_path = None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        target_pos = np.asarray(fk_target_pos_curve, dtype=np.float64)
        actual_pos = np.asarray(fk_actual_pos_curve, dtype=np.float64)
        if (
            target_pos.ndim == 2
            and actual_pos.ndim == 2
            and target_pos.shape == actual_pos.shape
            and np.isfinite(target_pos).any()
            and np.isfinite(actual_pos).any()
        ):
            labels = ("x", "y", "z")
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
            for axis_index, label in enumerate(labels):
                axes[0].plot(plot_x, target_pos[:, axis_index], linestyle=":", linewidth=1.0, label=f"target {label}")
                axes[0].plot(plot_x, actual_pos[:, axis_index], linewidth=1.0, label=f"actual {label}")
            axes[0].set_ylabel("EEF position (m)")
            axes[0].grid(True, alpha=0.25)
            axes[0].legend(loc="best", ncol=3, fontsize=8)

            axes[1].plot(plot_x, fk_curve, linewidth=1.2, label="position error (m)")
            axes[1].plot(plot_x, fk_rot_curve, linewidth=1.2, label="rotation error (rad)")
            axes[1].set_xlabel("recorded tracking sample")
            axes[1].set_ylabel("EEF error")
            axes[1].grid(True, alpha=0.25)
            axes[1].legend(loc="best")
            fig.suptitle(f"EEF FK target vs actual: {dataset_path.stem}")
            fig.savefig(eef_pose_png_path, dpi=180)
            plt.close(fig)
        else:
            eef_pose_png_path = None
    except Exception as exc:
        print(
            f"[JointTracking][curve] Warning: failed to write EEF pose plot {eef_pose_png_path}: {exc}",
            flush=True,
        )
        eef_pose_png_path = None

    print(f"[JointTracking][curve] csv={csv_path}", flush=True)
    if png_path is not None:
        print(f"[JointTracking][curve] png={png_path}", flush=True)
    if joint_values_png_path is not None:
        print(f"[JointTracking][curve] joint_values_png={joint_values_png_path}", flush=True)
    if eef_pose_png_path is not None:
        print(f"[JointTracking][curve] eef_pose_png={eef_pose_png_path}", flush=True)

    episode_summary_path, episode_plot_paths = _write_episode_trace_artifacts(
        records,
        joint_names,
        output_dir,
        base_name,
    )
    if episode_summary_path is not None:
        print(f"[JointTracking][curve] episode_summary_csv={episode_summary_path}", flush=True)
        print(
            f"[JointTracking][curve] episode_gap_plots={len(episode_plot_paths)} "
            f"dir={output_dir / f'{base_name}-episodes'}",
            flush=True,
        )
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


def _write_handle_tracking_artifacts(
    records: list[dict[str, Any]],
    dataset_file: str,
) -> tuple[Path | None, Path | None]:
    if not records:
        return None, None

    import numpy as np

    dataset_path = Path(dataset_file)
    output_dir = dataset_path.parent / "debug_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{dataset_path.stem}-handle-tracking"
    csv_path = _make_unique_output_path(output_dir / f"{base_name}.csv")
    png_path = _make_unique_output_path(output_dir / f"{base_name}.png")

    fieldnames = [
        "sample",
        "episode",
        "traj",
        "step",
        "handle_distance",
        "openness",
        "target_openness",
        "open_error",
    ]
    plot_x = []
    distances = []
    openness = []
    target_openness = []

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample, record in enumerate(records):
            target = record.get("target_openness")
            open_value = float(record["openness"])
            target_value = float(target) if target is not None else float("nan")
            distance = float(record["handle_distance"])
            row = {
                "sample": sample,
                "episode": record["episode"],
                "traj": record["traj"],
                "step": record["step"],
                "handle_distance": distance,
                "openness": open_value,
                "target_openness": target_value,
                "open_error": open_value - target_value if np.isfinite(target_value) else float("nan"),
            }
            writer.writerow(row)
            plot_x.append(sample)
            distances.append(distance)
            openness.append(open_value)
            target_openness.append(target_value)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        ax.plot(plot_x, distances, linewidth=1.4, label="handle distance (m)")
        ax.axhline(0.1, color="red", linestyle="--", linewidth=0.9, alpha=0.55, label="0.1m engagement")
        ax.set_xlabel("recorded tracking sample")
        ax.set_ylabel("handle distance (m)")
        ax.grid(True, alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(plot_x, openness, color="black", linewidth=1.2, label="door openness")
        if np.isfinite(np.asarray(target_openness, dtype=np.float64)).any():
            ax2.plot(
                plot_x,
                target_openness,
                color="black",
                linestyle=":",
                linewidth=1.0,
                label="target openness",
            )
        ax2.set_ylabel("door openness")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right")
        ax.set_title(f"Handle tracking: {dataset_path.stem}")
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"[HandleTracking][curve] Warning: failed to write plot {png_path}: {exc}", flush=True)
        png_path = None

    print(f"[HandleTracking][curve] csv={csv_path}", flush=True)
    if png_path is not None:
        print(f"[HandleTracking][curve] png={png_path}", flush=True)
    return csv_path, png_path


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
        self.handle_tracking_records: list[dict[str, Any]] = []
        self.episode_handle_tracking_records: list[dict[str, Any]] = []
        self.handle_tracking_disabled_reason: str | None = None
        self.episode_index = 0
        self.traj_index = 0

    @property
    def tracking_enabled(self) -> bool:
        return self.config.tracking_enabled

    def setup(self, env: Any, env_cfg: Any, policy: Any) -> None:
        if not self.tracking_enabled:
            return

        self.openable_object = _resolve_openable_object(env_cfg)

        if self.config.joint_tracking:
            term_name, self.joint_tracking_term = _resolve_robot_action_term(env)
            self.joint_tracking_joint_names = list(getattr(self.joint_tracking_term, "_joint_names", []))
            if not self.config.artifact_only:
                print(f"[JointTracking] action_term={term_name}", flush=True)
                print(f"[JointTracking] joint_names={self.joint_tracking_joint_names}", flush=True)
            if not self.config.joint_tracking_no_fk:
                try:
                    self.joint_tracking_fk = _PinocchioFkDebugger(
                        self.joint_tracking_joint_names,
                        self.config.joint_tracking_fk_link,
                    )
                    if not self.config.artifact_only:
                        print(
                            f"[JointTracking][fk] link={self.joint_tracking_fk.link_name} "
                            f"urdf={self.joint_tracking_fk.urdf_path}",
                            flush=True,
                        )
                except Exception as exc:
                    if not self.config.artifact_only:
                        print(
                            f"[JointTracking][fk] Warning: could not initialize FK diagnostics: {exc}",
                            flush=True,
                        )
            if not policy.set_state and not self.config.artifact_only:
                _print_drive_velocity_feasibility(policy, env, self.joint_tracking_term)
            if not self.config.artifact_only:
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
            if not self.config.artifact_only:
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
        if self.config.artifact_only:
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

        if self.config.handle_tracking and self.handle_tracking_disabled_reason is None:
            try:
                handle_stats = _handle_tracking_stats(env, self.openable_object, target_obj)
            except ValueError as exc:
                self.handle_tracking_disabled_reason = str(exc)
                print(f"[HandleTracking] disabled: {exc}", flush=True)
                return
            handle_stats.update(
                {
                    "episode": self.episode_index,
                    "traj": self.traj_index,
                    "step": step,
                }
            )
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
        if self.config.artifact_only:
            return
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
            if not self.config.artifact_only:
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
            if not self.config.artifact_only:
                _summarize_handle_tracking(self.handle_tracking_records)
            _write_handle_tracking_artifacts(self.handle_tracking_records, dataset_file)

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
