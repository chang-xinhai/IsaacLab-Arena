# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Replay AutoMoMa planner trajectories and record them to HDF5.

This entrypoint is intentionally record-only: it replays pre-computed joint
trajectories in Isaac Lab and exports observations/actions/images into an HDF5
file suitable for LeRobot conversion and policy training. Use
``replay_automoma_demos.py`` for metrics-only replay without HDF5 recording.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import add_example_environments_cli_args
from isaaclab_arena.scripts.automoma_replay_common import add_automoma_replay_args
from isaaclab_arena.utils.automoma_record_debug import add_record_debug_args, make_record_debugger

# ---- CLI arguments ----
parser = get_isaaclab_arena_cli_parser()
add_automoma_replay_args(parser)
parser.add_argument(
    "--dataset_file",
    type=str,
    required=True,
    help="Output HDF5 file path for recorded demonstrations.",
)
parser.add_argument(
    "--validate_record_success",
    action="store_true",
    default=False,
    help=(
        "After each replayed trajectory, evaluate the final state with the same "
        "door-open-and-final-engaged rule used by eval. Failed episodes are "
        "removed from the output HDF5 unless --keep_failed_record_demos is set."
    ),
)
parser.add_argument(
    "--keep_failed_record_demos",
    action="store_true",
    default=False,
    help=(
        "When used with --validate_record_success, keep failed replay demos in "
        "the output HDF5 and only annotate success metadata."
    ),
)

add_record_debug_args(parser)
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import h5py
import torch

from isaaclab_arena.scripts.automoma_replay_common import (
    build_automoma_replay_context,
    evaluate_replay_success,
    print_replay_header,
    run_automoma_replay,
    safe_hdf5_attr_value,
)


def _sorted_demo_keys(data_group) -> list[str]:
    def demo_index(key: str) -> int:
        try:
            return int(key.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 10**12

    return sorted(data_group.keys(), key=demo_index)


def _compact_demo_keys(data_group) -> None:
    for new_index, old_key in enumerate(_sorted_demo_keys(data_group)):
        new_key = f"demo_{new_index}"
        if old_key != new_key:
            data_group.move(old_key, new_key)


def _postprocess_record_success(
    dataset_file: str,
    episode_results: list[dict[str, bool | float | int | None]],
    *,
    validate_record_success: bool,
    handle_distance_threshold: float,
    keep_failed_record_demos: bool,
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
            elif validate_record_success and not keep_failed_record_demos:
                failed_keys.append(demo_key)

            demo.attrs["success"] = success
            demo.attrs["record_success_checked"] = bool(validate_record_success)
            demo.attrs["traj_index"] = int(result["traj_index"])
            if validate_record_success:
                demo.attrs["final_door_open"] = bool(result["final_door_open"])
                demo.attrs["final_engaged"] = bool(result["final_engaged"])
                demo.attrs["final_door_openness"] = safe_hdf5_attr_value(result["final_door_openness"])
                demo.attrs["final_openness"] = safe_hdf5_attr_value(result["final_openness"])
                demo.attrs["final_handle_distance"] = safe_hdf5_attr_value(result["final_handle_distance"])

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
            data.attrs["record_success_keep_failed_demos"] = bool(keep_failed_record_demos)

        return successful, len(remaining_keys)


def _postprocess_mobile_base_relative(dataset_file: str, base_dof: int = 3) -> None:
    """Convert recorded base actions from absolute targets to relative deltas."""

    print(f"\n[PostProcess] Converting base actions to relative (base_dof={base_dof}) ...")
    with h5py.File(dataset_file, "r+") as f:
        if "data" not in f:
            print("[PostProcess] Warning: no 'data' group in HDF5, skipping.")
            return
        for demo_key in sorted(f["data"].keys()):
            demo = f["data"][demo_key]
            if "actions" not in demo or "obs" not in demo or "joint_pos" not in demo["obs"]:
                continue

            actions = demo["actions"][:]
            processed = demo["processed_actions"][:]
            joint_pos = demo["obs"]["joint_pos"][:]
            if actions.shape[0] == 0 or actions.shape[-1] < base_dof or joint_pos.shape[-1] < base_dof:
                continue

            delta_base = actions[:, :base_dof] - joint_pos[:, :base_dof]
            actions[:, :base_dof] = delta_base
            processed[:, :base_dof] = delta_base
            demo["actions"][...] = actions
            demo["processed_actions"][...] = processed

    print("[PostProcess] Done.")


def _default_success_result() -> dict[str, bool | float | None]:
    return {
        "success": True,
        "final_door_open": None,
        "final_engaged": None,
        "final_door_openness": None,
        "final_openness": None,
        "final_handle_distance": None,
    }


def main():
    record_debugger = make_record_debugger(args_cli)
    ctx = build_automoma_replay_context(
        args_cli,
        recorder_mode="hdf5",
        dataset_file=args_cli.dataset_file,
        require_openable_object=args_cli.validate_record_success,
    )
    record_debugger.setup(ctx.env, ctx.env_cfg, ctx.policy)

    print_replay_header(ctx, args_cli, title="Recording AutoMoMa demos", output_path=args_cli.dataset_file)
    if args_cli.validate_record_success:
        print(
            "Record success validation: enabled "
            f"(openness_threshold={args_cli.openness_threshold}, "
            f"handle_distance_threshold={args_cli.record_eval_handle_distance_threshold}, "
            f"use_fingertips={not args_cli.disable_fingertip_proximity})"
        )
    else:
        print("Record success validation: disabled (saving all replayed episodes)")

    episode_results: list[dict[str, bool | float | int | None]] = []

    def after_episode(local_ctx, _ep_idx: int, actual_ep: int) -> None:
        if args_cli.validate_record_success:
            success_result = evaluate_replay_success(local_ctx.env, local_ctx.openable_object, args_cli)
            print(
                f"[RecordSuccess] traj={actual_ep} success={success_result['success']} "
                f"door_open={success_result['final_door_open']} "
                f"final_engaged={success_result['final_engaged']} "
                f"openness={success_result['final_door_openness']} "
                f"handle_distance={success_result['final_handle_distance']}",
                flush=True,
            )
        else:
            success_result = _default_success_result()

        success_result["traj_index"] = actual_ep
        episode_results.append(success_result)
        local_ctx.env.recorder_manager.set_success_to_episodes(
            [0],
            torch.tensor([[bool(success_result["success"])]], dtype=torch.bool, device=local_ctx.env.device),
        )

    def export_last_episode(local_ctx) -> None:
        local_ctx.env.recorder_manager.record_pre_reset(torch.tensor([0], device=local_ctx.env.device))

    recorded_count = run_automoma_replay(
        ctx,
        args_cli,
        record_debugger=record_debugger,
        after_episode=after_episode,
        after_last_episode=export_last_episode,
    )

    print(f"\n{'=' * 60}")
    print(f"Recording complete: {recorded_count} episodes saved to {args_cli.dataset_file}")
    print(f"{'=' * 60}")
    record_debugger.finish(args_cli.dataset_file)

    ctx.env.close()

    successful_count, saved_count = _postprocess_record_success(
        args_cli.dataset_file,
        episode_results,
        validate_record_success=args_cli.validate_record_success,
        handle_distance_threshold=args_cli.record_eval_handle_distance_threshold,
        keep_failed_record_demos=args_cli.keep_failed_record_demos,
    )
    if args_cli.validate_record_success:
        attempted_count = len(episode_results)
        success_rate = successful_count / attempted_count if attempted_count else 0.0
        print(
            f"[RecordSuccess] attempted={attempted_count} successful={successful_count} "
            f"saved={saved_count} success_rate={success_rate:.2%}",
            flush=True,
        )

    if args_cli.mobile_base_relative and os.path.exists(args_cli.dataset_file):
        _postprocess_mobile_base_relative(args_cli.dataset_file, base_dof=3)


if __name__ == "__main__":
    main()
    simulation_app.close()
