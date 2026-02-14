# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone collision-debug utility for Summit-Franka open-door scenes.

What this script does:
1) Builds the same environment pipeline used by replay/record scripts.
2) Audits collision APIs on robot and target object prim trees.
3) Optionally runs a gripper close/open probe and reports finger↔object contact force.
4) Best-effort enables collider debug visualization in the viewport.

Example:
    python isaaclab_arena/scripts/test_collision_meshes.py \
        --enable_cameras \
        --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
        summit_franka_open_door \
        --object_name microwave_7221 \
        --scene_name scene_0_seed_0 \
        --object_center
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--traj_file", type=str, default=None, help="Optional planner .pt trajectory file.")
parser.add_argument("--episode_index", type=int, default=10, help="Trajectory episode index if --traj_file is set.")
parser.add_argument("--steps_open", type=int, default=60, help="Open-gripper probe steps.")
parser.add_argument("--steps_close", type=int, default=80, help="Close-gripper probe steps.")
parser.add_argument("--steps_hold", type=int, default=80, help="Hold-closed probe steps.")
parser.add_argument(
    "--enable_collision_debug_vis",
    action="store_true",
    default=True,
    help="Best-effort enable PhysX collision visualization in viewport.",
)

add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
from collections import Counter

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg

from isaaclab_arena.assets.automoma_object_library import get_object_pose_from_metadata, load_scene_metadata
from isaaclab_arena.embodiments.summit_franka.summit_franka import SummitFrankaJointSpaceActionsCfg
from isaaclab_arena.policy.replay_automoma_trajectory_policy import ReplayAutomomaTrajectoryPolicy
from isaaclab_arena.utils.sim_utils import deactivate_prims_by_name, set_lighting_mode, sync_cameras_after_reset


def _try_enable_collision_debug_vis() -> bool:
    """Best-effort toggles for collision mesh visualization."""
    enabled = False
    try:
        import carb.settings

        settings = carb.settings.get_settings()
        candidate_keys = [
            "/persistent/physics/visualizationCollisionMeshes",
            "/persistent/physics/visualization/displayColliders",
            "/persistent/physics/visualization/collisionMeshes",
            "/persistent/physics/visualization/colliders",
        ]
        for key in candidate_keys:
            try:
                settings.set_bool(key, True)
                enabled = True
            except Exception:
                pass
    except Exception:
        pass

    try:
        import omni.kit.actions.core

        action_registry = omni.kit.actions.core.get_action_registry()
        candidate_actions = [
            ("omni.kit.viewport.menubar.physics", "toggle_collision_meshes"),
            ("omni.kit.viewport.menubar.physics", "toggle_colliders"),
        ]
        for ext, action_name in candidate_actions:
            action = action_registry.get_action(ext, action_name)
            if action is not None:
                action.execute()
                enabled = True
    except Exception:
        pass

    return enabled


def _resolve_first_prim(path_expr: str):
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    matches = sim_utils.find_matching_prims(path_expr, stage)
    return matches[0] if len(matches) > 0 else None


def _safe_attr_bool(attr, default: bool = True) -> bool:
    try:
        val = attr.Get()
        return default if val is None else bool(val)
    except Exception:
        return default


def _collect_collision_stats(root_prim_path: str) -> dict:
    from pxr import PhysxSchema, Usd, UsdPhysics

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        return {"root": root_prim_path, "valid": False}

    approx_counter = Counter()
    total_prims = 0
    collision_api_count = 0
    collision_enabled_count = 0
    collision_disabled_count = 0
    mesh_collision_api_count = 0
    physx_collision_api_count = 0

    for prim in Usd.PrimRange(root):
        total_prims += 1

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api_count += 1
            col_api = UsdPhysics.CollisionAPI(prim)
            if _safe_attr_bool(col_api.GetCollisionEnabledAttr(), default=True):
                collision_enabled_count += 1
            else:
                collision_disabled_count += 1

        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mesh_collision_api_count += 1
            approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            approx_counter[str(approx)] += 1

        if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            physx_collision_api_count += 1

    return {
        "root": root_prim_path,
        "valid": True,
        "total_prims": total_prims,
        "collision_api_count": collision_api_count,
        "collision_enabled_count": collision_enabled_count,
        "collision_disabled_count": collision_disabled_count,
        "mesh_collision_api_count": mesh_collision_api_count,
        "physx_collision_api_count": physx_collision_api_count,
        "mesh_approximation_histogram": dict(approx_counter),
    }


def _attach_finger_contact_sensor(env_cfg, object_name: str) -> str:
    sensor_name = "finger_object_contact_sensor"
    object_filter_paths = [
        f"{{ENV_REGEX_NS}}/{object_name}",
        f"{{ENV_REGEX_NS}}/{object_name}/.*",
        f"{{ENV_REGEX_NS}}/{object_name}.*",
    ]
    setattr(
        env_cfg.scene,
        sensor_name,
        ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_(left|right)finger",
            update_period=0.0,
            history_length=10,
            debug_vis=True,
            filter_prim_paths_expr=object_filter_paths,
        ),
    )
    return sensor_name


def _run_probe(env, sensor_name: str, object_name: str) -> dict:
    robot = env.scene["robot"]
    obj = env.scene[object_name]

    joint_names = list(robot.joint_names)
    finger_1_idx = joint_names.index("panda_finger_joint1")
    finger_2_idx = joint_names.index("panda_finger_joint2")

    def _step_with_finger_target(target_width: float):
        action = robot.data.joint_pos[0].clone()
        action[finger_1_idx] = target_width
        action[finger_2_idx] = target_width
        action = action.unsqueeze(0)
        env.step(action)

    max_force = 0.0
    min_obj_joint = float("inf")
    max_obj_joint = float("-inf")

    total_steps = args_cli.steps_open + args_cli.steps_close + args_cli.steps_hold
    for step in range(total_steps):
        if step < args_cli.steps_open:
            target = 0.04
        elif step < args_cli.steps_open + args_cli.steps_close:
            target = 0.0
        else:
            target = 0.0

        _step_with_finger_target(target)

        sensor = env.scene[sensor_name]
        if sensor.data.net_forces_w is not None:
            force_now = torch.linalg.norm(sensor.data.net_forces_w[0], dim=-1).max().item()
            max_force = max(max_force, float(force_now))

        if obj.num_joints > 0:
            joint0 = float(obj.data.joint_pos[0, 0].item())
            min_obj_joint = min(min_obj_joint, joint0)
            max_obj_joint = max(max_obj_joint, joint0)

    return {
        "max_finger_object_contact_force_N": max_force,
        "object_joint0_min": min_obj_joint,
        "object_joint0_max": max_obj_joint,
        "object_joint0_delta": (max_obj_joint - min_obj_joint) if min_obj_joint != float("inf") else 0.0,
    }


def main():
    if args_cli.num_envs != 1:
        print("[WARN] For collision debugging, forcing num_envs=1.")
        args_cli.num_envs = 1

    if args_cli.enable_collision_debug_vis:
        ok = _try_enable_collision_debug_vis()
        print(f"[INFO] Collision debug visualization requested: {'enabled' if ok else 'best-effort only'}")

    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    # Use direct joint-space actions so we can deterministically drive finger joints.
    env_cfg.actions = SummitFrankaJointSpaceActionsCfg()
    env_cfg.terminations.time_out = None

    sensor_name = _attach_finger_contact_sensor(env_cfg, args_cli.object_name)

    env = gym.make(env_name, cfg=env_cfg).unwrapped

    # Keep same post-creation behavior as recording pipeline.
    deactivate_prims_by_name(args_cli.object_name)
    set_lighting_mode(2)

    obs, _ = env.reset()
    obs = sync_cameras_after_reset(env)

    if args_cli.traj_file is not None:
        policy = ReplayAutomomaTrajectoryPolicy(
            traj_file=args_cli.traj_file,
            episode_index=args_cli.episode_index,
            set_state=False,
            device=args_cli.device,
            only_successful=True,
            interpolation_factor=1,
        )
        policy.set_initial_state(env)
        obs = sync_cameras_after_reset(env)

    # Resolve root prims for env_0.
    robot_root_prim = _resolve_first_prim("/World/envs/env_.*/Robot")
    object_root_prim = _resolve_first_prim(f"/World/envs/env_.*/{args_cli.object_name}")

    robot_root_path = str(robot_root_prim.GetPath()) if robot_root_prim is not None else ""
    object_root_path = str(object_root_prim.GetPath()) if object_root_prim is not None else ""

    robot_stats = _collect_collision_stats(robot_root_path) if robot_root_path else {"valid": False}
    object_stats = _collect_collision_stats(object_root_path) if object_root_path else {"valid": False}

    # Compare metadata scale vs spawned object scale.
    metadata = load_scene_metadata(args_cli.scene_name)
    _, metadata_scale = get_object_pose_from_metadata(
        metadata,
        asset_type="_".join(args_cli.object_name.split("_")[:-1]).capitalize(),
        asset_id=args_cli.object_name.split("_")[-1],
    )

    scene = getattr(env, "scene")
    spawned_scale = tuple(scene[args_cli.object_name].cfg.spawn.scale)

    probe_summary = _run_probe(env, sensor_name=sensor_name, object_name=args_cli.object_name)

    report = {
        "object_name": args_cli.object_name,
        "scene_name": args_cli.scene_name,
        "robot_root_prim": robot_root_path,
        "object_root_prim": object_root_path,
        "metadata_scale": metadata_scale,
        "spawned_object_scale": spawned_scale,
        "robot_collision_stats": robot_stats,
        "object_collision_stats": object_stats,
        "probe_summary": probe_summary,
    }

    print("\n" + "=" * 72)
    print("Collision debug report")
    print("=" * 72)
    print(json.dumps(report, indent=2))

    if object_stats.get("valid") and object_stats.get("collision_api_count", 0) == 0:
        print("[ALERT] Target object has zero CollisionAPI prims. Likely no physical collision mesh.")

    if probe_summary["max_finger_object_contact_force_N"] < 1e-3:
        print("[ALERT] Finger↔object contact force is near zero during close/hold probe.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
