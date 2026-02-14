# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Advanced collision + IK debug script for Summit-Franka open-door scenes.

This script extends test_collision_meshes.py with:
1) explicit collider AABB overlays (works even when native collider viewport toggle fails),
2) multiple IK seeds to approach handle candidates,
3) close-gripper contact checks at each IK target.

Example:
    python isaaclab_arena/scripts/test_collision_meshes_ik_debug.py \
      --enable_cameras \
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
parser.add_argument("--ik_iters", type=int, default=140, help="Max IK iterations per target.")
parser.add_argument("--ik_step", type=float, default=0.7, help="IK step size.")
parser.add_argument("--ik_damping", type=float, default=0.08, help="Damped least squares lambda.")
parser.add_argument("--ik_pos_tol", type=float, default=0.025, help="Position success tolerance (m).")
parser.add_argument("--show_collision_aabbs", action="store_true", default=True, help="Overlay collider AABBs.")
parser.add_argument("--max_aabbs", type=int, default=200, help="Maximum collision AABBs to draw.")
parser.add_argument("--gripper_close_steps", type=int, default=40, help="Steps to close/hold gripper at target.")

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
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg

from isaaclab_arena.embodiments.summit_franka.summit_franka import SummitFrankaJointSpaceActionsCfg
from isaaclab_arena.utils.sim_utils import deactivate_prims_by_name, set_lighting_mode, sync_cameras_after_reset


def _safe_attr_bool(attr, default: bool = True) -> bool:
    try:
        val = attr.Get()
        return default if val is None else bool(val)
    except Exception:
        return default


def _resolve_first_prim(path_expr: str):
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    matches = sim_utils.find_matching_prims(path_expr, stage)
    return matches[0] if len(matches) > 0 else None


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


def _get_world_aabb(prim):
    from pxr import Gf, Usd, UsdGeom

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    aligned: Gf.Range3d = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn = aligned.GetMin()
    mx = aligned.GetMax()
    center = ((mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5)
    extent = (max(mx[0] - mn[0], 1e-4), max(mx[1] - mn[1], 1e-4), max(mx[2] - mn[2], 1e-4))
    return center, extent


def _draw_collision_aabbs(root_prim_path: str, out_root: str, max_boxes: int = 200) -> int:
    from pxr import Usd, UsdGeom, UsdPhysics

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        return 0

    if not stage.GetPrimAtPath(out_root).IsValid():
        UsdGeom.Xform.Define(stage, out_root)

    count = 0
    for prim in Usd.PrimRange(root):
        if count >= max_boxes:
            break
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue

        center, extent = _get_world_aabb(prim)
        cube_path = f"{out_root}/box_{count:04d}"
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([(1.0, 0.25, 0.25)])

        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(center)
        xform.AddScaleOp().Set(extent)
        count += 1

    return count


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
            history_length=8,
            debug_vis=True,
            filter_prim_paths_expr=object_filter_paths,
        ),
    )
    return sensor_name


def _get_handle_candidates(object_root_prim_path: str) -> list[tuple[float, float, float]]:
    from pxr import Usd

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(object_root_prim_path)
    if not root.IsValid():
        return []

    # 1) Prefer prim names containing "handle"
    named = []
    for prim in Usd.PrimRange(root):
        if "handle" in prim.GetName().lower():
            center, _ = _get_world_aabb(prim)
            named.append(center)

    if named:
        return named[:6]

    # 2) Fallback: synthesize likely grasp points from object AABB
    center, extent = _get_world_aabb(root)
    cx, cy, cz = center
    ex, ey, ez = extent
    return [
        (cx + 0.50 * ex, cy, cz),
        (cx - 0.50 * ex, cy, cz),
        (cx, cy + 0.50 * ey, cz),
        (cx, cy - 0.50 * ey, cz),
        (cx, cy, cz + 0.20 * ez),
    ]


def _close_gripper_and_measure_contact(env, sensor_name: str, steps: int) -> float:
    robot = env.scene["robot"]
    joint_names = list(robot.joint_names)
    j1 = joint_names.index("panda_finger_joint1")
    j2 = joint_names.index("panda_finger_joint2")

    max_force = 0.0
    for _ in range(steps):
        action = robot.data.joint_pos[0].clone()
        action[j1] = 0.0
        action[j2] = 0.0
        env.step(action.unsqueeze(0))

        sensor = env.scene[sensor_name]
        if sensor.data.net_forces_w is not None:
            f = torch.linalg.norm(sensor.data.net_forces_w[0], dim=-1).max().item()
            max_force = max(max_force, float(f))
    return max_force


def _run_multi_seed_ik(env, sensor_name: str, object_root_prim_path: str) -> list[dict]:
    robot = env.scene["robot"]

    arm_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    arm_cfg.resolve(env.scene)

    if robot.is_fixed_base:
        ee_jac_idx = arm_cfg.body_ids[0] - 1
    else:
        ee_jac_idx = arm_cfg.body_ids[0]

    joint_ids = arm_cfg.joint_ids

    # Candidate targets around handle/object
    candidate_centers = _get_handle_candidates(object_root_prim_path)
    target_offsets = [
        (0.00, 0.00, 0.00),
        (0.00, 0.04, 0.00),
        (0.00, -0.04, 0.00),
        (0.03, 0.00, 0.00),
        (-0.03, 0.00, 0.00),
    ]
    targets = []
    for c in candidate_centers:
        for o in target_offsets:
            targets.append((c[0] + o[0], c[1] + o[1], c[2] + o[2]))

    # Multiple arm seeds to detect initialization sensitivity
    base = robot.data.default_joint_pos[0].clone()
    seed_deltas = [
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=base.device),
        torch.tensor([0.4, -0.2, 0.2, -0.4, 0.2, 0.3, 0.0], device=base.device),
        torch.tensor([-0.4, 0.2, -0.2, 0.4, -0.3, -0.2, 0.0], device=base.device),
        torch.tensor([0.2, -0.4, 0.3, -0.1, 0.4, -0.2, 0.1], device=base.device),
    ]

    records = []

    for seed_id, delta in enumerate(seed_deltas):
        q_all = base.clone()
        q_all[joint_ids] = q_all[joint_ids] + delta
        robot.write_joint_state_to_sim(q_all.unsqueeze(0), torch.zeros_like(q_all).unsqueeze(0))
        env.scene.write_data_to_sim()
        env.sim.step(render=True)
        env.scene.update(env.physics_dt)

        for target_id, target in enumerate(targets[:18]):  # keep runtime bounded
            target_t = torch.tensor(target, device=base.device)

            success = False
            last_err = None
            for _ in range(args_cli.ik_iters):
                jac = robot.root_physx_view.get_jacobians()[:, ee_jac_idx, :, joint_ids][0, :3, :]
                ee_pos = robot.data.body_pose_w[0, arm_cfg.body_ids[0], 0:3]
                err = target_t - ee_pos
                last_err = torch.linalg.norm(err).item()

                if last_err < args_cli.ik_pos_tol:
                    success = True
                    break

                jj_t = jac @ jac.T
                lam = args_cli.ik_damping
                dls = jac.T @ torch.linalg.solve(jj_t + (lam * lam) * torch.eye(3, device=jac.device), err)

                q_cur = robot.data.joint_pos[0].clone()
                q_cur[joint_ids] = q_cur[joint_ids] + args_cli.ik_step * dls
                robot.set_joint_position_target(q_cur[joint_ids].unsqueeze(0), joint_ids=joint_ids)

                env.scene.write_data_to_sim()
                env.sim.step(render=True)
                env.scene.update(env.physics_dt)

            # Close gripper and measure contact force at reached pose
            max_contact = _close_gripper_and_measure_contact(env, sensor_name, args_cli.gripper_close_steps)

            records.append(
                {
                    "seed_id": seed_id,
                    "target_id": target_id,
                    "target_xyz": [float(target_t[0]), float(target_t[1]), float(target_t[2])],
                    "ik_success": bool(success),
                    "final_pos_error_m": float(last_err if last_err is not None else 1e9),
                    "max_finger_object_contact_force_N": float(max_contact),
                }
            )

    return records


def main():
    if args_cli.num_envs != 1:
        print("[WARN] For collision/IK debugging, forcing num_envs=1.")
        args_cli.num_envs = 1

    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    env_cfg.actions = SummitFrankaJointSpaceActionsCfg()
    env_cfg.terminations.time_out = None
    sensor_name = _attach_finger_contact_sensor(env_cfg, args_cli.object_name)

    env = gym.make(env_name, cfg=env_cfg).unwrapped

    deactivate_prims_by_name(
        args_cli.object_name,
        exclude_paths=(),
        required_path_substrings=("/scene/",),
    )
    set_lighting_mode(2)

    env.reset()
    sync_cameras_after_reset(env)

    robot_root_prim = _resolve_first_prim("/World/envs/env_.*/Robot")
    object_root_prim = _resolve_first_prim(f"/World/envs/env_.*/{args_cli.object_name}")

    robot_root_path = str(robot_root_prim.GetPath()) if robot_root_prim is not None else ""
    object_root_path = str(object_root_prim.GetPath()) if object_root_prim is not None else ""

    robot_stats = _collect_collision_stats(robot_root_path) if robot_root_path else {"valid": False}
    object_stats = _collect_collision_stats(object_root_path) if object_root_path else {"valid": False}

    drawn = 0
    if args_cli.show_collision_aabbs and robot_root_path and object_root_path:
        drawn += _draw_collision_aabbs(robot_root_path, "/Visuals/CollisionAABBs/Robot", max_boxes=args_cli.max_aabbs)
        drawn += _draw_collision_aabbs(object_root_path, "/Visuals/CollisionAABBs/Object", max_boxes=args_cli.max_aabbs)

    ik_records = _run_multi_seed_ik(env, sensor_name=sensor_name, object_root_prim_path=object_root_path)

    n_success = sum(1 for r in ik_records if r["ik_success"])
    n_touch = sum(1 for r in ik_records if r["max_finger_object_contact_force_N"] > 1e-3)

    report = {
        "object_name": args_cli.object_name,
        "scene_name": args_cli.scene_name,
        "robot_root_prim": robot_root_path,
        "object_root_prim": object_root_path,
        "collision_aabbs_drawn": drawn,
        "robot_collision_stats": robot_stats,
        "object_collision_stats": object_stats,
        "ik_trials_total": len(ik_records),
        "ik_trials_success": n_success,
        "ik_trials_with_contact": n_touch,
        "ik_records": ik_records,
    }

    print("\n" + "=" * 88)
    print("Collision + multi-seed IK debug report")
    print("=" * 88)
    print(json.dumps(report, indent=2))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
