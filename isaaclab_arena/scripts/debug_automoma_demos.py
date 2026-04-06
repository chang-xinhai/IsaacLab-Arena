# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to debug-visualize automoma planner IK solutions or trajectories.

This script loads a .pt file (either ik_data.pt or traj_data.pt), replays it
in the Isaac Lab simulator for visual inspection, without recording to HDF5.

Usage::

    # Debug IK solutions
    python isaaclab_arena/scripts/debug_automoma_demos.py \\
        --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/grasp_0001/ik_data.pt \\
        summit_franka_open_door \\
        --object_name microwave_7221 \\
        --scene_name scene_0_seed_0

    # Debug per-grasp trajectory
    python isaaclab_arena/scripts/debug_automoma_demos.py \\
        --debug_file data/trajs/summit_franka/microwave_7221/scene_0_seed_0/grasp_0001/traj_data.pt \\
        --set_state \\
        summit_franka_open_door \\
        --object_name microwave_7221 \\
        --scene_name scene_0_seed_0
"""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)

# ---- CLI arguments ----
parser = get_isaaclab_arena_cli_parser()

parser.add_argument(
    "--debug_file",
    type=str,
    required=True,
    help="Path to the .pt file (ik_data.pt or traj_data.pt) to visualize.",
)
parser.add_argument(
    "--set_state",
    action="store_true",
    default=True,
    help="Default to set_state (teleport) for debugging. Toggle to False for physics drive.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=5,
    help="Number of episodes/solutions to visualize.",
)
parser.add_argument(
    "--start_episode",
    type=int,
    default=0,
    help="Starting index in the file.",
)
parser.add_argument(
    "--interpolated",
    type=int,
    default=1,
    help="Interpolation factor for trajectories.",
)

add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import tempfile
import torch
import numpy as np
import tqdm

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab_arena.embodiments.summit_franka.summit_franka import SummitFrankaJointSpaceActionsCfg
from isaaclab_arena.policy.replay_automoma_trajectory_policy import ReplayAutomomaTrajectoryPolicy
from isaaclab_arena.utils.sim_utils import (
    deactivate_prims_by_name,
    disable_all_collisions,
    set_lighting_mode,
    sync_cameras_after_reset,
)


def _convert_11d_to_12d(data: dict) -> dict:
    """Helper to convert per-grasp 11D traj data to IsaacLab 12D format."""
    # Constants from pipeline.py
    gripper_open = 0.04
    gripper_closed = 0.0
    prepend_steps = 4
    lift_offset = 0.0

    def _split(t: torch.Tensor):
        arm = t[..., :10].clone()
        obj = t[..., 10:].clone()
        arm[..., 1] += lift_offset 
        obj = -obj
        return arm, obj

    def _state(t: torch.Tensor, grip_val: float):
        arm, obj = _split(t)
        pad = list(arm.shape[:-1]) + [2]
        grip = torch.full(pad, grip_val, dtype=t.dtype)
        return torch.cat([arm, grip], dim=-1), obj

    def _traj(t: torch.Tensor):
        B, T, _ = t.shape
        arm, obj = _split(t)
        g_arm = arm[:, 0:1].repeat(1, prepend_steps, 1)
        g_obj = obj[:, 0:1].repeat(1, prepend_steps, 1)
        closing = torch.linspace(gripper_open, gripper_closed, steps=prepend_steps, dtype=t.dtype)
        g_grip = closing.view(1, -1, 1).repeat(B, 1, 2)
        g_robot = torch.cat([g_arm, g_grip], dim=-1)
        p_grip = torch.full((B, T, 2), gripper_closed, dtype=t.dtype)
        p_robot = torch.cat([arm, p_grip], dim=-1)
        return (
            torch.cat([g_robot, p_robot], dim=1),
            torch.cat([g_obj, obj], dim=1),
        )

    out = {}
    # Handles both plural and singular keys for robustness
    s_state = data.get("start_states", data.get("start_state"))
    g_state = data.get("goal_states", data.get("goal_state"))
    t_traj = data.get("trajectories", data.get("traj"))
    
    out["start_robot"], out["start_obj"] = _state(s_state, gripper_open)
    out["goal_robot"], out["goal_obj"] = _state(g_state, gripper_closed)
    out["traj_robot"], out["traj_obj"] = _traj(t_traj)
    out["traj_success"] = data["success"]
    return out


def main():
    # ---- Build environment ----
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    env_cfg.actions = SummitFrankaJointSpaceActionsCfg()
    env_cfg.observations.policy.joint_pos = mdp_isaac_lab.ObservationTermCfg(func=mdp_isaac_lab.joint_pos)
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    # Debug playback should not record dataset metrics; some recorder terms
    # (e.g. success recorder) expect corresponding termination terms.
    env_cfg.recorders = {}

    import gymnasium as gym
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    # Scene fixes
    object_name = getattr(args_cli, "object_name", None)
    if object_name:
        deactivate_prims_by_name(object_name, exclude_paths=(), required_path_substrings=("/scene/",))
    set_lighting_mode(2)
    
    # Load data
    print(f"Loading debug file: {args_cli.debug_file}")
    data = torch.load(args_cli.debug_file, map_location="cpu", weights_only=False)
    
    # Detect mode
    if "iks" in data:
        mode = "IK"
        print(f"Detected IK data. {len(data['iks'])} solutions found.")
    elif "trajectories" in data or "traj" in data:
        mode = "TRAJ"
        print("Detected Trajectory data.")
    elif "traj_robot" in data:
        mode = "TRAJ_12D"
        print("Detected IsaacLab 12D Trajectory data.")
    else:
        raise ValueError(f"Unknown data format in {args_cli.debug_file}. Keys: {data.keys()}")

    obs, _ = env.reset()
    obs = sync_cameras_after_reset(env)

    if mode == "IK":
        iks = data["iks"]
        n_solutions = min(args_cli.num_episodes, len(iks) - args_cli.start_episode)
        
        for i in range(n_solutions):
            idx = args_cli.start_episode + i
            ik = iks[idx].clone()
            
            # Convert to 12D for visualization
            # IK data is usually 10D (base + arm)
            # The second DOF is base_y, so no lift offset is applied here.
            ik[1] += 0.0
            gripper = torch.tensor([0.04, 0.04]) # open
            ik_12d = torch.cat([ik, gripper])
            
            print(f"Visualizing IK solution {idx}/{len(iks)}")
            
            # Teleport robot
            robot = env.scene["robot"]
            robot.write_joint_state_to_sim(ik_12d.unsqueeze(0).to(env.device), torch.zeros((1, 12), device=env.device))
            env.scene.write_data_to_sim()
            
            # Render a few frames
            for _ in range(50):
                simulation_app.update()
                
    else: # TRAJ mode
        if mode == "TRAJ":
            # Convert 11D to 12D
            converted = _convert_11d_to_12d(data)
            # Save to temp file because Policy expects a path
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                torch.save(converted, tmp.name)
                traj_path = tmp.name
        else:
            traj_path = args_cli.debug_file

        policy = ReplayAutomomaTrajectoryPolicy(
            traj_file=traj_path,
            episode_index=args_cli.start_episode,
            set_state=args_cli.set_state,
            device=env.device,
            only_successful=True,
            interpolation_factor=args_cli.interpolated,
        )
        
        num_episodes = min(args_cli.num_episodes, policy.n_episodes - args_cli.start_episode)
        
        for ep_idx in range(num_episodes):
            actual_ep = args_cli.start_episode + ep_idx
            policy.episode_index = actual_ep
            policy.reset()
            
            if not args_cli.set_state:
                policy.set_initial_state(env)
            
            print(f"Visualizing Trajectory {actual_ep}/{policy.n_episodes}")
            for step in tqdm.tqdm(range(policy.n_steps), leave=False):
                with torch.no_grad():
                    action = policy.get_action(env, obs)
                obs, _, _, _, _ = env.step(action)
            
            if ep_idx < num_episodes - 1:
                env.reset()

        if mode == "TRAJ" and 'traj_path' in locals():
            os.remove(traj_path)

    print("Visualization complete.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
