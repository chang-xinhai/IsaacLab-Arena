# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Replay Automoma Trajectory Policy

Loads pre-computed trajectory data from a .pt file produced by the automoma planner
and replays joint positions in the simulator. Supports two modes:

- **Drive mode** (default): Sends joint positions as actions to the robot's actuators,
  letting physics simulation handle contacts and friction.
- **Set-state mode** (``--set_state``): Returns robot and object joint targets as a
  single action. A matching IsaacLab action term applies them by directly writing
  joint state during ``env.step(action)``.

Additional options:

- **Interpolation** (``--interpolated X``): Linearly interpolate between trajectory
  keyframes by factor X, producing smoother motion (X-times more steps).
- **Mobile-base-relative** (``--mobile_base_relative``): Return base joints as
  relative deltas (Δx, Δy, Δθ) instead of absolute positions. Arm and gripper joints
  remain absolute.

The .pt file format is::

    {
        "start_robot":   Tensor[N_episodes, n_robot_joints],
        "start_obj":     Tensor[N_episodes, n_obj_joints],
        "goal_robot":    Tensor[N_episodes, n_robot_joints],
        "goal_obj":      Tensor[N_episodes, n_obj_joints],
        "traj_robot":    Tensor[N_episodes, T, n_robot_joints],
        "traj_obj":      Tensor[N_episodes, T, n_obj_joints],
        "traj_success":  Tensor[N_episodes]  (bool),
    }

Usage with policy_runner.py::

    python isaaclab_arena/examples/policy_runner.py \\
        --enable_cameras \\
        --policy_type replay_automoma \\
        --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \\
        --episode_index 0 \\
        summit_franka_open_door \\
        --object_name microwave_7221 \\
        --scene_name scene_0_seed_0
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict
from pathlib import Path

from isaaclab_arena.policy.policy_base import PolicyBase


class ReplayAutomomaTrajectoryPolicy(PolicyBase):
    """
    Replay joint-space trajectories from an automoma planner .pt file.

    In **drive mode** (default), the policy outputs joint position targets as actions
    at each step, allowing physics (friction, contacts) to determine the outcome.

    In **set-state mode** (``set_state=True``), the policy returns robot and
    object joint targets as one action. The environment must use the matching
    AutoMoMa set-state action term to apply those targets.

    Args:
        traj_file: Path to the .pt trajectory data file.
        episode_index: Which episode to replay (0-indexed). Defaults to 0.
        set_state: If True, return robot + object state targets as the action.
        device: Torch device string.
        only_successful: If True (default), skip episodes where traj_success is False.
        interpolation_factor: Factor to interpolate between trajectory keyframes.
            1 = no interpolation (default). 4 = 3 intermediate frames between each pair.
        mobile_base_relative: If True, return base joints (first ``base_dof`` dims) as
            relative deltas from the current state. Arm and gripper remain absolute.
        base_dof: Number of base degrees of freedom (default 3: x, y, θ).
    """

    def __init__(
        self,
        traj_file: str,
        episode_index: int = 0,
        set_state: bool = False,
        device: str = "cuda",
        only_successful: bool = True,
        interpolation_factor: int = 1,
        mobile_base_relative: bool = False,
        base_dof: int = 3,
    ):
        super().__init__()
        self.set_state = set_state
        self.device = device
        self.interpolation_factor = max(1, interpolation_factor)
        self.mobile_base_relative = mobile_base_relative
        self.base_dof = base_dof

        # Load trajectory data
        traj_path = Path(traj_file)
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {traj_file}")

        data = torch.load(traj_file, map_location=device, weights_only=True)
        self._validate_data(data)

        # Filter to successful episodes if requested
        if only_successful and "traj_success" in data:
            success_mask = data["traj_success"].bool()
            n_success = success_mask.sum().item()
            if n_success == 0:
                raise ValueError(f"No successful episodes found in {traj_file}")
            self._traj_robot = data["traj_robot"][success_mask]
            self._traj_obj = data["traj_obj"][success_mask]
            self._start_robot = data["start_robot"][success_mask]
            self._start_obj = data["start_obj"][success_mask]
            self._traj_success = data["traj_success"][success_mask]
            print(
                f"[ReplayAutomomaTrajectoryPolicy] "
                f"{n_success}/{len(data['traj_success'])} successful episodes available."
            )
        else:
            self._traj_robot = data["traj_robot"]
            self._traj_obj = data["traj_obj"]
            self._start_robot = data["start_robot"]
            self._start_obj = data["start_obj"]
            self._traj_success = data.get("traj_success", None)

        self._n_episodes = self._traj_robot.shape[0]
        self._n_raw_steps = self._traj_robot.shape[1]
        self._n_robot_joints = self._traj_robot.shape[2]
        self._n_obj_joints = self._traj_obj.shape[2]

        if episode_index >= self._n_episodes:
            raise ValueError(
                f"Episode index {episode_index} out of range. "
                f"Available: 0..{self._n_episodes - 1}"
            )

        self._episode_index = episode_index
        self._current_step = 0

        # Pre-compute interpolated trajectories if needed
        if self.interpolation_factor > 1:
            self._traj_robot = self._interpolate_trajectory(self._traj_robot)
            self._traj_obj = self._interpolate_trajectory(self._traj_obj)

        self._n_steps = self._traj_robot.shape[1]

        print(
            f"[ReplayAutomomaTrajectoryPolicy] Loaded {self._n_episodes} episodes, "
            f"{self._n_raw_steps} raw steps → {self._n_steps} effective steps "
            f"(interp={self.interpolation_factor}x), "
            f"{self._n_robot_joints} robot joints, {self._n_obj_joints} object joints. "
            f"Replaying episode {self._episode_index}. "
            f"Mode: {'set_state' if self.set_state else 'drive (physics)'}. "
            f"Mobile base relative: {self.mobile_base_relative}."
        )

    def _interpolate_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate a trajectory by ``interpolation_factor``.

        Args:
            traj: Tensor of shape ``(N, T, J)`` — N episodes, T timesteps, J joints.

        Returns:
            Interpolated tensor of shape ``(N, T * factor, J)``.
        """
        N, T, J = traj.shape
        factor = self.interpolation_factor
        new_T = (T - 1) * factor + 1  # e.g. T=32, factor=4 → 125 steps
        result = torch.zeros(N, new_T, J, dtype=traj.dtype, device=traj.device)

        for i in range(T - 1):
            for k in range(factor):
                alpha = k / factor
                idx = i * factor + k
                result[:, idx, :] = (1.0 - alpha) * traj[:, i, :] + alpha * traj[:, i + 1, :]
        # Last frame
        result[:, -1, :] = traj[:, -1, :]

        return result

    @staticmethod
    def _validate_data(data: dict) -> None:
        required_keys = ["traj_robot", "traj_obj", "start_robot", "start_obj"]
        for key in required_keys:
            if key not in data:
                raise KeyError(
                    f"Missing required key '{key}' in trajectory data. "
                    f"Found keys: {list(data.keys())}"
                )
        assert data["traj_robot"].dim() == 3, (
            f"traj_robot must be 3D [N, T, J], got shape {data['traj_robot'].shape}"
        )
        assert data["traj_obj"].dim() == 3, (
            f"traj_obj must be 3D [N, T, J], got shape {data['traj_obj'].shape}"
        )

    @property
    def n_steps(self) -> int:
        """Number of (effective) steps in the current episode."""
        return self._n_steps

    @property
    def n_raw_steps(self) -> int:
        """Number of raw (un-interpolated) steps per episode."""
        return self._n_raw_steps

    @property
    def n_episodes(self) -> int:
        """Total number of (filtered) episodes."""
        return self._n_episodes

    @property
    def episode_index(self) -> int:
        return self._episode_index

    @episode_index.setter
    def episode_index(self, value: int) -> None:
        if value >= self._n_episodes:
            raise ValueError(f"Episode index {value} out of range (max {self._n_episodes - 1})")
        self._episode_index = value
        self._current_step = 0

    def get_start_robot_joints(self) -> torch.Tensor:
        """Get the starting robot joint positions for the current episode."""
        return self._start_robot[self._episode_index]

    def get_start_obj_joints(self) -> torch.Tensor:
        """Get the starting object joint positions for the current episode."""
        return self._start_obj[self._episode_index]

    def get_current_robot_target(self) -> torch.Tensor | None:
        """Get the robot joint target for the current step."""
        if self._current_step >= self._n_steps:
            return None
        return self._traj_robot[self._episode_index, self._current_step]

    def get_current_obj_target(self) -> torch.Tensor | None:
        """Get the object joint target for the current step."""
        if self._current_step >= self._n_steps:
            return None
        return self._traj_obj[self._episode_index, self._current_step]

    def is_done(self) -> bool:
        """Return True if all steps in the current episode have been replayed."""
        return self._current_step >= self._n_steps

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """
        Get the action for the current step.

        Always returns trajectory joint targets as the action tensor so the
        recorder captures the planned command. In set-state mode the action
        includes the trailing object joint target; simulator writes are handled
        by the environment action term, not by this policy.

        Args:
            env: The gymnasium environment.
            observation: Current observation dict.

        Returns:
            Action tensor of shape ``(1, action_dim)``.
        """
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        dev = torch.device(unwrapped.device)

        if self._current_step >= self._n_steps:
            # Episode finished — return zeros to keep env alive
            return torch.zeros((getattr(unwrapped, "num_envs", 1), env.action_space.shape[-1]), device=dev)

        robot_target = self._traj_robot[self._episode_index, self._current_step].clone()
        obj_target = self._traj_obj[self._episode_index, self._current_step].clone()

        if self.set_state:
            action = torch.cat([robot_target, obj_target], dim=-1).unsqueeze(0).to(dev)
        else:
            action = robot_target.unsqueeze(0).to(dev)

        expected_dim = env.action_space.shape[-1]
        if action.shape[-1] != expected_dim:
            mode = "set-state" if self.set_state else "drive"
            raise ValueError(
                f"ReplayAutomomaTrajectoryPolicy produced a {action.shape[-1]}D {mode} action, "
                f"but the environment expects {expected_dim}D. "
                "Check that env_cfg.actions matches the replay mode."
            )

        self._current_step += 1
        return action

    def _set_robot_joint_state(self, env: gym.Env, joint_positions: torch.Tensor) -> None:
        """Directly set robot joint positions and hold targets."""
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        robot = unwrapped.scene["robot"]
        n_joints = min(joint_positions.shape[0], robot.num_joints)
        joint_pos = joint_positions[:n_joints].unsqueeze(0).to(unwrapped.device)
        joint_vel = torch.zeros_like(joint_pos)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)
        robot.set_joint_velocity_target(joint_vel)

    def _set_object_joint_state(self, env: gym.Env, joint_positions: torch.Tensor) -> None:
        """Directly set object (articulation) joint positions and hold targets."""
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        for key in unwrapped.scene.keys():
            if key == "robot":
                continue
            entity = unwrapped.scene[key]
            if hasattr(entity, "write_joint_state_to_sim") and hasattr(entity, "num_joints"):
                n_joints = min(joint_positions.shape[0], entity.num_joints)
                joint_pos = joint_positions[:n_joints].unsqueeze(0).to(unwrapped.device)
                joint_vel = torch.zeros_like(joint_pos)
                entity.write_joint_state_to_sim(joint_pos, joint_vel)
                if hasattr(entity, "set_joint_position_target"):
                    entity.set_joint_position_target(joint_pos)
                if hasattr(entity, "set_joint_velocity_target"):
                    entity.set_joint_velocity_target(joint_vel)
                break

    def _sync_written_state(self, env: gym.Env, *, step: bool = False, render: bool = True) -> None:
        """Flush recently written articulation state to sim-side buffers."""
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        unwrapped.scene.write_data_to_sim()
        if step:
            unwrapped.sim.step(render=render)
        elif render and hasattr(unwrapped.sim, "render"):
            unwrapped.sim.render()
        unwrapped.scene.update(unwrapped.physics_dt)

    def set_initial_state(self, env: gym.Env, init_steps: int = 1, render: bool = True) -> None:
        """Set the robot and object to the trajectory's starting state.

        Useful to align the first observation with the planned start
        configuration before replay begins.
        """
        if init_steps < 1:
            raise ValueError("init_steps must be >= 1.")

        for _ in range(init_steps):
            self._set_robot_joint_state(env, self.get_start_robot_joints())
            self._set_object_joint_state(env, self.get_start_obj_joints())
            self._sync_written_state(env, step=True, render=render)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the step counter for the current episode."""
        self._current_step = 0

    def advance_episode(self) -> bool:
        """
        Advance to the next episode. Returns True if there is a next episode,
        False if we've exhausted all episodes.
        """
        next_idx = self._episode_index + 1
        if next_idx >= self._n_episodes:
            return False
        self._episode_index = next_idx
        self._current_step = 0
        return True
