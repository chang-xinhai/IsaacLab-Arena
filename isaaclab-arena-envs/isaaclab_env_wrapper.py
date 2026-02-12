from __future__ import annotations

import atexit
import logging
import os
import signal
from contextlib import suppress
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .errors import IsaacLabArenaError


def cleanup_isaaclab(env, simulation_app) -> None:
    """Cleanup IsaacLab env and simulation app resources."""
    # Ignore signals during cleanup to prevent interruption
    old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    old_sigterm = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        with suppress(Exception):
            if env is not None:
                env.close()
        with suppress(Exception):
            if simulation_app is not None:
                simulation_app.app.close()
    finally:
        # Restore signal handlers
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


class IsaacLabEnvWrapper(gym.vector.AsyncVectorEnv):
    """Wrapper adapting IsaacLab batched GPU env to AsyncVectorEnv.
    IsaacLab handles vectorization internally on GPU. We inherit from
    AsyncVectorEnv for compatibility with LeRobot."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    _cleanup_in_progress = False  # Class-level flag for re-entrant protection

    def __init__(
        self,
        env,
        episode_length: int = 500,
        task: str | None = None,
        render_mode: str | None = "rgb_array",
        simulation_app=None,
        mobile_base_relative: bool = False,
        base_dof: int = 3,
        state_key: str = "joint_pos",
    ):
        self._env = env
        self._num_envs = env.num_envs
        self._episode_length = episode_length
        self._closed = False
        self.render_mode = render_mode
        self._simulation_app = simulation_app

        # Mobile-base-relative: convert policy delta-base actions to absolute
        self._mobile_base_relative = mobile_base_relative
        self._base_dof = base_dof
        self._state_key = state_key

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self.task = task

        if hasattr(env, "metadata") and env.metadata:
            self.metadata = {**self.metadata, **env.metadata}

        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        if IsaacLabEnvWrapper._cleanup_in_progress:
            return  # Prevent re-entrant cleanup
        IsaacLabEnvWrapper._cleanup_in_progress = True
        logging.info(f"Received signal {signum}, cleaning up...")
        self._cleanup()
        # Exit without raising to avoid propagating through callbacks
        os._exit(0)

    def _check_closed(self):
        if self._closed:
            raise IsaacLabArenaError()

    @property
    def unwrapped(self):
        return self

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def _max_episode_steps(self) -> int:
        return self._episode_length

    @property
    def device(self) -> str:
        return getattr(self._env, "device", "cpu")

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._check_closed()
        if isinstance(seed, (list, tuple, range)):
            seed = seed[0] if len(seed) > 0 else None

        try:
            obs, info = self._env.reset(seed=seed, options=options)
        except Exception as exc:
            if self._is_invalid_physx_view_error(exc):
                logging.warning(
                    "Detected invalid PhysX view during reset. Attempting recovery and retrying reset..."
                )
                self._try_recover_physx_views()
                obs, info = self._env.reset(seed=seed, options=options)
            else:
                raise

        if "final_info" not in info:
            zeros = np.zeros(self._num_envs, dtype=bool)
            info["final_info"] = {"is_success": zeros}

        return obs, info

    @staticmethod
    def _is_invalid_physx_view_error(exc: Exception) -> bool:
        message = str(exc)
        return (
            "Simulation view object is invalidated" in message
            or "Failed to set DOF positions in backend" in message
            or "setDofPositions" in message
        )

    def _try_recover_physx_views(self) -> None:
        """Best-effort recovery after USD/PhysX view invalidation."""
        try:
            if hasattr(self._env, "sim") and hasattr(self._env.sim, "reset"):
                self._env.sim.reset()
        except Exception as exc:
            logging.warning(f"PhysX recovery: sim.reset() failed: {exc}")

        try:
            if hasattr(self._env, "scene"):
                dt = getattr(self._env, "physics_dt", 0.0)
                self._env.scene.update(dt)
        except Exception as exc:
            logging.warning(f"PhysX recovery: scene.update() failed: {exc}")

    def step(
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
        self._check_closed()
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self._env.device)

        # Mobile-base-relative: integrate Δbase with current base state
        if self._mobile_base_relative and self._base_dof > 0:
            actions = self._integrate_base_deltas(actions)

        obs, reward, terminated, truncated, info = self._env.step(actions)

        # Convert to numpy for gym compatibility
        reward = reward.cpu().numpy().astype(np.float32)
        terminated = terminated.cpu().numpy().astype(bool)
        truncated = truncated.cpu().numpy().astype(bool)

        is_success = self._get_success(terminated, truncated)
        info["final_info"] = {"is_success": is_success}

        return obs, reward, terminated, truncated, info

    def _get_success(self, terminated: np.ndarray, truncated: np.ndarray) -> np.ndarray:
        is_success = np.zeros(self._num_envs, dtype=bool)

        if not hasattr(self._env, "termination_manager"):
            return is_success & (terminated | truncated)

        term_manager = self._env.termination_manager
        if not hasattr(term_manager, "get_term"):
            return is_success & (terminated | truncated)

        success_tensor = term_manager.get_term("success")
        if success_tensor is None:
            return is_success & (terminated | truncated)

        is_success = success_tensor.cpu().numpy().astype(bool)

        return is_success & (terminated | truncated)

    def _integrate_base_deltas(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert relative base deltas to absolute positions.

        Reads the current base joint state from the env's observation buffer,
        then replaces ``actions[:, :base_dof]`` with
        ``current_base + delta_base``.
        """
        try:
            obs_buf = self._env.obs_buf
            if isinstance(obs_buf, dict) and "policy" in obs_buf:
                policy_obs = obs_buf["policy"]
                if self._state_key in policy_obs:
                    current_state = policy_obs[self._state_key]  # (B, D)
                    new_actions = actions.clone()
                    new_actions[:, :self._base_dof] = (
                        current_state[:, :self._base_dof] + actions[:, :self._base_dof]
                    )
                    return new_actions
        except Exception as e:
            logging.warning(f"[MobileBaseRelative] Could not read base state: {e}")
        return actions

    def call(self, method_name: str, *args, **kwargs) -> list[Any]:
        if method_name == "_max_episode_steps":
            return [self._episode_length] * self._num_envs
        if method_name == "task":
            return [self.task] * self._num_envs
        if method_name == "render":
            return self.render_all()

        if hasattr(self._env, method_name):
            attr = getattr(self._env, method_name)
            result = attr(*args, **kwargs) if callable(attr) else attr
            if isinstance(result, list):
                return result
            return [result] * self._num_envs

        raise AttributeError(f"IsaacLab-Arena has no method/attribute '{method_name}'")

    def render_all(self) -> list[np.ndarray]:
        self._check_closed()
        frames = self.render()
        if frames is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            return [placeholder] * self._num_envs

        return [frames] * self._num_envs

    def render(self) -> np.ndarray | None:
        """Render all environments and return list of frames."""
        self._check_closed()
        if self.render_mode != "rgb_array":
            return None

        frames = self._env.render() if hasattr(self._env, "render") else None
        if frames is None:
            return None

        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        return frames[0] if frames.ndim == 4 else frames

    def _cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        IsaacLabEnvWrapper._cleanup_in_progress = True
        logging.info("Cleaning up IsaacLab Arena environment...")
        cleanup_isaaclab(self._env, self._simulation_app)

    def close(self) -> None:
        self._cleanup()

    @property
    def envs(self) -> list[IsaacLabEnvWrapper]:
        return [self] * self._num_envs

    def __del__(self):
        self._cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
        return False
