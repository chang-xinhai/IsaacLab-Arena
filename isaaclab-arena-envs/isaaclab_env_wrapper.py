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
        traj_file: str | None = None,
        traj_seed: int = 42,
        traj_selection_mode: str = "random",
        handle_distance_threshold: float = 0.1,
        interpolation_factor: int = 1,
        interpolation_type: str = "linear",
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
        self._handle_distance_threshold = handle_distance_threshold
        self.interpolation_factor = max(1, int(interpolation_factor))
        self.interpolation_type = str(interpolation_type)
        if self.interpolation_factor > 1 and self.interpolation_type != "none":
            logging.info(
                "[IsaacLabEnvWrapper] Action interpolation configured for external executor: "
                f"{self.interpolation_factor}x, type={self.interpolation_type}"
            )

        # Trajectory-based initial state for evaluation
        self._traj_data = None
        self._traj_rng = None
        self._traj_n_episodes = 0
        self._traj_selection_mode = str(traj_selection_mode)
        self._traj_next_episode = 0
        if self._traj_selection_mode not in {"random", "sequential"}:
            raise ValueError(
                "traj_selection_mode must be 'random' or 'sequential', "
                f"got {self._traj_selection_mode!r}"
            )
        if traj_file and os.path.exists(traj_file):
            data = torch.load(traj_file, map_location="cpu", weights_only=True)
            # Validate required keys
            for k in ("start_robot", "start_obj"):
                if k not in data:
                    logging.warning(
                        f"[IsaacLabEnvWrapper] Trajectory file missing key '{k}', "
                        f"initial state setting disabled."
                    )
                    data = None
                    break
            if data is not None:
                # Filter to successful episodes if available
                if "traj_success" in data:
                    mask = data["traj_success"].bool()
                    n_success = mask.sum().item()
                    if n_success > 0:
                        data["start_robot"] = data["start_robot"][mask]
                        data["start_obj"] = data["start_obj"][mask]
                        logging.info(
                            f"[IsaacLabEnvWrapper] Loaded {n_success} successful "
                            f"episodes from {traj_file}"
                        )
                    else:
                        logging.warning(
                            f"[IsaacLabEnvWrapper] No successful episodes in {traj_file}, "
                            f"using all episodes."
                        )
                else:
                    logging.info(
                        f"[IsaacLabEnvWrapper] Loaded {data['start_robot'].shape[0]} "
                        f"episodes from {traj_file} (no success filter)"
                    )
                self._traj_data = data
                self._traj_n_episodes = data["start_robot"].shape[0]
                self._traj_rng = np.random.RandomState(traj_seed)
                logging.info(
                    f"[IsaacLabEnvWrapper] Trajectory initial states enabled: "
                    f"{self._traj_n_episodes} episodes, seed={traj_seed}, "
                    f"selection_mode={self._traj_selection_mode}"
                )
        elif traj_file:
            logging.warning(
                f"[IsaacLabEnvWrapper] Trajectory file not found: {traj_file}"
            )

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self.task = task

        if hasattr(env, "metadata") and env.metadata:
            self.metadata = {**self.metadata, **env.metadata}

        self._episode_summaries = [self._make_empty_episode_summary() for _ in range(self._num_envs)]

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

    @staticmethod
    def _make_empty_episode_summary() -> dict[str, float | bool | None]:
        return {
            "final_openness": None,
            "final_door_openness": None,
            "final_door_open": False,
            "final_engaged": False,
            "final_handle_distance": None,
        }

    @staticmethod
    def _tensor_to_numpy(value: torch.Tensor | None) -> np.ndarray | None:
        if value is None:
            return None
        return value.detach().cpu().numpy()

    def _update_episode_summaries(self) -> None:
        diagnostics = None
        try:
            from isaaclab_arena.metrics.handle_proximity_rate import get_cached_handle_proximity_diagnostics

            task = getattr(getattr(self._env, "cfg", None), "isaaclab_arena_env", None)
            openable_object = getattr(getattr(task, "task", None), "openable_object", None)
            if openable_object is not None:
                diagnostics = get_cached_handle_proximity_diagnostics(self._env, openable_object)
        except Exception as exc:
            logging.debug(f"[IsaacLabEnvWrapper] Failed to read handle diagnostics cache: {exc}")
            diagnostics = None

        if diagnostics is None:
            return

        openness = self._tensor_to_numpy(diagnostics.get("openness"))
        door_open = self._tensor_to_numpy(diagnostics.get("door_open"))
        handle_distance = self._tensor_to_numpy(diagnostics.get("handle_distance"))

        for env_ix in range(self._num_envs):
            summary = self._episode_summaries[env_ix]
            if openness is not None:
                openness_value = float(openness[env_ix])
                summary["final_openness"] = openness_value
                summary["final_door_openness"] = openness_value
            if door_open is not None:
                summary["final_door_open"] = bool(door_open[env_ix])
            if handle_distance is not None:
                distance_value = float(handle_distance[env_ix])
                summary["final_handle_distance"] = distance_value
                summary["final_engaged"] = bool(distance_value <= self._handle_distance_threshold)

    def _finalize_episode_summaries(
        self,
        terminated: np.ndarray,
        truncated: np.ndarray,
        is_success: np.ndarray,
    ) -> dict[str, np.ndarray]:
        done = terminated | truncated
        final_info: dict[str, np.ndarray] = {"is_success": is_success}
        for key in self._episode_summaries[0]:
            if isinstance(self._episode_summaries[0][key], bool):
                values = np.zeros(self._num_envs, dtype=bool)
            else:
                values = np.full(self._num_envs, np.nan, dtype=np.float32)
            for env_ix in range(self._num_envs):
                if done[env_ix]:
                    value = self._episode_summaries[env_ix][key]
                    if isinstance(self._episode_summaries[0][key], bool):
                        values[env_ix] = bool(value)
                    elif value is not None:
                        values[env_ix] = float(value)
            final_info[key] = values
        for env_ix in range(self._num_envs):
            if done[env_ix]:
                self._episode_summaries[env_ix] = self._make_empty_episode_summary()
        return final_info

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

        # Set initial state from trajectory data (for evaluation with varying start poses)
        if self._traj_data is not None:
            obs = self._set_initial_state_from_traj(obs)

        self._episode_summaries = [self._make_empty_episode_summary() for _ in range(self._num_envs)]

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

    def _set_initial_state_from_traj(self, obs: dict) -> dict:
        """Set robot and object to a randomly sampled trajectory start state.

        Samples an episode index from the loaded trajectory data using the
        seeded RNG, then teleports the robot and object joints to the
        corresponding start positions.  After teleporting, a render pass
        and observation recompute ensure fresh camera images.

        Args:
            obs: Current observation dict (will be replaced with fresh obs).

        Returns:
            Recomputed observation dict reflecting the new initial state.
        """
        if self._traj_selection_mode == "sequential":
            ep_idx = self._traj_next_episode % self._traj_n_episodes
            self._traj_next_episode += 1
        else:
            # Sample a random episode index (reproducible via seeded RNG)
            ep_idx = self._traj_rng.randint(0, self._traj_n_episodes)
        start_robot = self._traj_data["start_robot"][ep_idx]
        start_obj = self._traj_data["start_obj"][ep_idx]

        dev = self._env.device

        # Set robot joint state
        if hasattr(self._env, "scene") and "robot" in self._env.scene.keys():
            robot = self._env.scene["robot"]
            n_joints = min(start_robot.shape[0], robot.num_joints)
            joint_pos = start_robot[:n_joints].unsqueeze(0).to(dev)
            joint_vel = torch.zeros_like(joint_pos)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # Set object joint state (first non-robot articulation in the scene)
        if hasattr(self._env, "scene"):
            for key in self._env.scene.keys():
                if key == "robot":
                    continue
                entity = self._env.scene[key]
                if hasattr(entity, "write_joint_state_to_sim") and hasattr(entity, "num_joints"):
                    n_joints = min(start_obj.shape[0], entity.num_joints)
                    joint_pos = start_obj[:n_joints].unsqueeze(0).to(dev)
                    joint_vel = torch.zeros_like(joint_pos)
                    entity.write_joint_state_to_sim(joint_pos, joint_vel)
                    break

        # Flush state to physics and re-render
        if hasattr(self._env, "scene"):
            self._env.scene.write_data_to_sim()
        if hasattr(self._env, "sim"):
            if hasattr(self._env.sim, "render"):
                self._env.sim.render()
            else:
                self._env.sim.step(render=True)
        if hasattr(self._env, "scene"):
            self._env.scene.update(getattr(self._env, "physics_dt", 0.0))

        # Recompute observations with fresh render
        if hasattr(self._env, "observation_manager"):
            obs = self._env.observation_manager.compute()
            self._env.obs_buf = obs

        logging.info(
            f"[IsaacLabEnvWrapper] Set initial state from trajectory episode {ep_idx}"
        )
        return obs

    def step(
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
        prepared_actions = self.prepare_action(actions)
        return self.step_prepared_action(prepared_actions)

    def prepare_action(self, actions: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert a caller action to the absolute action consumed by IsaacLab.

        This keeps ``step`` as a single environment transition while still
        allowing external executors to interpolate already-processed actions.
        """
        self._check_closed()
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self._env.device)

        # Mobile-base-relative: integrate Δbase with current base state
        if self._mobile_base_relative and self._base_dof > 0:
            actions = self._integrate_base_deltas(actions)
        return actions

    def step_prepared_action(
        self, actions: np.ndarray | torch.Tensor
    ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Execute one already-processed action without additional preprocessing."""
        self._check_closed()
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self._env.device)

        obs, reward, terminated, truncated, info = self._env.step(actions)
        self._update_episode_summaries()

        # Convert to numpy for gym compatibility
        reward = reward.cpu().numpy().astype(np.float32)
        terminated = terminated.cpu().numpy().astype(bool)
        truncated = truncated.cpu().numpy().astype(bool)

        is_success = self._get_success(terminated, truncated)
        info["final_info"] = self._finalize_episode_summaries(terminated, truncated, is_success)

        return obs, reward, terminated, truncated, info

    def make_action_executor(self, **kwargs):
        from isaaclab_arena.utils.action_execution import InterpolatedActionExecutor

        kwargs.setdefault("interpolation_factor", self.interpolation_factor)
        kwargs.setdefault("interpolation_type", self.interpolation_type)
        return InterpolatedActionExecutor(self, **kwargs)

    def _get_success(self, terminated: np.ndarray, truncated: np.ndarray) -> np.ndarray:
        done = terminated | truncated
        is_success = np.zeros(self._num_envs, dtype=bool)
        for env_ix in range(self._num_envs):
            if not done[env_ix]:
                continue
            summary = self._episode_summaries[env_ix]
            is_success[env_ix] = bool(summary["final_door_open"] and summary["final_engaged"])
        return is_success

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

        try:
            frames = self._env.render() if hasattr(self._env, "render") else None
        except RuntimeError as exc:
            logging.warning(f"[IsaacLabEnvWrapper] Render unavailable, returning placeholder frame: {exc}")
            return None
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
