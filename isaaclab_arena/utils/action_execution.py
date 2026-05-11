from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from isaaclab_arena.utils.action_interpolation import OnlineActionInterpolator


RenderFn = Callable[[], np.ndarray | None]
StateFn = Callable[[dict], np.ndarray]


@dataclass
class InterpolatedActionStep:
    obs: dict
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict
    policy_action: np.ndarray
    interpolated_action: np.ndarray
    sim_substep_ix: int
    sim_substeps_for_policy: int
    frame: np.ndarray | None = None
    sim_state_after: np.ndarray | None = None

    @property
    def done(self) -> bool:
        return bool(self.terminated[0] or self.truncated[0])


class InterpolatedActionExecutor:
    """Execute policy actions as one or more environment actions.

    The environment ``step`` remains a single-transition API.  This executor
    sits outside the environment, expands a policy action into interpolated
    absolute action targets, and steps the environment once for each target.
    """

    def __init__(
        self,
        env,
        *,
        interpolation_factor: int = 1,
        interpolation_type: str = "linear",
        render_fn: RenderFn | None = None,
        state_fn: StateFn | None = None,
    ) -> None:
        self.env = env
        self.render_fn = render_fn
        self.state_fn = state_fn
        self.interpolator = OnlineActionInterpolator(
            interpolation_factor=interpolation_factor,
            interpolation_type=interpolation_type,
        )

    @property
    def enabled(self) -> bool:
        return self.interpolator.enabled

    @property
    def interpolation_factor(self) -> int:
        return self.interpolator.interpolation_factor

    @property
    def interpolation_type(self) -> str:
        return self.interpolator.interpolation_type

    def reset(self) -> None:
        self.interpolator.reset()

    @staticmethod
    def _tensor_action_to_numpy(action: torch.Tensor) -> np.ndarray:
        action_np = action.detach().cpu().numpy()
        return action_np[0].astype(np.float32) if action_np.ndim == 2 else action_np.astype(np.float32)

    def _prepare_action(self, action: np.ndarray | torch.Tensor) -> torch.Tensor:
        if hasattr(self.env, "prepare_action"):
            return self.env.prepare_action(action)
        if isinstance(action, np.ndarray):
            device = getattr(self.env, "device", "cpu")
            return torch.from_numpy(action).to(device)
        return action

    def _step_prepared_action(self, action: torch.Tensor) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
        if hasattr(self.env, "step_prepared_action"):
            return self.env.step_prepared_action(action)
        return self.env.step(action)

    def execute(self, action: np.ndarray | torch.Tensor) -> list[InterpolatedActionStep]:
        """Run one policy action and return every executed environment step."""
        prepared_action = self._prepare_action(action)
        policy_action = self._tensor_action_to_numpy(prepared_action)
        expanded_actions = self.interpolator.expand(prepared_action)
        sim_substeps_for_policy = len(expanded_actions)

        results: list[InterpolatedActionStep] = []
        for sim_substep_ix, sim_action in enumerate(expanded_actions):
            obs, reward, terminated, truncated, info = self._step_prepared_action(sim_action)
            frame = self.render_fn() if self.render_fn is not None else None
            sim_state_after = self.state_fn(obs) if self.state_fn is not None else None
            result = InterpolatedActionStep(
                obs=obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                policy_action=policy_action,
                interpolated_action=self._tensor_action_to_numpy(sim_action),
                sim_substep_ix=sim_substep_ix,
                sim_substeps_for_policy=sim_substeps_for_policy,
                frame=frame,
                sim_state_after=sim_state_after,
            )
            results.append(result)
            if result.done:
                break

        return results
