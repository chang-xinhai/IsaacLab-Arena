from __future__ import annotations

from dataclasses import dataclass

import torch


SUPPORTED_INTERPOLATION_TYPES = (
    "none",
    "linear",
    "cubic",
    "smoothstep",
    "smootherstep",
    "minjerk",
)


def normalize_interpolation_type(interpolation_type: str | None) -> str:
    """Normalize and validate an interpolation type string."""
    if interpolation_type is None:
        return "linear"

    normalized = interpolation_type.strip().lower().replace("-", "_")
    aliases = {
        "minimum_jerk": "minjerk",
        "min_jerk": "minjerk",
        "quintic": "minjerk",
        "bicubic": "cubic",
        "catmull_rom": "cubic",
        "catmull-rom": "cubic",
        "catmullrom": "cubic",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_INTERPOLATION_TYPES:
        supported = ", ".join(SUPPORTED_INTERPOLATION_TYPES)
        raise ValueError(f"Unsupported interpolation_type '{interpolation_type}'. Supported: {supported}.")
    return normalized


def interpolation_alpha(alpha: torch.Tensor, interpolation_type: str | None) -> torch.Tensor:
    """Map linear alpha to the requested interpolation curve."""
    mode = normalize_interpolation_type(interpolation_type)
    if mode in ("none", "linear"):
        return alpha
    if mode == "smoothstep":
        return alpha * alpha * (3.0 - 2.0 * alpha)
    if mode == "cubic":
        return alpha * alpha * (3.0 - 2.0 * alpha)
    if mode == "smootherstep":
        return alpha * alpha * alpha * (alpha * (alpha * 6.0 - 15.0) + 10.0)
    if mode == "minjerk":
        return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5
    raise AssertionError(f"Unhandled interpolation type: {mode}")


def interpolate_actions(
    start: torch.Tensor,
    end: torch.Tensor,
    interpolation_factor: int,
    interpolation_type: str | None = "linear",
    *,
    include_start: bool,
) -> torch.Tensor:
    """Interpolate one action segment.

    Args:
        start: Segment start action.
        end: Segment end action. Must have the same shape as ``start``.
        interpolation_factor: Number of sim actions per original action interval.
        interpolation_type: Alpha curve to use.
        include_start: If True, returns alphas ``[0/f, ..., (f-1)/f]``.
            If False, returns alphas ``[1/f, ..., f/f]``.

    Returns:
        Tensor with shape ``(interpolation_factor, *start.shape)``. For
        ``interpolation_type='none'`` or factor <= 1, returns ``end[None]``
        unless ``include_start`` is true, where it returns ``start[None]``.
    """
    factor = max(1, int(interpolation_factor))
    mode = normalize_interpolation_type(interpolation_type)

    if mode == "none" or factor <= 1:
        return (start if include_start else end).unsqueeze(0)

    if start.shape != end.shape:
        raise ValueError(f"Action shapes must match, got {start.shape} and {end.shape}.")

    if include_start:
        alpha = torch.arange(factor, dtype=start.dtype, device=start.device) / factor
    else:
        alpha = torch.arange(1, factor + 1, dtype=start.dtype, device=start.device) / factor

    alpha = interpolation_alpha(alpha, mode)
    view_shape = (factor,) + (1,) * start.dim()
    alpha = alpha.reshape(view_shape)
    return (1.0 - alpha) * start.unsqueeze(0) + alpha * end.unsqueeze(0)


def interpolate_trajectory(
    trajectory: torch.Tensor,
    interpolation_factor: int,
    interpolation_type: str | None = "linear",
) -> torch.Tensor:
    """Interpolate a batched trajectory of shape ``(N, T, D)``."""
    if trajectory.dim() != 3:
        raise ValueError(f"trajectory must be 3D [N, T, D], got shape {trajectory.shape}.")

    factor = max(1, int(interpolation_factor))
    mode = normalize_interpolation_type(interpolation_type)
    if mode == "none" or factor <= 1 or trajectory.shape[1] <= 1:
        return trajectory

    if mode == "cubic":
        return _interpolate_trajectory_cubic(trajectory, factor)

    segments = [
        interpolate_actions(
            trajectory[:, i, :],
            trajectory[:, i + 1, :],
            factor,
            mode,
            include_start=True,
        )
        for i in range(trajectory.shape[1] - 1)
    ]
    result = torch.cat(segments + [trajectory[:, -1, :].unsqueeze(0)], dim=0)
    return result.permute(1, 0, 2).contiguous()


def _interpolate_trajectory_cubic(trajectory: torch.Tensor, interpolation_factor: int) -> torch.Tensor:
    """Catmull-Rom interpolation for full offline trajectories."""
    factor = max(1, int(interpolation_factor))
    pieces = []
    for i in range(trajectory.shape[1] - 1):
        p0 = trajectory[:, max(i - 1, 0), :]
        p1 = trajectory[:, i, :]
        p2 = trajectory[:, i + 1, :]
        p3 = trajectory[:, min(i + 2, trajectory.shape[1] - 1), :]
        alpha = torch.arange(factor, dtype=trajectory.dtype, device=trajectory.device) / factor
        t = alpha.reshape((factor,) + (1,) * p1.dim())
        t2 = t * t
        t3 = t2 * t
        segment = 0.5 * (
            2.0 * p1.unsqueeze(0)
            + (-p0 + p2).unsqueeze(0) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3).unsqueeze(0) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3).unsqueeze(0) * t3
        )
        # Preserve deliberate hold segments; Catmull-Rom tangents can otherwise
        # pull a flat p1->p2 interval toward the following waypoint.
        same_endpoints = torch.isclose(p1, p2, rtol=1e-7, atol=1e-9).unsqueeze(0)
        segment = torch.where(same_endpoints, p1.unsqueeze(0).expand_as(segment), segment)
        pieces.append(segment)
    result = torch.cat(pieces + [trajectory[:, -1, :].unsqueeze(0)], dim=0)
    return result.permute(1, 0, 2).contiguous()


@dataclass
class OnlineActionInterpolator:
    """Stateful interpolation for closed-loop policy actions."""

    interpolation_factor: int = 1
    interpolation_type: str = "linear"

    def __post_init__(self) -> None:
        self.interpolation_factor = max(1, int(self.interpolation_factor))
        self.interpolation_type = normalize_interpolation_type(self.interpolation_type)
        self._previous_action: torch.Tensor | None = None

    @property
    def enabled(self) -> bool:
        return self.interpolation_type != "none" and self.interpolation_factor > 1

    def reset(self) -> None:
        self._previous_action = None

    def expand(self, action: torch.Tensor) -> list[torch.Tensor]:
        """Return sim actions for the interval from previous action to action."""
        current_action = action.detach().clone()
        if not self.enabled or self._previous_action is None:
            self._previous_action = current_action
            return [action]

        expanded = interpolate_actions(
            self._previous_action.to(device=action.device, dtype=action.dtype),
            action,
            self.interpolation_factor,
            self.interpolation_type,
            include_start=False,
        )
        self._previous_action = current_action
        return [expanded[i] for i in range(expanded.shape[0])]
