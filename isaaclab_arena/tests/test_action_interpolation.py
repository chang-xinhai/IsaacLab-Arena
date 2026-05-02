import torch

from isaaclab_arena.utils.action_interpolation import OnlineActionInterpolator, interpolate_trajectory


def test_interpolate_trajectory_linear_shape_and_endpoints():
    traj = torch.tensor([[[0.0], [1.0], [3.0]]])

    result = interpolate_trajectory(traj, interpolation_factor=3, interpolation_type="linear")

    assert result.shape == (1, 7, 1)
    assert torch.allclose(result[0, :, 0], torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 5.0 / 3.0, 7.0 / 3.0, 3.0]))


def test_interpolate_trajectory_none_keeps_input():
    traj = torch.randn(2, 4, 3)

    result = interpolate_trajectory(traj, interpolation_factor=5, interpolation_type="none")

    assert result is traj


def test_interpolate_trajectory_cubic_uses_cubic_path():
    traj = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])

    result = interpolate_trajectory(traj, interpolation_factor=2, interpolation_type="cubic")

    assert result.shape == (1, 7, 1)
    assert torch.allclose(result[:, 0, :], traj[:, 0, :])
    assert torch.allclose(result[:, -1, :], traj[:, -1, :])


def test_online_action_interpolator_expands_after_first_action():
    interpolator = OnlineActionInterpolator(interpolation_factor=4, interpolation_type="minjerk")
    first = torch.tensor([[0.0, 0.0]])
    second = torch.tensor([[1.0, 2.0]])

    first_expanded = interpolator.expand(first)
    second_expanded = interpolator.expand(second)

    assert len(first_expanded) == 1
    assert torch.allclose(first_expanded[0], first)
    assert len(second_expanded) == 4
    assert torch.allclose(second_expanded[-1], second)
    assert torch.all(second_expanded[0] > first)
    assert torch.all(second_expanded[0] < second)


def test_online_action_interpolator_reset_discards_previous_action():
    interpolator = OnlineActionInterpolator(interpolation_factor=4, interpolation_type="linear")
    interpolator.expand(torch.tensor([[0.0]]))
    interpolator.reset()

    expanded = interpolator.expand(torch.tensor([[2.0]]))

    assert len(expanded) == 1
    assert torch.allclose(expanded[0], torch.tensor([[2.0]]))
