from isaaclab_env_wrapper import IsaacLabEnvWrapper
from unittest.mock import MagicMock
import gym
import numpy as np
import torch


def _create_mock_isaaclab_env(num_envs: int = 2, device: str = "cpu"):
    """Create a mock IsaacLab environment for testing."""
    mock_env = MagicMock()
    mock_env.num_envs = num_envs
    mock_env.device = device
    mock_env.observation_space = gym.spaces.Dict(
        {"policy": gym.spaces.Box(low=-1, high=1, shape=(num_envs, 54), dtype=np.float32)}
    )
    mock_env.action_space = gym.spaces.Box(low=-1, high=1, shape=(36,), dtype=np.float32)
    mock_env.metadata = {}
    return mock_env


def test_isaaclab_wrapper_init():
    """Test IsaacLabEnvWrapper initialization."""
    mock_env = _create_mock_isaaclab_env(num_envs=4)

    wrapper = IsaacLabEnvWrapper(
        mock_env,
        episode_length=300,
        task="Test task",
        render_mode="rgb_array",
    )

    assert wrapper.num_envs == 4
    assert wrapper._max_episode_steps == 300
    assert wrapper.task == "Test task"
    assert wrapper.render_mode == "rgb_array"
    assert wrapper.device == "cpu"
    assert len(wrapper.envs) == 4


def test_isaaclab_wrapper_reset():
    """Test IsaacLabEnvWrapper reset."""
    mock_env = _create_mock_isaaclab_env(num_envs=2)
    mock_obs = {"policy": torch.randn(2, 54)}
    mock_env.reset.return_value = (mock_obs, {})

    wrapper = IsaacLabEnvWrapper(mock_env, episode_length=100)
    obs, info = wrapper.reset(seed=42)

    mock_env.reset.assert_called_once_with(seed=42, options=None)
    assert "final_info" in info
    assert "is_success" in info["final_info"]
    assert len(info["final_info"]["is_success"]) == 2


def test_isaaclab_wrapper_reset_with_seed_list():
    """Test that seed list is handled correctly (IsaacLab expects single seed)."""
    mock_env = _create_mock_isaaclab_env(num_envs=2)
    mock_env.reset.return_value = ({"policy": torch.randn(2, 54)}, {})

    wrapper = IsaacLabEnvWrapper(mock_env)
    wrapper.reset(seed=[42, 43, 44])

    # Should extract first seed
    mock_env.reset.assert_called_once_with(seed=42, options=None)


def test_isaaclab_wrapper_step():
    """Test IsaacLabEnvWrapper step."""
    mock_env = _create_mock_isaaclab_env(num_envs=2)
    mock_env.step.return_value = (
        {"policy": torch.randn(2, 54)},
        torch.tensor([0.5, 0.3]),
        torch.tensor([False, False]),
        torch.tensor([False, True]),
        {},
    )
    # Mock termination manager
    mock_env.termination_manager.get_term.return_value = torch.tensor([False, True])

    wrapper = IsaacLabEnvWrapper(mock_env)
    actions = np.random.randn(2, 36).astype(np.float32)
    obs, reward, terminated, truncated, info = wrapper.step(actions)

    assert reward.dtype == np.float32
    assert terminated.dtype == bool
    assert truncated.dtype == bool
    assert len(reward) == 2
    assert "final_info" in info
    assert "is_success" in info["final_info"]


def test_isaaclab_wrapper_call_method():
    """Test IsaacLabEnvWrapper call method."""
    mock_env = _create_mock_isaaclab_env(num_envs=3)

    wrapper = IsaacLabEnvWrapper(mock_env, episode_length=200, task="My task")

    # Test _max_episode_steps
    result = wrapper.call("_max_episode_steps")
    assert result == [200, 200, 200]

    # Test task
    result = wrapper.call("task")
    assert result == ["My task", "My task", "My task"]


def test_isaaclab_wrapper_render():
    """Test IsaacLabEnvWrapper render."""
    mock_env = _create_mock_isaaclab_env(num_envs=2)
    mock_frames = torch.randint(0, 255, (2, 480, 640, 3), dtype=torch.uint8)
    mock_env.render.return_value = mock_frames

    wrapper = IsaacLabEnvWrapper(mock_env, render_mode="rgb_array")
    frame = wrapper.render()

    assert frame is not None
    assert frame.shape == (480, 640, 3)  # Returns first env frame


def test_isaaclab_wrapper_render_all():
    """Test IsaacLabEnvWrapper render_all."""
    mock_env = _create_mock_isaaclab_env(num_envs=2)
    mock_frames = torch.randint(0, 255, (2, 480, 640, 3), dtype=torch.uint8)
    mock_env.render.return_value = mock_frames

    wrapper = IsaacLabEnvWrapper(mock_env, render_mode="rgb_array")
    frames = wrapper.render_all()

    assert len(frames) == 2
    assert all(f.shape == (480, 640, 3) for f in frames)


def test_isaaclab_wrapper_render_none():
    """Test render returns None when render_mode is not rgb_array."""
    mock_env = _create_mock_isaaclab_env()

    wrapper = IsaacLabEnvWrapper(mock_env, render_mode=None)
    assert wrapper.render() is None


def test_isaaclab_wrapper_close():
    """Test IsaacLabEnvWrapper close."""
    mock_env = _create_mock_isaaclab_env()
    mock_app = MagicMock()

    wrapper = IsaacLabEnvWrapper(mock_env, simulation_app=mock_app)
    wrapper.close()

    mock_env.close.assert_called_once()
    mock_app.app.close.assert_called_once()
    assert wrapper._closed