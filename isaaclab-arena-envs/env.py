"""IsaacLab Arena EnvHub Environment.

For more information, visit https://huggingface.co/docs/lerobot/envhub_isaaclab_arena
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
import yaml

import gymnasium as gym

from huggingface_hub import hf_hub_download
from lerobot.envs.configs import EnvConfig

# Hub constants for downloading additional files
HUB_REPO_ID = "nvidia/isaaclab-arena-envs"
EXAMPLE_ENVS = "example_envs.yaml"

def _download_hub_file(filename: str):
    return hf_hub_download(repo_id=HUB_REPO_ID, filename=filename)

def _download_and_import(filename: str):
    """Download file from Hub and import as module."""
    local_path = _download_hub_file(filename)
    module_dir = os.path.dirname(local_path)

    # Add directory to sys.path so modules can import each other
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    module_name = filename.replace(".py", "")

    # Read file content and replace relative imports with absolute imports
    with open(local_path) as f:
        content = f.read()

    # Replace relative imports (e.g., "from .errors" -> "from errors")
    content = content.replace("from .errors import", "from errors import")
    content = content.replace("from .isaaclab_env_wrapper import", "from isaaclab_env_wrapper import")

    # Create module and execute modified content
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(module_name, local_path))
    sys.modules[module_name] = module

    # Compile and execute the modified code
    code = compile(content, local_path, "exec")
    exec(code, module.__dict__)  # noqa: S102

    return module


try:
    from .errors import IsaacLabArenaCameraKeyError, IsaacLabArenaStateKeyError
    from .isaaclab_env_wrapper import IsaacLabEnvWrapper, cleanup_isaaclab
except ImportError:
    _errors = _download_and_import("errors.py")
    _isaaclab_wrapper = _download_and_import("isaaclab_env_wrapper.py")
    IsaacLabEnvWrapper = _isaaclab_wrapper.IsaacLabEnvWrapper
    cleanup_isaaclab = _isaaclab_wrapper.cleanup_isaaclab
    IsaacLabArenaCameraKeyError = _errors.IsaacLabArenaCameraKeyError
    IsaacLabArenaStateKeyError = _errors.IsaacLabArenaStateKeyError


def validate_config(
    env,
    state_keys: tuple[str, ...],
    camera_keys: tuple[str, ...],
    cfg_state_dim: int,
    cfg_action_dim: int,
) -> None:
    """Validate observation keys and dimensions against IsaacLab managers."""
    obs_manager = env.observation_manager
    active_terms = obs_manager.active_terms
    policy_terms = set(active_terms.get("policy", []))
    camera_terms = set(active_terms.get("camera_obs", []))

    # Validate keys exist
    missing_state = [k for k in state_keys if k not in policy_terms]
    if missing_state:
        raise IsaacLabArenaStateKeyError(missing_state, policy_terms)

    missing_cam = [k for k in camera_keys if k not in camera_terms]
    if missing_cam:
        raise IsaacLabArenaCameraKeyError(missing_cam, camera_terms)

    # Validate dimensions
    env_action_dim = env.action_space.shape[-1]
    if cfg_action_dim != env_action_dim:
        raise ValueError(f"action_dim mismatch: config={cfg_action_dim}, env={env_action_dim}")

    # Compute expected state dimension
    policy_dims = obs_manager.group_obs_dim.get("policy", [])
    policy_names = active_terms.get("policy", [])
    term_dims = dict(zip(policy_names, policy_dims, strict=False))

    expected_state_dim = 0
    for key in state_keys:
        if key in term_dims:
            shape = term_dims[key]
            dim = 1
            for s in shape if isinstance(shape, (tuple, list)) else [shape]:
                dim *= s
            expected_state_dim += dim

    if cfg_state_dim != expected_state_dim:
        raise ValueError(
            f"state_dim mismatch: config={cfg_state_dim}, "
            f"computed={expected_state_dim}. "
            f"Term dims: {term_dims}"
        )

    logging.info(f"Validated: state_keys={state_keys}, camera_keys={camera_keys}")


def resolve_environment_alias(environment: str) -> str:
    envs_mapping = _download_hub_file(EXAMPLE_ENVS)
    with open(envs_mapping, "r") as f:
        envs_mapping = yaml.safe_load(f)

    module = (
        f"{envs_mapping['repo']['base_module']}.{envs_mapping['repo']['envs'][environment]}"
    )
    return module

def _create_isaaclab_env(config: dict, n_envs: int) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Create IsaacLab Arena environment from configuration.

    Args:
        config: Configuration dictionary.
        n_envs: Number of parallel environments.

    Returns:
        Dict mapping environment name to {task_id: VectorEnv}.
    """
    from isaaclab.app import AppLauncher

    if config.get("enable_pinocchio", True):
        import pinocchio  # noqa: F401

    # Override num_envs
    config["num_envs"] = n_envs

    # Create argparse namespace for IsaacLab
    as_isaaclab_argparse = argparse.Namespace(**config)

    logging.info("Launching IsaacLab simulation app...")
    app_launcher = AppLauncher(as_isaaclab_argparse)

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    environment = config.get("environment")
    if environment is None:
        raise ValueError("cfg.environment must be specified")

    # Resolve alias and create environment
    environment_path = resolve_environment_alias(environment)
    logging.info(f"Creating environment: {environment_path}")

    module_path, class_name = environment_path.rsplit(".", 1)
    environment_module = importlib.import_module(module_path)
    environment_class = getattr(environment_module, class_name)()

    env_builder = ArenaEnvBuilder(environment_class.get_env(as_isaaclab_argparse), as_isaaclab_argparse)

    # Determine render_mode
    render_mode = "rgb_array" if config.get("enable_cameras", False) else None

    raw_env = env_builder.make_registered()

    # Set render_mode on underlying env
    if render_mode and hasattr(raw_env, "render_mode"):
        raw_env.render_mode = render_mode
        logging.info(f"Set render_mode={render_mode} on underlying IsaacLab env")

    # Validate config
    state_keys = tuple(k.strip() for k in (config.get("state_keys") or "").split(",") if k.strip())
    camera_keys = tuple(k.strip() for k in (config.get("camera_keys") or "").split(",") if k.strip())
    try:
        validate_config(
            raw_env,
            state_keys,
            camera_keys,
            config.get("state_dim", 54),
            config.get("action_dim", 36),
        )
    except (IsaacLabArenaCameraKeyError, IsaacLabArenaStateKeyError, ValueError) as e:
        logging.error(f"Validation failed: {e}")
        cleanup_isaaclab(raw_env, app_launcher)
        raise
    except Exception as e:
        logging.error(f"Validation failed with unexpected error: {type(e).__name__}: {e}")
        cleanup_isaaclab(raw_env, app_launcher)
        raise

    # Get task description
    task = config.get("task")
    if task is None:
        task = f"Complete the {environment.replace('_', ' ')} task."

    # Wrap and return
    wrapped_env = IsaacLabEnvWrapper(
        raw_env,
        episode_length=config.get("episode_length", 300),
        task=task,
        render_mode=render_mode,
        simulation_app=app_launcher,
    )
    logging.info(f"Created: {environment} with {wrapped_env.num_envs} envs, render_mode={render_mode}")

    return {environment: {0: wrapped_env}}


def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,  # noqa: ARG001
    cfg: EnvConfig | None = None,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Create IsaacLab Arena environments (EnvHub-compatible API).

    This function provides the standard EnvHub interface for loading
    environments from the Hugging Face Hub. Configuration is passed
    directly via kwargs or loaded from configs/config.yaml.

    Args:
        n_envs: Number of parallel environments (default: 1).
        use_async_envs: Ignored for IsaacLab (GPU-based batched execution).
        cfg: Configuration object with environment parameters. Required keys:
            - environment: Environment name or alias (e.g., "gr1_microwave")
            - headless: Whether to run headless (default: True)
            - enable_cameras: Enable camera rendering (default: False)
            - episode_length: Max steps per episode (default: 300)
            - state_keys: Comma-separated observation keys
            - camera_keys: Comma-separated camera observation keys
            - state_dim: Expected state dimension
            - action_dim: Expected action dimension

    Returns:
        Dict mapping environment name to {task_id: VectorEnv}.
        Format: {suite_name: {0: wrapped_vector_env}}

    Note:
        IsaacLab environments use GPU-based batched execution, so
        `use_async_envs` is ignored. The returned wrapper provides
        VectorEnv compatibility.
    """
    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")

    if not hasattr(cfg, "environment") or cfg.environment is None:
        raise ValueError(
            "No 'environment' specified. Pass it via kwargs or create "
            "configs/config.yaml with environment settings."
        )
    # Build config from cfg attributes directly (hub path) or YAML (local dev)
    # if cfg is not None and hasattr(cfg, 'environment'):
    # # Extract config directly from EnvConfig attributes
    config = {
        "environment": cfg.environment,
        "embodiment": cfg.embodiment,
        "object": cfg.object,
        "mimic": cfg.mimic,
        "teleop_device": cfg.teleop_device,
        "seed": cfg.seed,
        "device": cfg.device,
        "disable_fabric": cfg.disable_fabric,
        "enable_cameras": cfg.enable_cameras,
        "headless": cfg.headless,
        "enable_pinocchio": cfg.enable_pinocchio,
        "episode_length": cfg.episode_length,
        "state_dim": cfg.state_dim,
        "action_dim": cfg.action_dim,
        "camera_height": cfg.camera_height,
        "camera_width": cfg.camera_width,
        "video": cfg.video,
        "video_length": cfg.video_length,
        "video_interval": cfg.video_interval,
        "state_keys": cfg.state_keys,
        "camera_keys": cfg.camera_keys or "",  # Pass empty string for no cameras
        "task": cfg.task,
    }

    logging.info(f"EnvHub make_env: environment={config.get('environment')}, n_envs={n_envs}")
    logging.info(f"Config: headless={config.get('headless')}, enable_cameras={config.get('enable_cameras')}")
    logging.info(f"EnvHub Config: {config}")

    return _create_isaaclab_env(config, n_envs)
