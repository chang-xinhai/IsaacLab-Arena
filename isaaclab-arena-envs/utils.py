import logging
from pathlib import Path
from typing import Any

import yaml

HUB_REPO_ID = "nvidia/isaaclab-arena-envs"
DEFAULT_CONFIG_FILE = "configs/config.yaml"


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
        raise ValueError(f"Invalid state_keys: {missing_state}. Available: {sorted(policy_terms)}")

    missing_cam = [k for k in camera_keys if k not in camera_terms]
    if missing_cam:
        raise ValueError(f"Invalid camera_keys: {missing_cam}. Available: {sorted(camera_terms)}")

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


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load environment config from YAML file.

    Args:
        config_path: Path to YAML config file.
            If None, uses default configs/config.yaml.

    Returns:
        Dictionary with environment configuration.
    """
    # Running from HF cache - download config from Hub
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(
        repo_id=HUB_REPO_ID, filename=DEFAULT_CONFIG_FILE
    )

    logging.info(f"Loading config from: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    return config