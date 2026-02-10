"""IsaacLab Arena EnvHub Environment.

This package provides an EnvHub-compatible environment for IsaacLab Arena integration
with LeRobot. Load it from the Hugging Face Hub with:

    from lerobot.envs.factory import make_env
    envs_dict = make_env("nvidia/isaaclab-arena-envs", n_envs=4, trust_remote_code=True)

See README.md for complete documentation and installation instructions.
"""

from .env import make_env

__all__ = ["make_env"]
__version__ = "0.1.0"

