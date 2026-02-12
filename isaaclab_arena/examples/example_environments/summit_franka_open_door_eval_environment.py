# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Summit Franka Open Door environment configured for LeRobot policy evaluation.

This environment variant uses 12-DOF joint-space actions and absolute joint
position observations, matching the data format produced by the automoma
recording pipeline.  It is intended for closed-loop evaluation of policies
trained on the recorded LeRobot datasets (ACT, SmolVLA, PI0, etc.).

Key differences from the base ``SummitFrankaOpenDoorEnvironment``:
    - Action space: 12-DOF joint positions (3 base + 7 arm + 2 gripper)
      instead of IK-based actions.
    - Observations: absolute ``joint_pos`` instead of ``joint_pos_rel``.
    - Post-creation hooks: deactivate duplicate object prims in scene USD,
      set grey lighting mode, optionally disable collisions.
    - Mobile-base-relative support: when enabled, policy outputs relative
      base deltas that are integrated with current base state during stepping.
"""

import argparse

from isaaclab_arena.examples.example_environments.summit_franka_open_door_environment import (
    SummitFrankaOpenDoorEnvironment,
)


class SummitFrankaOpenDoorEvalEnvironment(SummitFrankaOpenDoorEnvironment):
    """Summit Franka Open Door environment for LeRobot evaluation.

    Inherits scene/task setup from the base class but overrides the embodiment
    to use joint-space actions and absolute observations suitable for evaluating
    policies trained on automoma-recorded datasets.
    """

    name: str = "summit_franka_open_door_eval"

    def get_env(self, args_cli: argparse.Namespace):
        # Reuse the parent's environment construction
        arena_env = super().get_env(args_cli)

        # Override to joint-space actions
        from isaaclab_arena.embodiments.summit_franka.summit_franka import SummitFrankaJointSpaceActionsCfg

        arena_env.embodiment.action_config = SummitFrankaJointSpaceActionsCfg()

        # Override observations to use absolute joint positions
        import isaaclab.envs.mdp as mdp_isaac_lab
        from isaaclab.managers import ObservationTermCfg as ObsTerm

        obs_cfg = arena_env.embodiment.observation_config
        if hasattr(obs_cfg, "policy"):
            obs_cfg.policy.joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos)
            obs_cfg.policy.joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel)

        # Store flags for post-creation hooks (applied via modify_env_cfg or externally)
        self._disable_collision = getattr(args_cli, "disable_collision", False)
        self._mobile_base_relative = getattr(args_cli, "mobile_base_relative", False)

        return arena_env

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        # Inherit all parent CLI args
        SummitFrankaOpenDoorEnvironment.add_cli_args(parser)
        # Add eval-specific args
        parser.add_argument(
            "--disable_collision",
            action="store_true",
            default=False,
            help="Disable collision on robot and target object for evaluation.",
        )
        parser.add_argument(
            "--mobile_base_relative",
            action="store_true",
            default=False,
            help=(
                "If set, policy outputs base actions as relative deltas (Δx, Δy, Δθ). "
                "These are integrated with the current base state before sending to sim."
            ),
        )
