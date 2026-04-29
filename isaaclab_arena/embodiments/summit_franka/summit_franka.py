# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.utils.math as PoseUtils
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import ee_frame_pos, ee_frame_quat

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.mimic_utils import get_rigid_and_articulated_object_poses
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.franka.observations import gripper_pos
from isaaclab_arena.utils.pose import Pose

# Path to the summit_franka USD.
# Override via AUTOMOMA_ROBOT_ROOT environment variable.
_DEFAULT_ROBOT_ROOT = Path(__file__).resolve().parents[3] / "res_for_custom" / "automoma_assets" / "robot"
_ROBOT_ROOT = Path(os.environ.get("AUTOMOMA_ROBOT_ROOT", str(_DEFAULT_ROBOT_ROOT)))
_SUMMIT_FRANKA_USD_PATH = str(
    _ROBOT_ROOT / "summit_franka" / "summit_franka" / "summit_franka.usd"
)

# Default joint positions for summit_franka:
# [base_x, base_y, base_z (yaw),
#  panda_joint1..7,
#  panda_finger_joint1, panda_finger_joint2]
_SUMMIT_FRANKA_DEFAULT_JOINT_POS = [
    0.0, 0.0, 0.0,                                      # base x, y, yaw
    0.0, -0.785, -0.1107, -1.1775, 0.0, 0.785, 0.785,  # arm joints
    0.04, 0.04,                                           # finger joints
]

_SUMMIT_FRANKA_ACTION_JOINT_NAMES = [
    "base_x", "base_y", "base_z",
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
    "panda_finger_joint1", "panda_finger_joint2",
]


@register_asset
class SummitFrankaEmbodiment(EmbodimentBase):
    """Embodiment for the Summit + Franka mobile manipulator."""

    name = "summit_franka"

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config = SummitFrankaSceneCfg()
        self.action_config = SummitFrankaActionsCfg()
        self.observation_config = SummitFrankaObservationsCfg()
        self.event_config = SummitFrankaEventCfg()
        self.camera_config = SummitFrankaCameraCfg()
        self.mimic_env = SummitFrankaMimicEnv


@configclass
class SummitFrankaSceneCfg:
    """Scene configuration for the Summit Franka embodiment."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=UsdFileCfg(
            usd_path=_SUMMIT_FRANKA_USD_PATH,
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "base_x": 0.0,
                "base_y": 0.0,
                "base_z": 0.0,
                "panda_joint1": 0.0,
                "panda_joint2": -0.785,
                "panda_joint3": -0.1107,
                "panda_joint4": -1.1775,
                "panda_joint5": 0.0,
                "panda_joint6": 0.785,
                "panda_joint7": 0.785,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
            joint_vel={".*": 0.0},
        ),
        # TODO(walker): adjust stiffness and damping
        actuators={
            "base": ImplicitActuatorCfg(
                joint_names_expr=["base_x", "base_y", "base_z"],
                effort_limit_sim=5e3,
                velocity_limit_sim=1.5,
                stiffness=8e3,
                damping=2e3,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint.*"],
                effort_limit_sim=5e3,
                velocity_limit_sim=2.175,
                stiffness=1e4,
                damping=1e2,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=5e3,
                velocity_limit_sim=0.2,
                stiffness=1e6,  # gripper needs higher stiffness to maintain the grasp
                damping=1e3, # bigger damping to avoid oscillations
            ),
        },
    )

    # End-effector frame marker
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ],
    )

    def __post_init__(self):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


@configclass
class SummitFrankaCameraCfg:
    """Camera configuration for the Summit Franka embodiment.

    Three cameras are provided:
    - **ego_topdown**: Mounted on the base link, looking straight down from above.
    - **ego_wrist**: Mounted on the panda_hand, provides a wrist-eye view.
    - **fix_local**: Mounted on the Object prim (scene-fixed), provides a third-person view.

    Camera poses are specified in local frame of their parent prim.
    """

    # Top-down camera mounted on the robot base
    ego_topdown: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link_y/ego_topdown",
        update_period=0.0,
        height=240,
        width=320,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=50.0,
            clipping_range=(0.01, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 10.0),
            rot=(0.707, 0.0, 0.0, -0.707),  # looking down (−90° around X), with rxyz in isaacsim
            convention="opengl",
        ),
    )

    # Wrist-eye camera mounted on the gripper hand
    ego_wrist: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/ego_wrist",
        update_period=0.0,
        height=240,
        width=320,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15,
            clipping_range=(0.01, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.6, 0.0, -0.9),
            rot=(-0.171, 0.686, 0.686, -0.171),  # oriented to look forward from the wrist, with rxyz in isaacsim
            convention="opengl",
        ),
    )

    # Fixed local camera (scene-level, attached to env root for a third-person view)
    fix_local: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/fix_local",
        update_period=0.0,
        height=240,
        width=320,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0,
            clipping_range=(0.01, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-4.3, 4.7, 6.2),
            rot=(0.3805, 0.1047, -0.3413, -0.8531),  # oriented for third-person view, with rxyz in isaacsim
            convention="opengl",
        ),
    )


@configclass
class SummitFrankaActionsCfg:
    """Action specifications for Summit Franka.

    Uses differential IK for the Franka arm and binary joint positions for the gripper.
    The base joints (base_x, base_y, base_z) are not directly controlled in the action
    space for the table-top manipulation task; they remain at their initial configuration.
    """

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )

    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class SummitFrankaJointSpaceActionsCfg:
    """Joint-space action config for Summit Franka trajectory recording.

    All 12 DOF (3 base + 7 arm + 2 gripper) controlled via **absolute** joint
    position targets.  Used during trajectory recording/replay where the policy
    outputs complete joint configurations directly.

    ``processed_actions = raw_actions`` exactly (scale=1, offset=0).
    """

    joint_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=_SUMMIT_FRANKA_ACTION_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


class SummitFrankaAutomomaSetStateAction(ActionTerm):
    """Set robot and object articulation joint states from AutoMoMa joint targets.

    The action layout is robot joints followed by object articulation joints. For
    current open-door AutoMoMa assets this is 12D robot + 1D object = 13D.
    """

    cfg: "SummitFrankaAutomomaSetStateActionCfg"

    def __init__(self, cfg: "SummitFrankaAutomomaSetStateActionCfg", env):
        super().__init__(cfg, env)

        joint_names = self.cfg.joint_names or _SUMMIT_FRANKA_ACTION_JOINT_NAMES
        self._joint_ids, self._joint_names = self._asset.find_joints(
            joint_names,
            preserve_order=self.cfg.preserve_order,
        )
        self._num_robot_joints = len(self._joint_ids)

        self._object_asset_name, self._object_asset = self._resolve_object_asset()
        if self.cfg.object_joint_names is None:
            self._object_joint_ids = slice(None)
            self._object_joint_names = list(getattr(self._object_asset, "joint_names", []))
            self._num_object_joints = self._object_asset.num_joints
        else:
            self._object_joint_ids, self._object_joint_names = self._object_asset.find_joints(
                self.cfg.object_joint_names,
                preserve_order=self.cfg.object_preserve_order,
            )
            self._num_object_joints = len(self._object_joint_ids)

        if self._num_object_joints <= 0:
            raise ValueError(
                f"AutoMoMa set-state action requires an articulated object with at least one joint. "
                f"Resolved object '{self._object_asset_name}' has {self._num_object_joints} joints."
            )

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return self._num_robot_joints + self._num_object_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Invalid AutoMoMa set-state action shape: expected {self.action_dim}, "
                f"received {actions.shape[-1]}."
            )
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

    def apply_actions(self):
        robot_pos = self._processed_actions[:, :self._num_robot_joints]
        object_pos = self._processed_actions[:, self._num_robot_joints:]
        robot_vel = torch.zeros_like(robot_pos)
        object_vel = torch.zeros_like(object_pos)

        self._asset.write_joint_state_to_sim(robot_pos, robot_vel, joint_ids=self._joint_ids)
        self._asset.set_joint_position_target(robot_pos, joint_ids=self._joint_ids)
        self._asset.set_joint_velocity_target(robot_vel, joint_ids=self._joint_ids)

        self._object_asset.write_joint_state_to_sim(
            object_pos,
            object_vel,
            joint_ids=self._object_joint_ids,
        )
        self._object_asset.set_joint_position_target(object_pos, joint_ids=self._object_joint_ids)
        self._object_asset.set_joint_velocity_target(object_vel, joint_ids=self._object_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

    def _resolve_object_asset(self):
        object_asset_name = self.cfg.object_asset_name
        scene = self._env.scene

        if object_asset_name:
            if object_asset_name not in scene.keys():
                raise KeyError(
                    f"AutoMoMa object asset '{object_asset_name}' was not found in the scene. "
                    f"Available scene keys: {list(scene.keys())}"
                )
            object_asset = scene[object_asset_name]
            if not self._is_articulated_object(object_asset):
                raise TypeError(
                    f"Scene asset '{object_asset_name}' is not an articulated object with joint state writers."
                )
            return object_asset_name, object_asset

        candidates = []
        for name in scene.keys():
            if name == self.cfg.asset_name:
                continue
            asset = scene[name]
            if self._is_articulated_object(asset):
                candidates.append((name, asset))

        if len(candidates) != 1:
            raise ValueError(
                "AutoMoMa set-state action could not infer a unique articulated object. "
                f"Candidates: {[name for name, _ in candidates]}. "
                "Pass object_asset_name explicitly."
            )
        return candidates[0]

    @staticmethod
    def _is_articulated_object(asset) -> bool:
        return (
            hasattr(asset, "write_joint_state_to_sim")
            and hasattr(asset, "set_joint_position_target")
            and hasattr(asset, "set_joint_velocity_target")
            and hasattr(asset, "num_joints")
            and asset.num_joints > 0
        )


@configclass
class SummitFrankaAutomomaSetStateActionCfg(ActionTermCfg):
    """Configuration for AutoMoMa set-state replay actions."""

    class_type: type[ActionTerm] = SummitFrankaAutomomaSetStateAction

    joint_names: list[str] | None = None
    """Robot joint names controlled by the first action dimensions."""

    preserve_order: bool = True
    """Whether to preserve robot joint order."""

    object_asset_name: str | None = None
    """Name of the articulated object asset in the scene."""

    object_joint_names: list[str] | None = None
    """Object joint names controlled by the trailing action dimensions. None means all object joints."""

    object_preserve_order: bool = True
    """Whether to preserve object joint order."""


@configclass
class SummitFrankaAutomomaSetStateActionsCfg:
    """13D AutoMoMa set-state action config for Summit Franka open-door replay."""

    set_state_action: ActionTermCfg = SummitFrankaAutomomaSetStateActionCfg(
        asset_name="robot",
        joint_names=_SUMMIT_FRANKA_ACTION_JOINT_NAMES,
        object_asset_name=None,
    )

    def __init__(self, object_asset_name: str | None = None):
        self.set_state_action = SummitFrankaAutomomaSetStateActionCfg(
            asset_name="robot",
            joint_names=_SUMMIT_FRANKA_ACTION_JOINT_NAMES,
            object_asset_name=object_asset_name,
        )


@configclass
class SummitFrankaObservationsCfg:
    """Observation specifications for Summit Franka."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)
        gripper_pos = ObsTerm(func=gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class SummitFrankaEventCfg:
    """Configuration for Summit Franka reset events."""

    init_summit_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": _SUMMIT_FRANKA_DEFAULT_JOINT_POS,
        },
    )
    randomize_summit_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


class SummitFrankaMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic environment for Summit Franka."""

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        noise: float | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        delta_position = target_pos - curr_pos
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        (gripper_action,) = gripper_action_dict.values()

        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if noise is not None:
            noise = noise * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        target_pos = curr_pos + delta_position
        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle
        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)
        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        state = self.scene.get_state(is_relative=True)
        return get_rigid_and_articulated_object_poses(state, env_ids)
