# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import logging
import numpy as np
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
from isaaclab.utils import configclass

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.metrics.door_moved_rate import DoorMovedRateMetric
from isaaclab_arena.metrics.handle_proximity_rate import (
    HandleDiagnosticsRecorderCfg,
    HandleProximityRateMetric,
    compute_stable_open_and_joints,
)
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass


class OpenDoorTask(TaskBase):
    def __init__(
        self,
        openable_object: Openable,
        openness_threshold: float | None = None,
        reset_openness: float | None = None,
        episode_length_s: float | None = None,
        proximity_threshold: float = 0.12,
        proximity_window_steps: int = 8,
        proximity_required_steps: int = 5,
        stability_window_steps: int = 8,
        openness_stability_epsilon: float = 1e-3,
        joint_stability_epsilon: float = 1e-3,
        use_fingertips: bool = True,
        debug_visualize_handle: bool = False,
        debug_record_handle_diagnostics: bool = False,
        debug_marker_scale: float = 1.0,
    ):
        super().__init__(episode_length_s=episode_length_s)
        assert isinstance(openable_object, Openable), "Openable object must be an instance of Openable"
        self.openable_object = openable_object
        self.openness_threshold = openness_threshold
        self.reset_openness = reset_openness
        self.proximity_threshold = proximity_threshold
        self.proximity_window_steps = proximity_window_steps
        self.proximity_required_steps = proximity_required_steps
        self.stability_window_steps = stability_window_steps
        self.openness_stability_epsilon = openness_stability_epsilon
        self.joint_stability_epsilon = joint_stability_epsilon
        self.use_fingertips = use_fingertips
        self.debug_visualize_handle = debug_visualize_handle
        self.debug_record_handle_diagnostics = debug_record_handle_diagnostics
        self.debug_marker_scale = debug_marker_scale
        self.scene_config = None
        self.events_cfg = OpenDoorEventCfg(self.openable_object, reset_openness=self.reset_openness)
        self.termination_cfg = self.make_termination_cfg()

        logging.info(
            "[OpenDoorTask] handle_link=%s handle_local_position=%s openness_threshold=%s proximity_threshold=%s "
            "proximity_window_steps=%s proximity_required_steps=%s stability_window_steps=%s "
            "openness_stability_epsilon=%s joint_stability_epsilon=%s use_fingertips=%s debug_visualize_handle=%s "
            "debug_record_handle_diagnostics=%s debug_marker_scale=%s",
            self.openable_object.handle_link_name,
            self.openable_object.handle_local_position,
            self.openness_threshold,
            self.proximity_threshold,
            self.proximity_window_steps,
            self.proximity_required_steps,
            self.stability_window_steps,
            self.openness_stability_epsilon,
            self.joint_stability_epsilon,
            self.use_fingertips,
            self.debug_visualize_handle,
            self.debug_record_handle_diagnostics,
            self.debug_marker_scale,
        )

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def make_termination_cfg(self):
        params = {
            "openable_object": self.openable_object,
            "stability_window_steps": self.stability_window_steps,
            "openness_stability_epsilon": self.openness_stability_epsilon,
            "joint_stability_epsilon": self.joint_stability_epsilon,
            "use_fingertips": self.use_fingertips,
            "proximity_threshold": self.proximity_threshold,
            "openness_threshold": self.openness_threshold,
            "debug_visualize_handle": self.debug_visualize_handle,
            "debug_marker_scale": self.debug_marker_scale,
        }
        success = TerminationTermCfg(
            func=compute_stable_open_and_joints,
            params=params,
        )
        return TerminationsCfg(success=success)

    def get_events_cfg(self):
        return self.events_cfg

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_mimic_env_cfg(self, embodiment_name: str):
        return OpenDoorMimicEnvCfg(
            embodiment_name=embodiment_name,
            openable_object_name=self.openable_object.name,
        )

    def get_metrics(self) -> list[MetricBase]:
        return [
            SuccessRateMetric(),
            DoorMovedRateMetric(
                self.openable_object,
                reset_openness=self.reset_openness,
            ),
            HandleProximityRateMetric(
                self.openable_object,
                proximity_threshold=self.proximity_threshold,
                use_fingertips=self.use_fingertips,
            ),
        ]

    def get_recorder_term_cfg(self) -> RecorderManagerBaseCfg | None:
        if not self.debug_record_handle_diagnostics:
            return None

        recorder_cfg = HandleDiagnosticsRecorderCfg(
            openable_object=self.openable_object,
            use_fingertips=self.use_fingertips,
            proximity_threshold=self.proximity_threshold,
        )
        recorder_manager_cfg_cls = make_configclass(
            "OpenDoorRecorderManagerCfg",
            [("handle_diagnostics", HandleDiagnosticsRecorderCfg, recorder_cfg)],
            bases=(RecorderManagerBaseCfg,),
        )
        return recorder_manager_cfg_cls()

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(lookat_object=self.openable_object, offset=np.array([-1.3, -1.3, 1.3]))


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

    # Dependent on the openable object, so this is passed in from the task at
    # construction time. We keep the field name `success` because IsaacLab
    # recorder and mimic utilities expect this termination term to exist.
    success: TerminationTermCfg = MISSING


@configclass
class OpenDoorEventCfg:
    """Configuration for Open Door."""

    reset_door_state: EventTermCfg = MISSING

    reset_openable_object_pose: EventTermCfg = MISSING

    def __init__(self, openable_object: Openable, reset_openness: float | None):
        assert isinstance(openable_object, Openable), "Object pose must be an instance of Openable"
        params = {}
        if reset_openness is not None:
            params["percentage"] = reset_openness
        self.reset_door_state = EventTermCfg(
            func=openable_object.close,
            mode="reset",
            params=params,
        )
        initial_pose = openable_object.get_initial_pose()
        if initial_pose is not None:
            self.reset_openable_object_pose = EventTermCfg(
                func=set_object_pose,
                mode="reset",
                params={
                    "pose": initial_pose,
                    "asset_cfg": SceneEntityCfg(openable_object.name),
                },
            )


@configclass
class OpenDoorMimicEnvCfg(MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Open Door env.
    """

    embodiment_name: str = "franka"

    openable_object_name: str = "openable_object"

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "demo_src_opendoor_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # The following are the subtask configurations for the pick and place task.
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=self.openable_object_name,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="grasp_1",
                # Specifies time offsets for data generation when splitting a trajectory into
                # subtask segments. Random offsets are added to the termination boundary.
                subtask_term_offset_range=(10, 20),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.005,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                # TODO(alexmillane, 2025.09.02): This is currently broken. FIX.
                # We need a way to pass in a reference to an object that exists in the
                # scene.
                object_ref=self.openable_object_name,
                # End of final subtask does not need to be detected
                subtask_term_signal=None,
                # No time offsets for the final subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.005,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        if self.embodiment_name == "franka":
            self.subtask_configs["robot"] = subtask_configs
        elif self.embodiment_name == "summit_franka":
            self.subtask_configs["robot"] = subtask_configs
        # We need to add the left and right subtasks for GR1.
        elif self.embodiment_name == "gr1_pink":
            self.subtask_configs["right"] = subtask_configs
            # EEF on opposite side (arm is static)
            subtask_configs = []
            subtask_configs.append(
                SubTaskConfig(
                    # Each subtask involves manipulation with respect to a single object frame.
                    object_ref=self.openable_object_name,
                    # Corresponding key for the binary indicator in "datagen_info" for completion
                    subtask_term_signal=None,
                    # Time offsets for data generation when splitting a trajectory
                    subtask_term_offset_range=(0, 0),
                    # Selection strategy for source subtask segment
                    selection_strategy="nearest_neighbor_object",
                    # Optional parameters for the selection strategy function
                    selection_strategy_kwargs={"nn_k": 3},
                    # Amount of action noise to apply during this subtask
                    action_noise=0.005,
                    # Number of interpolation steps to bridge to this subtask segment
                    num_interpolation_steps=0,
                    # Additional fixed steps for the robot to reach the necessary pose
                    num_fixed_steps=0,
                    # If True, apply action noise during the interpolation phase and execution
                    apply_noise_during_interpolation=False,
                )
            )
            self.subtask_configs["left"] = subtask_configs

        else:
            raise ValueError(f"Embodiment name {self.embodiment_name} not supported")
