# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase


class SummitFrankaOpenDoorEnvironment(ExampleEnvironmentBase):
    """
    Generic environment for the Summit Franka mobile manipulator opening an
    automoma articulated door (microwave, dishwasher, oven, etc.) in an automoma scene.

    This environment is **not** limited to microwaves. Any automoma openable object
    that follows the PartNet-Mobility ``joint_0`` convention works — the object is
    selected at runtime via ``--object_name``.

    Supported objects include (but are not limited to):
        - microwave_7221
        - dishwasher_11622
        - oven_101773
        - Any <type>_<id> with a URDF/USD under automoma_assets/object/

    Usage:
        python isaaclab_arena/scripts/replay_demos.py \\
            --device cpu --enable_cameras \\
            --dataset_file $DATASET_DIR/summit_franka_open_door.hdf5 \\
            summit_franka_open_door \\
            --object_name microwave_7221 \\
            --scene_name scene_0_seed_0
    """

    name: str = "summit_franka_open_door"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.assets.automoma_background_library import get_automoma_scene
        from isaaclab_arena.assets.automoma_object_library import (
            get_automoma_object,
            get_object_pose_from_metadata,
            load_scene_metadata,
        )
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask
        from isaaclab_arena.utils.pose import Pose

        # ---- Parse CLI arguments ----
        object_name = args_cli.object_name  # e.g. "microwave_7221", "dishwasher_11622", "oven_101773"
        scene_name = args_cli.scene_name    # e.g. "scene_0_seed_0"
        object_center = getattr(args_cli, "object_center", False)

        # Parse object type and id from the object_name (format: <type>_<id>)
        # The last segment that is purely numeric is the asset_id
        parts = object_name.split("_")
        asset_id = parts[-1]
        asset_type = "_".join(parts[:-1])  # e.g. "microwave", "dishwasher", "oven"
        asset_type_capitalized = asset_type.capitalize()

        # ---- Load scene background ----
        background = get_automoma_scene(scene_name)

        # ---- Load object ----
        target_object = get_automoma_object(
            asset_type=asset_type_capitalized,
            asset_id=asset_id,
        )

        # ---- Read object pose from scene metadata ----
        metadata = load_scene_metadata(scene_name)
        object_pose, object_scale = get_object_pose_from_metadata(
            metadata,
            asset_type=asset_type_capitalized,
            asset_id=asset_id,
        )

        # Keep original pose for robot placement in object_center mode
        original_object_pose = Pose(
            position_xyz=object_pose.position_xyz,
            rotation_wxyz=object_pose.rotation_wxyz,
        )

        # ---- object_center mode: shift world origin so object is at (0, 0, z, 1, 0, 0, 0) ----
        if object_center:
            from isaaclab_arena.utils.pose import compose_poses

            # Build a correction transform that:
            #   1. Counter-rotates the object's rotation to identity
            #   2. Translates so object x,y become 0 (keep z unchanged)
            #
            # The inverse of the object's pose (ignoring z translation) gives us
            # the transform to apply to all prims so the object ends up at
            # (0, 0, z_original, 1, 0, 0, 0).
            obj_pose_xy = Pose(
                position_xyz=(object_pose.position_xyz[0], object_pose.position_xyz[1], 0.0),
                rotation_wxyz=object_pose.rotation_wxyz,
            )
            # Inverse: R^-1 and -R^-1 * t
            import torch
            from isaaclab.utils.math import matrix_from_quat, quat_from_matrix
            R = matrix_from_quat(torch.tensor(obj_pose_xy.rotation_wxyz))
            R_inv = R.T
            q_inv = quat_from_matrix(R_inv)
            t_inv = -(R_inv @ torch.tensor(obj_pose_xy.position_xyz))
            correction = Pose(
                position_xyz=tuple(t_inv.tolist()),
                rotation_wxyz=tuple(q_inv.tolist()),
            )

            # Object pose becomes (0, 0, z_original) with identity rotation
            object_pose = Pose(
                position_xyz=(0.0, 0.0, object_pose.position_xyz[2]),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )

            # Apply correction to background pose
            bg_pose = background.get_initial_pose()
            if bg_pose is None:
                bg_pose = Pose.identity()
            background.set_initial_pose(compose_poses(correction, bg_pose))
        # ---------------------------------------------------------------

        target_object.set_initial_pose(object_pose)
        assets = [background, target_object]

        # ---- Load embodiment ----
        embodiment = self.asset_registry.get_asset_by_name("summit_franka")(
            enable_cameras=args_cli.enable_cameras
        )

        # If cameras are enabled, attach the fixed local camera under the target object prim.
        # This keeps the camera parented to the object even when the object name changes.
        if args_cli.enable_cameras and embodiment.camera_config is not None:
            object_prim_path = target_object.get_prim_path()
            embodiment.camera_config.fix_local.prim_path = f"{object_prim_path}/fix_local"

        # Set the robot's initial pose (position it near the object)
        # if object_center: (backup, not used)
        #     # In object_center mode, apply the same correction transform to
        #     # a default robot pose relative to the original object position.
        #     # The robot is placed 0.6m in front of the object in its local frame.
        #     from isaaclab_arena.utils.pose import compose_poses as _compose
        #     raw_robot_pose = Pose(
        #         position_xyz=(
        #             original_object_pose.position_xyz[0] - 2.5,  
        #             original_object_pose.position_xyz[1],
        #             0.0,
        #         ),
        #         rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        #     )
        #     robot_initial_pose = _compose(correction, raw_robot_pose)
        robot_initial_pose = Pose(
            position_xyz=(
                object_pose.position_xyz[0],  # TODO: make this configurable: -1.0
                object_pose.position_xyz[1],
                0.0,  # robot base on the ground
            ),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        embodiment.set_initial_pose(robot_initial_pose)

        # ---- Optional teleop device ----
        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # ---- Optionally add an extra object ----
        if args_cli.object is not None:
            extra_object = self.asset_registry.get_asset_by_name(args_cli.object)()
            extra_object_pose = Pose(
                position_xyz=(
                    object_pose.position_xyz[0] + 0.3,
                    object_pose.position_xyz[1] - 0.3,
                    object_pose.position_xyz[2],
                ),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
            extra_object.set_initial_pose(extra_object_pose)
            assets.append(extra_object)

        # ---- Compose the scene ----
        scene = Scene(assets=assets)

        # ---- Create the task ----
        task = OpenDoorTask(
            target_object,
            openness_threshold=0.8,
            reset_openness=0.3,
            episode_length_s=2.0,
        )

        # ---- Create the environment ----
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )

        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--object_name",
            type=str,
            default="microwave_7221",
            help=(
                "Automoma object name in <type>_<id> format. Supported examples: "
                "microwave_7221, dishwasher_11622, oven_101773"
            ),
        )
        parser.add_argument(
            "--scene_name",
            type=str,
            default="scene_0_seed_0",
            help="Automoma scene name, e.g. scene_0_seed_0, scene_1_seed_1",
        )
        parser.add_argument(
            "--object_center",
            action="store_true",
            default=False,
            help=(
                "If set, shift the world origin so the object is at (0, 0, z). "
                "The robot and scene background are offset accordingly."
            ),
        )
        parser.add_argument("--object", type=str, default=None, help="Optional additional object from asset registry")
        parser.add_argument("--teleop_device", type=str, default=None, help="Teleoperation device name")
