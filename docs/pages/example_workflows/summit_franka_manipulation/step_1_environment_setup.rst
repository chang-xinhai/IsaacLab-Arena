Environment Setup and Validation
--------------------------------

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:


Environment Description
^^^^^^^^^^^^^^^^^^^^^^^

The ``SummitFrankaOpenDoorEnvironment`` loads assets dynamically from the
``automoma_assets`` collection. Objects and scenes are specified by name and can
be swapped without modifying code.

This is a **generic** environment — it works with any automoma openable object
(microwave, dishwasher, oven, etc.), not just microwaves. The legacy name
``summit_franka_open_microwave`` is kept as a backward-compatible alias.

.. dropdown:: The Summit Franka Open Door Environment
   :animate: fade-in

   .. code-block:: python

      class SummitFrankaOpenDoorEnvironment(ExampleEnvironmentBase):
          name: str = "summit_franka_open_door"

          def get_env(self, args_cli: argparse.Namespace):
              from isaaclab_arena.assets.automoma_background_library import get_automoma_scene
              from isaaclab_arena.assets.automoma_object_library import (
                  get_automoma_object, get_object_pose_from_metadata, load_scene_metadata,
              )
              from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
              from isaaclab_arena.scene.scene import Scene
              from isaaclab_arena.tasks.open_door_task import OpenDoorTask
              from isaaclab_arena.utils.pose import Pose, compose_poses

              object_name = args_cli.object_name   # e.g. "microwave_7221"
              scene_name = args_cli.scene_name      # e.g. "scene_0_seed_0"
              object_center = getattr(args_cli, "object_center", False)

              # Parse type and id
              parts = object_name.split("_")
              asset_id = parts[-1]
              asset_type = "_".join(parts[:-1]).capitalize()

              # Load scene and object
              background = get_automoma_scene(scene_name)
              target_object = get_automoma_object(asset_type=asset_type, asset_id=asset_id)

              # Load pose from scene metadata
              metadata = load_scene_metadata(scene_name)
              object_pose, _ = get_object_pose_from_metadata(metadata, asset_type, asset_id)
              original_object_pose = object_pose  # keep for robot placement

              # Optional: shift origin to object center with identity rotation
              if object_center:
                  # Compute inverse of (x, y, rotation) to place object at (0, 0, z, 1, 0, 0, 0)
                  obj_pose_xy = Pose(
                      position_xyz=(object_pose.position_xyz[0], object_pose.position_xyz[1], 0.0),
                      rotation_wxyz=object_pose.rotation_wxyz,
                  )
                  # correction = inverse(obj_pose_xy) — counter-rotates AND translates
                  correction = ...  # see full source for inverse computation
                  object_pose = Pose(
                      position_xyz=(0.0, 0.0, object_pose.position_xyz[2]),
                      rotation_wxyz=(1.0, 0.0, 0.0, 0.0),  # identity rotation
                  )
                  background.set_initial_pose(compose_poses(correction, bg_pose))

              target_object.set_initial_pose(object_pose)

              # Load robot — placed relative to the corrected/original object pose
              embodiment = self.asset_registry.get_asset_by_name("summit_franka")(
                  enable_cameras=args_cli.enable_cameras
              )
              # In object_center mode, the robot pose is also transformed
              if object_center:
                  robot_initial_pose = compose_poses(correction, raw_robot_pose)
              else:
                  robot_initial_pose = Pose(
                      position_xyz=(object_pose.position_xyz[0]-0.6,
                                    object_pose.position_xyz[1], 0.0),
                      rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
                  )
              embodiment.set_initial_pose(robot_initial_pose)

              scene = Scene(assets=[background, target_object])
              task = OpenDoorTask(target_object, openness_threshold=0.8,
                                  reset_openness=0.3, episode_length_s=2.0)
              return IsaacLabArenaEnvironment(
                  name=self.name, embodiment=embodiment, scene=scene, task=task,
              )


Step-by-Step Breakdown
^^^^^^^^^^^^^^^^^^^^^^^

**1. Dynamic Asset Loading**

.. code-block:: python

   background = get_automoma_scene(scene_name)          # e.g. "scene_0_seed_0"
   target_object = get_automoma_object(
       asset_type="Microwave", asset_id="7221"
   )

Objects and scenes are loaded from ``res_for_custom/automoma_assets/`` by name.
You can swap the object by changing ``--object_name`` (e.g. ``dishwasher_11622``,
``oven_101773``) and the scene by changing ``--scene_name`` (e.g. ``scene_1_seed_1``).

**2. Pose from Metadata**

.. code-block:: python

   metadata = load_scene_metadata(scene_name)
   object_pose, object_scale = get_object_pose_from_metadata(
       metadata, asset_type="Microwave", asset_id="7221"
   )
   target_object.set_initial_pose(object_pose)

The object pose is read from ``res_for_custom/automoma_assets/scene/<scene>/info/metadata.json``,
not hardcoded. This ensures consistency with the Blender-exported scene layout.

**3. Object-Center Mode**

When ``--object_center`` is passed, the world origin shifts so the object is at
``(0, 0, z, 1, 0, 0, 0)`` — that is, position ``(0, 0, z)`` with **identity rotation**.
This means both the position (x, y) and the rotation of the object are corrected:
the object's original rotation is counter-rotated to identity, and all other prims
(robot, background) are transformed by the same inverse correction. This is useful
for policies that expect the object to always be at a canonical, axis-aligned pose.

.. code-block:: bash

   # Run with object-center mode
   python isaaclab_arena/examples/policy_runner.py \
     --device cpu --enable_cameras --num_steps 100 \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_0_seed_0 \
     --object_center

**4. Robot & Cameras**

The Summit Franka embodiment provides **12 DOF** (3 base + 7 arm + 2 gripper) for
coordinated whole-body manipulation. Three cameras are available:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Camera
     - Resolution
     - Description
   * - ``ego_topdown``
     - 320 x 240
     - Mounted on ``base_link_y``, looking straight down from 10 m above.
   * - ``ego_wrist``
     - 320 x 240
     - Mounted on ``panda_hand``, wrist-eye view with wide FOV (focal_length=1.5).
   * - ``fix_local``
     - 320 x 240
     - Scene-fixed third-person view (attached to Object prim).

Cameras are activated by passing ``--enable_cameras``.

**5. Task and Environment**

.. code-block:: python

   task = OpenDoorTask(target_object, openness_threshold=0.8,
                       reset_openness=0.3, episode_length_s=2.0)
   env = IsaacLabArenaEnvironment(
       name="summit_franka_open_door",
       embodiment=embodiment, scene=scene, task=task,
   )


Step 1: Validate the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run a zero-action test to verify the environment loads correctly:

.. code-block:: bash

   python isaaclab_arena/examples/policy_runner.py \
     --enable_cameras \
     --num_steps 100 \
     --policy_type zero_action \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_0_seed_0

You should see the Summit Franka robot and the microwave in the automoma scene.
The robot will not move (zero actions), but the environment should load without errors.

.. note::

   If the environment fails to load, check that:

   - The USD files exist at the expected paths under ``res_for_custom/automoma_assets/``
   - The object name and scene name match existing directories
   - The metadata.json contains the expected object entry
   - For physics tuning issues, see :doc:`physics_tuning_guide`
