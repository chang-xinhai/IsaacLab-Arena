Adding Custom Robots, Objects, and Scenes
==========================================

This guide explains how to add your own robot, object, or scene to the
Isaac Lab Arena automoma pipeline.

.. contents:: Table of Contents
   :depth: 2
   :local:


Directory Structure
--------------------

All automoma assets live under ``res_for_custom/automoma_assets/``:

.. code-block:: text

   res_for_custom/automoma_assets/
   ├── robot/
   │   └── <robot_name>/
   │       ├── <robot_name>.urdf
   │       └── <robot_name>/
   │           └── <robot_name>.usd
   ├── object/
   │   └── <type>_<id>/
   │       ├── <id>_0_scaling.urdf
   │       ├── <id>_0_scaling/
   │       │   └── <id>_0_scaling.usd
   │       ├── bounding_box.json
   │       ├── meta.json
   │       └── semantics.txt
   └── scene/
       └── <scene_name>/
           ├── export/
           │   └── export_scene.blend/
           │       └── export_scene.usdc
           └── info/
               └── metadata.json


Adding a New Object
---------------------

Requirements
^^^^^^^^^^^^

Your openable object must:

1. Have a URDF and corresponding USD under ``object/<type>_<id>/``
2. Use ``joint_0`` as the door hinge joint (PartNet-Mobility convention)
3. Have proper collision geometry on the handle for friction-based grasping
4. Be referenced in the scene's ``metadata.json`` with position, rotation, and scale

Steps
^^^^^

1. **Place the USD/URDF assets:**

   .. code-block:: bash

      # Example: adding a new fridge model
      mkdir -p res_for_custom/automoma_assets/object/fridge_12345/12345_0_scaling/
      cp your_fridge.usd res_for_custom/automoma_assets/object/fridge_12345/12345_0_scaling/12345_0_scaling.usd
      cp your_fridge.urdf res_for_custom/automoma_assets/object/fridge_12345/12345_0_scaling.urdf

2. **Verify the joint naming.** Open the USD/URDF and confirm the door hinge is
   named ``joint_0``. If it has a different name, either rename it or create a
   subclass of ``AutomomaOpenableObject`` with a custom ``openable_joint_name``.

3. **Add to scene metadata.** In
   ``res_for_custom/automoma_assets/scene/<scene>/info/metadata.json``,
   add an entry under ``static_objects``:

   .. code-block:: json

      "StaticCategoryFactory(Fridge_12345_some_id_mobility)": {
          "name": "StaticCategoryFactory(Fridge_12345_some_id_mobility)",
          "asset_type": "Fridge",
          "asset_id": "12345",
          "position": [1.5, 0.3, 0.0],
          "rotation": [0.0, 0.0, 1.5708],
          "scale": [1.0, 1.0, 1.0]
      }

4. **Use it from the CLI:**

   .. code-block:: bash

      python isaaclab_arena/examples/policy_runner.py \
        --device cpu --enable_cameras --num_steps 100 \
        summit_franka_open_door \
        --object_name fridge_12345 \
        --scene_name scene_0_seed_0

5. **(Optional) Pre-register the object** for convenience:

   In ``isaaclab_arena/assets/automoma_object_library.py``, add:

   .. code-block:: python

      @register_asset
      class Fridge12345(AutomomaOpenableObject):
          name = "fridge_12345"
          def __init__(self, prim_path=None, initial_pose=None, **kwargs):
              super().__init__(
                  name=self.name,
                  asset_type="Fridge",
                  asset_id="12345",
                  prim_path=prim_path,
                  initial_pose=initial_pose,
                  **kwargs,
              )

   This allows ``self.asset_registry.get_asset_by_name("fridge_12345")``.


Adding a New Robot
-------------------

Requirements
^^^^^^^^^^^^

Your robot must:

1. Have a USD file (converted from URDF if needed)
2. Define all actuated joints explicitly
3. Have collision geometry on any parts interacting with objects (especially gripper)

Steps
^^^^^

1. **Place the USD:** ``res_for_custom/automoma_assets/robot/<robot_name>/<robot_name>/<robot_name>.usd``

2. **Create an embodiment class** in ``isaaclab_arena/embodiments/<robot_name>/<robot_name>.py``.
   Follow the pattern of ``SummitFrankaEmbodiment``:

   - Define ``ArticulationCfg`` with joint names, actuator parameters, initial state
   - Define ``ActionsCfg`` (IK, joint position, or other control mode)
   - Define ``ObservationsCfg`` with needed observations
   - Define ``CameraCfg`` with your camera setup
   - Register with ``@register_asset``

3. **Create joint space configs** in ``isaaclab_arena_gr00t/config/<robot_name>/``:

   - ``<N>dof_action_joint_space.yaml`` — joints in the action space
   - ``<N>dof_state_joint_space.yaml`` — joints in the state observation
   - ``gr00t_<N>dof_joint_space.yaml`` — joints for GR00T policy prediction
   - ``modality.json`` — maps state/action/video fields to tensor indices
   - ``info.json`` — LeRobot dataset template

4. **Create a DataConfig** in ``isaaclab_arena_gr00t/data_config.py`` (extend ``BaseDataConfig``).

5. **Create a YAML config** for HDF5→LeRobot conversion and closed-loop inference.


Adding a New Scene
-------------------

Requirements
^^^^^^^^^^^^

1. A USDC file exported from Blender (or any DCC tool)
2. A ``metadata.json`` listing all objects placed in the scene with their poses

Steps
^^^^^

1. **Place the scene:**

   .. code-block:: bash

      mkdir -p res_for_custom/automoma_assets/scene/<scene_name>/export/export_scene.blend/
      cp your_scene.usdc res_for_custom/automoma_assets/scene/<scene_name>/export/export_scene.blend/export_scene.usdc
      mkdir -p res_for_custom/automoma_assets/scene/<scene_name>/info/
      cp your_metadata.json res_for_custom/automoma_assets/scene/<scene_name>/info/metadata.json

2. **Metadata format.** The ``metadata.json`` must have a ``static_objects`` dict:

   .. code-block:: json

      {
          "static_objects": {
              "<unique_key>": {
                  "asset_type": "Microwave",
                  "asset_id": "7221",
                  "position": [x, y, z],
                  "rotation": [roll, pitch, yaw],
                  "scale": [sx, sy, sz]
              }
          }
      }

   - ``position``: world coordinates in meters
   - ``rotation``: Euler angles (roll, pitch, yaw) in radians
   - ``scale``: uniform or non-uniform scale factors

3. **(Optional) Pre-register the scene** in
   ``isaaclab_arena/assets/automoma_background_library.py``:

   .. code-block:: python

      @register_asset
      class MySceneBackground(AutomomaSceneBackground):
          name = "my_scene"
          def __init__(self, prim_path=None, initial_pose=None, **kwargs):
              super().__init__(
                  name=self.name,
                  scene_name="my_scene",
                  prim_path=prim_path,
                  initial_pose=initial_pose,
                  **kwargs,
              )


Camera Configuration Guide
----------------------------

The Summit Franka uses 3 cameras defined in ``SummitFrankaCameraCfg``:

.. list-table::
   :widths: 15 20 15 50
   :header-rows: 1

   * - Camera
     - Parent Prim
     - Resolution
     - Purpose
   * - ego_topdown
     - Robot/base_link_y
     - 320 x 240
     - Top-down bird's eye view of the workspace
   * - ego_wrist
     - Robot/panda_hand
     - 320 x 240
     - Wrist-eye view for close-up manipulation
   * - fix_local
     - Object prim
     - 320 x 240
     - Scene-fixed third-person observation

To **add or modify cameras** for your robot:

1. Define a ``CameraCfg`` in your embodiment's camera config class
2. Set the ``prim_path`` to attach the camera to the correct link
3. Configure ``offset`` (position and rotation in parent frame)
4. Set ``data_types`` (``["rgb"]``, ``["rgb", "depth"]``, etc.)
5. Ensure the matching video keys are added to your ``modality.json`` and ``DataConfig``

.. code-block:: python

   my_camera: CameraCfg = CameraCfg(
       prim_path="{ENV_REGEX_NS}/Robot/my_link/my_camera",
       update_period=0.0,
       height=240,
       width=320,
       data_types=["rgb"],
       spawn=sim_utils.PinholeCameraCfg(
           focal_length=24.0,
           clipping_range=(0.01, 1.0e5),
       ),
       offset=CameraCfg.OffsetCfg(
           pos=(0.0, 0.0, 0.5),
           rot=(1.0, 0.0, 0.0, 0.0),
           convention="opengl",
       ),
   )
