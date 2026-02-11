Closed-Loop Policy Inference and Evaluation
-------------------------------------------

This workflow demonstrates running the trained GR00T N1.5 policy in closed-loop
and evaluating it in the Summit Franka Open Door environment.

**Docker Container**: Base + GR00T (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_gr00t:

Once inside the container, set directories:

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/summit_franka_tutorial
    export MODELS_DIR=/models/isaaclab_arena/summit_franka_tutorial


Step 1: Run Single Environment Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GR00T model is configured by
``isaaclab_arena_gr00t/summit_franka_manip_gr00t_closedloop_config.yaml``.

.. dropdown:: Configuration file (``summit_franka_manip_gr00t_closedloop_config.yaml``):
   :animate: fade-in

   .. code-block:: yaml

      model_path: /models/isaaclab_arena/summit_franka_tutorial/checkpoint-20000
      language_instruction: "Reach out to the microwave and open it."
      action_horizon: 16
      action_chunk_length: 16
      embodiment_tag: summit_franka
      data_config: isaaclab_arena_gr00t.data_config:SummitFrankaDataConfig
      video_backend: decord
      policy_joints_config_path: isaaclab_arena_gr00t/config/summit_franka/gr00t_12dof_joint_space.yaml
      action_joints_config_path: isaaclab_arena_gr00t/config/summit_franka/12dof_action_joint_space.yaml
      state_joints_config_path: isaaclab_arena_gr00t/config/summit_franka/12dof_state_joint_space.yaml
      task_mode_name: summit_franka_manipulation
      pov_cam_names_sim:
        - "ego_topdown_rgb"
        - "ego_wrist_rgb"
        - "fix_local_rgb"
      original_image_size: [512, 512, 3]
      target_image_size: [512, 512, 3]


Test the policy in a single environment:

.. code-block:: bash

   python isaaclab_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaaclab_arena_gr00t/summit_franka_manip_gr00t_closedloop_config.yaml \
     --num_steps 2000 \
     --enable_cameras \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_0_seed_0

Expected output:

.. code-block:: text

   Metrics: {'success_rate': ..., 'door_moved_rate': ..., 'num_episodes': ...}


Step 2: Run Parallel Environments Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Test the policy in 10 parallel environments:

.. code-block:: bash

   python isaaclab_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaaclab_arena_gr00t/summit_franka_manip_gr00t_closedloop_config.yaml \
     --num_steps 2000 \
     --num_envs 10 \
     --enable_cameras \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_0_seed_0

During evaluation you should see:

.. code-block:: text

   Resetting policy for terminated env_ids: tensor([...]) and truncated env_ids: tensor([...])


Swapping Objects and Scenes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One key advantage of this setup is that you can easily swap the object and scene:

.. code-block:: bash

   # Use a different object (e.g. dishwasher)
   python isaaclab_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaaclab_arena_gr00t/summit_franka_manip_gr00t_closedloop_config.yaml \
     --num_steps 2000 \
     --enable_cameras \
     summit_franka_open_door \
     --object_name dishwasher_11622 \
     --scene_name scene_0_seed_0

   # Use a different scene
   python isaaclab_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaaclab_arena_gr00t/summit_franka_manip_gr00t_closedloop_config.yaml \
     --num_steps 2000 \
     --enable_cameras \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_1_seed_1


Using Object-Center Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^

Add ``--object_center`` to shift the world origin to the object center:

.. code-block:: bash

   python isaaclab_arena/examples/policy_runner.py \
     --policy_type gr00t_closedloop \
     --policy_config_yaml_path isaaclab_arena_gr00t/summit_franka_manip_gr00t_closedloop_config.yaml \
     --num_steps 2000 \
     --enable_cameras \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_0_seed_0 \
     --object_center


.. note::

   - Objects must follow the automoma openable convention (URDF/USD with ``joint_0`` as the door hinge).
   - Objects and scenes must exist under ``res_for_custom/automoma_assets/``.
   - Scene metadata must contain a matching entry for the specified object type and id.
   - For different objects, you may want to retrain the policy or adjust the language instruction.
   - For physics tuning, see :doc:`physics_tuning_guide`.
   - For adding custom assets, see :doc:`custom_assets_guide`.
