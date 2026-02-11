Policy Post-training
--------------------

This workflow covers post-training a GR00T N1.5 policy using the generated dataset
for the Summit Franka task.

**Docker Container**: Base + GR00T (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_gr00t:

Once inside the container, set the dataset and models directories:

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/summit_franka_tutorial
    export MODELS_DIR=/models/isaaclab_arena/summit_franka_tutorial


Step 1: Convert to LeRobot Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convert the HDF5 dataset to LeRobot format:

.. code-block:: bash

   python isaaclab_arena_gr00t/data_utils/convert_hdf5_to_lerobot.py \
     --yaml_file isaaclab_arena_gr00t/config/summit_franka_manip_config.yaml

This creates a folder
``$DATASET_DIR/summit_franka_open_microwave_7221_generated/lerobot``
containing parquet files with states/actions, MP4 camera recordings (3 views), and
dataset metadata.

.. dropdown:: Configuration file (``summit_franka_manip_config.yaml``)
   :animate: fade-in

   .. code-block:: yaml

      data_root: /datasets/isaaclab_arena/summit_franka_tutorial
      hdf5_name: "summit_franka_open_microwave_7221_generated.hdf5"
      language_instruction: "Reach out to the microwave and open it."
      task_index: 0
      state_name_sim: "robot_joint_pos"
      action_name_sim: "processed_actions"

      # 3 camera views
      pov_cam_names_sim:
        - "ego_topdown_rgb"
        - "ego_wrist_rgb"
        - "fix_local_rgb"

      fps: 50
      chunks_size: 1000

      # 12 DOF joint space configs
      policy_joints_config_path: "isaaclab_arena_gr00t/config/summit_franka/gr00t_12dof_joint_space.yaml"
      action_joints_config_path: "isaaclab_arena_gr00t/config/summit_franka/12dof_action_joint_space.yaml"
      state_joints_config_path: "isaaclab_arena_gr00t/config/summit_franka/12dof_state_joint_space.yaml"
      robot_type: "summit_franka"

.. dropdown:: Modality mapping (``modality.json``)
   :animate: fade-in

   .. code-block:: json

      {
          "state": {
              "base":    {"original_key": "observation.state", "start": 0,  "end": 3},
              "arm":     {"original_key": "observation.state", "start": 3,  "end": 10},
              "gripper": {"original_key": "observation.state", "start": 10, "end": 12}
          },
          "action": {
              "base":    {"start": 0,  "end": 3},
              "arm":     {"start": 3,  "end": 10},
              "gripper": {"start": 10, "end": 12}
          },
          "video": {
              "ego_topdown": {"original_key": "observation.images.ego_topdown"},
              "ego_wrist":   {"original_key": "observation.images.ego_wrist"},
              "fix_local":   {"original_key": "observation.images.fix_local"}
          },
          "annotation": {
              "human.action.task_description": {"original_key": "task_index"}
          }
      }


Step 2: Post-train Policy
^^^^^^^^^^^^^^^^^^^^^^^^^

Post-train the GR00T N1.5 policy on the Summit Franka dataset.

The data configuration class ``SummitFrankaDataConfig`` is already registered at
``isaaclab_arena_gr00t.data_config:SummitFrankaDataConfig``. It defines:

- **video_keys**: ``video.ego_topdown``, ``video.ego_wrist``, ``video.fix_local``
- **state_keys**: ``state.base``, ``state.arm``, ``state.gripper`` (12 DOF total)
- **action_keys**: ``action.base``, ``action.arm``, ``action.gripper`` (12 DOF total)

.. tabs::

   .. tab:: Single GPU

      .. code-block:: bash

         cd submodules/Isaac-GR00T

         python scripts/gr00t_finetune.py \
         --dataset_path=$DATASET_DIR/summit_franka_open_microwave_7221_generated/lerobot \
         --output_dir=$MODELS_DIR \
         --data_config=isaaclab_arena_gr00t.data_config:SummitFrankaDataConfig \
         --batch_size=24 \
         --max_steps=20000 \
         --num_gpus=1 \
         --save_steps=5000 \
         --base_model_path=nvidia/GR00T-N1.5-3B \
         --no_tune_llm \
         --tune_visual \
         --tune_projector \
         --tune_diffusion_model \
         --no-resume \
         --dataloader_num_workers=16 \
         --report_to=wandb \
         --embodiment_tag=summit_franka \
         --lora_rank=128

   .. tab:: Multi-GPU (Best Quality)

      .. code-block:: bash

         cd submodules/Isaac-GR00T

         python scripts/gr00t_finetune.py \
         --dataset_path=$DATASET_DIR/summit_franka_open_microwave_7221_generated/lerobot \
         --output_dir=$MODELS_DIR \
         --data_config=isaaclab_arena_gr00t.data_config:SummitFrankaDataConfig \
         --batch_size=24 \
         --max_steps=20000 \
         --num_gpus=8 \
         --save_steps=5000 \
         --base_model_path=nvidia/GR00T-N1.5-3B \
         --no_tune_llm \
         --tune_visual \
         --tune_projector \
         --tune_diffusion_model \
         --no-resume \
         --dataloader_num_workers=16 \
         --report_to=wandb \
         --embodiment_tag=summit_franka

See the `GR00T fine-tuning guidelines <https://github.com/NVIDIA/Isaac-GR00T#3-fine-tuning>`_
for adjusting the training configuration.
