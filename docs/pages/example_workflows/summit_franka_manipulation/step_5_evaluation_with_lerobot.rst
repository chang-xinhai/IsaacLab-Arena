Evaluation with LeRobot
-----------------------

This workflow demonstrates how to evaluate a trained policy (ACT, SmolVLA, PI0,
etc.) using **lerobot-eval** against the Summit Franka Open Door environment in
IsaacLab Arena.  Unlike :doc:`step_5_evaluation` (which uses GR00T via
``policy_runner.py`` inside Docker), this approach runs directly through the
LeRobot evaluation pipeline and works fully locally — no Docker or Hub uploads
required.


Architecture Overview
^^^^^^^^^^^^^^^^^^^^^

The ``lerobot-eval`` pipeline connects to IsaacLab Arena via the
``isaaclab-arena-envs`` bridge package:

.. code-block:: text

   lerobot-eval CLI
        │
        ▼
   EvalPipelineConfig  (--env.type=isaaclab_arena)
        │
        ▼
   make_env()  ──►  _download_hub_file()  ──►  isaaclab-arena-envs/env.py
        │                                              │
        ▼                                              ▼
   IsaaclabArenaProcessorStep           ArenaEnvBuilder → IsaacLabEnvWrapper
   (obs ↔ policy format)                  (IsaacSim environment)

Key components:

- **IsaaclabArenaEnv** (``--env.type=isaaclab_arena``): LeRobot's ``EnvConfig``
  subclass that carries all environment parameters.
- **Hub bridge** (``isaaclab-arena-envs/env.py``): The ``make_env`` entrypoint
  that creates the IsaacLab Arena environment and wraps it in a Gymnasium
  ``VectorEnv``.
- **IsaaclabArenaProcessorStep**: Transforms raw IsaacLab observations (joint
  positions, camera images) into the format expected by the trained policy.
- **SummitFrankaOpenDoorEvalEnvironment**: Custom environment variant with
  12-DOF joint-space actions and absolute observations, matching the data format
  produced by the automoma recording pipeline.


Prerequisites
^^^^^^^^^^^^^

1. **IsaacSim 5.1.0** or later installed and activated.
2. **LeRobot** installed in editable mode:

   .. code-block:: bash

      cd lerobot
      pip install -e ".[dev]"

3. **IsaacLab Arena** installed in editable mode:

   .. code-block:: bash

      cd IsaacLab-Arena
      pip install -e .

4. A **trained policy checkpoint** — this tutorial uses the ACT model trained in
   :doc:`step_4_policy_training`:

   .. code-block:: text

      lerobot/outputs/train/act_summit_franka_open_microwave_7221_setstate/
        checkpoints/010000/pretrained_model/
          config.json        ← policy architecture & feature definitions
          model.safetensors  ← trained weights
          config.yaml        ← training run config

   The trained ACT model expects:

   .. list-table::
      :widths: 40 60
      :header-rows: 1

      * - Feature
        - Shape
      * - ``observation.state``
        - ``(12,)`` — 3 base + 7 arm + 2 gripper joint positions
      * - ``observation.img_state_delta``
        - ``(1,)`` — dummy zero feature (from dataset conversion)
      * - ``observation.images.ego_topdown``
        - ``(3, 240, 320)``
      * - ``observation.images.ego_wrist``
        - ``(3, 240, 320)``
      * - ``observation.images.fix_local``
        - ``(3, 240, 320)``
      * - ``action`` (output)
        - ``(12,)`` — 12-DOF joint positions


Step 1: Local Hub Path Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``lerobot-eval`` pipeline uses ``--env.hub_path`` to locate the bridge code.
By default this points to a HuggingFace Hub repo (``nvidia/isaaclab-arena-envs``),
but **local filesystem paths are also supported**:

.. code-block:: bash

   # Absolute path to the isaaclab-arena-envs directory
   --env.hub_path=/path/to/IsaacLab-Arena/isaaclab-arena-envs

This skips all Hub downloads and loads ``env.py`` directly from the local
directory.  The ``example_envs.yaml``, environment classes, and all other
dependencies resolve from the same directory tree.

.. tip::

   If you later want to share your env bridge with others, you can upload the
   ``isaaclab-arena-envs`` directory as a HuggingFace Hub repo:

   .. code-block:: bash

      cd IsaacLab-Arena/isaaclab-arena-envs
      huggingface-cli repo create <your-username>/isaaclab-arena-envs --type space
      git init && git add . && git commit -m "initial"
      git remote add origin https://huggingface.co/spaces/<your-username>/isaaclab-arena-envs
      git push origin main

   Then use ``--env.hub_path=<your-username>/isaaclab-arena-envs``.


Step 2: Understanding the Eval Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The evaluation environment ``summit_franka_open_door_eval`` is defined in
``isaaclab_arena/examples/example_environments/summit_franka_open_door_eval_environment.py``.
It inherits from the base ``SummitFrankaOpenDoorEnvironment`` and makes the
following changes for policy evaluation:

1. **Action space**: Switched to ``SummitFrankaJointSpaceActionsCfg`` (12-DOF
   absolute joint positions) instead of IK-based actions.
2. **Observations**: Uses absolute ``joint_pos`` instead of relative positions.
3. **Extra CLI args**: ``--disable_collision`` and ``--mobile_base_relative``.

These changes ensure the environment produces observations and accepts actions
in the same format as the recorded training dataset.


Step 3: Run lerobot-eval
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd IsaacLab-Arena

   lerobot-eval \
     --policy.path=../lerobot/outputs/train/act_summit_franka_open_microwave_7221_setstate/checkpoints/010000/pretrained_model \
     --policy.device=cuda \
     --env.type=isaaclab_arena \
     --env.hub_path=$(pwd)/isaaclab-arena-envs \
     --env.environment=summit_franka_open_door_eval \
     --env.headless=false \
     --env.enable_cameras=true \
     --env.state_keys=joint_pos \
     --env.camera_keys=ego_topdown_rgb,ego_wrist_rgb,fix_local_rgb \
     --env.state_dim=12 \
     --env.action_dim=12 \
     --env.camera_height=240 \
     --env.camera_width=320 \
     --env.episode_length=300 \
     --env.kwargs='{"object_name": "microwave_7221", "scene_name": "scene_0_seed_0", "object_center": true, "disable_collision": true, "mobile_base_relative": true}' \
     --rename_map='{"observation.images.ego_topdown_rgb": "observation.images.ego_topdown", "observation.images.ego_wrist_rgb": "observation.images.ego_wrist", "observation.images.fix_local_rgb": "observation.images.fix_local"}' \
     --trust_remote_code=true \
     --eval.batch_size=1 \
     --eval.n_episodes=10


Parameter Reference
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Parameter
     - Description
   * - ``--policy.path``
     - Path to the trained model directory containing ``config.json`` and
       ``model.safetensors``.
   * - ``--policy.device``
     - Device for policy inference (``cuda`` or ``cpu``).
   * - ``--env.type``
     - Must be ``isaaclab_arena`` to use the IsaacLab Arena bridge.
   * - ``--env.hub_path``
     - Path to the ``isaaclab-arena-envs`` directory or a HF Hub repo ID.
   * - ``--env.environment``
     - Environment name (registered in ``example_envs.yaml``). Use
       ``summit_franka_open_door_eval`` for joint-space evaluation.
   * - ``--env.headless``
     - ``false`` to see the Isaac Sim viewport, ``true`` for headless.
   * - ``--env.enable_cameras``
     - Must be ``true`` for vision-based policies.
   * - ``--env.state_keys``
     - Comma-separated observation keys from the ``policy`` obs group.
       For the eval environment: ``joint_pos`` (absolute positions).
   * - ``--env.camera_keys``
     - Comma-separated camera keys from the ``camera_obs`` group.
       Note the ``_rgb`` suffix (``ego_topdown_rgb``, not ``ego_topdown``).
   * - ``--env.state_dim``
     - State vector dimension. Must match the policy's ``observation.state``
       shape. Summit Franka = 12 (3 base + 7 arm + 2 gripper).
   * - ``--env.action_dim``
     - Action vector dimension. Must match the policy's ``action`` shape.
       Summit Franka = 12.
   * - ``--env.camera_height/width``
     - Camera resolution. Must match training data (240 × 320).
   * - ``--env.episode_length``
     - Maximum steps per episode before truncation.
   * - ``--env.kwargs``
     - JSON dict of extra parameters passed through to the IsaacLab
       environment CLI. See below.
   * - ``--rename_map``
     - JSON dict mapping env observation keys to policy expected keys.
       Needed because camera keys in the env have ``_rgb`` suffix but the
       trained model expects names without it.
   * - ``--trust_remote_code``
     - Must be ``true`` to allow loading the hub bridge code.
   * - ``--eval.batch_size``
     - Number of parallel environments (use 1 for debugging).
   * - ``--eval.n_episodes``
     - Total number of evaluation episodes.


Environment kwargs
^^^^^^^^^^^^^^^^^^^

The ``--env.kwargs`` JSON dict passes extra parameters to the IsaacLab Arena
environment. These are promoted to CLI arguments of the environment class:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``object_name``
     - str
     - Automoma object name (e.g. ``microwave_7221``, ``dishwasher_11622``).
   * - ``scene_name``
     - str
     - Scene name (e.g. ``scene_0_seed_0``).
   * - ``object_center``
     - bool
     - If ``true``, shifts world origin to the object center (must match
       training data setup).
   * - ``disable_collision``
     - bool
     - Disables collision between robot and object. Useful for set-state
       recordings or when large action jumps cause interpenetration.
       See :doc:`physics_tuning_guide` §8.
   * - ``mobile_base_relative``
     - bool
     - If ``true``, the first ``base_dof`` (default 3) action dimensions are
       treated as relative deltas (Δx, Δy, Δθ). The wrapper integrates them
       with the current base state before sending absolute positions to sim.
       Must match the recording mode (``--mobile_base_relative`` flag during
       ``record_automoma_demos.py``).


Rename Map Explained
^^^^^^^^^^^^^^^^^^^^^

The ``--rename_map`` bridges a naming mismatch:

- The IsaacLab camera observation group names cameras with a ``_rgb`` suffix
  (e.g. ``ego_topdown_rgb``), which is how ``make_camera_observation_cfg``
  constructs observation terms (``<camera_name>_<data_type>``).
- During HDF5→LeRobot dataset conversion, camera keys are stored **without**
  the ``_rgb`` suffix (e.g. ``ego_topdown``).
- The trained policy therefore expects ``observation.images.ego_topdown``, but
  the env processor produces ``observation.images.ego_topdown_rgb``.

The rename map resolves this:

.. code-block:: json

   {
     "observation.images.ego_topdown_rgb": "observation.images.ego_topdown",
     "observation.images.ego_wrist_rgb": "observation.images.ego_wrist",
     "observation.images.fix_local_rgb": "observation.images.fix_local"
   }


Swapping Objects and Scenes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Change the ``object_name`` and ``scene_name`` in ``--env.kwargs``:

.. code-block:: bash

   # Evaluate on a dishwasher
   --env.kwargs='{"object_name": "dishwasher_11622", "scene_name": "scene_0_seed_0", "object_center": true, "disable_collision": true, "mobile_base_relative": true}'

   # Evaluate on a different scene
   --env.kwargs='{"object_name": "microwave_7221", "scene_name": "scene_1_seed_1", "object_center": true, "disable_collision": true, "mobile_base_relative": true}'

.. important::

   The policy was trained on a specific object and scene. Cross-object or
   cross-scene generalization depends on the training data diversity and
   the policy architecture.


Video Recording
^^^^^^^^^^^^^^^^

Add ``--env.video=true`` to record evaluation videos:

.. code-block:: bash

   lerobot-eval \
     --policy.path=../lerobot/outputs/train/act_summit_franka_open_microwave_7221_setstate/checkpoints/010000/pretrained_model \
     --env.type=isaaclab_arena \
     --env.hub_path=$(pwd)/isaaclab-arena-envs \
     --env.environment=summit_franka_open_door_eval \
     --env.headless=true \
     --env.enable_cameras=true \
     --env.video=true \
     --env.video_length=10 \
     --env.video_interval=5 \
     --env.state_keys=joint_pos \
     --env.camera_keys=ego_topdown_rgb,ego_wrist_rgb,fix_local_rgb \
     --env.state_dim=12 \
     --env.action_dim=12 \
     --env.camera_height=240 \
     --env.camera_width=320 \
     --env.kwargs='{"object_name": "microwave_7221", "scene_name": "scene_0_seed_0", "object_center": true, "disable_collision": true, "mobile_base_relative": true}' \
     --rename_map='{"observation.images.ego_topdown_rgb": "observation.images.ego_topdown", "observation.images.ego_wrist_rgb": "observation.images.ego_wrist", "observation.images.fix_local_rgb": "observation.images.fix_local"}' \
     --trust_remote_code=true \
     --eval.batch_size=1 \
     --eval.n_episodes=50 \
     --policy.device=cuda


Troubleshooting
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Symptom
     - Fix
   * - ``KeyError: 'joint_pos'``
     - Ensure ``--env.environment=summit_franka_open_door_eval`` (not the base
       ``summit_franka_open_door``). The eval variant overrides observations to
       use absolute ``joint_pos``.
   * - Observation shape mismatch
     - Check ``--env.state_dim`` matches the model's ``observation.state``
       shape (12 for Summit Franka).
   * - Camera image shape mismatch
     - Verify ``--env.camera_height=240 --env.camera_width=320`` matches the
       model config. Check ``--rename_map`` maps camera keys correctly.
   * - ``RuntimeError: trust_remote_code``
     - Add ``--trust_remote_code=true``.
   * - ``FileNotFoundError: env.py``
     - Check ``--env.hub_path`` points to the correct directory or repo.
   * - Robot explodes / objects fly away
     - Add ``"disable_collision": true`` to ``--env.kwargs``. See
       :doc:`physics_tuning_guide`.
   * - Base drifts during evaluation
     - Ensure ``"mobile_base_relative": true`` is set in kwargs if the policy
       was trained with ``--mobile_base_relative`` during recording.
   * - ``observation.img_state_delta`` missing
     - This dummy feature is automatically injected by
       ``IsaaclabArenaProcessorStep``. If you see this error, ensure the
       env processor is being loaded correctly (``--env.type=isaaclab_arena``).


Complete Pipeline Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For reference, the full pipeline from recording to evaluation:

.. code-block:: bash

   # 1. Record demonstrations
   python isaaclab_arena/scripts/record_automoma_demos.py \
     --enable_cameras --set_state --disable_collision \
     --interpolated 4 --mobile_base_relative \
     --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
     --dataset_file data/automoma/summit_franka_open_microwave_7221_setstate.hdf5 \
     --num_episodes 30 \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_0_seed_0 \
     --object_center

   # 2. Convert HDF5 → LeRobot format
   python isaaclab_arena_gr00t/data_utils/convert_hdf5_to_lerobot_v30.py \
     --yaml_file isaaclab_arena_gr00t/config/summit_franka_manip_config.yaml \
     --data_root data/automoma \
     --hdf5_name summit_franka_open_microwave_7221_setstate.hdf5 \
     --repo_id automoma/summit_franka_open_microwave_7221_setstate \
     --output_dir data/lerobot/automoma/summit_franka_open_microwave_7221_setstate

   # 3. Train ACT policy
   exp_name="summit_franka_open_microwave_7221_setstate"
   dataset_root=data/lerobot/automoma/$exp_name
   lerobot-train \
     --policy.type=act \
     --batch_size=128 \
     --steps=10000 \
     --log_freq=50 \
     --eval_freq=500 \
     --save_freq=1000 \
     --job_name=act_$exp_name \
     --dataset.repo_id=$exp_name \
     --dataset.root=$dataset_root \
     --policy.chunk_size=16 \
     --policy.n_action_steps=16 \
     --policy.optimizer_lr=1e-4 \
     --policy.push_to_hub=false \
     --policy.device=cuda \
     --wandb.enable=true \
     --output_dir=outputs/train/act_$exp_name \
     --dataset.preload=true \
     --dataset.preload_cache=true \
     --dataset.filter_features_by_policy=true

   # 4. Evaluate with lerobot-eval
   lerobot-eval \
     --policy.path=../lerobot/outputs/train/act_$exp_name/checkpoints/010000/pretrained_model \
     --policy.device=cuda \
     --env.type=isaaclab_arena \
     --env.hub_path=$(pwd)/isaaclab-arena-envs \
     --env.environment=summit_franka_open_door_eval \
     --env.headless=false \
     --env.enable_cameras=true \
     --env.state_keys=joint_pos \
     --env.camera_keys=ego_topdown_rgb,ego_wrist_rgb,fix_local_rgb \
     --env.state_dim=12 \
     --env.action_dim=12 \
     --env.camera_height=240 \
     --env.camera_width=320 \
     --env.episode_length=300 \
     --env.kwargs='{"object_name": "microwave_7221", "scene_name": "scene_0_seed_0", "object_center": true, "disable_collision": true, "mobile_base_relative": true}' \
     --rename_map='{"observation.images.ego_topdown_rgb": "observation.images.ego_topdown", "observation.images.ego_wrist_rgb": "observation.images.ego_wrist", "observation.images.fix_local_rgb": "observation.images.fix_local"}' \
     --trust_remote_code=true \
     --eval.batch_size=1 \
     --eval.n_episodes=10


.. note::

   - This pipeline runs entirely locally — no Docker container or Hub uploads
     needed.
   - The ``--env.hub_path`` accepts both local filesystem paths and HuggingFace
     Hub repo IDs (e.g. ``nvidia/isaaclab-arena-envs``).
   - For GR00T N1.5 evaluation via ``policy_runner.py``, see :doc:`step_5_evaluation`.
   - For physics parameter tuning, see :doc:`physics_tuning_guide`.
