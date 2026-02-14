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


Automoma Evaluation Notes
^^^^^^^^^^^^^^^^^^^^^^^^^

For automoma evaluation (when ``traj_file`` is provided in ``--env.kwargs``),
the env bridge applies the same scene post-fixes used in recording:

1. Deactivate duplicate object prims by ``object_name``.
2. Set lighting to grey mode by default (mode ``2``), unless overridden by
  ``lighting_mode``.
3. If ``disable_collision=true``, disable collisions globally.

These hooks are intentionally gated by ``traj_file`` so default IsaacLab-Arena
examples are not affected.


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
     --env.kwargs='{"object_name": "microwave_7221", "scene_name": "scene_0_seed_0", "object_center": true, "disable_collision": true, "mobile_base_relative": true, "traj_file": "res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data_test.pt", "traj_seed": 42}' \
     --rename_map='{"observation.images.ego_topdown_rgb": "observation.images.ego_topdown", "observation.images.ego_wrist_rgb": "observation.images.ego_wrist", "observation.images.fix_local_rgb": "observation.images.fix_local"}' \
     --trust_remote_code=true \
     --eval.batch_size=1 \
     --eval.n_episodes=10

.. important::

   **Initial state from trajectory file**: Unlike standard benchmarks where the
   robot always starts at the same pose, automoma tasks require different initial
   states per episode (the planner places the robot at different positions relative
   to the object). The ``traj_file`` kwarg loads start states from a ``.pt`` file
   and teleports the robot/object at the beginning of each episode. The ``traj_seed``
   kwarg controls the random sampling order for reproducibility.


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
     - Disables **all** collisions in the entire simulation stage.
       Every prim with ``CollisionAPI`` or ``PhysxCollisionAPI`` is disabled.
       Useful for set-state recordings or when policy actions cause
       interpenetration.
   * - ``mobile_base_relative``
     - bool
     - If ``true``, the first ``base_dof`` (default 3) action dimensions are
       treated as relative deltas (Δx, Δy, Δθ). The wrapper integrates them
       with the current base state before sending absolute positions to sim.
       Must match the recording mode (``--mobile_base_relative`` flag during
       ``record_automoma_demos.py``).
   * - ``traj_file``
     - str
     - Path to a ``.pt`` trajectory file for setting initial robot/object
       states at the beginning of each evaluation episode. On each ``reset()``,
       a random episode is sampled from this file and the robot+object are
       teleported to the corresponding start positions. This is essential for
       automoma tasks where the robot's starting pose varies per episode
       (unlike standard benchmarks where the initial state is always the same).
   * - ``traj_seed``
     - int
     - Random seed for sampling episodes from ``traj_file`` (default: 42).
       Ensures reproducible evaluation across runs.


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

   # 4. Evaluate with lerobot-eval (with traj-based initial state)
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
     --env.kwargs='{"object_name": "microwave_7221", "scene_name": "scene_0_seed_0", "object_center": true, "disable_collision": true, "mobile_base_relative": true, "traj_file": "res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data_test.pt", "traj_seed": 42}' \
     --rename_map='{"observation.images.ego_topdown_rgb": "observation.images.ego_topdown", "observation.images.ego_wrist_rgb": "observation.images.ego_wrist", "observation.images.fix_local_rgb": "observation.images.fix_local"}' \
     --trust_remote_code=true \
     --eval.batch_size=1 \
     --eval.n_episodes=10


Understanding Success Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluation Call Flow (End-to-End)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The runtime path for ``lerobot-eval`` with ``--env.type=isaaclab_arena`` is:

1. ``lerobot_eval.py::eval_main`` loads env and policy.
2. ``lerobot.envs.factory.make_env`` detects ``hub_path`` and calls
  ``isaaclab-arena-envs/env.py::make_env``.
3. ``env.py::_create_isaaclab_env`` builds IsaacLab env, applies automoma
  scene hooks (if ``traj_file`` is set), and wraps with ``IsaacLabEnvWrapper``.
4. ``rollout()`` runs the loop:

  a. ``observation = env.reset(...)``

  b. ``observation -> IsaaclabArenaProcessorStep -> policy input``

  c. ``action = policy.select_action(observation)``

  d. ``env.step(action)``

  e. read ``info["final_info"]["is_success"]`` for metrics

5. ``eval_policy`` aggregates rewards/success over episodes and reports
  ``pc_success``.


What ``traj_file`` does (and does NOT do)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the current evaluation implementation:

- ``traj_file`` is read only inside ``IsaacLabEnvWrapper`` during init/reset.
- On each ``reset()``, one trajectory episode index is sampled (seeded by
  ``traj_seed``), and only ``start_robot`` / ``start_obj`` are used to set the
  initial pose.
- During stepping, actions come from ``policy.select_action(...)`` in
  LeRobot's rollout loop, then passed to ``env.step(action)``.
- No evaluation code path reads ``traj_robot`` / ``traj_obj`` to drive actions.

Therefore, for eval, ``traj_file`` currently provides only initial state
sampling; control execution is policy-driven, not replay-driven.

During evaluation, **lerobot-eval** reports ``is_success`` for each episode. The
success signal flows through several layers:

.. code-block:: text

   IsaacLabEnvWrapper._get_success()
        │
        ▼
   termination_manager.get_term("success")
        │
        ▼
   OpenDoorTask → openable_object.is_open(env, threshold=...)
        │
        ▼
   Openable.get_openness(env) > threshold

**How it works step by step:**

1. ``OpenDoorTask`` (in ``isaaclab_arena/tasks/open_door_task.py``) creates a
   ``success`` termination term using ``openable_object.is_open()``.

2. ``Openable.is_open()`` (in ``isaaclab_arena/affordances/openable.py``) reads
   the normalized joint position of the door hinge and checks if
   ``openness > threshold``.

3. The wrapper's ``_get_success()`` method queries ``termination_manager.get_term("success")``
   each step. When the door openness exceeds the threshold, the termination fires and
   the episode is marked as successful.

4. **Default threshold**: The ``OpenDoorTask`` is constructed with
   ``openness_threshold=0.8`` (80% open) in the environment class. This means the
   door must open to at least 80% of its full range for the episode to count as
   successful.

**How to modify the success criteria:**

To change when an episode is considered successful, follow these steps:

1. **Change the openness threshold** (simplest):

   Edit ``isaaclab_arena/examples/example_environments/summit_franka_open_door_environment.py``.
   In the ``get_env()`` method, find the ``OpenDoorTask`` constructor:

   .. code-block:: python

      task = OpenDoorTask(
          target_object,
          openness_threshold=0.8,   # ← Change this value
          reset_openness=0.3,
          episode_length_s=2.0,
      )

   For example, set ``openness_threshold=0.5`` to succeed when the door is 50% open.

   You can also pass the threshold at runtime via ``--env.kwargs``:

   .. code-block:: python

      # In summit_franka_open_door_eval_environment.py, read from args:
      openness_threshold = getattr(args_cli, "openness_threshold", 0.8)
      # Then pass to OpenDoorTask(... openness_threshold=openness_threshold ...)

2. **Use a custom angle-based criterion** (e.g., "door opened > N degrees"):

   a. Read the raw (un-normalized) joint position in radians:

      .. code-block:: python

         # In a new termination function:
         def door_angle_exceeds(env, asset_cfg, min_angle_rad: float = 0.5):
             asset = env.scene[asset_cfg.name]
             joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
             return joint_pos.squeeze(-1).abs() > min_angle_rad

   b. Register it as the ``success`` termination term in ``OpenDoorTask.make_termination_cfg()``.

   c. Adjust the ``TerminationTermCfg`` to use your function:

      .. code-block:: python

         success = TerminationTermCfg(
             func=door_angle_exceeds,
             params={"min_angle_rad": 0.785, "asset_cfg": SceneEntityCfg(self.openable_object.name)},
         )

3. **Completely replace the success logic**:

   Create a new task class inheriting from ``OpenDoorTask`` and override
   ``make_termination_cfg()`` with any custom logic (e.g., combine door angle
   with gripper contact, robot proximity, etc.).

.. tip::

   The ``is_open`` function uses **normalized** joint positions (0.0 = fully closed,
   1.0 = fully open). If you want to use raw joint angles in radians or degrees,
   read directly from ``asset.data.joint_pos`` instead.


.. note::

   - This pipeline runs entirely locally — no Docker container or Hub uploads
     needed.
   - The ``--env.hub_path`` accepts both local filesystem paths and HuggingFace
     Hub repo IDs (e.g. ``nvidia/isaaclab-arena-envs``).
   - For GR00T N1.5 evaluation via ``policy_runner.py``, see :doc:`step_5_evaluation`.
   - For physics parameter tuning, see :doc:`physics_tuning_guide`.
