Data Generation
---------------

This workflow covers two methods for generating training datasets:

1. **Isaac Lab Mimic** — augments teleoperated demonstrations using rigid body
   transformations (requires prior teleoperation step).
2. **Automoma Planner Replay** — replays pre-computed joint trajectories from a
   ``.pt`` file, recording observations, actions, and camera images to HDF5.


Method A: Isaac Lab Mimic (from Teleop)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method assumes you've completed the preceding teleoperation step, or you have a
pre-recorded dataset available.

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:


Step 1: Annotate Demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Annotate the demonstrations by segmenting into two subtasks (reach, open door):

.. code-block:: bash

   python isaaclab_arena/scripts/annotate_demos.py \
     --device cpu \
     --input_file $DATASET_DIR/summit_franka_open_microwave_7221_recorded.hdf5 \
     --output_file $DATASET_DIR/summit_franka_open_microwave_7221_annotated.hdf5 \
     --enable_pinocchio \
     --mimic \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_0_seed_0

Follow the CLI instructions to mark subtask boundaries:

1. **Reach:** Robot reaches toward the door handle
2. **Open door:** Robot grasps and opens the door


Step 2: Generate Augmented Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isaac Lab Mimic generates additional demonstrations from annotated demonstrations
using rigid body transformations:

.. code-block:: bash

   python isaaclab_arena/scripts/generate_dataset.py \
     --device cpu \
     --generation_num_trials 50 \
     --num_envs 10 \
     --input_file $DATASET_DIR/summit_franka_open_microwave_7221_annotated.hdf5 \
     --output_file $DATASET_DIR/summit_franka_open_microwave_7221_generated.hdf5 \
     --enable_pinocchio \
     --enable_cameras \
     --headless \
     --mimic \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_0_seed_0

Data generation takes 30-60 minutes depending on hardware. Remove ``--headless``
to visualize the data generation process.

The generated dataset includes all 3 camera views (ego_topdown, ego_wrist, fix_local).


Step 3: Validate Generated Data (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the generated dataset to verify:

.. code-block:: bash

   python isaaclab_arena/scripts/replay_demos.py \
     --device cpu \
     --enable_cameras \
     --dataset_file $DATASET_DIR/summit_franka_open_microwave_7221_generated.hdf5 \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_0_seed_0


Method B: Automoma Planner Replay
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have pre-computed trajectories from the automoma planner (a ``.pt`` file),
you can replay them in the simulator and record to HDF5 directly — no teleoperation
required.

**Trajectory file format**

The ``.pt`` file contains a dict with these tensors:

.. code-block:: text

   start_robot:   [N_episodes, n_robot_joints]    # initial robot joint positions
   start_obj:     [N_episodes, n_obj_joints]       # initial object joint positions
   goal_robot:    [N_episodes, n_robot_joints]     # goal robot joint positions
   goal_obj:      [N_episodes, n_obj_joints]       # goal object joint positions
   traj_robot:    [N_episodes, T, n_robot_joints]  # per-step robot targets
   traj_obj:      [N_episodes, T, n_obj_joints]    # per-step object targets
   traj_success:  [N_episodes]                     # success flag per episode

For the Summit Franka + microwave example: ``n_robot_joints = 12``,
``n_obj_joints = 1``, ``T = 32``.

Trajectory files are stored at::

   res_for_custom/automoma_trajs/<robot>/<object>/<scene>/traj_data.pt


**Recording options**

The recording script supports several options that affect the output HDF5:

- ``--set_state``: Teleport robot/object joints directly (no physics). Guarantees
  the recorded demo exactly follows the planner's trajectory.
- ``--interpolated X``: Linearly interpolate between trajectory keyframes by
  factor X (default 1 = none). E.g. ``--interpolated 4`` produces 4x more frames,
  yielding smoother motion in set-state mode.
- ``--mobile_base_relative``: Store base actions as relative deltas (Δx, Δy, Δθ)
  instead of absolute positions. Arm / gripper remain absolute. This is the
  recommended format for policy training — the policy learns "move forward a bit"
  which generalises better than "go to world coordinate (x, y)".

**Action space convention**

When ``--mobile_base_relative`` is set, the recorded action space is:

- **Base** (dims 0–2): Relative joint positions (Δx, Δy, Δθ)
- **Arm** (dims 3–9): Absolute joint positions (next frame's panda_joint1–7)
- **Gripper** (dims 10–11): Absolute joint positions (finger joints)

During evaluation, relative base actions are integrated with the current state::

    new_base = current_base + delta_base

**Environment loading fixes**

The recording script automatically handles two common issues:

1. **Duplicate object prims**: Background scene USDs often contain the target object.
   The script deactivates these duplicate prims by matching the object name
   (case-insensitive) in the USD stage, so only the separately-spawned articulation
   is active.
2. **Lighting**: The script sets the viewport lighting to grey mode (mode 2) for
   consistent rendering across different scenes.


**Step 1: Record with set-state mode (recommended for initial data)**

.. code-block:: bash

   python isaaclab_arena/scripts/record_automoma_demos.py \
     --enable_cameras --set_state \
     --interpolated 4 --mobile_base_relative \
     --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
     --dataset_file $DATASET_DIR/summit_franka_open_microwave_7221_setstate.hdf5 \
     --num_episodes 10 \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_0_seed_0 \
     --object_center

With ``--set_state``, robot joints and object articulation are teleported each step.
Adding ``--interpolated 4`` smooths the original 32-step trajectory to ~125 steps.

**Step 2: Record with drive mode (physics-based, optional)**

.. code-block:: bash

   python isaaclab_arena/scripts/record_automoma_demos.py \
     --enable_cameras \
     --interpolated 4 --mobile_base_relative \
     --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
     --dataset_file $DATASET_DIR/summit_franka_open_microwave_7221_drive.hdf5 \
     --num_episodes 50 \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_0_seed_0 \
     --object_center

In drive mode, the robot is first teleported to the trajectory's starting pose,
then joint targets are sent to the actuators with physics simulation. This is more
realistic but may result in some episodes deviating from the planned trajectory.

**Step 3: Preview recorded data**

You can use the ``replay`` policy type to replay any recorded HDF5:

.. code-block:: bash

   python isaaclab_arena/examples/policy_runner.py \
     --enable_cameras \
     --policy_type replay \
     --replay_file_path $DATASET_DIR/summit_franka_open_microwave_7221_setstate.hdf5 \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_0_seed_0 \
     --object_center

You can also replay automoma trajectories directly (without recording to HDF5):

.. code-block:: bash

   python isaaclab_arena/examples/policy_runner.py \
     --device cpu --enable_cameras \
     --policy_type replay_automoma \
     --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
     --episode_index 0 \
     --interpolated 4 \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_0_seed_0

You should see the robot successfully performing the task.

.. note::

   The dataset was generated using CPU device physics, therefore the replay uses
   ``--device cpu`` to ensure reproducibility.
