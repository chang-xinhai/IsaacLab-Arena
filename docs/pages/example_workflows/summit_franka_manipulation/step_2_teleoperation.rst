Teleoperation Data Collection
-----------------------------

This workflow covers collecting demonstrations for the Summit Franka using
Isaac Lab Teleop.

**Docker Container**: Base (see :doc:`../../quickstart/docker_containers` for more details)

:docker_run_default:


.. note::

    For teleop with Apple Vision Pro, follow the same CloudXR setup as described in
    the GR1 tutorial. The key difference is the environment and embodiment names.
    You can also use other supported teleop devices.

Step 1: Start Recording
^^^^^^^^^^^^^^^^^^^^^^^

To start the recording session:

.. code-block:: bash

   python isaaclab_arena/scripts/record_demos.py \
     --device cpu \
     --dataset_file $DATASET_DIR/summit_franka_open_microwave_7221_recorded.hdf5 \
     --num_demos 10 \
     --num_success_steps 2 \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_0_seed_0 \
     --teleop_device avp_handtracking

.. note::

   Replace ``--teleop_device avp_handtracking`` with your available teleop device.
   Use ``--teleop_device keyboard`` if no VR device is available (limited capability).

   You can also add ``--object_center`` to shift the world origin to the object center,
   which may improve consistency across different scenes.


Step 2: Record Demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow the same procedure as the GR1 tutorial:

1. Connect your teleoperation device
2. Complete the task by reaching toward and opening the door
3. The environment will automatically reset on task completion
4. Repeat for ``num_demos`` demonstrations (set to 10 above)

The script saves successful demonstrations to the HDF5 file at
``$DATASET_DIR/summit_franka_open_microwave_7221_recorded.hdf5``.


.. hint::

   For best results:

   - Move slowly and smoothly — coordinated base + arm motion is critical
   - Keep within the robot's workspace
   - Ensure the gripper approaches the door handle before grasping
   - Complete at least 10 successful demonstrations

Using Different Objects
^^^^^^^^^^^^^^^^^^^^^^^^

You can collect demonstrations for any openable object:

.. code-block:: bash

   # Dishwasher
   python isaaclab_arena/scripts/record_demos.py \
     --device cpu \
     --dataset_file $DATASET_DIR/summit_franka_open_dishwasher_11622_recorded.hdf5 \
     --num_demos 10 --num_success_steps 2 \
     summit_franka_open_door \
     --object_name dishwasher_11622 \
     --scene_name scene_0_seed_0 \
     --teleop_device avp_handtracking

   # Oven
   python isaaclab_arena/scripts/record_demos.py \
     --device cpu \
     --dataset_file $DATASET_DIR/summit_franka_open_oven_101773_recorded.hdf5 \
     --num_demos 10 --num_success_steps 2 \
     summit_franka_open_door \
     --object_name oven_101773 \
     --scene_name scene_0_seed_0 \
     --teleop_device avp_handtracking
