Summit Franka Open Door Task
============================

This example demonstrates the complete workflow for the **Summit Franka mobile manipulator
opening an articulated door** (microwave, dishwasher, oven, etc.) using custom automoma
assets in Isaac Lab - Arena. It covers environment setup and validation, teleoperation data
collection, data generation with Isaac Lab Mimic, policy post-training (GR00T N1.5), and
closed-loop policy inference and evaluation.

The walkthrough uses **microwave_7221** and **scene_0_seed_0** as the canonical example,
but the same pipeline applies to any ``<type>_<id>`` object and any ``<scene_N_seed_M>``
scene — just swap the ``--object_name`` and ``--scene_name`` CLI arguments.

Task Overview
-------------

**Task ID:** ``summit_franka_open_door``

(Legacy alias ``summit_franka_open_microwave`` also works.)

**Task Description:** The Summit Franka mobile manipulator reaches toward an openable
object (microwave, dishwasher, oven) and opens its door. The robot, object, and scene
are loaded from the ``automoma_assets`` collection and can be dynamically swapped by name.

**Key Specifications:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Mobile manipulation, Coordinated whole-body control
   * - **Skills**
     - Reach, Grasp handle, Open door
   * - **Embodiment**
     - Summit Franka (3 DOF base + 7 DOF arm + 2 DOF gripper = **12 DOF**)
   * - **Action space**
     - 12 DOF joint positions (all joints, whole-body)
   * - **State space**
     - 12 DOF joint positions (base + arm + gripper)
   * - **Cameras**
     - 3 views: ego_topdown (320×240), ego_wrist (320×240), fix_local (320×240)
   * - **Scene**
     - Automoma scene (e.g. ``scene_0_seed_0``)
   * - **Objects**
     - Automoma openable objects (e.g. ``microwave_7221``, ``dishwasher_11622``, ``oven_101773``)
   * - **Policy**
     - GR00T N1.5 (vision-language foundation model)
   * - **Post-training**
     - Imitation Learning
   * - **Physics**
     - PhysX (200Hz @ 4 decimation)
   * - **Closed-loop**
     - Yes (50Hz control)
   * - **Metrics**
     - Success rate, Door moved rate


Workflow
--------

This tutorial covers the pipeline between creating an environment, generating training data,
fine-tuning a policy (GR00T N1.5), and evaluating the policy in closed-loop.
A user can follow the whole pipeline, or start at any intermediate step.

Prerequisites
^^^^^^^^^^^^^

Start the isaaclab docker container:

:docker_run_default:

We store data on Hugging Face, so you'll need to log in:

.. code-block:: bash

    hf auth login

Create the folders for data and models:

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/summit_franka_tutorial
    mkdir -p $DATASET_DIR
    export MODELS_DIR=/models/isaaclab_arena/summit_franka_tutorial
    mkdir -p $MODELS_DIR

Workflow Steps
^^^^^^^^^^^^^^

Follow these steps to complete the workflow:

- :doc:`step_1_environment_setup`
- :doc:`step_2_teleoperation`
- :doc:`step_3_data_generation`
- :doc:`step_4_policy_training`
- :doc:`step_5_evaluation`
- :doc:`step_5_evaluation_with_lerobot`
- :doc:`physics_tuning_guide`
- :doc:`custom_assets_guide`


.. toctree::
   :maxdepth: 1
   :hidden:

   step_1_environment_setup
   step_2_teleoperation
   step_3_data_generation
   step_4_policy_training
   step_5_evaluation
   step_5_evaluation_with_lerobot
   physics_tuning_guide
   custom_assets_guide
