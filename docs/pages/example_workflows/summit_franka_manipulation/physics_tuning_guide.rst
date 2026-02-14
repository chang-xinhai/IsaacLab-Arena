Physics Simulation Parameter Tuning Guide
==========================================

This document provides a comprehensive guide for tuning physics simulation
parameters in Isaac Lab Arena, with a focus on enabling reliable friction-based
grasping and articulated object manipulation (e.g., opening microwave doors with
a gripper).

.. contents:: Table of Contents
   :depth: 3
   :local:


Overview
--------

Achieving realistic manipulation in simulation — especially grasping a door
handle via friction and opening an articulated joint — requires careful tuning
across **three layers**:

1. **Robot actuator parameters** (stiffness, damping, effort/velocity limits)
2. **Object physics material** (friction, restitution, density) and joint parameters
3. **Contact and solver settings** (PhysX solver iterations, time step, decimation)

Small changes in any of these can cause:

- The gripper to slip off the handle
- The robot arm to oscillate or explode
- Joints to behave unrealistically (too stiff or too loose)
- Contact forces to be unstable or not generated at all

Below we go through each layer in detail.


1. Robot Actuator Parameters
----------------------------

Located in the ``ArticulationCfg.actuators`` section of the embodiment
(e.g., ``SummitFrankaSceneCfg.robot.actuators``).

.. code-block:: python

   actuators={
       "base": ImplicitActuatorCfg(
           joint_names_expr=["base_x", "base_y", "base_z"],
           effort_limit=87.0,
           velocity_limit=1.5,
           stiffness=800.0,
           damping=40.0,
       ),
       "arm": ImplicitActuatorCfg(
           joint_names_expr=["panda_joint.*"],
           effort_limit=87.0,
           velocity_limit=2.175,
           stiffness=800.0,
           damping=40.0,
       ),
       "gripper": ImplicitActuatorCfg(
           joint_names_expr=["panda_finger_joint.*"],
           effort_limit=20.0,
           velocity_limit=0.2,
           stiffness=2e3,
           damping=1e2,
       ),
   }


1.1 Stiffness & Damping
^^^^^^^^^^^^^^^^^^^^^^^^

These define PD controller gains for implicit actuators:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Parameter
     - Effect when **too low**
     - Effect when **too high**
   * - ``stiffness``
     - Joints are sluggish, cannot hold position, gripper cannot maintain grip
     - Joints overshoot aggressively, oscillate, may become unstable
   * - ``damping``
     - No velocity damping → oscillation and jitter
     - Over-damped → sluggish response, slow movements

**Tuning strategy:**

- **Arm joints**: Start with ``stiffness=800``, ``damping=40``. Increase stiffness
  if the arm cannot hold position under load. Increase damping if oscillation occurs.
- **Gripper joints**: Gripper needs **high stiffness** (``2000+``) to maintain
  friction-based grasps. The ``effort_limit`` determines max grip force.
  A ``damping=100`` prevents finger oscillation during grasps.
- **Base joints**: Similar to arm. For a mobile base that should not drift,
  use high stiffness (``800+``).


1.2 Effort & Velocity Limits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``effort_limit``
     - Maximum torque/force the joint can apply (Nm or N). If the robot cannot
       open a heavy door or maintain a grasp, increase this.
   * - ``velocity_limit``
     - Maximum joint speed (rad/s or m/s). If the robot moves too fast and
       destabilizes contacts, reduce this. If it is too slow to complete
       the task in time, increase it.

**For grasping handles:**

- Gripper ``effort_limit`` should be ≥ 20 N. Values below 10 N typically
  result in the gripper slipping.
- Gripper ``velocity_limit`` should be low (0.1–0.3) for stable closing.


2. Object Physics Parameters
-----------------------------

These are set via the USD file or programmatically through
``ArticulationCfg`` / ``RigidObjectCfg``.

# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.schemas.html#isaaclab.sim.schemas.RigidBodyPropertiesCfg
# https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationCfg

2.1 Friction
^^^^^^^^^^^^

Critical for friction-based grasping. Set on the **collision geometry**
of both the gripper fingers AND the object handle.

.. code-block:: python

   # In the USD or via PhysicsMaterialCfg:
   static_friction = 1.0   # Minimum 0.8 for reliable grasps
   dynamic_friction = 1.0  # Usually set equal to static

   # Or via rigid body properties:
   sim_utils.RigidBodyPropertiesCfg(
       disable_gravity=False,
   )

**Key guidelines:**

- If the gripper slips, increase ``static_friction`` on **both** the finger
  colliders and the handle colliders. Values of 1.0–2.0 are common in sim.
- ``dynamic_friction`` controls sliding resistance. Usually set equal to
  ``static_friction`` or slightly lower.
- You can modify friction directly in the USD via the ``PhysicsMaterial``
  prim, or apply it programmatically via ``sim_utils.RigidBodyMaterialCfg``.

.. important::

   Friction is a **per-contact-pair** property. PhysX computes the effective
   friction as ``combine(material_A, material_B)``. The default combine mode
   is **average**, so if one side has friction 0, the effective friction is
   halved. Ensure both sides have high friction.


2.2 Object Joint Parameters (Articulated Objects)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For openable objects (microwave, dishwasher, oven), the door hinge is
typically ``joint_0``. Key parameters:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``joint damping``
     - Resistance to door motion. Higher → harder to open. A microwave door
       might use 0.1–1.0. A heavy oven door might need 2.0–5.0.
   * - ``joint friction``
     - Static friction in the hinge. Prevents the door from swinging freely.
       Values of 0.0–0.5 are typical.
   * - ``joint stiffness``
     - Spring stiffness pulling the door back to its rest position. Usually
       0 for doors (no spring return), but can be set > 0 for spring-loaded
       mechanisms.
   * - ``drive max force``
     - Maximum force the joint drive can apply. If using a position/velocity
       drive on the hinge (e.g., for auto-close), this limits the force.
   * - ``joint limits``
     - ``lower`` and ``upper`` bounds in radians. For a microwave door,
       typically [0, 1.57] (0 to 90°). Ensure these match the real geometry.

**Tuning strategy:**

- If the door opens **too easily** (no resistance), increase ``joint damping``.
- If the door is **impossible** to open, decrease ``joint damping`` and ensure
  the robot's ``effort_limit`` is sufficient.
- If the door **swings** open when barely touched, add ``joint friction``.


2.3 Collision Geometry
^^^^^^^^^^^^^^^^^^^^^^^

- Ensure the door handle has **convex decomposition** collision geometry,
  not just a bounding box. A simple box collider on a handle will make
  grasping unreliable.
- Use ``mesh_collision_resolution`` in the USD to control collision mesh quality.
- For very thin handles, PhysX may generate poor contact normals.
  Consider thickening the collision geometry slightly.

.. code-block:: python

   # Example: setting collision approximation to convex decomposition
   # in the USD or via spawn config:
   spawn=UsdFileCfg(
       usd_path=usd_path,
       activate_contact_sensors=True,
       # If needed, override rigid body properties:
       rigid_props=sim_utils.RigidBodyPropertiesCfg(
           disable_gravity=False,
           max_depenetration_velocity=5.0,
       ),
       # Override collision properties
       collision_props=sim_utils.CollisionPropertiesCfg(
           contact_offset=0.005,
           rest_offset=0.0,
       ),
   )


2.4 Mass and Density
^^^^^^^^^^^^^^^^^^^^^

- Object mass affects how easily the gripper can hold and move it.
- If the object is too heavy for the gripper, reduce ``density`` in the USD.
- If the object flies away on contact, it may be too light.
- Typical microwave door: 0.5–2.0 kg.
- Gripper mass is set in the robot URDF/USD. Franka gripper is ~0.73 kg.


3. PhysX Solver and Simulation Settings
-----------------------------------------

These are set in the environment configuration or ``sim`` settings.


3.1 Time Step and Decimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # simulation_dt = physics_dt × decimation
   # For 50 Hz control: physics_dt=0.005, decimation=4 → sim_dt=0.02
   sim: SimulationCfg = SimulationCfg(
       dt=1.0 / 200.0,          # 200 Hz physics
       render_interval=4,       # 50 Hz rendering / control
   )

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Guideline
   * - ``dt`` (physics)
     - 1/120 to 1/240 is typical. Smaller → more stable but slower.
       For manipulation with contact, use 1/200 or smaller.
   * - ``decimation``
     - Number of physics steps per control step. 4 is standard.
       Increase to 8 if stability issues persist.


3.2 Solver Iterations
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   physx: PhysxCfg = PhysxCfg(
       solver_type=1,              # 0=PGS, 1=TGS (recommended)
       num_position_iterations=8,  # Default 4; increase for stable contacts
       num_velocity_iterations=1,
       max_stabilization_iterations=1,
       bounce_threshold_velocity=0.5,
       enable_gyroscopic_forces=True,
   )

- **TGS solver** (type 1) is more stable for manipulation than PGS.
- **Position iterations**: 4 is default, 8–16 for complex grasps.
  More iterations = more stable contacts but slower simulation.
- If the gripper "phases through" objects, increase position iterations.


3.3 Contact Settings
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   collision_props=sim_utils.CollisionPropertiesCfg(
       contact_offset=0.005,  # Distance at which contact is detected
       rest_offset=0.0,       # Resting penetration depth
   )

- ``contact_offset``: Increase if contacts are not detected reliably.
  Default 0.02 is often too large for small objects; use 0.005.
- ``rest_offset``: Usually 0. A small positive value can prevent jitter
  at the cost of visible gaps.

.. code-block:: python

   rigid_props=sim_utils.RigidBodyPropertiesCfg(
       max_depenetration_velocity=5.0,
       disable_gravity=False,
   )

- ``max_depenetration_velocity``: Limits how fast objects are pushed apart
  when overlapping. Default (often 10+) can cause objects to fly apart.
  Use 1.0–5.0 for manipulation.


4. Tuning Workflow
-------------------

Follow this systematic approach when setting up a new robot + object combination:


Step 1: Validate Robot in Isolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Load only the robot, no scene or objects.
2. Command the gripper to open/close. Verify smooth motion.
3. Command the arm to move to various poses. Check for oscillation.
4. If issues:

   - Oscillation → increase ``damping``, decrease ``stiffness``
   - Cannot hold position → increase ``stiffness``
   - Too slow → increase ``velocity_limit``


Step 2: Validate Object in Isolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Load only the object (e.g., microwave).
2. Manually (via USD editor or script) set the door joint to various angles.
3. Verify the joint limits are correct.
4. Apply a force to the door. Check damping and friction feel reasonable.
5. If issues:

   - Door won't move → reduce ``joint damping`` / ``joint friction``
   - Door swings too freely → increase ``joint damping``


Step 3: Robot + Object Interaction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Load both. Command the robot to approach the handle.
2. Close the gripper on the handle. Check if it grips.
3. If the gripper slips:

   a. Increase ``static_friction`` on gripper fingers AND handle colliders
   b. Increase gripper ``stiffness`` and ``effort_limit``
   c. Ensure the handle collider is not a simple box
   d. Check ``contact_offset`` (may be too small to detect contact)

4. With a successful grip, command the arm to pull the door open.
5. If the door doesn't open:

   a. Check that gripper ``effort_limit`` > door ``joint_damping``
   b. Increase arm ``effort_limit`` if needed
   c. Reduce door ``joint_damping``


Step 4: Full Environment Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Load the complete scene (background + object + robot).
2. Run zero-action test to verify scene loads.
3. Run a teleop session. Confirm:

   - Robot can reach the handle
   - Gripper can grasp the handle
   - Pulling motion opens the door
   - Task success criterion triggers

4. If the simulation is slow:

   - Reduce ``num_position_iterations``
   - Increase ``dt`` (less frequent physics)
   - Reduce collision mesh complexity


5. Object-Specific Tuning Cheat Sheet
---------------------------------------

.. list-table::
   :widths: 15 20 20 20 25
   :header-rows: 1

   * - Object
     - Door Damping
     - Hinge Friction
     - Handle Friction
     - Notes
   * - Microwave
     - 0.5–1.0
     - 0.1–0.3
     - 1.0–2.0
     - Lightweight door, small handle
   * - Dishwasher
     - 1.0–3.0
     - 0.2–0.5
     - 1.0–2.0
     - Heavier door, larger handle, pulls down
   * - Oven
     - 2.0–5.0
     - 0.3–0.8
     - 1.0–2.0
     - Heavy door, may need higher arm effort

.. note::

   These values are starting points. Every object model has different
   geometry and mass distribution, so fine-tuning is always required.


6. Common Failure Modes and Fixes
-----------------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Symptom
     - Fix
   * - Gripper slips off handle
     - Increase friction on both gripper and handle; increase gripper stiffness;
       ensure handle has convex decomposition collider
   * - Robot arm oscillates
     - Increase arm damping; reduce stiffness; increase physics dt
   * - Object flies away on contact
     - Reduce ``max_depenetration_velocity``; increase solver iterations;
       check object mass is reasonable
   * - Door won't open
     - Reduce door joint damping; increase arm effort_limit; check joint limits
   * - Simulation explodes
     - Reduce dt (increase physics frequency); increase solver iterations;
       check for interpenetrating collision geometry at initialization
   * - Gripper phases through handle
     - Increase ``num_position_iterations``; reduce collision ``contact_offset``;
       ensure contact sensors are activated
   * - Task never succeeds
     - Check ``openness_threshold`` vs actual joint range; verify joint_0
       is the correct hinge joint; check that the metric is being computed


7. Modifying Parameters Programmatically
------------------------------------------

Besides editing USD files directly, many parameters can be overridden
in the ``ArticulationCfg``:

.. code-block:: python

   # Override articulation root properties
   spawn=UsdFileCfg(
       usd_path=usd_path,
       activate_contact_sensors=True,
       articulation_props=sim_utils.ArticulationRootPropertiesCfg(
           enabled_self_collisions=False,
           solver_position_iteration_count=8,
           solver_velocity_iteration_count=1,
       ),
   )

   # Per-joint overrides can be done via init_state and actuators
   init_state=ArticulationCfg.InitialStateCfg(
       joint_pos={"joint_0": 0.0},
   )

For rigid objects, use ``RigidObjectCfg``:

.. code-block:: python

   spawn=UsdFileCfg(
       usd_path=usd_path,
       rigid_props=sim_utils.RigidBodyPropertiesCfg(
           disable_gravity=False,
           max_depenetration_velocity=5.0,
       ),
       mass_props=sim_utils.MassPropertiesCfg(
           density=500.0,
       ),
       collision_props=sim_utils.CollisionPropertiesCfg(
           contact_offset=0.005,
           rest_offset=0.0,
       ),
   )


8. Disabling Collision for Set-State Recording / Evaluation
-------------------------------------------------------------

When using ``--set_state`` mode (teleporting the robot to pre-planned poses)
or during evaluation where the policy may command large position jumps,
collision between the robot and object can cause PhysX to apply enormous
depenetration forces, sending objects flying.

The solution is to **disable collision** between the robot and the object
while keeping self-collision and ground-plane collision intact.

8.1 How It Works
^^^^^^^^^^^^^^^^^

The utility ``disable_collision_for_prim_and_descendants`` in
``isaaclab_arena.utils.sim_utils`` traverses the entire USD subtree of a
given prim and disables all collision APIs:

- ``UsdPhysics.CollisionAPI`` — the core collision flag
- ``PhysxSchema.PhysxCollisionAPI`` — PhysX-specific collision extensions
- ``UsdPhysics.MeshCollisionAPI`` — mesh-level collision approximation

For mesh/Xform prims that do **not** already have a ``CollisionAPI`` applied,
the function applies the API first, then disables it. This is necessary
because some USD assets from PartNet-Mobility only have collision defined
on a subset of descendant meshes, and simply calling ``RemoveAPI`` is not
sufficient to prevent PhysX from generating contacts.

.. code-block:: python

   from isaaclab_arena.utils.sim_utils import (
       disable_collision_for_prim_and_descendants,
       disable_collision_for_env,
   )

   # Low-level: disable collision on a specific prim subtree
   count = disable_collision_for_prim_and_descendants("/World/envs/env_0/Object")

   # High-level: disable collision for both robot and object in an env
   disable_collision_for_env(env, object_name="microwave_7221")


8.2 When to Use
^^^^^^^^^^^^^^^^

- **Recording with ``--set_state``**: Robot is teleported to grasp poses,
  which may interpenetrate the object. Pass ``--disable_collision``.
- **Evaluation with ``lerobot-eval``**: Policy actions may overshoot and
  cause interpenetration. Add ``disable_collision: true`` to the env config
  kwargs.
- **NOT recommended** for force-closure tasks where contact forces
  between the gripper and handle are essential for the task.

8.3 Common Pitfall: Silent Failure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A previous implementation using ``try/except`` around ``RemoveAPI`` calls
silently failed because:

1. ``RemoveAPI`` does not raise an exception on failure — it returns ``False``.
2. Some prims lack a ``CollisionAPI`` entirely, so removal has no effect.
3. PhysX caches collision shapes at scene initialization; removing APIs
   after that point may not take effect without a scene rebuild.

The current implementation avoids these issues by **applying then
disabling** the API (setting the ``collision:enabled`` attribute to
``False``), which PhysX respects even after initialization.


9. Debugging Tools
-------------------

- **Contact visualization**: Enable ``activate_contact_sensors=True`` on the
  robot USD and use Isaac Sim's contact visualization to see force magnitudes.
- **Joint state logging**: Print ``env.scene["robot"].data.joint_pos`` every step
  to check for NaN or exploding values.
- **PhysX debug visualization**: In Isaac Sim UI, enable
  *Physics* → *Debug Visualization* to see collision shapes, contacts, and forces.
- **Single-step debugging**: Set ``decimation=1`` and step the simulation manually
  to observe contact behavior frame by frame.

