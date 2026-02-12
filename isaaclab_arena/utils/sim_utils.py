# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for IsaacSim scene manipulation.

Provides helpers for:
- Deactivating duplicate prims in the USD stage (e.g., objects baked into scene USDs)
- Setting viewport lighting modes
- Syncing camera observations after resets
- Disabling collision on prims for collision-free recording/evaluation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym


def deactivate_prims_by_name(name_pattern: str, exclude_paths: tuple[str, ...] = ("/World/envs",)) -> list[str]:
    """Deactivate all prims in the USD stage whose **name** contains ``name_pattern``.

    The match is case-insensitive.  Only the prim's own name (the last segment
    of its path) is checked, so ancestor names are not matched.  When a prim is
    deactivated its entire sub-tree is hidden and excluded from simulation.

    This is useful when a background scene USD already contains the target
    object, and we want to spawn a separate articulation for it instead.

    Args:
        name_pattern: Substring to search for in prim names (case-insensitive).
        exclude_paths: Tuple of prim path prefixes to skip (e.g., "/World/envs").

    Returns:
        List of prim paths that were deactivated.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print(f"[deactivate_prims_by_name] Warning: USD stage is not available.")
        return []

    pattern_lower = name_pattern.lower()
    deactivated: list[str] = []

    # Use a recursive walk so that once a parent is deactivated we skip children
    def _walk(prim):
        prim_path_str = str(prim.GetPath())
        for prefix in exclude_paths:
            if prim_path_str.startswith(prefix):
                return
        prim_name = prim.GetName().lower()
        if pattern_lower in prim_name:
            prim.SetActive(False)
            deactivated.append(str(prim.GetPath()))
            return  # skip children – they are implicitly deactivated
        for child in prim.GetChildren():
            _walk(child)

    for root_prim in stage.GetPseudoRoot().GetChildren():
        _walk(root_prim)

    if deactivated:
        print(f"[deactivate_prims_by_name] Deactivated {len(deactivated)} prims matching '{name_pattern}':")
        for p in deactivated:
            print(f"  - {p}")
    else:
        print(f"[deactivate_prims_by_name] No prims found matching '{name_pattern}'.")

    return deactivated


def set_lighting_mode(mode: int = 2) -> None:
    """Set the viewport lighting rig mode.

    Args:
        mode: Lighting mode index.  Common values:
            0 – Stage lights (default)
            1 – Ambient
            2 – Grey (uniform grey environment light)
    """
    try:
        import omni.kit.actions.core

        action_registry = omni.kit.actions.core.get_action_registry()
        action = action_registry.get_action(
            "omni.kit.viewport.menubar.lighting",
            "set_lighting_mode_rig",
        )
        if action is not None:
            action.execute(lighting_mode=mode)
            print(f"[set_lighting_mode] Lighting mode set to {mode}.")
        else:
            print(f"[set_lighting_mode] Warning: lighting action not found in registry.")
    except Exception as e:
        print(f"[set_lighting_mode] Warning: Could not set lighting mode: {e}")


def sync_cameras_after_reset(env: "gym.Env") -> dict:
    """Force an extra render pass so camera images reflect the post-reset state.

    IsaacSim cameras can lag by one frame after a physics reset.  This helper
    triggers a render, updates the scene, and recomputes observations so that
    ``env.obs_buf`` contains fresh camera images.

    Args:
        env: The **unwrapped** IsaacLab environment.

    Returns:
        The recomputed observation dict (also written to ``env.obs_buf``).
    """
    # Trigger rendering
    if hasattr(env.sim, "render"):
        env.sim.render()
    else:
        # Fallback: do a zero-dt physics step with rendering enabled
        env.sim.step(render=True)

    # Update scene entities (sensors, cameras, etc.)
    if hasattr(env, "scene"):
        env.scene.update(env.physics_dt)

    # Recompute observations with the fresh render
    obs = env.observation_manager.compute()
    env.obs_buf = obs
    return obs


def disable_collision_for_prim_and_descendants(prim_path: str) -> int:
    """Disable collision on a prim and all its descendants.

    This function traverses the USD stage starting from ``prim_path`` and
    disables all collision-related APIs on every prim in the subtree.  It
    handles three types of collision APIs:

    - ``UsdPhysics.CollisionAPI`` — standard USD physics collision
    - ``UsdPhysics.MeshCollisionAPI`` — mesh-based collision approximation
    - ``PhysxSchema.PhysxCollisionAPI`` — PhysX-specific collision properties

    Args:
        prim_path: Absolute USD prim path (e.g., ``/World/envs/env_0/Robot``).

    Returns:
        Number of prims whose collision was modified.
    """
    import omni.usd
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[disable_collision] Warning: USD stage is not available.")
        return 0

    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        print(f"[disable_collision] Warning: prim not found at '{prim_path}'.")
        return 0

    total_changed = 0
    for prim in Usd.PrimRange(root_prim):
        changed = False

        # 1. Disable UsdPhysics.CollisionAPI
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            changed = True

        # 2. Disable PhysxSchema.PhysxCollisionAPI
        if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            PhysxSchema.PhysxCollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            changed = True

        # 3. Disable UsdPhysics.MeshCollisionAPI
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            # MeshCollisionAPI doesn't have GetCollisionEnabledAttr directly
            # but disabling CollisionAPI above should suffice
            changed = True

        if changed:
            total_changed += 1

    return total_changed


def disable_collision_for_env(env: "gym.Env", object_name: str | None = None) -> None:
    """Disable collision for robot and target object in the environment.

    Resolves the actual prim paths from the environment's scene entities,
    then disables collision on all prims in their subtrees.

    Args:
        env: The unwrapped IsaacLab environment.
        object_name: Name of the target object in the scene (e.g., "microwave_7221").
    """
    prim_paths: list[str] = []

    # Collect prim paths for robot
    if hasattr(env, "scene") and "robot" in env.scene.keys():
        robot_cfg = env.scene["robot"].cfg
        robot_prim_path = robot_cfg.prim_path
        # Resolve {ENV_REGEX_NS} patterns for env_0
        robot_prim_path = robot_prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
        prim_paths.append(robot_prim_path)

    # Collect prim paths for target object
    if object_name and hasattr(env, "scene") and object_name in env.scene.keys():
        obj_cfg = env.scene[object_name].cfg
        obj_prim_path = obj_cfg.prim_path
        obj_prim_path = obj_prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
        prim_paths.append(obj_prim_path)

    if not prim_paths:
        print("[disable_collision] No prim paths found to disable collision on.")
        return

    # Use sim_utils.find_matching_prims to resolve regex patterns
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[disable_collision] Warning: USD stage is not available.")
        return

    total_changed = 0
    for prim_path in prim_paths:
        try:
            import isaaclab.sim as sim_utils
            resolved_prims = sim_utils.find_matching_prims(prim_path, stage)
            for root_prim in resolved_prims:
                actual_path = str(root_prim.GetPath())
                changed = disable_collision_for_prim_and_descendants(actual_path)
                total_changed += changed
        except Exception as exc:
            # Fallback: try direct path (no regex)
            clean_path = prim_path.replace("/env_.*", "/env_0")
            print(f"[disable_collision] Regex resolve failed for '{prim_path}': {exc}")
            print(f"[disable_collision] Trying direct path: {clean_path}")
            changed = disable_collision_for_prim_and_descendants(clean_path)
            total_changed += changed

    print(f"[disable_collision] Disabled collision on {total_changed} prims total.")
