# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for IsaacSim scene manipulation.

Provides helpers for:
- Deactivating duplicate prims in the USD stage (e.g., objects baked into scene USDs)
- Setting a repo-managed lighting rig that works in both GUI and headless runs
- Syncing camera observations after resets
- Disabling collision on prims for collision-free recording/evaluation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym


def deactivate_prims_by_name(
    name_pattern: str,
    exclude_paths: tuple[str, ...] = (),
    required_path_substrings: tuple[str, ...] = (),
) -> list[str]:
    """Deactivate all prims in the USD stage whose **name** contains ``name_pattern``.

    The match is case-insensitive.  Only the prim's own name (the last segment
    of its path) is checked, so ancestor names are not matched.  When a prim is
    deactivated its entire sub-tree is hidden and excluded from simulation.

    This is useful when a background scene USD already contains the target
    object, and we want to spawn a separate articulation for it instead.

    Args:
        name_pattern: Substring to search for in prim names (case-insensitive).
        exclude_paths: Tuple of prim path prefixes to skip.
        required_path_substrings: Optional tuple of substrings.  If provided,
            only prims whose full path contains at least one of these
            substrings are considered for deactivation.

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
        if required_path_substrings and not any(s in prim_path_str for s in required_path_substrings):
            for child in prim.GetChildren():
                _walk(child)
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


def _set_prim_active(stage, prim_path: str, active: bool) -> bool:
    """Set a prim active/inactive if it exists."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    prim.SetActive(active)
    return True


def set_lighting_mode(mode: int = 2) -> None:
    """Set a repo-managed lighting rig in a headless-safe way.

    Unlike the old viewport menubar action, this implementation authors stage
    lights directly so camera output is consistent between GUI and headless
    runs under Isaac Sim 5.1.

    Args:
        mode: Lighting mode index. Common values:
            0 – Stage lights (scene-authored/default stage lights only)
            1 – Ambient dome light
            2 – Grey neutral dome light
    """
    try:
        from pathlib import Path
        import sys

        import isaacsim
        import omni.usd
        from pxr import Sdf, Usd, UsdGeom, UsdLux

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[set_lighting_mode] Warning: USD stage is not available.")
            return

        rig_prim_path = "/OmniKit_Viewport_LightRig"
        extscache_root = Path(isaacsim.__file__).resolve().parent / "extscache"
        lighting_ext_root = extscache_root / "omni.kit.viewport.menubar.lighting-107.3.1+107.3.0"
        if str(lighting_ext_root) not in sys.path:
            sys.path.append(str(lighting_ext_root))

        from omni.kit.viewport.menubar.lighting.actions import _add_rig_reference
        from omni.kit.viewport.menubar.lighting.utility import VisibilityEdit

        def is_a_light(prim: Usd.Prim, prim_path):
            return prim.HasAPI(UsdLux.LightAPI)

        def _clear_usd_references(prim_path: str):
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsDefined():
                prim.GetReferences().SetReferences([])
            return prim

        def _rig_asset_path(rig_name: str) -> str:
            return str(lighting_ext_root / "data" / "usd" / f"{rig_name}.usda")

        if mode == 0:
            _clear_usd_references(rig_prim_path)
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                VisibilityEdit(stage, is_a_light, None, api_types_for_prune=["LightAPI"]).run()
            print("[set_lighting_mode] Lighting mode set to 0 (stage lights restored).")
            return

        rig_profiles = {
            1: "Default",
            2: "Grey_Studio",
        }
        rig_name = rig_profiles.get(mode)
        if rig_name is None:
            print(f"[set_lighting_mode] Warning: unsupported lighting mode {mode}, leaving stage unchanged.")
            return

        asset_path = _rig_asset_path(rig_name)
        if not Path(asset_path).exists():
            print(f"[set_lighting_mode] Warning: rig asset not found: {asset_path}")
            return

        _clear_usd_references(rig_prim_path)
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            VisibilityEdit(stage, None, is_a_light, api_types_for_prune=["LightAPI"]).run()
            rig_prim = stage.OverridePrim(rig_prim_path)
            xformable, adjustment = _add_rig_reference(rig_prim, asset_path)
            omni.usd.editor.set_hide_in_stage_window(rig_prim, True)
            omni.usd.editor.set_no_delete(rig_prim, True)
            omni.usd.get_context().set_pickable(rig_prim_path, False)

        if xformable and adjustment:
            xformable.AddXformOp(UsdGeom.XformOp.TypeTransform).Set(adjustment)

        print(f"[set_lighting_mode] Lighting mode set to {mode} using rig {rig_name} ({asset_path}).")
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


def _format_material_value(value: float | None) -> str:
    return "unchanged" if value is None else f"{value:g}"


def set_asset_material_friction(
    asset,
    static_friction: float | None = None,
    dynamic_friction: float | None = None,
    label: str | None = None,
) -> bool:
    """Set contact material friction on every shape in an IsaacLab asset.

    This updates the PhysX material property buffer directly, matching the
    approach used by IsaacLab's material randomization events.  It applies to
    all rigid shapes in the asset.

    Args:
        asset: IsaacLab ``RigidObject`` or ``Articulation`` scene entity.
        static_friction: Static friction coefficient. ``None`` leaves it unchanged.
        dynamic_friction: Dynamic friction coefficient. ``None`` leaves it unchanged.
        label: Optional label used in logs.

    Returns:
        ``True`` if material properties were updated, otherwise ``False``.
    """
    if static_friction is None and dynamic_friction is None:
        return False
    root_physx_view = getattr(asset, "root_physx_view", None)
    if root_physx_view is None:
        print(f"[set_material_friction] Warning: asset {label or asset!r} has no root_physx_view.")
        return False

    materials = root_physx_view.get_material_properties()
    if static_friction is not None:
        materials[..., 0] = float(static_friction)
    if dynamic_friction is not None:
        materials[..., 1] = float(dynamic_friction)

    # IsaacLab expects env ids for material updates on the CPU.
    import torch

    num_envs = int(materials.shape[0])
    env_ids = torch.arange(num_envs, device="cpu")
    root_physx_view.set_material_properties(materials, env_ids)

    name = label or getattr(asset, "name", asset.__class__.__name__)
    print(
        "[set_material_friction] "
        f"{name}: static={_format_material_value(static_friction)} "
        f"dynamic={_format_material_value(dynamic_friction)} "
        f"envs={num_envs} shapes={int(materials.shape[1]) if materials.ndim >= 2 else 'unknown'}"
    )
    return True


def set_robot_object_material_friction(
    env: "gym.Env",
    object_name: str | None,
    static_friction: float | None = None,
    dynamic_friction: float | None = None,
) -> int:
    """Set material friction on the whole robot and target object.

    This is intentionally broad: every rigid shape on ``scene["robot"]`` and
    ``scene[object_name]`` receives the same friction values.

    Args:
        env: The unwrapped IsaacLab environment.
        object_name: Target object scene key, e.g. ``"microwave_7221"``.
        static_friction: Static friction coefficient. ``None`` leaves it unchanged.
        dynamic_friction: Dynamic friction coefficient. ``None`` leaves it unchanged.

    Returns:
        Number of scene assets updated.
    """
    if static_friction is None and dynamic_friction is None:
        return 0
    if static_friction is None:
        static_friction = dynamic_friction
    if dynamic_friction is None:
        dynamic_friction = static_friction
    if not hasattr(env, "scene"):
        print("[set_robot_object_material_friction] Warning: env has no scene.")
        return 0

    updated = 0
    scene_keys = set(env.scene.keys())
    if "robot" in scene_keys:
        updated += int(
            set_asset_material_friction(
                env.scene["robot"],
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                label="robot",
            )
        )
    else:
        print("[set_robot_object_material_friction] Warning: scene has no 'robot' asset.")

    if object_name and object_name in scene_keys:
        updated += int(
            set_asset_material_friction(
                env.scene[object_name],
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                label=object_name,
            )
        )
    elif object_name:
        print(f"[set_robot_object_material_friction] Warning: scene has no target object '{object_name}'.")
    else:
        print("[set_robot_object_material_friction] Warning: object_name was not provided.")

    print(f"[set_robot_object_material_friction] Updated {updated} assets.")
    return updated


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
        else:
            # Fallback: some assets expose the attribute without applied schema
            attr = prim.GetAttribute("physics:collisionEnabled")
            if attr.IsValid():
                attr.Set(False)
                changed = True

        # 2. Disable PhysxSchema.PhysxCollisionAPI
        if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            PhysxSchema.PhysxCollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            changed = True
        else:
            # Fallback: some assets expose the attribute without applied schema
            attr = prim.GetAttribute("physxCollision:collisionEnabled")
            if attr.IsValid():
                attr.Set(False)
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


def disable_all_collisions() -> int:
    """Disable collision on **every** prim in the USD stage.

    This is a brute-force approach that guarantees no collision responses
    anywhere in the simulation.  Useful for set-state recording or evaluation
    where physics-based contacts are unwanted.

    The function traverses the entire USD stage and disables:

    - ``UsdPhysics.CollisionAPI``
    - ``PhysxSchema.PhysxCollisionAPI``

    on every prim that has them.

    Returns:
        Number of prims whose collision was disabled.
    """
    import omni.usd
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[disable_all_collisions] Warning: USD stage is not available.")
        return 0

    total_changed = 0
    api_hits = 0
    attr_hits = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        changed = False
        
        print(f"Checking prim: {prim.GetPath()}")  # Debug log to trace which prims are checked

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            api_hits += 1
            changed = True
        else:
            attr = prim.GetAttribute("physics:collisionEnabled")
            if attr.IsValid():
                attr.Set(False)
                attr_hits += 1
                changed = True
                print("attr", attr.GetPath())

        if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            PhysxSchema.PhysxCollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            api_hits += 1
            changed = True
        else:
            attr = prim.GetAttribute("physxCollision:collisionEnabled")
            if attr.IsValid():
                attr.Set(False)
                attr_hits += 1
                changed = True
                print("Phy_attr", attr.GetPath())

        if changed:
            total_changed += 1

    print(
        "[disable_all_collisions] Disabled collision on "
        f"{total_changed} prims in entire stage (api_hits={api_hits}, attr_hits={attr_hits})."
    )
    if total_changed == 0:
        print(
            "[disable_all_collisions] Warning: no collision-enabled APIs/attrs found. "
            "If this is unexpected, call this after env.reset() when runtime prims are fully initialized."
        )
    return total_changed
