# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Automoma Object Library

Provides dynamically-registered openable objects (microwave, dishwasher, oven, etc.)
from the automoma_assets collection. Objects are looked up by name in the form
``<type>_<id>`` (e.g. ``microwave_7221``, ``dishwasher_11622``, ``oven_101773``).
"""

import json
import os
from pathlib import Path

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

# Root directories for automoma assets.
# Override via environment variables to point to an external assets directory.
_DEFAULT_ASSETS_ROOT = Path(__file__).resolve().parents[2] / "res_for_custom" / "automoma_assets"
_AUTOMOMA_OBJECT_ROOT = Path(os.environ.get("AUTOMOMA_OBJECT_ROOT", str(_DEFAULT_ASSETS_ROOT / "object")))
_AUTOMOMA_SCENE_ROOT = Path(os.environ.get("AUTOMOMA_SCENE_ROOT", str(_DEFAULT_ASSETS_ROOT / "scene")))


class AutomomaOpenableObject(Object, Openable):
    """
    An openable articulated object loaded from the automoma_assets collection.

    The USD is expected at:
        res_for_custom/automoma_assets/object/<type>_<id>/<id>_0_scaling/<id>_0_scaling.usd

    The openable joint is ``joint_0`` (the hinge door), consistent with the
    PartNet-Mobility convention used by all automoma openable assets.
    """

    # Openable affordance parameters
    openable_joint_name: str = "joint_0"
    openable_open_threshold: float = 0.5

    def __init__(
        self,
        name: str,
        asset_type: str,
        asset_id: str,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        openable_joint_name: str | None = None,
        openable_open_threshold: float | None = None,
        **kwargs,
    ):
        # Try new layout first: {root}/{Type}/{id}/mobility/mobility.usd
        # Fallback to legacy layout: {root}/{type_lower}_{id}/mobility/mobility.usd
        new_usd_path = _AUTOMOMA_OBJECT_ROOT / asset_type / asset_id / "mobility" / "mobility.usd"
        legacy_usd_path = _AUTOMOMA_OBJECT_ROOT / f"{asset_type.lower()}_{asset_id}" / "mobility" / "mobility.usd"
        if new_usd_path.exists():
            usd_path = str(new_usd_path)
        elif legacy_usd_path.exists():
            usd_path = str(legacy_usd_path)
        else:
            raise FileNotFoundError(
                f"USD file not found for automoma object {name}. "
                f"Searched:\n  {new_usd_path}\n  {legacy_usd_path}\n"
                "Set AUTOMOMA_OBJECT_ROOT env var or ensure the asset exists."
            )
        if openable_joint_name is None:
            openable_joint_name = self.openable_joint_name
        if openable_open_threshold is None:
            openable_open_threshold = self.openable_open_threshold
        super().__init__(
            name=name,
            prim_path=prim_path,
            tags=["object", "openable", "automoma"],
            usd_path=usd_path,
            object_type=ObjectType.ARTICULATION,
            scale=scale,
            initial_pose=initial_pose,
            openable_joint_name=openable_joint_name,
            openable_open_threshold=openable_open_threshold,
            **kwargs,
        )


def get_automoma_object(
    asset_type: str,
    asset_id: str,
    prim_path: str | None = None,
    initial_pose: Pose | None = None,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> AutomomaOpenableObject:
    """
    Factory function to create an automoma openable object by type and id.

    Args:
        asset_type: The object category, e.g. "Microwave", "Dishwasher", "Oven".
        asset_id: The PartNet asset id, e.g. "7221", "11622", "101773".
        prim_path: Optional USD prim path override.
        initial_pose: Optional initial pose.
        scale: Scale tuple.

    Returns:
        An AutomomaOpenableObject instance.
    """
    name = f"{asset_type.lower()}_{asset_id}"
    return AutomomaOpenableObject(
        name=name,
        asset_type=asset_type,
        asset_id=asset_id,
        prim_path=prim_path,
        initial_pose=initial_pose,
        scale=scale,
    )


# ---------------------------------------------------------------------------
# Pre-registered automoma objects
# These are registered so they can be retrieved via:
#   self.asset_registry.get_asset_by_name("microwave_7221")()
# ---------------------------------------------------------------------------

@register_asset
class Microwave7221(AutomomaOpenableObject):
    """Automoma Microwave 7221."""
    name = "microwave_7221"

    def __init__(self, prim_path=None, initial_pose=None, **kwargs):
        super().__init__(
            name=self.name,
            asset_type="Microwave",
            asset_id="7221",
            prim_path=prim_path,
            initial_pose=initial_pose,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Scene metadata utilities
# ---------------------------------------------------------------------------

def load_scene_metadata(scene_name: str) -> dict:
    """
    Load the metadata.json for a given automoma scene.

    Args:
        scene_name: e.g. "scene_0_seed_0"

    Returns:
        The parsed metadata dict.
    """
    metadata_path = _AUTOMOMA_SCENE_ROOT / scene_name / "info" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Scene metadata not found: {metadata_path}. "
            "Set AUTOMOMA_SCENE_ROOT env var or ensure the scene exists."
        )
    with open(metadata_path, "r") as f:
        return json.load(f)


def get_object_pose_from_metadata(
    metadata: dict,
    asset_type: str,
    asset_id: str,
) -> tuple[Pose, tuple[float, float, float]]:
    """
    Extract the initial pose of an object from scene metadata.

    Searches through ``metadata["static_objects"]`` for an entry matching the
    given asset_type and asset_id.

    Args:
        metadata: Parsed metadata.json dict.
        asset_type: e.g. "Microwave"
        asset_id:   e.g. "7221"

    Returns:
        A Pose object with position and rotation (euler-to-quat converted).
    """
    import math

    for key, obj_info in metadata.get("static_objects", {}).items():
        if obj_info.get("asset_type") == asset_type and str(obj_info.get("asset_id")) == str(asset_id):
            pos = obj_info["position"]
            rot_euler = obj_info["rotation"]  # [roll, pitch, yaw] in radians
            scale = obj_info["scale"]

            # Convert euler (XYZ convention) to quaternion (w, x, y, z)
            roll, pitch, yaw = rot_euler
            cr, sr = math.cos(roll / 2), math.sin(roll / 2)
            cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy

            return Pose(
                position_xyz=tuple(pos),
                rotation_wxyz=(w, x, y, z),
            ), tuple(scale)

    raise ValueError(
        f"Object {asset_type} with id {asset_id} not found in scene metadata. "
        f"Available objects: {list(metadata.get('static_objects', {}).keys())}"
    )
