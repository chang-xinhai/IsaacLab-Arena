# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Automoma Scene Background Library

Provides dynamically-registered scene backgrounds from the automoma_assets
collection. Scenes are looked up by name (e.g. ``scene_0_seed_0``).
"""

import os
from pathlib import Path

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

# Root of the automoma assets relative to the project root
_AUTOMOMA_ASSETS_ROOT = Path(__file__).resolve().parents[2] / "res_for_custom" / "automoma_assets"


class AutomomaSceneBackground(Background):
    """
    A scene background loaded from the automoma_assets collection.

    The USDC is expected at:
        res_for_custom/automoma_assets/scene/<scene_name>/export/export_scene.blend/export_scene.usdc
    """

    def __init__(
        self,
        scene_name: str,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        object_min_z: float = -0.2,
        **kwargs,
    ):
        usdc_path = str(
            _AUTOMOMA_ASSETS_ROOT / "scene" / scene_name / "export" / "export_scene.blend" / "export_scene.usdc"
        )
        if not os.path.exists(usdc_path):
            raise FileNotFoundError(
                f"USDC file not found for automoma scene {scene_name}: {usdc_path}. "
                "Make sure the scene exists under res_for_custom/automoma_assets/scene/."
            )
        if initial_pose is None:
            initial_pose = Pose.identity()
        super().__init__(
            name=scene_name,
            usd_path=usdc_path,
            initial_pose=initial_pose,
            object_min_z=object_min_z,
            prim_path=prim_path,
            tags=["background", "automoma"],
            **kwargs,
        )


def get_automoma_scene(
    scene_name: str,
    initial_pose: Pose | None = None,
    object_min_z: float = -0.2,
) -> AutomomaSceneBackground:
    """
    Factory function to create an automoma scene background by name.

    Args:
        scene_name: e.g. "scene_0_seed_0", "scene_1_seed_1"
        initial_pose: Optional initial pose for the scene. Defaults to identity.
        object_min_z: Z height below which objects are considered "dropped".

    Returns:
        An AutomomaSceneBackground instance.
    """
    return AutomomaSceneBackground(
        scene_name=scene_name,
        initial_pose=initial_pose,
        object_min_z=object_min_z,
    )


# ---------------------------------------------------------------------------
# Pre-registered automoma scenes
# ---------------------------------------------------------------------------

@register_asset
class Scene0Seed0Background(AutomomaSceneBackground):
    """Automoma scene_0_seed_0 background."""
    name = "scene_0_seed_0"

    def __init__(self, **kwargs):
        super().__init__(scene_name=self.name, **kwargs)
