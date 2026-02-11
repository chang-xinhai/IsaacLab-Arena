import os
import sys
import json
import numpy as np
from pathlib import Path
from pxr import Usd, UsdGeom, Gf, Sdf

# --- Configuration ---
# DATASET_ROOT = "assets/scene/infinigen/kitchen_1130" 
DATASET_ROOT = os.path.expanduser("/home/xinhai/projects/lerobot-arena/IsaacLab-Arena/res_for_custom/automoma_assets/scene")
Z_OFFSET = -0.12  # Amount to shift Z (add this value)

class ScenePoseFixer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)

    def process_metadata(self, scene_dir):
        """
        Updates metadata.json: shifts position, matrix, and bbox_corners by Z_OFFSET.
        """
        metadata_path = scene_dir / "info" / "metadata.json"
        
        if not metadata_path.exists():
            print(f"    ⚠️ Metadata not found: {metadata_path}")
            return False

        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)

            if "static_objects" not in data:
                print("    ℹ️ No static_objects in metadata")
                return True

            modified_count = 0
            
            for obj_name, obj_data in data["static_objects"].items():
                # 1. Update Position [x, y, z]
                if "position" in obj_data and len(obj_data["position"]) == 3:
                    obj_data["position"][2] += Z_OFFSET
                
                # 2. Update Matrix (4x4)
                # The translation Z component is at index [2][3]
                if "matrix" in obj_data:
                    matrix = np.array(obj_data["matrix"])
                    if matrix.shape == (4, 4):
                        matrix[2, 3] += Z_OFFSET
                        obj_data["matrix"] = matrix.tolist()
                
                # 3. Update BBox Corners
                # List of lists [[x,y,z], ...]
                if "bbox_corners" in obj_data:
                    new_corners = []
                    for corner in obj_data["bbox_corners"]:
                        if len(corner) == 3:
                            new_c = [corner[0], corner[1], corner[2] + Z_OFFSET]
                            new_corners.append(new_c)
                        else:
                            new_corners.append(corner)
                    obj_data["bbox_corners"] = new_corners
                
                modified_count += 1

            # Save changes
            with open(metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"    ✅ Updated metadata for {modified_count} objects")
            return True

        except Exception as e:
            print(f"    ❌ Error processing metadata: {e}")
            return False

    def process_usd(self, scene_dir):
        """
        Updates export_scene.usdc: Applies translation to /World/scene prim.
        """
        usd_path = scene_dir / "export" / "export_scene.blend" / "export_scene.usdc"
        
        if not usd_path.exists():
            print(f"    ⚠️ USD file not found: {usd_path}")
            return False

        try:
            stage = Usd.Stage.Open(str(usd_path))
            if not stage:
                print(f"    ❌ Failed to open stage: {usd_path}")
                return False

            scene_prim_path = "/World/scene"
            scene_prim = stage.GetPrimAtPath(scene_prim_path)

            if not scene_prim:
                print(f"    ⚠️ Prim '{scene_prim_path}' not found. (Did you run batch_usd_preprocess.py?)")
                return False

            # Make the prim Xformable
            xform = UsdGeom.Xformable(scene_prim)
            
            # Check for existing translate op or add a new one
            # We specifically want to add to the existing transform if it exists, 
            # or define a new one. Here we set a specific translation.
            
            # Helper to find or create translateOp
            translate_op = None
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            
            if not translate_op:
                translate_op = xform.AddTranslateOp()

            # Get current value (default is 0,0,0 if not set)
            current_val = translate_op.Get()
            if not current_val:
                current_val = Gf.Vec3d(0, 0, 0)
            
            # Calculate new value
            new_val = Gf.Vec3d(
                current_val[0],
                current_val[1],
                current_val[2] + Z_OFFSET
            )
            
            translate_op.Set(new_val)
            
            stage.Save()
            print(f"    ✅ Applied transform to /World/scene: {current_val} -> {new_val}")
            return True

        except Exception as e:
            print(f"    ❌ Error processing USD: {e}")
            return False

    def process_all(self):
        # Validate base directory
        if not self.base_dir.exists():
            print(f"❌ Base directory does not exist: {self.base_dir}")
            return

        # Find scene directories
        scene_dirs = [
            d for d in self.base_dir.iterdir()
            if d.is_dir() and d.name.startswith("scene_")
        ]
        
        # Sort naturally
        scene_dirs.sort(key=lambda x: [
            int(c) if c.isdigit() else c
            for c in __import__('re').split(r'(\d+)', x.name)
        ])

        print(f"🚀 Starting Pose Fix (Z_OFFSET = {Z_OFFSET})")
        print(f"📂 Found {len(scene_dirs)} scenes in {self.base_dir}")
        print("="*60)

        success_count = 0

        for scene_dir in scene_dirs:
            print(f"Processing: {scene_dir.name}")
            
            meta_ok = self.process_metadata(scene_dir)
            usd_ok = self.process_usd(scene_dir)

            if meta_ok and usd_ok:
                success_count += 1
            
            print("-" * 40)

        print(f"\n🎉 Finished. Successfully processed {success_count}/{len(scene_dirs)} scenes.")

def main():
    fixer = ScenePoseFixer(DATASET_ROOT)
    fixer.process_all()

if __name__ == "__main__":
    main()