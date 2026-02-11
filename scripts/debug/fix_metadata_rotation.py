
import os
import json
import glob
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path

# Configuration
# DATASET_ROOT = os.path.expanduser("~/projects/automoma_cvpr26/assets/scene/infinigen/kitchen_1130")
DATASET_ROOT = os.path.expanduser("/home/xinhai/projects/lerobot-arena/IsaacLab-Arena/res_for_custom/automoma_assets/scene")

def single_axis_self_rotation(matrix: np.ndarray, axis: str, angle: float) -> np.ndarray:
    """Apply a single-axis rotation to the matrix (intrinsic/local rotation)."""
    rot = R.from_euler(axis, angle)
    R_local = rot.as_matrix()
    
    # Post-multiply to rotate around the object's local axis
    # M_new = M_old @ R_local_4x4
    # (Since R_local is 3x3, we apply it to the top-left block)
    result = matrix.copy()
    result[:3, :3] = matrix[:3, :3] @ R_local
    return result

def transform_corners_correctly(corners, old_matrix, new_matrix):
    """
    Transforms world-space corners from the old pose to the new pose.
    Uses full 4x4 matrix inversion to handle Scale, Rotation, and Translation correctly.
    """
    if not corners:
        return []

    # Convert corners to homogeneous coordinates (N x 4)
    corners_np = np.array(corners) # Shape (8, 3)
    ones = np.ones((corners_np.shape[0], 1))
    corners_homo = np.hstack([corners_np, ones]) # Shape (8, 4)

    # 1. Transform World -> Local (using Old Matrix Inverse)
    # P_local = Old_Matrix_Inv @ P_world
    try:
        old_inv = np.linalg.inv(old_matrix)
    except np.linalg.LinAlgError:
        print("  Warning: Matrix singular, cannot invert. Skipping BBox update.")
        return corners

    local_homo = (old_inv @ corners_homo.T).T # Shape (8, 4)

    # 2. Transform Local -> World New (using New Matrix)
    # P_new = New_Matrix @ P_local
    new_corners_homo = (new_matrix @ local_homo.T).T # Shape (8, 4)

    # Convert back to Cartesian (x, y, z)
    new_corners = new_corners_homo[:, :3].tolist()
    return new_corners

def process_metadata_file(file_path):
    print(f"Processing: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    if "static_objects" not in data:
        return

    objects_modified = 0

    for obj_key, obj_data in data["static_objects"].items():
        # Load current matrix
        old_matrix = np.array(obj_data["matrix"]) # 4x4
        
        # --- 1. Fix Matrix ---
        new_matrix = single_axis_self_rotation(old_matrix, axis='z', angle=np.pi)
        
        # --- 2. Fix Rotation Field (Euler XYZ) ---
        # Extract the pure rotation component (ignoring scale for Euler calculation)
        # Note: We assume uniform scale for extraction stability, but just grabbing the 3x3 
        # and passing to scipy often works if we normalize columns, 
        # BUT simplest is to take the new matrix and re-calculate.
        # To get pure rotation for Euler, we must remove scale.
        upper_3x3 = new_matrix[:3, :3]
        scale = np.linalg.norm(upper_3x3, axis=0)
        pure_rotation = upper_3x3 / scale
        
        r = R.from_matrix(pure_rotation)
        new_euler = r.as_euler('xyz').tolist()
        
        # --- 3. Fix Bounding Box Corners ---
        if "bbox_corners" in obj_data:
            new_corners = transform_corners_correctly(
                obj_data["bbox_corners"], 
                old_matrix, 
                new_matrix
            )
            obj_data["bbox_corners"] = new_corners

        # Apply updates
        obj_data["matrix"] = new_matrix.tolist()
        obj_data["rotation"] = new_euler
        
        objects_modified += 1

    # Save
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    root_path = Path(DATASET_ROOT)
    if not root_path.exists():
        print(f"Error: Path not found: {root_path}")
        return

    # Recursive search for metadata.json
    files = list(root_path.rglob("metadata.json"))
    
    print(f"Found {len(files)} metadata files.")
    
    for json_path in files:
        # Backup logic
        backup_path = json_path.parent / "metadata_backup.json"
        if not backup_path.exists():
            shutil.copy2(json_path, backup_path)
        
        try:
            process_metadata_file(json_path)
        except Exception as e:
            print(f"  ERROR in {json_path}: {e}")

if __name__ == "__main__":
    main()