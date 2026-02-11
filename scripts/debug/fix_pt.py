import torch

# --- Configuration ---
input_path = "/home/xinhai/projects/lerobot-arena/IsaacLab-Arena/res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data_11d.pt"
output_path = input_path.replace("traj_data_11d.pt", "traj_data.pt")
def process_data(tensor):
    """
    Transforms (..., 11) tensor into:
    - Robot (..., 12): [j0, j1-1.5, ...j9, 0.04, 0.04]
    - Object (..., 1): [-obj_state]
    """
    # 1. Separate Robot (0-9) and Object (10)
    # Clone is important to avoid modifying the original data in memory if used later
    robot_joints = tensor[..., :10].clone()
    obj_state = tensor[..., 10:].clone()

    # 2. Adjust Lift Joint (Index 1): Subtract 1.5
    robot_joints[..., 1] -= 1.5

    # 3. Reverse Object State: Negate the value
    obj_state = -obj_state

    # 4. Add Gripper Padding: [0.04, 0.04]
    # pad_shape logic handles both (Batch, 10) and (Batch, Time, 10)
    pad_shape = list(robot_joints.shape[:-1]) + [2]
    gripper_padding = torch.full(pad_shape, 0.04, device=tensor.device, dtype=tensor.dtype)
    
    robot_final = torch.cat([robot_joints, gripper_padding], dim=-1)

    return robot_final, obj_state

# --- Execution ---
print(f"Loading from: {input_path}")
data = torch.load(input_path)
clean_data = {}

# Process start, goal, and trajectory
clean_data['start_robot'], clean_data['start_obj'] = process_data(data['start_state'])
clean_data['goal_robot'],  clean_data['goal_obj']  = process_data(data['goal_state'])
clean_data['traj_robot'],  clean_data['traj_obj']  = process_data(data['traj'])

# Copy success metric
clean_data['traj_success'] = data['success']

# --- Verification ---
print("\nTransformation Checks (First Index):")
# 1. Check Lift Joint Adjustment
orig_lift = data['traj'][0, 0, 1].item()
new_lift  = clean_data['traj_robot'][0, 0, 1].item()
print(f"Lift Joint: {orig_lift:.4f} -> {new_lift:.4f} (Expected: {orig_lift - 1.5:.4f})")

# 2. Check Object Reversal
orig_obj = data['traj'][0, 0, 10].item()
new_obj  = clean_data['traj_obj'][0, 0].item()
print(f"Obj State:  {orig_obj:.4f} -> {new_obj:.4f} (Expected: {-orig_obj:.4f})")

# 3. Check Dimensions
print(f"\nFinal Shapes:")
for k, v in clean_data.items():
    print(f"{k:15}: {v.shape}")

# Save
torch.save(clean_data, output_path)
print(f"\nSaved to: {output_path}")