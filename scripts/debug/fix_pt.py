import torch

# --- Configuration ---
# 1. Load the original data
path = "/home/xinhai/projects/lerobot-arena/IsaacLab-Arena/res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data_11d.pt"
data = torch.load(path)

# Set this to True to subtract 1.5 from the lift joint (index 1)
ADJUST_LIFT = True 

def split_and_expand(tensor, adjust_lift=False):
    """
    Input:  Tensor of shape (..., 11) [j0...j9, obj_state]
    Output: Tuple (joints, obj_state)
            joints:    (..., 12) -> [j0, j1_adj, ...j9, 0.04, 0.04]
            obj_state: (..., 1)  -> [obj_state]
    """
    # 1. Separate the components
    # We clone the joints part so we can modify it safely without affecting the original data or the object part
    joints_original = tensor[..., :10].clone() 
    obj_state = tensor[..., 10:]  # Last 1 dim (keep dim for shape consistency)

    # 2. Optional: Adjust the lift joint (Index 1)
    if adjust_lift:
        # Subtract 1.5 from index 1. 
        # The '...' ensures it works for both (N, 11) and (N, T, 11) shapes
        joints_original[..., 1] -= 1.5

    # 3. Create the gripper padding (0.04)
    # We use shape[:-1] + [2] to automatically handle both (N, 10) and (N, T, 10) inputs
    pad_shape = list(joints_original.shape[:-1]) + [2]
    gripper_padding = torch.full(pad_shape, 0.04, device=tensor.device, dtype=tensor.dtype)

    # 4. Concatenate to make the 12D joint vector
    joints_12d = torch.cat([joints_original, gripper_padding], dim=-1)

    return joints_12d, obj_state

# Initialize new dictionary
clean_data = {}

# Process start_state
s_joints, s_state = split_and_expand(data['start_state'], adjust_lift=ADJUST_LIFT)
clean_data['start_robot'] = s_joints       # Shape: (N, 12)
clean_data['start_obj'] = s_state          # Shape: (N, 1)

# Process goal_state
g_joints, g_state = split_and_expand(data['goal_state'], adjust_lift=ADJUST_LIFT)
clean_data['goal_robot'] = g_joints        # Shape: (N, 12)
clean_data['goal_obj'] = g_state           # Shape: (N, 1)

# Process trajectory
t_joints, t_state = split_and_expand(data['traj'], adjust_lift=ADJUST_LIFT)
clean_data['traj_robot'] = t_joints        # Shape: (N, T, 12)
clean_data['traj_obj'] = t_state           # Shape: (N, T, 1)

# Keep success as is (renamed to traj_success per your request)
clean_data['traj_success'] = data['success']

# --- Verification & Advice ---
print(f"Transformation Complete (Lift Adjustment: {ADJUST_LIFT}). New Structure:")
for k, v in clean_data.items():
    print(f"{k:20} : {v.shape}")

print("\nExample check (First step of first trajectory):")
# Check original vs new to confirm the -1.5 shift
orig_lift = data['traj'][0, 0, 1]
new_lift = clean_data['traj_robot'][0, 0, 1]
print(f"Original Lift Joint: {orig_lift:.4f}")
print(f"New Lift Joint:      {new_lift:.4f} (Should be {orig_lift - 1.5:.4f})")
print(f"Full New Joints:     {clean_data['traj_robot'][0, 0]}")

# Save the new clean file
save_path = path.replace("traj_data_11d.pt", "traj_data.pt")
torch.save(clean_data, save_path)
print(f"\nSaved to: {save_path}")