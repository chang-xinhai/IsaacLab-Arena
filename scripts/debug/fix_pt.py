import torch
import copy

# --- Configuration ---
INPUT_PATH = "/home/xinhai/projects/lerobot-arena/IsaacLab-Arena/res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/filtered_traj_data.pt"
OUTPUT_PATH = INPUT_PATH.replace("filtered_traj_data.pt", "traj_data.pt")

# Physics / Gripper Constants
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0
PREPEND_STEPS = 4  # Number of steps to close the gripper before pulling

def transform_arm_object(tensor):
    """
    Applies the base transformation to raw data:
    - Splits Arm (0-9) and Object (10)
    - Adjusts Arm Lift Joint (Index 1 -= 1.5)
    - Negates Object State
    """
    # Clone to avoid reference issues
    robot_joints = tensor[..., :10].clone()
    obj_state = tensor[..., 10:].clone()

    # 1. Adjust Lift Joint
    robot_joints[..., 1] -= 1.5

    # 2. Reverse Object State
    obj_state = -obj_state
    
    return robot_joints, obj_state

def process_single_state(tensor, gripper_val):
    """
    Processes a single state (Start/Goal).
    Args:
        tensor: Shape (Batch, 11)
        gripper_val: Float, the value to fill the gripper joints with (e.g. 0.04 or 0.0)
    """
    robot_arm, obj_state = transform_arm_object(tensor)
    
    # Create gripper tensor: Shape (Batch, 2)
    pad_shape = list(robot_arm.shape[:-1]) + [2]
    gripper_state = torch.full(pad_shape, gripper_val, device=tensor.device, dtype=tensor.dtype)
    
    # Concat: Arm (10) + Gripper (2) -> (12)
    robot_final = torch.cat([robot_arm, gripper_state], dim=-1)
    
    return robot_final, obj_state

def process_trajectory(tensor):
    """
    Processes the time-series trajectory.
    - Prepends 'PREPEND_STEPS' where arm is static but gripper closes.
    - Appends original trajectory where gripper stays closed.
    
    Args:
        tensor: Shape (Batch, Time, 11)
    """
    batch_size, orig_time, _ = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    # --- 1. Transform Base Data ---
    # raw_arm: (B, T, 10), raw_obj: (B, T, 1)
    raw_arm, raw_obj = transform_arm_object(tensor)

    # --- 2. Create Prepend Phase (Grasp) ---
    # Concept: Robot stays at Frame 0, Gripper closes from Open to Closed
    
    # Take Frame 0 of arm/obj and repeat it for PREPEND_STEPS
    # Shape: (B, 1, 10) -> (B, Prepend, 10)
    grasp_arm = raw_arm[:, 0:1, :].repeat(1, PREPEND_STEPS, 1)
    grasp_obj = raw_obj[:, 0:1, :].repeat(1, PREPEND_STEPS, 1)

    # Generate Gripper closing trajectory
    # Linear interpolation from 0.04 to 0.0
    # Shape: (Prepend,)
    gripper_closing_1d = torch.linspace(GRIPPER_OPEN, GRIPPER_CLOSED, steps=PREPEND_STEPS, device=device, dtype=dtype)
    # Expand to (B, Prepend, 2)
    # We use None to add dimensions: (1, Prepend, 1) -> repeat -> (B, Prepend, 2)
    grasp_gripper = gripper_closing_1d.view(1, -1, 1).repeat(batch_size, 1, 2)

    # Combine Grasp Phase
    grasp_robot_full = torch.cat([grasp_arm, grasp_gripper], dim=-1) # (B, P, 12)

    # --- 3. Create Main Phase (Pull) ---
    # Concept: Robot moves as recorded, Gripper stays Closed (0.0)
    
    # Generate Gripper holding trajectory
    # Shape: (B, T, 2) filled with GRIPPER_CLOSED
    pull_gripper = torch.full((batch_size, orig_time, 2), GRIPPER_CLOSED, device=device, dtype=dtype)
    
    # Combine Pull Phase
    pull_robot_full = torch.cat([raw_arm, pull_gripper], dim=-1) # (B, T, 12)

    # --- 4. Concatenate Phases ---
    final_robot_traj = torch.cat([grasp_robot_full, pull_robot_full], dim=1) # Time dim is 1
    final_obj_traj = torch.cat([grasp_obj, raw_obj], dim=1)

    return final_robot_traj, final_obj_traj

# --- Execution ---
if __name__ == "__main__":
    print(f"Loading from: {INPUT_PATH}")
    data = torch.load(INPUT_PATH)
    clean_data = {}

    print("Processing data...")
    
    # 1. Start State: Gripper should be OPEN before starting
    clean_data['start_robot'], clean_data['start_obj'] = process_single_state(
        data['start_state'], gripper_val=GRIPPER_OPEN
    )

    # 2. Goal State: Gripper should be CLOSED (holding the door)
    clean_data['goal_robot'], clean_data['goal_obj'] = process_single_state(
        data['goal_state'], gripper_val=GRIPPER_CLOSED
    )

    # 3. Trajectory: Prepend closing + Original Pulling
    clean_data['traj_robot'], clean_data['traj_obj'] = process_trajectory(
        data['traj']
    )

    # 4. Copy Metadata
    clean_data['traj_success'] = data['success']

    # --- Verification ---
    print("\n--- Verification ---")
    
    orig_steps = data['traj'].shape[1]
    new_steps = clean_data['traj_robot'].shape[1]
    print(f"Time Steps: {orig_steps} -> {new_steps} (Expected: {orig_steps} + {PREPEND_STEPS} = {orig_steps + PREPEND_STEPS})")
    
    # Check Gripper Interpolation
    print("\nGripper Values Check (First batch, Left Finger):")
    g_traj = clean_data['traj_robot'][0, :, 10] # Index 10 is first gripper joint
    
    print(f"Step 0 (Start): {g_traj[0]:.4f} (Expected {GRIPPER_OPEN})")
    print(f"Step {PREPEND_STEPS-1} (End of Grasp): {g_traj[PREPEND_STEPS-1]:.4f} (Expected near {GRIPPER_CLOSED})")
    print(f"Step {PREPEND_STEPS} (Start of Pull): {g_traj[PREPEND_STEPS]:.4f} (Expected {GRIPPER_CLOSED})")
    print(f"Step -1 (End): {g_traj[-1]:.4f} (Expected {GRIPPER_CLOSED})")

    # Save
    torch.save(clean_data, OUTPUT_PATH)
    print(f"\nSaved to: {OUTPUT_PATH}")