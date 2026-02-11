# docs/pages/example_workflows/summit_franka_manipulation/step_1_environment_setup.rst
python isaaclab_arena/examples/policy_runner.py \
  --enable_cameras \
  --num_steps 1000000 \
  --policy_type zero_action \
  summit_franka_open_door \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --object_center


python isaaclab_arena/examples/policy_runner.py \
  --headless \
  --enable_cameras \
  --num_steps 100 \
  --policy_type zero_action \
  summit_franka_open_door \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0

python isaaclab_arena/examples/policy_runner.py \
  --policy_type zero_action \
  --num_steps 2000 \
  --enable_cameras \
  gr1_open_microwave \
  --embodiment gr1_joint


python isaaclab_arena/scripts/record_automoma_demos.py \
  --enable_cameras --set_state \
  --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
  --dataset_file data/automoma/summit_franka_open_microwave_7221_setstate.hdf5 \
  --num_episodes 50 \
  summit_franka_open_door \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --object_center


python isaaclab_arena/examples/policy_runner.py \
     --enable_cameras \
     --policy_type replay \
     --replay_file_path data/automoma/summit_franka_open_microwave_7221_setstate.hdf5 \
     summit_franka_open_door \
     --object_name microwave_7221 \
     --scene_name scene_0_seed_0 \
     --object_center

DATASET_DIR="/home/xinhai/projects/lerobot-arena/IsaacLab-Arena/data/nvidia/Arena-G1-Loco-Manipulation-Task"

python isaaclab_arena/scripts/replay_demos.py \
  --device cpu \
  --enable_cameras \
  --dataset_file $DATASET_DIR/arena_g1_loco_manipulation_dataset_generated_small.hdf5 \
  galileo_g1_locomanip_pick_and_place \
  --object brown_box \
  --embodiment g1_wbc_pink

# Generate 10 demonstrations
python isaaclab_arena/scripts/generate_dataset.py \
  --headless \
  --enable_cameras \
  --mimic \
  --input_file $DATASET_DIR/arena_g1_loco_manipulation_dataset_annotated.hdf5 \
  --output_file $DATASET_DIR/arena_g1_loco_manipulation_dataset_generated.hdf5 \
  --generation_num_trials 10 \
  --device cpu \
  galileo_g1_locomanip_pick_and_place \
  --object brown_box \
  --embodiment g1_wbc_pink

python isaaclab_arena/scripts/replay_demos.py \
  --device cpu \
  --enable_cameras \
  --dataset_file $DATASET_DIR/arena_g1_loco_manipulation_dataset_generated.hdf5 \
  galileo_g1_locomanip_pick_and_place \
  --object brown_box \
  --embodiment g1_wbc_pink