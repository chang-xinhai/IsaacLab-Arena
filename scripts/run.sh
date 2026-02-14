# pipeline from recording to training to evaluation for summit_franka open_microwave

# Record
python isaaclab_arena/scripts/record_automoma_demos.py \
  --enable_cameras \
  --mobile_base_relative \
  --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
  --dataset_file data/automoma/summit_franka_open_microwave_7221_drive.hdf5 \
  --num_episodes 30 \
  summit_franka_open_door \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --object_center

python isaaclab_arena/scripts/record_automoma_demos.py \
  --enable_cameras \
  --interpolated 4 \
  --mobile_base_relative \
  --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
  --dataset_file data/automoma/summit_franka_open_microwave_7221_drive.hdf5 \
  --num_episodes 30 \
  summit_franka_open_door \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --object_center

# Convert to lerobot format
python isaaclab_arena_gr00t/data_utils/convert_hdf5_to_lerobot_v30.py \
  --yaml_file isaaclab_arena_gr00t/config/summit_franka_manip_config.yaml \
  --data_root  data/automoma \
  --hdf5_name summit_franka_open_microwave_7221_drive.hdf5 \
  --repo_id  automoma/summit_franka_open_microwave_7221_drive \
  --output_dir data/lerobot/automoma/summit_franka_open_microwave_7221_drive


# Visualize dataset
lerobot-dataset-viz \
    --repo-id automoma/summit_franka_open_microwave_7221_drive \
    --root data/lerobot/automoma/summit_franka_open_microwave_7221_drive \
    --episode-index 0 \
    --video-backend pyav

exp_name="summit_franka_open_microwave_7221_drive"
dataset_root=data/lerobot/automoma/$exp_name
rm -rf outputs/train/act_$exp_name
lerobot-train \
  --policy.type=act \
  --batch_size=128 \
  --steps=10000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=1000 \
  --job_name=act_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.optimizer_lr=1e-4 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/act_$exp_name \
  --dataset.preload=true \
  --dataset.preload_cache=true \
  --dataset.filter_features_by_policy=true


conda activate lerobot-arena
cd IsaacLab-Arena
lerobot-eval \
     --policy.path=outputs/train/act_$exp_name/checkpoints/010000/pretrained_model \
     --policy.device=cuda \
     --env.type=isaaclab_arena \
     --env.hub_path=$(pwd)/isaaclab-arena-envs \
     --env.environment=summit_franka_open_door_eval \
     --env.headless=false \
     --env.enable_cameras=true \
     --env.state_keys=joint_pos \
     --env.camera_keys=ego_topdown_rgb,ego_wrist_rgb,fix_local_rgb \
     --env.state_dim=12 \
     --env.action_dim=12 \
     --env.camera_height=240 \
     --env.camera_width=320 \
     --env.episode_length=300 \
     --env.kwargs='{"object_name": "microwave_7221", "scene_name": "scene_0_seed_0", "object_center": true, "mobile_base_relative": true, "traj_file": "res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data_test.pt", "traj_seed": 42}' \
     --rename_map='{"observation.images.ego_topdown_rgb": "observation.images.ego_topdown", "observation.images.ego_wrist_rgb": "observation.images.ego_wrist", "observation.images.fix_local_rgb": "observation.images.fix_local"}' \
     --trust_remote_code=true \
     --eval.batch_size=1 \
     --eval.n_episodes=10























# docs/pages/example_workflows/summit_franka_manipulation/step_1_environment_setup.rst
python isaaclab_arena/examples/policy_runner.py \
  --device cpu \
  --enable_cameras \
  --num_steps 1000000 \
  --policy_type zero_action \
  summit_franka_open_door \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --object_center

python isaaclab_arena/examples/policy_runner.py \
  --device cpu \
  --enable_cameras \
  --num_steps 1000000 \
  --policy_type zero_action \
  gr1_open_microwave \
  --embodiment gr1_joint

git clone -b main https://github.com/moveit/moveit2_tutorials

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

# Train pipeline
python scripts/debug/fix_pt.py

python isaaclab_arena/scripts/record_automoma_demos.py \
  --device cpu \
  --enable_cameras --set_state --disable_collision \
  --interpolated 4 --mobile_base_relative \
  --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
  --dataset_file data/automoma/summit_franka_open_microwave_7221_setstate.hdf5 \
  --num_episodes 30 \
  summit_franka_open_door \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --object_center

python isaaclab_arena/scripts/record_automoma_demos.py \
  --enable_cameras --disable_collision \
  --interpolated 4 --mobile_base_relative \
  --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
  --dataset_file data/automoma/summit_franka_open_microwave_7221_drive.hdf5 \
  --num_episodes 30 \
  summit_franka_open_door \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --object_center

python isaaclab_arena/scripts/record_automoma_demos.py \
  --device cpu \
  --enable_cameras \
  --interpolated 4 --mobile_base_relative \
  --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
  --dataset_file data/automoma/summit_franka_open_microwave_7221_drive.hdf5 \
  --num_episodes 30 \
  summit_franka_open_door \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --object_center

python isaaclab_arena/scripts/record_automoma_demos.py \
  --device cpu \
  --enable_cameras \
  --mobile_base_relative \
  --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
  --dataset_file data/automoma/summit_franka_open_microwave_7221_drive.hdf5 \
  --num_episodes 30 \
  summit_franka_open_door \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --object_center

python scripts/debug/test_collision_meshes.py \
  --enable_cameras \
  --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
  summit_franka_open_door \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --object_center

python isaaclab_arena_gr00t/data_utils/convert_hdf5_to_lerobot_v30.py \
  --yaml_file isaaclab_arena_gr00t/config/summit_franka_manip_config.yaml \
  --data_root  data/automoma \
  --hdf5_name summit_franka_open_microwave_7221_setstate.hdf5 \
  --repo_id  automoma/summit_franka_open_microwave_7221_setstate \
  --output_dir data/lerobot/automoma/summit_franka_open_microwave_7221_setstate


lerobot-dataset-viz \
    --repo-id automoma/summit_franka_open_microwave_7221_setstate \
    --root data/lerobot/automoma/summit_franka_open_microwave_7221_setstate \
    --episode-index 0 \
    --video-backend pyav


exp_name="summit_franka_open_microwave_7221_setstate"
dataset_root=data/lerobot/automoma/$exp_name
rm -rf outputs/train/act_$exp_name
lerobot-train \
  --policy.type=act \
  --batch_size=128 \
  --steps=10000 \
  --log_freq=50 \
  --eval_freq=500 \
  --save_freq=1000 \
  --job_name=act_$exp_name \
  --dataset.repo_id=$exp_name \
  --dataset.root=$dataset_root \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.optimizer_lr=1e-4 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --wandb.enable=true \
  --output_dir=outputs/train/act_$exp_name \
  --dataset.preload=true \
  --dataset.preload_cache=true \
  --dataset.filter_features_by_policy=true



lerobot-eval \
    --policy.path=outputs/train/act_Arena-GR1-Manipulation-Task-v3/checkpoints/005000/pretrained_model \
    --env.type=isaaclab_arena \
    --env.hub_path=nvidia/isaaclab-arena-envs \
    --rename_map='{"observation.images.robot_pov_cam_rgb": "observation.images.robot_pov_cam"}' \
    --policy.device=cuda \
    --env.environment=gr1_microwave \
    --env.embodiment=gr1_pink \
    --env.object=mustard_bottle \
    --env.headless=false \
    --env.enable_cameras=true \
    --env.video=true \
    --env.video_length=10 \
    --env.video_interval=15 \
    --env.state_keys=robot_joint_pos \
    --env.camera_keys=robot_pov_cam_rgb \
    --trust_remote_code=True \
    --eval.batch_size=1 

conda activate lerobot-arena
cd IsaacLab-Arena
lerobot-eval \
  --policy.path=../lerobot/outputs/train/act_summit_franka_open_microwave_7221_setstate/checkpoints/010000/pretrained_model \
  --env.type=isaaclab_arena \
  --env.hub_path=$(pwd)/isaaclab-arena-envs \
  --env.environment=summit_franka_open_door_eval \
  --env.enable_cameras=true \
  --env.state_keys=joint_pos \
  --env.camera_keys=ego_topdown_rgb,ego_wrist_rgb,fix_local_rgb \
  --env.state_dim=12 --env.action_dim=12 \
  --env.camera_height=240 --env.camera_width=320 \
  --env.kwargs='{"object_name":"microwave_7221","scene_name":"scene_0_seed_0","object_center":true,"disable_collision":true,"mobile_base_relative":true}' \
  --rename_map='{"observation.images.ego_topdown_rgb":"observation.images.ego_topdown","observation.images.ego_wrist_rgb":"observation.images.ego_wrist","observation.images.fix_local_rgb":"observation.images.fix_local"}' \
  --trust_remote_code=true --eval.batch_size=1 --eval.n_episodes=10



















python isaaclab_arena/scripts/record_automoma_demos.py \
  --enable_cameras --set_state \
  --interpolated 4 --mobile_base_relative \
  --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
  --dataset_file data/automoma/summit_franka_open_microwave_7221_setstate.hdf5 \
  --num_episodes 10 \
  summit_franka_open_door \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --object_center


python isaaclab_arena/scripts/record_automoma_demos.py \
  --enable_cameras --set_state --disable_collision \
  --interpolated 4 --mobile_base_relative \
  --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
  --dataset_file data/automoma/summit_franka_open_microwave_7221_setstate.hdf5 \
  --num_episodes 10 \
  summit_franka_open_door \
  --object_name microwave_7221 \
  --scene_name scene_0_seed_0 \
  --object_center

python isaaclab_arena/scripts/record_automoma_demos.py \
  --enable_cameras \
  --interpolated 4 --mobile_base_relative \
  --traj_file res_for_custom/automoma_trajs/summit_franka/microwave_7221/scene_0_seed_0/traj_data.pt \
  --dataset_file data/automoma/summit_franka_open_microwave_7221_drive.hdf5 \
  --num_episodes 10 \
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


python isaaclab_arena_gr00t/data_utils/convert_hdf5_to_lerobot_v30.py \
  --yaml_file isaaclab_arena_gr00t/config/summit_franka_manip_config.yaml \
  --data_root  data/automoma \
  --hdf5_name summit_franka_open_microwave_7221_setstate.hdf5 \
  --repo_id  automoma/summit_franka_open_microwave_7221_setstate \
  --output_dir data/lerobot

lerobot-dataset-viz \
    --repo-id automoma/summit_franka_open_microwave_7221_setstate \
    --root data/lerobot \
    --episode-index 0 \
    --video-backend pyav


python isaaclab_arena_gr00t/data_utils/convert_hdf5_to_lerobot_v30.py \
  --yaml_file isaaclab_arena_gr00t/g1_locomanip_gr00t_closedloop_config.yaml \
  --data_root  data/nvidia/Arena-G1-Loco-Manipulation-Task \
  --hdf5_name arena_g1_loco_manipulation_dataset_generated_small.hdf5 \
  --repo_id  nvidia/arena_g1_loco_manipulation_dataset_generated_small \
  --output_dir data/lerobot


lerobot-dataset-viz \
    --repo-id nvidia/arena_g1_loco_manipulation_dataset_generated_small \
    --root data/lerobot \
    --episode-index 0 \
    --video-backend pyav












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


