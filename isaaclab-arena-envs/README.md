# IsaacLab-Arena Environments

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EnvHub-blue)](https://huggingface.co/nvidia/isaaclab-arena-envs)

[Isaac Lab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) is a robotics simulation framework that enhances [NVIDIA IsaacLab](https://github.com/isaac-sim/IsaacLab) by providing a composable, scalable system for creating diverse simulation environments and evaluating robot learning policies. GPU-accelerated simulation environments for robotics learning, compatible with [LeRobot](https://github.com/huggingface/lerobot) and the [EnvHub](https://huggingface.co/docs/lerobot/envhub) ecosystem.


## Available Environments

The following are example environments built with NVIDIA IsaacLab Arena. The community can create custom environments using the IsaacLab Arena framework:

| Preview                                                                                                                                                                                    | Environment        | Description                                                                                                          |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------- | :------------------------------------------------------------------------------------------------------------------- |
| <img src="https://huggingface.co/nvidia/isaaclab-arena-envs/resolve/main/assets/Gr1OpenMicrowaveEnvironment.png" alt="GR1 Microwave" width="200" />                      | `gr1_microwave`    | Reach out to the microwave and open it.                                                                              |
| <img src="https://huggingface.co/nvidia/isaaclab-arena-envs/resolve/main/assets/GalileoPickAndPlaceEnvironment.png" alt="Galileo Pick and Place" width="200" />          | `galileo_pnp`      | Pick objects and place in target location                                                                            |
| <img src="https://huggingface.co/nvidia/isaaclab-arena-envs/resolve/main/assets/GalileoG1LocomanipPickAndPlaceEnvironment.png" alt="G1 Loco-manipulation" width="200" /> | `g1_locomanip_pnp` | Pick up the brown box from the shelf, and place it into the blue bin on the table located at the right of the shelf. |
| <img src="https://huggingface.co/nvidia/isaaclab-arena-envs/resolve/main/assets/KitchenPickAndPlaceEnvironment.png" alt="Kitchen Pick and Place" width="200" />          | `kitchen_pnp`      | Kitchen object manipulation tasks                                                                                    |
| <img src="https://huggingface.co/nvidia/isaaclab-arena-envs/resolve/main/assets/PressButtonEnvironment.png" alt="Press Button" width="200" />                            | `press_button`     | Locate and press button                                                                                              |


## Quick Start

Evaluate a trained policy e.g. ```SmolVLA```:

```bash
lerobot-eval \
    --policy.path=nvidia/smolvla-arena-gr1-microwave \
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
```

## Related Links

- [LeRobot](https://github.com/huggingface/lerobot) - Open-source robotics learning library
- [EnvHub Documentation](https://huggingface.co/docs/lerobot/envhub) - LeRobot environment hub
- [IsaacLab](https://github.com/isaac-sim/IsaacLab) - NVIDIA's robot learning framework
- [IsaacLab Arena](https://github.com/isaac-sim/IsaacLab-Arena) - Community environments for IsaacLab

