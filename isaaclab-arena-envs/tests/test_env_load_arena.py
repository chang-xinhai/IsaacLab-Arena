import logging
from dataclasses import asdict, dataclass
from pprint import pformat
import torch
import tqdm
from lerobot import envs
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig


@parser.wrap()
def main(cfg: EvalPipelineConfig):
    """Run zero action rollout for IsaacLab Arena environment."""
    logging.info(pformat(asdict(cfg)))

    from lerobot.envs.factory import make_env

    # hub_path = cfg.env.hub_path
    env_dict = make_env(
        cfg.env,
        n_envs=cfg.env.num_envs,
        trust_remote_code=True,
    )
    env = next(iter(env_dict.values()))[0]

    try:
        env.reset()

        for _ in tqdm.tqdm(range(cfg.env.episode_length)):
            with torch.inference_mode():
                action_dim = env.action_space.shape[-1]
                actions = torch.zeros(
                    (env.num_envs, action_dim), device=env.device
                )
                obs, rewards, terminated, truncated, info = env.step(actions)
                print(obs.keys())
                print(obs["policy"].keys())
                print(obs["camera_obs"].keys())
    finally:
        env.close()


if __name__ == "__main__":
    main()
