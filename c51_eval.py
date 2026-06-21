import random
from argparse import Namespace
from typing import Callable

import gymnasium as gym
import numpy as np
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    START_EVAL_SEED = 123

    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model_data = torch.load(model_path, map_location="cpu")
    args = Namespace(**model_data["args"])
    model = Model(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max)
    model.load_state_dict(model_data["model_weights"])
    model = model.to(device)
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    episodic_lengths = []
    num_episode = 0
    while len(episodic_returns) < eval_episodes:
        num_episode += 1
        seed = START_EVAL_SEED + num_episode
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _ = model.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()
        next_obs, _, terminations, truncations, infos = envs.step(actions)
        autoresets = np.logical_or(terminations, truncations)
        if autoresets[0]:
            if "episode" not in infos.keys():
                continue
            print(f"eval_episode={len(episodic_returns)}, episodic_return={infos['episode']['r']}")
            episodic_lengths.append(infos["episode"]["l"])
            episodic_returns.append(infos["custom_episode_return"])
        obs = next_obs

    print(f"Mean reward: {np.mean(episodic_returns)}, std: {np.std(episodic_returns)}")
    print(f"Mean length: {np.mean(episodic_lengths)}, std: {np.std(episodic_lengths)}")

    return episodic_returns

import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, env, n_atoms=101, v_min=-100, v_max=100):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = env.single_action_space.n
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.n * n_atoms),
        )

    def get_action(self, x, action=None):
        logits = self.network(x)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]

if __name__ == "__main__":
    # from huggingface_hub import hf_hub_download

    from c51 import make_env

    model_path = "Provide_path"
    evaluate(
        model_path,
        make_env,
        "MountainCar-v0",
        eval_episodes=20,
        run_name=f"eval",
        Model=QNetwork,
        device="cpu",
        capture_video=True,
    )
