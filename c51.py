# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/c51/#c51py
import os
import random
import time
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from buffers import ReplayBuffer
from td3_baseline.td3_baseline import ExtendedKalmanFilterWrapper

MAX_STEPS = 1000
NOISE = np.array([0.08, 0.008])
PROCESS_NOISE = np.array([1e-4, 1e-5])

@dataclass
class Args:
    exp_name: str = "c51"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rl-mountain-car"
    """the wandb's project name"""
    wandb_entity: str = "tdomagala-agh"
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "MountainCar-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    n_atoms: int = 101
    """the number of atoms"""
    v_min: float = -100
    """the return lower bound"""
    v_max: float = 100
    """the return upper bound"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.6
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


class HeightRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.current_obs = None
        self.continuous_episode_return = 0.0

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.current_obs = obs
        self.continuous_episode_return = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.current_obs = obs

        modified_reward = self.reward(reward)
        continuous_reward = self.logging_reward(action)
        self.continuous_episode_return += continuous_reward

        if terminated or truncated:
            if terminated:
                self.continuous_episode_return += 100
            info["custom_episode_return"] = self.continuous_episode_return

        return obs, modified_reward, terminated, truncated, info

    def reward(self, reward):
        if self.current_obs is not None:
            position = self.current_obs[0]
            height = np.sin(3 * position)
            reward += 0.5 * height

        return reward

    def logging_reward(self, action):
        if self.current_obs is None:
            return 0.0

        return -0.1 * action**2

class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, noise_std):
        super().__init__(env)
        self.noise_std = noise_std

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs + np.random.normal(0.0, self.noise_std, size=obs.shape)


class KalmanFilter:
    _POWER = 0.0015

    def __init__(self, obs_dim: int, obs_noise: np.ndarray, process_noise: np.ndarray):
        n = obs_dim
        # State transition
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]])
        # Control-input matrix
        self.B = np.array([[0.0],
                           [self._POWER]])
        self.H = np.eye(n)                      # observation model
        self.Q = np.diag(process_noise)         # process noise covariance
        self.R = np.diag(obs_noise ** 2)        # measurement noise covariance
        self.x = np.zeros(n)                    # state estimate
        self.P = np.eye(n)                      # error covariance

    def reset(self, initial_obs: np.ndarray) -> None:
        self.x = initial_obs.copy()
        self.P = np.eye(len(initial_obs))

    def update(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        # Predict state
        u = np.atleast_1d(u).reshape(-1, 1)
        x_pred = self.F @ self.x + (self.B @ u).squeeze()
        P_pred = self.F @ self.P @ self.F.T + self.Q
        # Update statte
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ (z - self.H @ x_pred)

        I_KH = np.eye(len(self.x)) - K @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        return self.x.copy()


class KalmanFilterWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, obs_noise: np.ndarray, process_noise: np.ndarray):
        super().__init__(env)
        obs_dim = env.observation_space.shape[0]
        self.kf = KalmanFilter(obs_dim, obs_noise, process_noise)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.kf.reset(obs)
        return self.kf.update(obs, u=np.zeros(1)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.kf.update(obs, u=action), reward, terminated, truncated, info


class MyStatisticsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.num_timestep = 0

        self.last_50_success = deque(maxlen=50)
        self.last_50_returns = deque(maxlen=50)
        self.max_return = -np.inf
        self.first_success = None
    
    def reset(self, *, seed = None, options = None):
        print(f"Ret50: {np.mean(self.last_50_returns)}, SR50: {np.mean(self.last_50_success)}, MaxRet: {self.max_return}, FirstSuccess: {self.first_success if self.first_success is not None else -1}")

        return super().reset(seed=seed, options=options)
    
    def step(self, action):
        self.num_timestep += 1
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            reward = info["custom_episode_return"]
            self.last_50_returns.append(reward)
            self.last_50_success.append(1.0 if terminated else 0)
            if reward > self.max_return:
                self.max_return = reward
            if terminated and self.first_success is None:
                self.first_success = self.num_timestep

        return obs, reward, terminated, truncated, info


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=MAX_STEPS)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, max_episode_steps=MAX_STEPS)
        env = HeightRewardWrapper(env)
        env = NoisyObservationWrapper(env, NOISE)
        env = ExtendedKalmanFilterWrapper(env, NOISE, PROCESS_NOISE)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = MyStatisticsWrapper(env)

        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
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


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = Args()
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, pmf = q_network.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        autoresets = np.logical_or(terminations, truncations)
        if autoresets[0]:
            if "episode" in infos.keys():
                episodic_return = infos["episode"]["r"][0]
                episodic_length = infos["episode"]["l"][0]

                print(f"global_step={global_step}, episodic_return={episodic_return}")
                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", episodic_length, global_step)

                if "custom_episode_return" in infos:
                    custom_episode_return = infos["custom_episode_return"][0]
                    print(f"custom_episode_return={custom_episode_return}")
                    writer.add_scalar("charts/custom_episode_return", custom_episode_return, global_step)

        # TRY NOT TO MODIFY: save data to reply buffer
        rb.add(obs, next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    _, next_pmfs = target_network.get_action(data.next_observations)
                    next_atoms = data.rewards + args.gamma * target_network.atoms * (1 - data.dones)
                    # projection
                    delta_z = target_network.atoms[1] - target_network.atoms[0]
                    tz = next_atoms.clamp(args.v_min, args.v_max)

                    b = (tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)
                    # (l == u).float() handles the case where bj is exactly an integer
                    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs
                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                _, old_pmfs = q_network.get_action(data.observations, data.actions.flatten())
                loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    old_val = (old_pmfs * q_network.atoms).sum(1)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")
        from c51_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=20,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()