import gymnasium as gym
import numpy as np
import random
import torch
import time
from collections import deque

from enum import IntEnum
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

SEED = 1
EVAL_START_SEED = 123
ENV_NAME = "MountainCarContinuous-v0"
RUN_NAME = "heightreward_noised2_2_kalmanextended"
NOISE = np.array([0.02, 0.002])
PROCESS_NOISE = np.array([1e-4, 1e-5])

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class RunType(IntEnum):
    Training = 1
    Evaluation = 2


class TimeRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        reward -= -1.0

        return obs, reward, terminated, truncated, info


class VelocityRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, turncated, info = super().step(action)

        _, velocity = obs
        reward += abs(velocity) * 6

        return obs, reward, terminated, turncated, info


class HeightRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.original_reward = 0.0
    
    def reset(self, *, seed = None, options = None):
        self.original_reward = 0.0
        return super().reset(seed=seed, options=options)
    

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info = self.log_original_reward(reward, terminated, truncated, info)
        position = obs[0]
        height = np.sin(3 * position)  # get hill height
        reward += 0.5 * height        # bonus for being higher
        return obs, reward, terminated, truncated, info

    def log_original_reward(self, reward, terminated, truncated, info):
        self.original_reward += reward

        if terminated or truncated:
            info["custom_episode_return"] = self.original_reward
        
        return info


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
        # Update state
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
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.kf.update(obs, u=action), reward, terminated, truncated, info


class ExtendedKalmanFilter:
    POWER = 0.0015
    GRAVITY = 0.0025

    def __init__(self, obs_noise: np.ndarray, process_noise: np.ndarray):
        self.n = 2
        self.H = np.eye(2)
        self.Q = np.diag(process_noise)
        self.R = np.diag(obs_noise ** 2)
        self.x = np.zeros(2)
        self.P = np.eye(2)

    def reset(self, initial_obs: np.ndarray):
        self.x = initial_obs.astype(np.float64).copy()
        self.P = np.eye(2)

    def f(self, x: np.ndarray, action: float) -> np.ndarray:
        pos = x[0]
        vel = x[1]

        vel_next = (vel + self.POWER * action - self.GRAVITY * np.cos(3.0 * pos))
        vel_next = np.clip(vel_next, -0.07, 0.07)

        pos_next = pos + vel_next
        pos_next = np.clip(pos_next, -1.2, 0.6)

        if pos_next <= -1.2 and vel_next < 0:
            vel_next = 0.0

        return np.array([pos_next, vel_next])

    def jacobian_F(self, x: np.ndarray) -> np.ndarray:
        pos = x[0]

        dvel_dpos = 3.0 * self.GRAVITY * np.sin(3.0 * pos)

        return np.array([[1.0 + dvel_dpos, 1.0],
                         [dvel_dpos,       1.0]])

    def update(self, z: np.ndarray, action: float) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).reshape(2,)
        action = float(np.asarray(action).squeeze())

        x_pred = self.f(self.x, action)
        F = self.jacobian_F(self.x)
        P_pred = F @ self.P @ F.T + self.Q

        innovation = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ innovation
        I = np.eye(self.n)

        self.P = (I - K @ self.H) @ P_pred @ (I - K @ self.H).T + K @ self.R @ K.T
        return self.x.copy()


class ExtendedKalmanFilterWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, obs_noise: np.ndarray, process_noise: np.ndarray):
        super().__init__(env)
        self.kf = ExtendedKalmanFilter(obs_noise, process_noise)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.kf.reset(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.kf.update(obs, action=action), reward, terminated, truncated, info


class TrainingStatsCallback(BaseCallback):
    def __init__(self):
        super().__init__()

        self.last_50_success = deque(maxlen=50)
        self.last_50_returns = deque(maxlen=50)

        self.sr50 = 0.0
        self.ret50 = 0.0
        self.maxRet = -np.inf
        self.firstSuccess = None

    def _on_step(self) -> bool:

        infos = self.locals["infos"]

        for info in infos:
            if "episode" in info:

                ep_return = info["custom_episode_return"]

                self.last_50_returns.append(ep_return)

                self.ret50 = np.mean(self.last_50_returns)
                self.maxRet = max(self.maxRet, ep_return)

                success = bool(info.get("is_success", False))

                self.last_50_success.append(int(success))
                self.sr50 = np.mean(self.last_50_success)

                if success and self.firstSuccess is None:
                    self.firstSuccess = self.num_timesteps

                self.logger.record("custom/sr50", self.sr50)
                self.logger.record("custom/ret50", self.ret50)
                self.logger.record("custom/maxRet", self.maxRet)

                if self.firstSuccess is not None:
                    self.logger.record("custom/firstSuccess", self.firstSuccess)

        return True


class SuccessWrapper(gym.Wrapper):

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        success = False

        if terminated:
            success = True

        info["is_success"] = success

        return obs, reward, terminated, truncated, info


def create_env(type: RunType):
    match type:
        case RunType.Training:
            env = gym.make(ENV_NAME)
        case RunType.Evaluation:
            env = gym.make(ENV_NAME, render_mode="rgb_array")
            env = RecordVideo(env, video_folder=f"videos/td3_2_{ENV_NAME}_{RUN_NAME}-{int(time.time())}")

    # env = TimeRewardWrapper(env)
    # env = VelocityRewardWrapper(env)
    if type == RunType.Training:
        env = HeightRewardWrapper(env)
    env = NoisyObservationWrapper(env, NOISE)
    # env = KalmanFilterWrapper(env, NOISE, PROCESS_NOISE)
    env = ExtendedKalmanFilterWrapper(env, NOISE, PROCESS_NOISE)
    env = SuccessWrapper(env)
    return env


def train():
    run_type = RunType.Training

    env = create_env(run_type)
    env_eval = create_env(RunType.Evaluation)

    n_actions = env.action_space.shape[-1]

    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    td3 = TD3("MlpPolicy", env, tensorboard_log="./logs/",learning_rate=3e-4,
        batch_size=256,
        buffer_size=100000,
        learning_starts=10000,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        action_noise=action_noise) #, seed=SEED)
    eval_callback = EvalCallback(env_eval, eval_freq=5000, deterministic=True)
    stats_callback = TrainingStatsCallback()
    td3.learn(total_timesteps=50000, log_interval=10, tb_log_name=f"td3_2_mountaincar_{RUN_NAME}", callback=[eval_callback, stats_callback], progress_bar=True)
    td3.save(f"td3_2_{ENV_NAME}_{RUN_NAME}-{int(time.time())}")

    env.close()

    return td3

def evaluate(model, num_episodes=20):
    run_type = RunType.Evaluation

    eval_seed = EVAL_START_SEED
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)

    env = create_env(run_type)

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        episode_seed = eval_seed + episode
        obs, _ = env.reset(seed=episode_seed)
        done = False
        total_reward = 0.0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            done = terminated or truncated

            episode_length += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode + 1}/{num_episodes}: reward={total_reward:.2f}")

    env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    print("\nEvaluation Results")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward:  {std_reward:.2f}")
    print(f"Mean length: {mean_length:.2f}")
    print(f"Std length:  {std_length:.2f}")

def main():
    model = train()

    # model = TD3.load("Provide path")
    evaluate(model)


if __name__ == "__main__":
    main()