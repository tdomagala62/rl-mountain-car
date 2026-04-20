import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper


class NoisyObservationWrapper(ObservationWrapper):
    """Adds Gaussian noise to every observation."""

    def __init__(self, env: gym.Env, noise_std):
        super().__init__(env)
        self.noise_std = noise_std

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs + np.random.normal(0.0, self.noise_std, size=obs.shape)


class RewardWrapper(gym.Wrapper):
    """Adds potential-based height bonus to guide exploration."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        position = obs[0] 
        height = np.sin(3 * position)  # get hill height
        reward += 0.5 * height        # bonus for being higher
        return obs, reward, terminated, truncated, info

class KalmanFilter:

    def __init__(self, obs_dim:int, obs_noise:np.ndarray, process_noise:float = 0.01):
        n = obs_dim
        self.F = np.eye(n)          # state transition 
        self.H = np.eye(n)          # observation model
        self.Q = np.eye(n) * process_noise
        print(type(obs_noise))
        self.R = np.diag(obs_noise ** 2)
        self.x = np.zeros(n)        # state estimate
        self.P = np.eye(n)          # error covariance

    def reset(self, initial_obs: np.ndarray) -> None:
        self.x = initial_obs.copy()
        self.P = np.eye(len(initial_obs))

    def update(self, z: np.ndarray) -> np.ndarray:
        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        # Update
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ (z - self.H @ x_pred)
        self.P = (np.eye(len(self.x)) - K @ self.H) @ P_pred
        return self.x.copy()


class KalmanFilterWrapper(gym.Wrapper):
    """Applies a Kalman filter to smooth noisy observations."""

    def __init__(self, env: gym.Env, obs_noise, process_noise: float = 0.01):
        super().__init__(env)
        obs_dim = env.observation_space.shape[0]
        self.kf = KalmanFilter(obs_dim, obs_noise, process_noise)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.kf.reset(obs)
        return self.kf.update(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.kf.update(obs), reward, terminated, truncated, info


#observation state 1 min,max == [-1.2 0.6]
#observation state 2 min,max == [-0.07 0.07]
# NOISE_STD = np.array([0.05, 0.01])
NOISE_STD = np.array([0.15, 0.03])

def make_env(case: int) -> gym.Env:
    """
    Case 1: clean environment, no Kalman filter
    Case 2: noisy environment, no Kalman filter
    Case 3: noisy environment + Kalman filter
    """
    env = gym.make("MountainCarContinuous-v0")
    env = RewardWrapper(env)
    if case == 1:
        return env
    elif case == 2:
        return NoisyObservationWrapper(env, noise_std=NOISE_STD)
    elif case == 3:
        env = NoisyObservationWrapper(env, noise_std=NOISE_STD)
        return KalmanFilterWrapper(env, NOISE_STD)
    else:
        raise ValueError(f"Unknown case: {case}. Choose 1, 2, or 3.")
