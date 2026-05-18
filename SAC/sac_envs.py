import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from collections import deque
from sac_utils import load_config, SAC_Database


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
    # MountainCarContinuous dynamics:
    #   position += velocity          (dt = 1 step)
    #   velocity += action * power    (power = 0.0015)
    # gravity term cos(3*pos) is nonlinear so it is absorbed into process noise Q.
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
        # Predict
        u = np.atleast_1d(u).reshape(-1, 1)
        x_pred = self.F @ self.x + (self.B @ u).squeeze()
        P_pred = self.F @ self.P @ self.F.T + self.Q
        # Update
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ (z - self.H @ x_pred)

        I_KH = np.eye(len(self.x)) - K @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        return self.x.copy()


class KalmanFilterWrapper(gym.Wrapper):
    """Applies a Kalman filter to smooth noisy observations."""
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


class ExtendedKalmanFilter:
    """
    EKF with analytical Jacobian for the MountainCar gravity term.
    Dynamics: vel += action*POWER - 0.0025*cos(3*pos),  pos += vel
    """
    _POWER    = 0.0015
    _GRAVITY  = 0.0025
    _POS_CLIP = (-1.2,  0.6)
    _VEL_CLIP = (-0.07, 0.07)
 
    def __init__(self, obs_dim: int, obs_noise: np.ndarray, process_noise: np.ndarray):
        self.H = np.eye(obs_dim)
        self.Q = np.diag(process_noise)
        self.R = np.diag(obs_noise ** 2)
        self.x = np.zeros(obs_dim)
        self.P = np.eye(obs_dim)
 
    def reset(self, initial_obs: np.ndarray) -> None:
        self.x = initial_obs.copy()
        self.P = np.eye(len(initial_obs))
 
    def _f(self, x: np.ndarray, u: float) -> np.ndarray:
        pos, vel = x
        vel = np.clip(vel + u * self._POWER - self._GRAVITY * np.cos(3.0 * pos), *self._VEL_CLIP)
        pos = np.clip(pos + vel, *self._POS_CLIP)
        return np.array([pos, vel])
 
    def _F_jac(self, x: np.ndarray) -> np.ndarray:
        dvel_dpos = self._GRAVITY * 3.0 * np.sin(3.0 * x[0])
        return np.array([[1.0 + dvel_dpos, 1.0],
                         [dvel_dpos,       1.0]])
 
    def update(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        u      = float(np.atleast_1d(u).ravel()[0])
        F      = self._F_jac(self.x)
        x_pred = self._f(self.x, u)
        P_pred = F @ self.P @ F.T + self.Q
        S      = self.H @ P_pred @ self.H.T + self.R
        K      = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ (z - self.H @ x_pred)
        I_KH   = np.eye(len(self.x)) - K @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        return self.x.copy()
 
 
class EKFWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, obs_noise: np.ndarray, process_noise: np.ndarray):
        super().__init__(env)
        self.ekf = ExtendedKalmanFilter(env.observation_space.shape[0], obs_noise, process_noise)
 
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.ekf.reset(obs)
        return self.ekf.update(obs, u=np.zeros(1)), info
 
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.ekf.update(obs, u=action), reward, terminated, truncated, info
 
 
class FrameStackWrapper(ObservationWrapper):
    """Stacks the last n_frames observations into a single flat vector."""
    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        self._frames  = deque(maxlen=n_frames)
        low  = np.tile(env.observation_space.low,  n_frames)
        high = np.tile(env.observation_space.high, n_frames)
        self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)
 
    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._frames.append(obs)
        return np.concatenate(list(self._frames), axis=0)
 
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self._frames.append(obs.copy())
        return np.concatenate(list(self._frames), axis=0), info
 
 
def make_env(case: int,
             SAC_Data_config: str = '',
             render_mode: str = "rgb_array",
             obs_noise: np.ndarray = np.array([]),
             add_noise_to_case_1:bool = False) -> gym.Env:
    """
    Case 1: clean
    Case 2: noisy
    Case 3: noisy + Linear Kalman Filter
    Case 4: noisy + Frame Stacking
    Case 5: noisy + Extended Kalman Filter
    """
    env = gym.make("MountainCarContinuous-v0", render_mode=render_mode)
    env = RewardWrapper(env)
 
    cfg:SAC_Database = SAC_Database()
    if SAC_Data_config:
        cfg = load_config(path=SAC_Data_config, cfg_class=SAC_Database) # type: ignore
 
    noise_std     = cfg.NOISE_STD
    process_noise = cfg.PROCESS_NOISE
    n_frames      = cfg.FRAME_STACK_N

    if obs_noise.size == 0:
        obs_noise = noise_std
 
    if case == 1:
        if not add_noise_to_case_1:
            return env
        else:
            return NoisyObservationWrapper(env, obs_noise)
    elif case == 2:
        return NoisyObservationWrapper(env, obs_noise)
    elif case == 3:
        return KalmanFilterWrapper(NoisyObservationWrapper(env, obs_noise), noise_std, process_noise)
    elif case == 4:
        return FrameStackWrapper(NoisyObservationWrapper(env, obs_noise), n_frames=n_frames)
    elif case == 5:
        return EKFWrapper(NoisyObservationWrapper(env, obs_noise), noise_std, process_noise)
    else:
        raise ValueError(f"Unknown case: {case}. Choose 1–5.")