import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json


NOISE_TYPE = "gaussian"
FILTER_TYPE = "KF"
SCENARIO = 1
RUN_MODE = "noise_sweep"


ENV_ID = "MountainCarContinuous-v0"
SEED = 42
TOTAL_STEPS = 100_000
ROLLOUT_STEPS = 1024
EPOCHS = 10
MINIBATCH_SIZE = 128
GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
LR = 3e-4
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NOISE_STD = 0.05
PROCESS_NOISE = 1e-4

torch.manual_seed(SEED)
np.random.seed(SEED)


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        total = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.mean = new_mean
        self.var = new_var
        self.count = total

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x):
        return np.clip((x - self.mean) / self.std, -10, 10)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.mu = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(64, 1)

        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.zeros_(self.mu.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def get_dist(self, x):
        feat = self.shared(x)
        return Normal(self.mu(feat), self.log_std.exp()), self.critic(feat).squeeze(-1)

    def act(self, obs):
        with torch.no_grad():
            dist, val = self.get_dist(obs)
            action = dist.sample()
            return action, dist.log_prob(action).sum(-1), val


@torch.no_grad()
def compute_gae(rewards, values, dones, next_value):
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nv = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + GAMMA * nv * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * LAM * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages, advantages + values


def ppo_update(model, optimizer, obs, actions, log_probs_old, returns, advantages):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    idx = np.arange(len(obs))
    for _ in range(EPOCHS):
        np.random.shuffle(idx)
        for start in range(0, len(obs), MINIBATCH_SIZE):
            mb = idx[start:start + MINIBATCH_SIZE]
            dist, vals = model.get_dist(obs[mb])
            log_probs = dist.log_prob(actions[mb]).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            ratio = (log_probs - log_probs_old[mb]).exp()
            surr1 = ratio * advantages[mb]
            surr2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * advantages[mb]
            loss_pi = -torch.min(surr1, surr2).mean()
            loss_vf = (vals - returns[mb]).pow(2).mean()
            loss = loss_pi + VF_COEF * loss_vf - ENT_COEF * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()


def add_noise(obs, noise_std, noise_type):
    if noise_std == 0.0:
        return obs.copy()
    if noise_type == "gaussian":
        return obs + np.random.normal(0, noise_std, obs.shape)
    elif noise_type == "jednostajny":
        half = noise_std * np.sqrt(3)
        return obs + np.random.uniform(-half, half, obs.shape)
    elif noise_type == "laplacea":
        scale = noise_std / np.sqrt(2)
        return obs + np.random.laplace(0, scale, obs.shape)
    elif noise_type == "impulsowy":
        noisy = obs.copy()
        mask = np.random.rand(*obs.shape) < 0.05
        noisy[mask] += np.random.choice([-10 * noise_std, 10 * noise_std], size=mask.sum())
        return noisy

def mcc_dynamics(x, u):
    pos, vel = x
    vel_new = vel + 0.001 * u - 0.0025 * np.cos(3 * pos)
    vel_new = np.clip(vel_new, -0.07, 0.07)
    pos_new = pos + vel_new
    pos_new = np.clip(pos_new, -1.2, 0.6)
    return np.array([pos_new, vel_new])


def mcc_jacobian(x, u):
    pos, vel = x
    dvel_dpos = 0.0025 * 3 * np.sin(3 * pos)
    F = np.array([
        [1 + dvel_dpos, 1.0],
        [dvel_dpos,     1.0]
    ])
    return F

class KalmanFilter:
    def __init__(self, obs_dim, noise_std, process_noise):
        self.obs_dim = obs_dim
        self.F = np.array([[1, 1], [0, 1]])
        self.H = np.eye(2) if obs_dim == 2 else np.array([[1, 0]])
        self.R = (np.eye(2) if obs_dim == 2 else np.array([[1]])) * noise_std**2
        self.Q = np.eye(2) * process_noise
        self.x = None
        self.P = None

    def reset(self, first_obs):
        self.x = first_obs.copy() if self.obs_dim == 2 else np.array([first_obs[0], 0.0])
        self.P = np.diag([1.0, 10.0])

    def step(self, noisy_obs, action):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        z = noisy_obs if self.obs_dim == 2 else noisy_obs[:1]
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        return self.x.copy()


class ExtendedKalmanFilter:
    def __init__(self, obs_dim, noise_std, process_noise):
        self.obs_dim = obs_dim
        self.H = np.eye(2) if obs_dim == 2 else np.array([[1, 0]])
        self.R = (np.eye(2) if obs_dim == 2 else np.array([[1]])) * noise_std**2
        self.Q = np.eye(2) * process_noise
        self.x = None
        self.P = None

    def reset(self, first_obs):
        self.x = first_obs.copy() if self.obs_dim == 2 else np.array([first_obs[0], 0.0])
        self.P = np.diag([1.0, 10.0])

    def step(self, noisy_obs, action):
        self.x = mcc_dynamics(self.x, action)
        F = mcc_jacobian(self.x, action)
        self.P = F @ self.P @ F.T + self.Q

        z = noisy_obs if self.obs_dim == 2 else noisy_obs[:1]
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        return self.x.copy()


def make_filter(filter_type, obs_dim, noise_std, process_noise):
    if filter_type == "KF":
        return KalmanFilter(obs_dim, noise_std, process_noise)
    elif filter_type == "EKF":
        return ExtendedKalmanFilter(obs_dim, noise_std, process_noise)
    elif filter_type == "brak":
        return None


class Metrics:
    def __init__(self, label):
        self.label = label
        self.episode_returns = []
        self.episode_lengths = []
        self.success_flags = []
        self.wall_time = []
        self.steps_at_episode = []
        self.t_start = None

    def start(self):
        self.t_start = time.time()

    def record(self, ret, length, step):
        self.episode_returns.append(ret)
        self.episode_lengths.append(length)
        self.success_flags.append(float(ret > 90))
        self.wall_time.append(time.time() - self.t_start)
        self.steps_at_episode.append(step)

    def summary(self):
        r = np.array(self.episode_returns)
        s = np.array(self.success_flags)
        return {
            "label": self.label,
            "mean_return": float(np.mean(r)),
            "std_return": float(np.std(r)),
            "max_return": float(np.max(r)),
            "mean_return_last50": float(np.mean(r[-50:])) if len(r) >= 50 else float(np.mean(r)),
            "success_rate": float(np.mean(s)),
            "success_rate_last50": float(np.mean(s[-50:])) if len(s) >= 50 else float(np.mean(s)),
            "total_episodes": len(r),
            "total_wall_time_s": float(self.wall_time[-1]) if self.wall_time else 0.0,
            "steps_to_first_success": int(self.steps_at_episode[np.argmax(s > 0)]) if s.any() else -1,
        }

    def save(self, path):
        data = {
            "label": self.label,
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "success_flags": self.success_flags,
            "wall_time": self.wall_time,
            "steps_at_episode": self.steps_at_episode,
            "summary": self.summary()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def train(scenario, noise_std, noise_type, filter_type, save_path, label=None):
    obs_dim_sensor = 2 if scenario == 1 else 1
    label = label or f"PPO | {filter_type} | {noise_type} σ={noise_std}"
    metrics = Metrics(label)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Scenariusz: {'pos+vel z szumem' if scenario==1 else 'tylko pos z szumem, vel estymowana'}")
    print(f"{'='*60}")

    env = gym.make(ENV_ID)
    act_dim = env.action_space.shape[0]
    agent_obs_dim = 2

    model = ActorCritic(agent_obs_dim, act_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    obs_rms = RunningMeanStd(shape=(agent_obs_dim,))
    ret_rms = RunningMeanStd(shape=())
    filt = make_filter(filter_type, obs_dim_sensor, noise_std, PROCESS_NOISE)

    obs_buf = torch.zeros(ROLLOUT_STEPS, agent_obs_dim, device=DEVICE)
    act_buf = torch.zeros(ROLLOUT_STEPS, act_dim, device=DEVICE)
    logp_buf = torch.zeros(ROLLOUT_STEPS, device=DEVICE)
    rew_buf = torch.zeros(ROLLOUT_STEPS, device=DEVICE)
    done_buf = torch.zeros(ROLLOUT_STEPS, device=DEVICE)
    val_buf = torch.zeros(ROLLOUT_STEPS, device=DEVICE)

    ep_ret, ep_len = 0.0, 0

    def get_filtered_obs(raw_obs, action=0.0):
        noisy = add_noise(raw_obs, noise_std, noise_type)
        if filt is None:
            return noisy
        return filt.step(noisy, action)

    true_obs_np, _ = env.reset(seed=SEED)
    noisy_init = add_noise(true_obs_np, noise_std, noise_type)
    if filt is not None:
        filt.reset(noisy_init)
    filtered_obs = get_filtered_obs(true_obs_np, 0.0)

    obs_rms.update(filtered_obs[None])
    obs = torch.tensor(obs_rms.normalize(filtered_obs), dtype=torch.float32, device=DEVICE)

    step = 0
    metrics.start()

    while step < TOTAL_STEPS:
        raw_rewards = []

        for t in range(ROLLOUT_STEPS):
            action, logp, val = model.act(obs.unsqueeze(0))
            action = action.squeeze(0)
            clipped_action = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)

            true_next_obs, raw_reward, terminated, truncated, _ = env.step(clipped_action)
            done = terminated or truncated

            filtered_next_obs = get_filtered_obs(true_next_obs, clipped_action.squeeze())

            raw_rewards.append(raw_reward)
            ep_ret += raw_reward
            ep_len += 1

            obs_buf[t] = obs
            act_buf[t] = action
            logp_buf[t] = logp
            rew_buf[t] = raw_reward
            done_buf[t] = float(done)
            val_buf[t] = val.squeeze()

            obs_rms.update(filtered_next_obs[None])
            obs = torch.tensor(obs_rms.normalize(filtered_next_obs), dtype=torch.float32, device=DEVICE)

            if done:
                metrics.record(ep_ret, ep_len, step)
                ep_ret = ep_len = 0

                true_obs_np, _ = env.reset()
                noisy_init = add_noise(true_obs_np, noise_std, noise_type)
                if filt is not None:
                    filt.reset(noisy_init)
                filtered_obs = get_filtered_obs(true_obs_np, 0.0)
                obs_rms.update(filtered_obs[None])
                obs = torch.tensor(obs_rms.normalize(filtered_obs), dtype=torch.float32, device=DEVICE)

        disc_rets = []
        running_return = 0.0
        for r, d in zip(raw_rewards, done_buf.cpu().numpy()):
            running_return = r + GAMMA * running_return * (1 - d)
            disc_rets.append(running_return)
        ret_rms.update(np.array(disc_rets))

        rew_buf_norm = (rew_buf / ret_rms.std).clamp(-10, 10)

        with torch.no_grad():
            _, next_val = model.get_dist(obs.unsqueeze(0))
        advantages, returns = compute_gae(rew_buf_norm, val_buf, done_buf, next_val.squeeze())
        ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, returns, advantages)

        step += ROLLOUT_STEPS
        if metrics.episode_returns:
            mean_ret = np.mean(metrics.episode_returns[-10:])
            sr = np.mean(metrics.success_flags[-10:]) * 100
            print(f"step {step:>8} | eps {len(metrics.episode_returns):>4} | "
                  f"ret(10) {mean_ret:>8.2f} | success(10) {sr:>5.1f}%")

    env.close()
    torch.save({"model": model.state_dict(), "obs_rms": obs_rms.state_dict()}, save_path)
    metrics.save(str(save_path).replace(".pt", "_metrics.json"))
    print(f"\nZapisano → {save_path}")
    print_summary(metrics.summary())
    return metrics


def print_summary(s):
    print(f"\n  {'Metryka':<35} {'Wartość':>12}")
    print(f"  {'-'*50}")
    for k, v in s.items():
        if k == "label":
            continue
        if isinstance(v, float):
            print(f"  {k:<35} {v:>12.4f}")
        else:
            print(f"  {k:<35} {v:>12}")


def plot_metrics_comparison(metrics_list, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=13)

    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_list)))

    for m, c in zip(metrics_list, colors):
        returns = np.array(m.episode_returns)
        steps = np.array(m.steps_at_episode)

        if len(returns) > 0:
            plt.plot(
                range(len(returns)),
                returns,
                color=c,
                lw=1.5,
                label=m.label
            )

    

    plt.xlabel("Kroki środowiska")
    plt.ylabel("Return")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"Wykres zapisany → {save_path}")


def print_comparison_table(metrics_list):
    keys = ["mean_return_last50", "success_rate_last50", "steps_to_first_success",
            "total_wall_time_s", "max_return"]
    headers = ["Konfiguracja", "Ret(50)", "SR(50)", "1stSuccess", "Czas[s]", "MaxRet"]

    col_w = [35, 10, 8, 12, 10, 10]
    header_line = "  ".join(f"{h:<{w}}" for h, w in zip(headers, col_w))
    print(f"\n{'='*90}")
    print(f"  TABELA PORÓWNAWCZA")
    print(f"{'='*90}")
    print(f"  {header_line}")
    print(f"  {'-'*88}")

    for m in metrics_list:
        s = m.summary()
        vals = [
            m.label[:col_w[0]],
            f"{s['mean_return_last50']:.2f}",
            f"{s['success_rate_last50']*100:.1f}%",
            str(s["steps_to_first_success"]),
            f"{s['total_wall_time_s']:.0f}",
            f"{s['max_return']:.2f}",
        ]
        print("  " + "  ".join(f"{v:<{w}}" for v, w in zip(vals, col_w)))
    print(f"{'='*90}")


def run_single():
    m = train(
        scenario=SCENARIO,
        noise_std=NOISE_STD,
        noise_type=NOISE_TYPE,
        filter_type=FILTER_TYPE,
        save_path=Path(f"ppo_{FILTER_TYPE}_{NOISE_TYPE}_s{SCENARIO}.pt"),
        label=f"PPO | {FILTER_TYPE} | {NOISE_TYPE} | σ={NOISE_STD} | S{SCENARIO}"
    )
    plot_metrics_comparison([m], m.label, "single_run.png")
    return [m]


def run_noise_sweep():
    noise_types = ["gaussian", "jednostajny", "laplacea", "impulsowy"]
    metrics_list = []
    for nt in noise_types:
        m = train(
            scenario=SCENARIO,
            noise_std=NOISE_STD,
            noise_type=nt,
            filter_type=FILTER_TYPE,
            save_path=Path(f"ppo_{FILTER_TYPE}_{nt}.pt"),
            label=f"{nt}"
        )
        metrics_list.append(m)
    plot_metrics_comparison(metrics_list, f"Porównanie typów szumu (filtr: {FILTER_TYPE})", "noise_sweep.png")
    print_comparison_table(metrics_list)
    return metrics_list


def run_filter_sweep():
    filter_types = ["brak", "KF", "EKF"]
    metrics_list = []
    for ft in filter_types:
        m = train(
            scenario=SCENARIO,
            noise_std=NOISE_STD,
            noise_type=NOISE_TYPE,
            filter_type=ft,
            save_path=Path(f"ppo_{ft}_{NOISE_TYPE}.pt"),
            label=f"{ft}"
        )
        metrics_list.append(m)
    plot_metrics_comparison(metrics_list, f"Brak filtra vs KF vs EKF (szum: {NOISE_TYPE} σ={NOISE_STD})", "filter_sweep.png")
    print_comparison_table(metrics_list)
    return metrics_list





if __name__ == "__main__":
    runners = {
        "single":       run_single,
        "noise_sweep":  run_noise_sweep,
        "filter_sweep": run_filter_sweep,
    }
    results = runners[RUN_MODE]()
