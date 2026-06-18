import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path
import random

FILTER_TYPE = "EKF"
RUN_MODE = "filter_sweep"

ENV_ID = "MountainCarContinuous-v0"
BASE_SEED = 123
TOTAL_STEPS = 100_000
ROLLOUT_STEPS = 4096
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

PROCESS_NOISE = 1e-4
EVAL_EPISODES = 20
NOISE_CONFIGS = [
    (0.00, 0.000),
    (0.02, 0.002),
    (0.04, 0.004),
    (0.08, 0.008),
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": float(self.count)}

    def load_state_dict(self, d):
        self.mean = np.array(d["mean"])
        self.var = np.array(d["var"])
        self.count = float(d["count"])


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


def add_noise(obs, sigma_pos, sigma_vel):
    noisy = obs.copy()
    if sigma_pos > 0:
        noisy[0] += np.random.normal(0, sigma_pos)
    if sigma_vel > 0:
        noisy[1] += np.random.normal(0, sigma_vel)
    return noisy

def mcc_dynamics(x, u):
    pos, vel = x
    vel_new = vel + 0.001 * u - 0.0025 * np.cos(3 * pos)
    vel_new = np.clip(vel_new, -0.07, 0.07)
    pos_new = np.clip(pos + vel_new, -1.2, 0.6)
    return np.array([pos_new, vel_new])


def mcc_jacobian(x, u):
    pos = x[0]
    dv_dp = 0.0025 * 3 * np.sin(3 * pos)
    return np.array([[1 + dv_dp, 1.0], [dv_dp, 1.0]])

class KalmanFilter:
    def __init__(self, sigma_pos, sigma_vel):
        self.F = np.array([[1, 1], [0, 1]])
        self.H = np.eye(2)
        self.R = np.diag([sigma_pos**2 if sigma_pos > 0 else 1e-6,
                          sigma_vel**2 if sigma_vel > 0 else 1e-6])
        self.Q = np.eye(2) * PROCESS_NOISE
        self.x = None
        self.P = None

    def reset(self, obs):
        self.x = obs.copy()
        self.P = np.diag([1.0, 10.0])

    def step(self, noisy_obs, action=0.0):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = noisy_obs - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        return self.x.copy()


class ExtendedKalmanFilter:
    def __init__(self, sigma_pos, sigma_vel):
        self.H = np.eye(2)
        self.R = np.diag([sigma_pos**2 if sigma_pos > 0 else 1e-6,
                          sigma_vel**2 if sigma_vel > 0 else 1e-6])
        self.Q = np.eye(2) * PROCESS_NOISE
        self.x = None
        self.P = None

    def reset(self, obs):
        self.x = obs.copy()
        self.P = np.diag([1.0, 10.0])

    def step(self, noisy_obs, action=0.0):
        self.x = mcc_dynamics(self.x, action)
        F_jac = mcc_jacobian(self.x, action)
        self.P = F_jac @ self.P @ F_jac.T + self.Q
        y = noisy_obs - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        return self.x.copy()


def make_filter(filter_type, sigma_pos, sigma_vel):
    if filter_type == "KF":
        return KalmanFilter(sigma_pos, sigma_vel)
    elif filter_type == "EKF":
        return ExtendedKalmanFilter(sigma_pos, sigma_vel)
    return None


def train(label, sigma_pos=0.0, sigma_vel=0.0, filter_type="brak", save_path=None):
    set_seed(BASE_SEED)
    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    ret_rms = RunningMeanStd(shape=())
    filt = make_filter(filter_type, sigma_pos, sigma_vel)

    obs_buf = torch.zeros(ROLLOUT_STEPS, obs_dim, device=DEVICE)
    act_buf = torch.zeros(ROLLOUT_STEPS, act_dim, device=DEVICE)
    logp_buf = torch.zeros(ROLLOUT_STEPS, device=DEVICE)
    rew_buf = torch.zeros(ROLLOUT_STEPS, device=DEVICE)
    done_buf = torch.zeros(ROLLOUT_STEPS, device=DEVICE)
    val_buf = torch.zeros(ROLLOUT_STEPS, device=DEVICE)

    ep_returns, ep_lengths, ep_steps, ep_success = [], [], [], []
    ep_ret, ep_len = 0.0, 0
    ep_idx = 0

    obs_np, _ = env.reset(seed=BASE_SEED + ep_idx)
    noisy = add_noise(obs_np, sigma_pos, sigma_vel)
    if filt is not None:
        filt.reset(noisy)
        obs_in = filt.step(noisy)
    else:
        obs_in = noisy
    obs_rms.update(obs_in[None])
    obs = torch.tensor(obs_rms.normalize(obs_in), dtype=torch.float32, device=DEVICE)

    step = 0
    while step < TOTAL_STEPS:
        raw_rewards = []
        for t in range(ROLLOUT_STEPS):
            action, logp, val = model.act(obs.unsqueeze(0))
            action = action.squeeze(0)
            clipped = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)

            next_obs_np, raw_reward, terminated, truncated, _ = env.step(clipped)
            done = terminated or truncated

            noisy_next = add_noise(next_obs_np, sigma_pos, sigma_vel)
            if filt is not None:
                obs_in_next = filt.step(noisy_next, float(clipped[0]))
            else:
                obs_in_next = noisy_next

            raw_rewards.append(raw_reward)
            ep_ret += raw_reward
            ep_len += 1

            obs_buf[t] = obs
            act_buf[t] = action
            logp_buf[t] = logp
            rew_buf[t] = raw_reward
            done_buf[t] = float(done)
            val_buf[t] = val.squeeze()

            obs_rms.update(obs_in_next[None])
            obs = torch.tensor(obs_rms.normalize(obs_in_next), dtype=torch.float32, device=DEVICE)

            if done:
                ep_returns.append(ep_ret)
                ep_lengths.append(ep_len)
                ep_steps.append(step + t)
                ep_success.append(float(ep_ret > 90))
                ep_idx += 1
                ep_ret = ep_len = 0

                obs_np, _ = env.reset(seed=BASE_SEED + ep_idx)
                noisy = add_noise(obs_np, sigma_pos, sigma_vel)
                if filt is not None:
                    filt.reset(noisy)
                    obs_in = filt.step(noisy)
                else:
                    obs_in = noisy
                obs_rms.update(obs_in[None])
                obs = torch.tensor(obs_rms.normalize(obs_in), dtype=torch.float32, device=DEVICE)

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
        if ep_returns:
            n = len(ep_returns)
            last = slice(max(0, n - 10), n)
            print(f"[{label[:30]}] step {step:>8} | eps {n:>4} | "
                  f"ret(10) {np.mean(ep_returns[last]):>8.2f} | "
                  f"SR(10) {np.mean(ep_success[last])*100:>5.1f}%")

    env.close()

    result = {
        "label": label,
        "ep_returns": np.array(ep_returns),
        "ep_lengths": np.array(ep_lengths),
        "ep_steps": np.array(ep_steps),
        "ep_success": np.array(ep_success),
        "sigma_pos": sigma_pos,
        "sigma_vel": sigma_vel,
        "filter_type": filter_type,
        "model_state": model.state_dict(),
        "obs_rms": obs_rms.state_dict(),
    }
    if save_path:
        torch.save({k: v for k, v in result.items()
                    if k not in ("ep_returns", "ep_lengths", "ep_steps", "ep_success")},
                   save_path)
    return result


def evaluate(result):
    sigma_pos = result["sigma_pos"]
    sigma_vel = result["sigma_vel"]
    filter_type = result["filter_type"]

    env = gym.make(ENV_ID)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(DEVICE)
    model.load_state_dict(result["model_state"])
    model.eval()
    obs_rms = RunningMeanStd(shape=(env.observation_space.shape[0],))
    obs_rms.load_state_dict(result["obs_rms"])

    returns, lengths = [], []
    for ep in range(EVAL_EPISODES):
        filt = make_filter(filter_type, sigma_pos, sigma_vel)
        obs_np, _ = env.reset(seed=BASE_SEED + 10_000 + ep)
        noisy = add_noise(obs_np, sigma_pos, sigma_vel)
        if filt is not None:
            filt.reset(noisy)
            obs_in = filt.step(noisy)
        else:
            obs_in = noisy

        done, total, length = False, 0.0, 0
        while not done:
            obs_t = torch.tensor(obs_rms.normalize(obs_in), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                dist, _ = model.get_dist(obs_t)
                action = dist.mean.squeeze(0).cpu().numpy()
            next_obs_np, reward, terminated, truncated, _ = env.step(
                action.clip(env.action_space.low, env.action_space.high))
            done = terminated or truncated
            total += reward
            length += 1
            noisy_next = add_noise(next_obs_np, sigma_pos, sigma_vel)
            obs_in = filt.step(noisy_next, float(action[0])) if filt is not None else noisy_next

        returns.append(total)
        lengths.append(length)

    env.close()
    return np.array(returns), np.array(lengths)


def metrics(result):
    r = result["ep_returns"]
    s = result["ep_success"]
    st = result["ep_steps"]
    n = len(r)
    last50 = slice(max(0, n - 50), n)
    first = int(st[np.argmax(s > 0)]) if s.any() else -1
    return {
        "Ret(50)": float(np.mean(r[last50])),
        "SR(50)": float(np.mean(s[last50]) * 100),
        "MaxRet": float(np.max(r)) if n > 0 else 0.0,
        "1st sukces": first,
    }


def plot_training(result, save_path="plot_training.png"):
    r = result["ep_returns"]
    l = result["ep_lengths"]
    st = result["ep_steps"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"Trening PPO - wersja bazowa", fontsize=13)

    axes[0].plot(st, r, color="steelblue", lw=0.8, alpha=0.8)
    axes[0].set_ylabel("Return")
    axes[0].set_title("Wykres nagrody")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(st, l, color="darkorange", lw=0.8, alpha=0.8)
    axes[1].set_ylabel("Długość epizodu")
    axes[1].set_xlabel("Kroki środowiska")
    axes[1].set_title("Wykres długości epizodu")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def print_results_table(results_list, eval_dict=None):
    col0, cw = 38, 12
    header = f"{'Konfiguracja':<{col0}}{'Ret(50)':>{cw}}{'SR(50)%':>{cw}}{'MaxRet':>{cw}}{'1st sukces':>{cw}}"
    if eval_dict:
        header += f"{'Śr.ret (val)':>{cw+2}}{'Śr.dł (val)':>{cw+2}}"

    print(f"\n{'='*100}")
    print("  WYNIKI")
    print(f"{'='*100}")
    print(f"  {header}")
    print(f"  {'-'*98}")

    for res in results_list:
        m = metrics(res)
        row = f"  {res['label']:<{col0}}{m['Ret(50)']:>{cw}.2f}{m['SR(50)']:>{cw}.1f}{m['MaxRet']:>{cw}.2f}{str(m['1st sukces']):>{cw}}"
        if eval_dict and res["label"] in eval_dict:
            er, el = eval_dict[res["label"]]
            row += f"{f'{er.mean():.2f}±{er.std():.2f}':>{cw+2}}{f'{el.mean():.1f}±{el.std():.1f}':>{cw+2}}"
        print(row)
    print(f"{'='*100}")

def run_single():
    res = train(
        label=f"PPO | {FILTER_TYPE} | σ_pos=0.02 σ_vel=0.002",
        sigma_pos=0.02, sigma_vel=0.002,
        filter_type=FILTER_TYPE,
        save_path=Path(f"ppo_{FILTER_TYPE}.pt")
    )
    er, el = evaluate(res)
    print_results_table([res], {res["label"]: (er, el)})
    plot_training(res, "plot_single.png")


def run_noise_sweep():
    results = {}
    evals = {}

    for sp, sv in NOISE_CONFIGS:
        for ft in ["brak", "KF", "EKF"]:
            if sp == 0.0 and sv == 0.0 and ft != "brak":
                continue
            label = "Brak szumu" if sp == 0.0 else f"σ_pos={sp} σ_vel={sv} [{ft}]"
            print(f"\n>>> {label}")
            res = train(label=label, sigma_pos=sp, sigma_vel=sv, filter_type=ft,
                        save_path=Path(f"ppo_sp{sp}_sv{sv}_{ft}.pt"))
            results[(sp, sv, ft)] = res
            er, el = evaluate(res)
            evals[label] = (er, el)

    print_results_table(list(results.values()), evals)


def run_filter_sweep():
    results = []
    evals = {}

    for ft in ["brak", "KF", "EKF"]:
        label = f"σ_pos=0.04 σ_vel=0.004 [{ft}]"
        print(f"\n>>> {label}")
        res = train(label=label, sigma_pos=0.04, sigma_vel=0.004, filter_type=ft,
                    save_path=Path(f"ppo_{ft}.pt"))
        results.append(res)
        er, el = evaluate(res)
        evals[label] = (er, el)

    print_results_table(results, evals)


if __name__ == "__main__":
    runners = {
        "single": run_single,
        "noise_sweep": run_noise_sweep,
        "filter_sweep": run_filter_sweep,
    }
    runners[RUN_MODE]()
