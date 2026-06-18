import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter1d

# testowane usprawnienia
VERSIONS = [
    ("Baseline", False, False, False, False),
    ("+ norm_obs", True, False, False, False),
    ("+ norm_rew", False, True,  False, False),
    ("+ entropy", False, False, True,  False),
    ("+ duzy_rollout", False, False, False, True),
    ("+ norm_obs + norm_rew", True, True, False, False),
    ("+ norm_obs + entropy", True, False, True, False),
    ("+ norm_obs + norm_rew + entropy", True, True, True, False),
    ("Wszystkie razem", True, True, True, True),
]

ENV_ID = "MountainCarContinuous-v0"
BASE_SEED = 123
TOTAL_STEPS = 100_000
ROLLOUT_SMALL = 2048
ROLLOUT_LARGE = 4096
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x):
        return np.clip((x - self.mean) / self.std, -10, 10)


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


def ppo_update(model, optimizer, obs, actions, log_probs_old, returns, advantages, ent_coef):
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
            loss = loss_pi + VF_COEF * loss_vf - ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()


def train(label, use_norm_obs, use_norm_rew, use_entropy, use_large_rollout):
    set_seed(BASE_SEED)
    rollout_steps = ROLLOUT_LARGE if use_large_rollout else ROLLOUT_SMALL
    ent_coef = ENT_COEF if use_entropy else 0.0

    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    ret_rms = RunningMeanStd(shape=())

    obs_buf = torch.zeros(rollout_steps, obs_dim, device=DEVICE)
    act_buf = torch.zeros(rollout_steps, act_dim, device=DEVICE)
    logp_buf = torch.zeros(rollout_steps, device=DEVICE)
    rew_buf = torch.zeros(rollout_steps, device=DEVICE)
    done_buf = torch.zeros(rollout_steps, device=DEVICE)
    val_buf = torch.zeros(rollout_steps, device=DEVICE)

    ep_returns, ep_steps, ep_success = [], [], []
    ep_ret, ep_len, ep_idx = 0.0, 0, 0

    obs_np, _ = env.reset(seed=BASE_SEED + ep_idx)
    if use_norm_obs:
        obs_rms.update(obs_np[None])
        obs_in = obs_rms.normalize(obs_np)
    else:
        obs_in = obs_np
    obs = torch.tensor(obs_in, dtype=torch.float32, device=DEVICE)

    step = 0
    while step < TOTAL_STEPS:
        raw_rewards = []
        for t in range(rollout_steps):
            action, logp, val = model.act(obs.unsqueeze(0))
            action = action.squeeze(0)
            clipped = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)
            next_obs_np, raw_reward, terminated, truncated, _ = env.step(clipped)
            done = terminated or truncated

            raw_rewards.append(raw_reward)
            ep_ret += raw_reward
            ep_len += 1

            obs_buf[t] = obs
            act_buf[t] = action
            logp_buf[t] = logp
            rew_buf[t] = raw_reward
            done_buf[t] = float(done)
            val_buf[t] = val.squeeze()

            if use_norm_obs:
                obs_rms.update(next_obs_np[None])
                obs_in = obs_rms.normalize(next_obs_np)
            else:
                obs_in = next_obs_np
            obs = torch.tensor(obs_in, dtype=torch.float32, device=DEVICE)

            if done:
                ep_returns.append(ep_ret)
                ep_steps.append(step + t)
                ep_success.append(float(ep_ret > 90))
                ep_idx += 1
                ep_ret = ep_len = 0

                obs_np, _ = env.reset(seed=BASE_SEED + ep_idx)
                if use_norm_obs:
                    obs_rms.update(obs_np[None])
                    obs_in = obs_rms.normalize(obs_np)
                else:
                    obs_in = obs_np
                obs = torch.tensor(obs_in, dtype=torch.float32, device=DEVICE)

        if use_norm_rew:
            disc_rets = []
            running_return = 0.0
            for r, d in zip(raw_rewards, done_buf.cpu().numpy()):
                running_return = r + GAMMA * running_return * (1 - d)
                disc_rets.append(running_return)
            ret_rms.update(np.array(disc_rets))
            rew_buf_use = (rew_buf / ret_rms.std).clamp(-10, 10)
        else:
            rew_buf_use = rew_buf

        with torch.no_grad():
            _, next_val = model.get_dist(obs.unsqueeze(0))
        advantages, returns = compute_gae(rew_buf_use, val_buf, done_buf, next_val.squeeze())
        ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, returns, advantages, ent_coef)

        step += rollout_steps
        if ep_returns:
            n = len(ep_returns)
            last = slice(max(0, n - 10), n)
            sr = np.mean(ep_success[last]) * 100
            print(f"[{label[:32]}] step {step:>8} | eps {n:>4} | "
                  f"ret(10) {np.mean(ep_returns[last]):>8.2f} | SR {sr:>5.1f}%")

    env.close()
    return {
        "label": label,
        "ep_returns": np.array(ep_returns),
        "ep_steps": np.array(ep_steps),
        "ep_success": np.array(ep_success),
    }


def compute_metrics(res):
    r = res["ep_returns"]
    s = res["ep_success"]
    st = res["ep_steps"]
    n = len(r)
    last50 = slice(max(0, n - 50), n)
    first = int(st[np.argmax(s > 0)]) if s.any() else -1
    return {
        "Ret(50)": float(np.mean(r[last50])),
        "SR(50)%": float(np.mean(s[last50]) * 100),
        "MaxRet": float(np.max(r)) if n > 0 else 0.0,
        "1st sukces": first,
    }


def print_table(results):
    col0, cw = 40, 13
    print(f"\n{'='*95}")
    print("  TABELA ABLACYJNA — wpływ usprawnień PPO")
    print(f"{'='*95}")
    print(f"  {'Konfiguracja':<{col0}}{'Ret(50)':>{cw}}{'SR(50)%':>{cw}}{'MaxRet':>{cw}}{'1st sukces':>{cw}}")
    print(f"  {'-'*93}")
    for res in results:
        m = compute_metrics(res)
        print(f"  {res['label']:<{col0}}{m['Ret(50)']:>{cw}.2f}"
              f"{m['SR(50)%']:>{cw}.1f}{m['MaxRet']:>{cw}.2f}{str(m['1st sukces']):>{cw}}")
    print(f"{'='*95}")


def smooth(returns, steps, n=400, sigma_frac=0.05):
    if len(returns) < 2:
        return steps, returns
    grid = np.linspace(steps[0], steps[-1], n)
    interp = np.interp(grid, steps, returns)
    sig = max(1, int(n * sigma_frac))
    return grid, gaussian_filter1d(interp, sigma=sig)

def plot_versions(results, save_path="results_versions.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Wpływ usprawnień", fontsize=13)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for res, c in zip(results, colors):
        r = res["ep_returns"]
        s = res["ep_success"]
        steps = res["ep_steps"]
        label = res["label"]

        if len(r) < 2:
            continue

        sg, sm = smooth(r, steps)
        axes[0].plot(steps, r, color=c, alpha=0.12, lw=0.8)
        axes[0].plot(sg, sm, color=c, lw=2.0, label=label)

        win = 30
        if len(s) >= win:
            sr = np.convolve(s, np.ones(win) / win, mode="valid") * 100
            axes[1].plot(range(win - 1, len(s)), sr, color=c, lw=2.0, label=label)

    for ax, (title, xl, yl) in zip(axes, [
        ("Return vs kroki środowiska", "Kroki", "Return"),
        (f"Success rate (okno 30 eps)", "Epizod", "SR [%]"),
    ]):
        #ax.axhline(90, color="green", linestyle="--", lw=1, alpha=0.5, label="solved=90")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    all_results = []
    for label, norm_obs, norm_rew, entropy, large_rollout in VERSIONS:
        print(f"\n>>> {label}")
        res = train(label, norm_obs, norm_rew, entropy, large_rollout)
        all_results.append(res)

    print_table(all_results)
    plot_versions(all_results)
