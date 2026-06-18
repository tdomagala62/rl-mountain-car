import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path
import random
import json
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

ENV_ID = "MountainCarContinuous-v0"
BASE_SEED = 123
STEPS_TRIAL = 20_000
STEPS_FINAL = 100_000
N_TRIALS = 10
EPOCHS = 10
MINIBATCH_SIZE = 128
GAMMA_DEFAULT = 0.99
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE_HP = {
    "lr": 3e-4,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_eps": 0.2,
    "ent_coef": 0.01,
    "rollout_steps": 4096,
}


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
def compute_gae(rewards, values, dones, next_value, gamma, lam):
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nv = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages, advantages + values


def ppo_update(model, optimizer, obs, actions, log_probs_old, returns, advantages, hp):
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
            surr2 = ratio.clamp(1 - hp["clip_eps"], 1 + hp["clip_eps"]) * advantages[mb]
            loss_pi = -torch.min(surr1, surr2).mean()
            loss_vf = (vals - returns[mb]).pow(2).mean()
            loss = loss_pi + VF_COEF * loss_vf - hp["ent_coef"] * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()


def train(hp, total_steps, label="", verbose=True):
    set_seed(BASE_SEED)
    rollout_steps = hp["rollout_steps"]

    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], eps=1e-5)
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
    obs_rms.update(obs_np[None])
    obs = torch.tensor(obs_rms.normalize(obs_np), dtype=torch.float32, device=DEVICE)

    step = 0
    while step < total_steps:
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

            obs_rms.update(next_obs_np[None])
            obs = torch.tensor(obs_rms.normalize(next_obs_np), dtype=torch.float32, device=DEVICE)

            if done:
                ep_returns.append(ep_ret)
                ep_steps.append(step + t)
                ep_success.append(float(ep_ret > 90))
                ep_idx += 1
                ep_ret = ep_len = 0
                obs_np, _ = env.reset(seed=BASE_SEED + ep_idx)
                obs_rms.update(obs_np[None])
                obs = torch.tensor(obs_rms.normalize(obs_np), dtype=torch.float32, device=DEVICE)

        disc_rets = []
        running_return = 0.0
        for r, d in zip(raw_rewards, done_buf.cpu().numpy()):
            running_return = r + hp["gamma"] * running_return * (1 - d)
            disc_rets.append(running_return)
        ret_rms.update(np.array(disc_rets))
        rew_buf_norm = (rew_buf / ret_rms.std).clamp(-10, 10)

        with torch.no_grad():
            _, next_val = model.get_dist(obs.unsqueeze(0))
        advantages, returns = compute_gae(
            rew_buf_norm, val_buf, done_buf, next_val.squeeze(), hp["gamma"], hp["lam"])
        ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, returns, advantages, hp)

        step += rollout_steps
        if verbose and ep_returns:
            n = len(ep_returns)
            last = slice(max(0, n - 10), n)
            print(f"  [{label[:30]}] step {step:>8} | eps {n:>4} | "
                  f"ret(10) {np.mean(ep_returns[last]):>8.2f} | "
                  f"SR(10) {np.mean(ep_success[last])*100:>5.1f}%")

    env.close()
    return {
        "label": label,
        "hp": hp,
        "ep_returns": np.array(ep_returns),
        "ep_steps": np.array(ep_steps),
        "ep_success": np.array(ep_success),
        "model_state": model.state_dict(),
        "obs_rms": obs_rms.state_dict(),
    }


def objective(trial):
    hp = {
        **BASELINE_HP,
        "lr": trial.suggest_float("lr", 1e-4, 8e-4, log=True),
        "gamma": trial.suggest_float("gamma", 0.985, 0.999),
        "lam": trial.suggest_float("lam", 0.9, 0.99),
        "clip_eps": trial.suggest_float("clip_eps", 0.15, 0.35),
        "ent_coef": trial.suggest_float("ent_coef", 5e-3, 0.05, log=True),
        "rollout_steps": trial.suggest_categorical("rollout_steps", [4096, 8192])
    }

    results = train(hp, total_steps=STEPS_TRIAL, verbose=False)

    r = np.array(results["ep_returns"])
    s = np.array(results["ep_success"])

    if len(r) == 0:
        return 0.0

    sr_last = np.mean(s[-50:]) if len(s) >= 50 else np.mean(s)
    ret_last = np.mean(r[-50:]) if len(r) >= 50 else np.mean(r)
    score = sr_last * 100 + ret_last * 0.1
    return -score


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
    col0, cw = 28, 13
    print(f"\n{'='*82}")
    print("  PORÓWNANIE: Baseline vs Optuna TPE")
    print(f"{'='*82}")
    print(f"  {'Konfiguracja':<{col0}}{'Ret(50)':>{cw}}{'SR(50)%':>{cw}}{'MaxRet':>{cw}}{'1st sukces':>{cw}}")
    print(f"  {'-'*80}")
    for res in results:
        m = compute_metrics(res)
        print(f"  {res['label']:<{col0}}{m['Ret(50)']:>{cw}.2f}"
              f"{m['SR(50)%']:>{cw}.1f}{m['MaxRet']:>{cw}.2f}{str(m['1st sukces']):>{cw}}")
    print(f"{'='*82}")


def plot_comparison(results, save_path="optuna_comparison.png"):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["steelblue", "darkorange"]
    for res, c in zip(results, colors):
        ax.plot(res["ep_steps"], res["ep_returns"], color=c, lw=0.8, alpha=0.8, label=res["label"])
    ax.set_xlabel("Kroki")
    ax.set_ylabel("Return")
    ax.set_title("Baseline vs Optuna TPE")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("trening baseline")
    print("="*60)
    res_baseline = train(BASELINE_HP, total_steps=STEPS_FINAL, label="Baseline")

    print("\n" + "="*60)
    print(f"Optymalizacja TPE ({N_TRIALS} prób, {STEPS_TRIAL} kroków/próba)")
    print("="*60)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=BASE_SEED, n_startup_trials=5),
        study_name="ppo_tpe"
    )

    def callback(study, trial):
        print(f"Próba {trial.number:>3}/{N_TRIALS} | wynik: {-trial.value:>8.2f} | " f"najlepszy: {-study.best_value:>8.2f}")

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])

    best_hp = study.best_params
    print(f"\nNajlepsze hiperparametry:")
    for k, v in best_hp.items():
        print(f"  {k:<20} {v}  (baseline: {BASELINE_HP.get(k, '—')})")

    with open("best_hp_tpe.json", "w") as f:
        json.dump(best_hp, f, indent=2)

    print("\n" + "="*60)
    print(" Trening z najlepszymi HP")
    print("="*60)
    res_optuna = train(best_hp, total_steps=STEPS_FINAL, label="Optuna TPE")

    print_table([res_baseline, res_optuna])
    plot_comparison([res_baseline, res_optuna], "optuna_comparison.png")
