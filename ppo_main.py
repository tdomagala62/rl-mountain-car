import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path

ENV_ID = "MountainCarContinuous-v0"
SEED = 42
TOTAL_STEPS = 80_000
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
SAVE_PATH = Path("ppo_mcc.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RENDER_EVERY_N_EPISODES = 50
RECORD_VIDEO = True
VIDEO_PATH = "eval_video_v2"
VIDEO_EPISODES = 3

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


def render_episode(model, obs_rms, episode_num):
    env = gym.make(ENV_ID, render_mode="human")
    obs, _ = env.reset()
    done, total = False, 0.0
    while not done:
        obs_norm = obs_rms.normalize(obs)
        obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            dist, _ = model.get_dist(obs_t)
            action = dist.mean.squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, _ = env.step(action.clip(env.action_space.low, env.action_space.high))
        total += reward
        done = terminated or truncated
    env.close()
    print(f"  [render] ep {episode_num} | return {total:.2f}")


def train():
    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    ret_rms = RunningMeanStd(shape=())

    obs_buf = torch.zeros(ROLLOUT_STEPS, obs_dim, device=DEVICE)
    act_buf = torch.zeros(ROLLOUT_STEPS, act_dim, device=DEVICE)
    logp_buf = torch.zeros(ROLLOUT_STEPS, device=DEVICE)
    rew_buf = torch.zeros(ROLLOUT_STEPS, device=DEVICE)
    done_buf = torch.zeros(ROLLOUT_STEPS, device=DEVICE)
    val_buf = torch.zeros(ROLLOUT_STEPS, device=DEVICE)

    episode_returns = []
    ep_ret, ep_len = 0.0, 0
    running_return = 0.0

    obs_np, _ = env.reset(seed=SEED)
    obs_rms.update(obs_np[None])
    obs = torch.tensor(obs_rms.normalize(obs_np), dtype=torch.float32, device=DEVICE)

    step = 0
    while step < TOTAL_STEPS:
        raw_rewards = []

        for t in range(ROLLOUT_STEPS):
            action, logp, val = model.act(obs.unsqueeze(0))
            action = action.squeeze(0)

            clipped_action = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)
            next_obs_np, raw_reward, terminated, truncated, _ = env.step(clipped_action)
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
            obs_np = next_obs_np
            obs = torch.tensor(obs_rms.normalize(obs_np), dtype=torch.float32, device=DEVICE)

            if done:
                ep_idx = len(episode_returns)
                episode_returns.append(ep_ret)

                if RENDER_EVERY_N_EPISODES and ep_idx % RENDER_EVERY_N_EPISODES == 0:
                    model.eval()
                    render_episode(model, obs_rms, ep_idx)
                    model.train()

                ep_ret = ep_len = 0
                running_return = 0.0
                obs_np, _ = env.reset()
                obs_rms.update(obs_np[None])
                obs = torch.tensor(obs_rms.normalize(obs_np), dtype=torch.float32, device=DEVICE)

        disc_rets = []
        running_return = 0.0
        for r, d in zip(raw_rewards, done_buf.cpu().numpy()):
            running_return = r + GAMMA * running_return * (1 - d)
            disc_rets.append(running_return)
        ret_rms.update(np.array(disc_rets))

        rew_buf_norm = rew_buf / ret_rms.std
        rew_buf_norm = rew_buf_norm.clamp(-10, 10)

        with torch.no_grad():
            _, next_val = model.get_dist(obs.unsqueeze(0))
        advantages, returns = compute_gae(rew_buf_norm, val_buf, done_buf, next_val.squeeze())
        ppo_update(model, optimizer, obs_buf, act_buf, logp_buf, returns, advantages)

        step += ROLLOUT_STEPS
        if episode_returns:
            mean_ret = np.mean(episode_returns[-10:])
            print(f"step {step:>8} | eps {len(episode_returns):>4} | raw_ret(10) {mean_ret:>8.2f}")

    env.close()
    torch.save({"model": model.state_dict(), "obs_rms": obs_rms.state_dict()}, SAVE_PATH)

    return episode_returns


def evaluate(n_episodes=10, record=False):
    if record:
        base_env = gym.make(ENV_ID, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(base_env, video_folder=VIDEO_PATH, episode_trigger=lambda ep: ep < VIDEO_EPISODES, name_prefix="ppo_mcc")
    else:
        env = gym.make(ENV_ID, render_mode=None)

    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    obs_rms.load_state_dict(checkpoint["obs_rms"])

    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            obs_norm = obs_rms.normalize(obs)
            obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                dist, _ = model.get_dist(obs_t)
                action = dist.mean.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action.clip(env.action_space.low, env.action_space.high))
            total += reward
            done = terminated or truncated
        returns.append(total)

    env.close()
    
    return returns


def plot_returns(episode_returns, save_path="reward_curve_v2.png"):
    returns = np.array(episode_returns)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(returns, alpha=0.3, color="blue", label="episode return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    episode_returns = train()
    plot_returns(episode_returns)
    evaluate(n_episodes=10, record=RECORD_VIDEO)
