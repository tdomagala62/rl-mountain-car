import argparse
import numpy as np
from stable_baselines3 import SAC
from envs import make_env
import gymnasium as gym
import envs as _envs

def evaluate(case: int, n_episodes: int = 10, render: bool = False) -> None:
    print(f"\n{'='*55}")
    print(f"  Evaluating SAC – Case {case}")
    print(f"  Episodes  : {n_episodes}")
    print(f"{'='*55}\n")

    render_mode = "human" if render else None
    env = make_env(case)        
    if render:
        base = gym.make("MountainCarContinuous-v0", render_mode="human")
        if case == 2 or case == 4:
            env = _envs.NoisyObservationWrapper(base, noise_std=_envs.NOISE_STD)
        elif case == 3:
            env = _envs.NoisyObservationWrapper(base, noise_std=_envs.NOISE_STD)
            env = _envs.KalmanFilterWrapper(env, obs_noise=_envs.NOISE_STD)
        else:
            env = base
    
    if case == 4:
        model = SAC.load(f"models/sac_case_{1}", env=env)
    else:
        model = SAC.load(f"models/sac_case_{case}", env=env)

    rewards, lengths = [], []

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward, steps = 0.0, 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        lengths.append(steps)
        print(f"  Episode {ep:3d}: reward={total_reward:8.2f}  length={steps}")

    print(f"\n  Mean reward : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean length : {np.mean(lengths):.1f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAC on MountainCarContinuous")
    parser.add_argument("--case", type=int, choices=[1, 2, 3, 4], required=True,
                        help="1=clean  2=noisy  3=noisy+Kalman")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment visually")
    args = parser.parse_args()

    evaluate(case=args.case, n_episodes=args.episodes, render=args.render)
