import argparse
import numpy as np
from stable_baselines3 import SAC
from sac_envs import make_env
import sac_envs as _envs
import gymnasium as gym
import random
import torch
import os
import imageio

def evaluate(case: int, SAC_Data_config='', 
             direct_model_path:str="",
             n_episodes: int = 10, 
             render: bool = False, 
             seed:int = 133) -> None:
    print(f"\n{'='*55}")
    print(f"  Evaluating SAC – Case {case}")
    print(f"  Episodes  : {n_episodes}")
    print(f"{'='*55}\n")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    #create environment based on the selected case
    env = make_env(case, SAC_Data_config=SAC_Data_config)        
    #load the trained model from the appropriate case
    try:
        if not direct_model_path:
            model = SAC.load(f"SAC/models/mountain_car_case_{case}", env=env)
        else:
            model = SAC.load(direct_model_path, env=env)
    except Exception as e:
        raise Exception(f"Coud not load model: {e}")
    rewards, lengths = [], []

    gif_dir = f"SAC/gifs/case_{case}"
    os.makedirs(gif_dir, exist_ok=True)

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=seed + ep + 23)
        done = False
        total_reward, steps = 0.0, 0
        frames = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            frame = env.render()
            frames.append(frame)

        rewards.append(total_reward)
        lengths.append(steps)
        print(f"  Episode {ep:3d}: reward={total_reward:8.2f}  length={steps}")

        if render:
            gif_path = os.path.join(gif_dir, f"case{case}_episode_{ep:03d}.gif")
            imageio.mimsave(gif_path, frames, fps=30)  
        
    print(f"\n  Mean reward : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean length : {np.mean(lengths):.1f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAC on MountainCarContinuous")
    parser.add_argument("--case", type=int, choices=[1, 2, 3], required=True,
                        help="1=clean  2=noisy  3=noisy+Kalman")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment visually and store as gifs")
    parser.add_argument("--model_path",  type=str, default='',
                        help="Path to the model file")
    args = parser.parse_args()

    evaluate(case=args.case, n_episodes=args.episodes, render=args.render, direct_model_path=args.model_path)
