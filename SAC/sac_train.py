import argparse
import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from pytorch_lightning.loggers import TensorBoardLogger
from envs import make_env
import torch

class TrainingLogger(BaseCallback):

    def __init__(self, pl_logger: TensorBoardLogger):
        super().__init__()
        self.pl_logger = pl_logger
        self._ep_reward = 0.0
        self._ep_length = 0
        self._episode = 0

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals["rewards"][0])
        self._ep_length += 1

        if self.locals["dones"][0]:
            self._episode += 1
            metrics = {
                "train/episode_reward": self._ep_reward,
                "train/episode_length": self._ep_length,
            }
            self.pl_logger.log_metrics(metrics, step=self._episode)
            print(
                f"  [ep {self._episode:4d} | step {self.num_timesteps:7d}] "
                f"reward={self._ep_reward:8.2f}  length={self._ep_length}"
            )
            self._ep_reward = 0.0
            self._ep_length = 0

        return True


#Training
def train(case:int, total_timesteps:int = 150000) -> None:
    print('='*55)
    print(f"Training SAC: {case}")
    print(f"Timesteps: {total_timesteps}")
    print('='*55)

    env = make_env(case)
    os.makedirs("models", exist_ok=True)

    pl_logger = TensorBoardLogger("logs", name=f"case_{case}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    model = SAC(
        "MlpPolicy",
        env,
        device=device,
        verbose=0,
        learning_rate=1e-3,
        buffer_size=100_000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
    )

    callback = TrainingLogger(pl_logger)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    save_path = f"models/sac_case_{case}"
    model.save(save_path)
    print(f"\nModel saved {save_path}.zip")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on MountainCarContinuous")
    parser.add_argument("--case", type=int, choices=[1, 2, 3], required=True,
                        help="1=clean  2=noisy  3=noisy+Kalman")
    parser.add_argument("--timesteps", type=int, default=150_000,
                        help="Total environment steps (default: 150 000)")
    args = parser.parse_args()

    train(case=args.case, total_timesteps=args.timesteps)
