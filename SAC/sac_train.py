import argparse
import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from pytorch_lightning.loggers import TensorBoardLogger
from sac_envs import make_env
import torch
from dataclasses import dataclass


@dataclass
class SAC_TRAIN_ENV:
    verbose:int = 0
    learning_rate:float = 1e-3
    buffer_size:int = 100_000
    batch_size:int = 256
    ent_coef:str = "auto"
    gamma:float = 0.99
    tau:float =  0.005


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
def train(case:int, 
          SAC_Data_config='', 
          SAC_Train_config='', 
          total_timesteps:int = 150000
          ) -> None:
    print('='*55)
    print(f"Training SAC: {case}")
    print(f"Timesteps: {total_timesteps}")
    print('='*55)

    env = make_env(case, SAC_Data_config=SAC_Data_config)
    os.makedirs("SAC/models", exist_ok=True)

    pl_logger = TensorBoardLogger("SAC/logs", name=f"case_{case}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    #initialize variables
    SAC_Tenvs = SAC_TRAIN_ENV()
    if SAC_Train_config:
        pass
    model = SAC(
        "MlpPolicy",
        env,
        device=device,
        verbose=SAC_Tenvs.verbose,
        learning_rate=SAC_Tenvs.learning_rate,
        buffer_size=SAC_Tenvs.buffer_size,
        batch_size=SAC_Tenvs.batch_size,
        ent_coef=SAC_Tenvs.ent_coef,
        gamma=SAC_Tenvs.gamma,
        tau=SAC_Tenvs.tau,
    )

    callback = TrainingLogger(pl_logger)
    checkpoint_callback = CheckpointCallback(
                                save_freq=2000,
                                save_path="SAC/models",
                                name_prefix=f"mountain_car_case_{case}",
                                )
    
    model.learn(total_timesteps=total_timesteps, callback=[callback, checkpoint_callback])

    save_path = f"SAC/models/mountain_car_case_{case}"
    model.save(save_path)
    print(f"\nModel saved {save_path}.zip")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on MountainCarContinuous")
    parser.add_argument("--case", type=int, choices=[1, 2, 3], required=True,
                        help="1=clean  2=noisy  3=noisy+Kalman")
    parser.add_argument("--timesteps", type=int, default=150_000,
                        help="Total environment steps (default: 150 000)")
    parser.add_argument("--data_cfg", type=str, default='',
                        help="Path to the SAC DATA config file")
    parser.add_argument("--train_cfg", type=str, default='',
                        help="Path to the SAC TRAINING config file")
    args = parser.parse_args()

    train(case=args.case, 
          SAC_Data_config=args.data_cfg, 
          SAC_Train_config=args.train_cfg, 
          total_timesteps=args.timesteps,
          )
