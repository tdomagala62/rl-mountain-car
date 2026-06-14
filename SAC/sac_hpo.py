import argparse
import multiprocessing as mp
import os
import time
import warnings
from typing import Optional

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

from sac_envs import make_env


N_EVAL_EPISODES     = 10
EVAL_SEED           = 42
STORAGE_DIR         = "SAC/hpo"
DB_PATH             = f"sqlite:///{STORAGE_DIR}/hpo.db"
HB_MIN_RESOURCE     = 20_000
HB_MAX_RESOURCE     = 100_000
HB_REDUCTION_FACTOR = 3


class PruningCallback(BaseCallback):
    def __init__(self, trial: optuna.Trial, eval_env, n_eval: int):
        super().__init__()
        self.trial     = trial
        self.eval_env  = eval_env
        self.n_eval    = n_eval
        self.rungs     = self._build_rungs()
        self._rung_idx = 0

    def _build_rungs(self) -> list[int]:
        rungs, r = [], HB_MIN_RESOURCE
        while r < HB_MAX_RESOURCE:
            rungs.append(r)
            r = int(r * HB_REDUCTION_FACTOR)
        return rungs

    def _on_step(self) -> bool:
        if self._rung_idx >= len(self.rungs):
            return True
        if self.num_timesteps < self.rungs[self._rung_idx]:
            return True

        mean_reward, _ = evaluate_policy(
            self.model, self.eval_env,
            n_eval_episodes=self.n_eval,
            deterministic=True, warn=False,
        )
        self.trial.report(mean_reward, step=self.num_timesteps) # type: ignore
        self._rung_idx += 1

        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return True


def objective(trial: optuna.Trial, case: int, data_cfg: str) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    buffer_size   = trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000])
    batch_size    = trial.suggest_categorical("batch_size", [128, 256, 512])
    gamma         = trial.suggest_float("gamma", 0.97, 0.999)
    tau           = trial.suggest_float("tau", 1e-3, 0.05, log=True)
    ent_coef      = trial.suggest_categorical("ent_coef", ["auto", 0.01, 0.1])

    env      = make_env(case, SAC_Data_config=data_cfg)
    eval_env = make_env(case, SAC_Data_config=data_cfg)
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    model = SAC(
        "MlpPolicy", env, device=device, verbose=0,
        learning_rate=learning_rate, buffer_size=buffer_size,
        batch_size=batch_size, ent_coef=ent_coef, gamma=gamma, tau=tau, # type: ignore
    )

    mean_reward = float("-inf")
    try:
        model.learn(total_timesteps=HB_MAX_RESOURCE,
                    callback=PruningCallback(trial, eval_env, N_EVAL_EPISODES))
        mean_reward, _ = evaluate_policy(
            model, eval_env,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True, warn=False,
        )
    except optuna.TrialPruned:
        raise
    except Exception as exc:
        print(f"[case {case}] trial {trial.number} failed: {exc}", flush=True)
    finally:
        env.close()
        eval_env.close()

    return mean_reward # type: ignore


def _make_sampler(name: str):
    if name == "tpe":
        return TPESampler(seed=EVAL_SEED, multivariate=True)
    if name == "cmaes":
        return CmaEsSampler(seed=EVAL_SEED)
    return RandomSampler(seed=EVAL_SEED)


def _worker(case: int, n_trials: int, sampler_name: str, data_cfg: str,
            queue: mp.Queue) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=f"sac_case_{case}",
        direction="maximize",
        storage=DB_PATH,
        load_if_exists=True,
        sampler=_make_sampler(sampler_name),
        pruner=HyperbandPruner(
            min_resource=HB_MIN_RESOURCE,
            max_resource=HB_MAX_RESOURCE,
            reduction_factor=HB_REDUCTION_FACTOR,
        ),
    )

    print(f"[case {case}] starting {n_trials} trials …", flush=True)
    study.optimize(lambda t: objective(t, case, data_cfg),
                   n_trials=n_trials, gc_after_trial=True)

    try:
        b = study.best_trial
        queue.put((case, {"value": b.value, "number": b.number, "params": b.params}))
    except ValueError:
        queue.put((case, None))


def run_parallel(cases, n_trials, n_jobs, sampler_name, data_cfg) -> dict:
    queue   = mp.Queue()
    pending = list(cases)
    active  = []
    results = {}

    while pending or active:
        while pending and len(active) < n_jobs:
            case = pending.pop(0)
            p = mp.Process(target=_worker,
                           args=(case, n_trials, sampler_name, data_cfg, queue),
                           daemon=True)
            p.start()
            active.append((case, p))
            print(f"[orchestrator] launched case {case} (pid={p.pid})", flush=True)

        while not queue.empty():
            case_id, info = queue.get_nowait()
            results[case_id] = info

        active = [(c, p) for c, p in active if p.is_alive() or (p.join(), False)[1]]
        time.sleep(1.0)

    while not queue.empty():
        case_id, info = queue.get_nowait()
        results[case_id] = info

    return results


def print_summary(results: dict) -> None:
    print(f"\n{'='*60}\n  HPO Summary\n{'='*60}")
    for case in sorted(results):
        info = results[case]
        if info is None:
            print(f"  Case {case}: no successful trial")
            continue
        p = info["params"]
        print(f"  Case {case}  trial #{info['number']:>3d}  reward={info['value']:>8.2f}  "
              f"lr={p['learning_rate']:.2e}  γ={p['gamma']:.4f}  "
              f"batch={p['batch_size']}  τ={p['tau']:.4f}  ent={p['ent_coef']}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cases",    type=int, nargs="+", choices=[1,2,3,4,5], default=[1,2,3,4,5])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--n_jobs",   type=int, default=2)
    parser.add_argument("--sampler",  choices=["tpe", "cmaes", "random"], default="tpe")
    parser.add_argument("--data_cfg", type=str, default="")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(STORAGE_DIR, exist_ok=True)

    # Initialise the DB schema once before workers spawn to avoid a race
    # condition where two processes try to CREATE TABLE simultaneously.
    optuna.create_study(study_name="__init__", storage=DB_PATH,
                        direction="maximize", load_if_exists=True)

    print(f"{'='*60}\n  Cases={args.cases}  trials={args.n_trials}  "
          f"jobs={args.n_jobs}  sampler={args.sampler}\n{'='*60}")

    t0 = time.time()
    results = run_parallel(args.cases, args.n_trials, args.n_jobs,
                           args.sampler, args.data_cfg)
    print_summary(results)
    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()