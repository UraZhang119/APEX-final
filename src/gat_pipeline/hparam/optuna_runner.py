from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import optuna

from ..config import load_config
from ..training import train_fold


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna search for GAT hyperparameters")
    parser.add_argument("--config", default="configs/fungi.yaml")
    parser.add_argument("--folds", nargs="*", type=int, default=[0])
    parser.add_argument("--model", default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1029)
    parser.add_argument("--pruner", default="median", choices=["none", "median", "halving"])
    parser.add_argument("--storage", default=None)
    parser.add_argument("--study-name", default="gat-hpo")
    parser.add_argument("--direction", default="minimize", choices=["minimize", "maximize"])
    parser.add_argument("--n-startup-trials", type=int, default=5)
    parser.add_argument("--wandb", action="store_true", help="Log each trial to WandB (uses configured project)")
    return parser.parse_args()


def _make_pruner(name: str, n_startup_trials: int) -> optuna.pruners.BasePruner:
    if name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials)
    if name == "halving":
        return optuna.pruners.SuccessiveHalvingPruner()
    return optuna.pruners.NopPruner()


def _trial_objective(args: argparse.Namespace, trial: optuna.trial.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-6, 5e-4, log=True)
    drop_prob = trial.suggest_float("drop_prob", 0.1, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    ratio = trial.suggest_float("ratio", 0.1, 0.3)
    batch_size = trial.suggest_categorical("batch_size", [4, 6, 8])

    overrides = {
        "lr": lr,
        "drop_prob": drop_prob,
        "weight_decay": weight_decay,
        "ratio": ratio,
        "train_batch_size": batch_size,
        "test_batch_size": batch_size,
    }

    config = load_config(args.config, overrides)

    best_losses: List[float] = []
    for fold_idx, fold in enumerate(args.folds):
        summary = train_fold(
            config,
            fold=fold,
            model_name=args.model,
            use_wandb=args.wandb,
            wandb_run_name=f"optuna_trial{trial.number}_fold{fold}",
        )
        val_losses = [epoch["val_loss"] for epoch in summary.history["epochs"]]
        best_loss = float(np.min(val_losses))
        best_losses.append(best_loss)
        trial.report(best_loss, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(best_losses))


def main() -> None:
    args = _parse_args()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = _make_pruner(args.pruner, args.n_startup_trials)

    study = optuna.create_study(
        direction=args.direction,
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=bool(args.storage),
    )

    try:
        study.optimize(lambda trial: _trial_objective(args, trial), n_trials=args.trials)
    except KeyboardInterrupt:
        pass

    print("Best trial:")
    best = study.best_trial
    print(f"  value: {best.value}")
    print("  params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
