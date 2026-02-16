from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..config import PipelineConfig
from ..utils import ensure_dir


def _kfold_split(n_splits: int, samples: np.ndarray):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1029)
    trains = []
    tests = []
    for train_index, test_index in kf.split(samples):
        trains.append(samples[train_index])
        tests.append(samples[test_index])
    return trains, tests


def generate_kfold_splits(config: PipelineConfig) -> None:
    """Generate train/test gene lists for each fold."""

    pos_samples = pd.read_excel(config.pos_samples_path)["GeneSymbol"].values
    neg_samples = pd.read_excel(config.neg_samples_path)["GeneSymbol"].values

    pos_trains, pos_tests = _kfold_split(config.n_splits, pos_samples)
    neg_trains, neg_tests = _kfold_split(config.n_splits, neg_samples)

    for idx in range(config.n_splits):
        fold_dir = config.kfold_root_path / f"fold{idx}"
        ensure_dir(fold_dir)
        fold_train = np.vstack((pos_trains[idx].reshape(-1, 1), neg_trains[idx].reshape(-1, 1)))
        fold_test = np.vstack((pos_tests[idx].reshape(-1, 1), neg_tests[idx].reshape(-1, 1)))

        train_df = pd.DataFrame(fold_train, columns=["GeneSymbol"])
        test_df = pd.DataFrame(fold_test, columns=["GeneSymbol"])
        train_df.to_csv(fold_dir / "train_data.txt", sep="\t", index=False)
        test_df.to_csv(fold_dir / "test_data.txt", sep="\t", index=False)
