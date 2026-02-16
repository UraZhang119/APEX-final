from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig, load_config
from .utils import compute_classification_metrics, ensure_dir, load_pt

LOGGER = logging.getLogger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class FoldSplit:
    fold_id: int
    train_ids: List[str]
    test_ids: List[str]


def _read_gene_symbols(file_path: Path) -> List[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing gene list: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines()]
    if not lines:
        return []
    # Drop header if present
    if lines[0].lower().startswith("genesymbol"):
        lines = lines[1:]
    return [line for line in lines if line]


def _load_gene_table(config: PipelineConfig) -> pd.DataFrame:
    table_path = config.root_path / config.species / "orig_sample_list" / "gene_list.txt"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing gene_list.txt at {table_path}")
    df = pd.read_csv(table_path, sep="\t")
    expected_cols = {"GeneSymbol", "Fasta", "Target"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"gene_list.txt must contain columns {expected_cols}, got {df.columns.tolist()}")
    df = df.dropna(subset=["GeneSymbol"]).copy()
    df["GeneSymbol"] = df["GeneSymbol"].astype(str)
    df["Target"] = df["Target"].astype(int)
    return df


def _load_fold_splits(config: PipelineConfig, fold_indices: Sequence[int]) -> List[FoldSplit]:
    splits: List[FoldSplit] = []
    for fold_id in fold_indices:
        fold_dir = config.kfold_root_path / f"fold{fold_id}"
        train_path = fold_dir / "train_data.txt"
        test_path = fold_dir / "test_data.txt"
        train_ids = _read_gene_symbols(train_path)
        test_ids = _read_gene_symbols(test_path)
        if not train_ids or not test_ids:
            LOGGER.warning("Fold %s is missing training or test genes (train=%d, test=%d).", fold_id, len(train_ids), len(test_ids))
        splits.append(FoldSplit(fold_id=fold_id, train_ids=train_ids, test_ids=test_ids))
    return splits


def _mean_esm_vector(tensor) -> np.ndarray:
    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D ESM representation, got shape {arr.shape}")
    return arr.mean(axis=0).astype(np.float32)


def _precompute_esm_lookup(raw_dir: Path, gene_ids: Iterable[str]) -> Dict[str, np.ndarray]:
    lookup: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for gene in gene_ids:
        sample_path = raw_dir / f"{gene}.pt"
        if not sample_path.exists():
            missing.append(gene)
            continue
        sample = load_pt(sample_path)
        lookup[gene] = _mean_esm_vector(sample["feature_representation"])
    if missing:
        LOGGER.warning("Missing ESM embeddings for %d genes (showing first 5): %s", len(missing), missing[:5])
    return lookup


def _sequence_features(sequence: str) -> np.ndarray:
    seq = (sequence or "").strip().upper()
    if not seq:
        return np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    length = float(len(seq))
    counts = [seq.count(aa) / length for aa in AMINO_ACIDS]
    return np.array(counts, dtype=np.float32)


def _precompute_sequence_lookup(sequence_map: Dict[str, str], gene_ids: Iterable[str]) -> Dict[str, np.ndarray]:
    lookup: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for gene in gene_ids:
        sequence = sequence_map.get(gene)
        if sequence is None:
            missing.append(gene)
            continue
        lookup[gene] = _sequence_features(sequence)
    if missing:
        LOGGER.warning("Missing raw sequences for %d genes (showing first 5): %s", len(missing), missing[:5])
    return lookup


def _make_classifier(random_state: int = 42):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=random_state,
        ),
    )


def _run_fold_evaluation(
    baseline_name: str,
    split: FoldSplit,
    feature_lookup: Dict[str, np.ndarray],
    target_lookup: Dict[str, int],
):
    train_ids = [gene for gene in split.train_ids if gene in feature_lookup and gene in target_lookup]
    test_ids = [gene for gene in split.test_ids if gene in feature_lookup and gene in target_lookup]
    if not train_ids or not test_ids:
        LOGGER.warning("Skipping fold %s for %s (usable train=%d, test=%d).", split.fold_id, baseline_name, len(train_ids), len(test_ids))
        return None

    X_train = np.stack([feature_lookup[gene] for gene in train_ids])
    y_train = np.asarray([target_lookup[gene] for gene in train_ids], dtype=np.int64)
    X_test = np.stack([feature_lookup[gene] for gene in test_ids])
    y_test = np.asarray([target_lookup[gene] for gene in test_ids], dtype=np.int64)

    unique_labels = np.unique(y_train)
    if unique_labels.size < 2:
        LOGGER.warning(
            "Fold %s for %s has a single class (label=%s); using constant baseline instead of logistic regression.",
            split.fold_id,
            baseline_name,
            unique_labels.tolist(),
        )
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        clf = DummyClassifier(strategy="constant", constant=int(unique_labels[0]))
        clf.fit(X_train_scaled, y_train)
        probs = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        clf = _make_classifier()
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]

    tp, fp, fn, tn, _, _, auc, aupr, f1, accuracy, recall, specificity, precision = compute_classification_metrics(
        y_test, probs, threshold=0.5
    )
    return {
        "model": baseline_name,
        "fold": split.fold_id,
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "auc": float(auc),
        "aupr": float(aupr),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
    }


def run_baseline_comparisons(
    config_path: str | Path,
    output_dir: str | Path,
    folds: Optional[Iterable[int]] = None,
    baselines: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    """Train and evaluate lightweight baselines for the configured dataset."""

    config = load_config(config_path)
    fold_indices = list(folds) if folds is not None else list(range(config.n_splits))
    if not fold_indices:
        raise ValueError("At least one fold must be provided.")

    gene_table = _load_gene_table(config)
    target_lookup = gene_table.set_index("GeneSymbol")["Target"].to_dict()
    sequence_lookup = gene_table.set_index("GeneSymbol")["Fasta"].to_dict()
    splits = _load_fold_splits(config, fold_indices)
    genes_in_use = sorted({gene for split in splits for gene in (split.train_ids + split.test_ids)})

    baseline_names = list(baselines) if baselines else ["esm-only", "sequence-only"]
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    summary_payload: Dict[str, Dict[str, float]] = {}
    artifact_paths: Dict[str, Path] = {}

    for baseline_name in baseline_names:
        if baseline_name == "esm-only":
            feature_lookup = _precompute_esm_lookup(Path(config.raw_data_path), genes_in_use)
        elif baseline_name == "sequence-only":
            feature_lookup = _precompute_sequence_lookup(sequence_lookup, genes_in_use)
        else:
            raise ValueError(f"Unknown baseline '{baseline_name}'")

        records = []
        for split in splits:
            record = _run_fold_evaluation(baseline_name, split, feature_lookup, target_lookup)
            if record is not None:
                records.append(record)

        if not records:
            LOGGER.warning("No valid folds were evaluated for baseline '%s'.", baseline_name)
            continue

        df = pd.DataFrame.from_records(records)
        metric_columns = [col for col in df.columns if col not in {"model", "fold"}]

        folds_only = df[df["fold"].apply(lambda value: value != "mean")]

        mean_row = {"model": baseline_name, "fold": "mean"}
        std_row = {"model": baseline_name, "fold": "std"}
        mean_stats: Dict[str, float] = {}
        std_stats: Dict[str, float] = {}
        for column in metric_columns:
            values = folds_only[column].astype(float).to_numpy()
            mean_value = float(values.mean()) if values.size else 0.0
            std_value = float(values.std(ddof=1)) if values.size > 1 else 0.0
            mean_row[column] = mean_value
            std_row[column] = std_value
            mean_stats[column] = mean_value
            std_stats[column] = std_value

        df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

        destination = output_dir / f"{baseline_name.replace(' ', '_')}_metrics.csv"
        df.to_csv(destination, index=False)
        artifact_paths[baseline_name] = destination
        summary_payload[baseline_name] = {
            "mean": mean_stats,
            "std": std_stats,
        }

    if summary_payload:
        summary_path = output_dir / "baseline_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2)
        artifact_paths["summary"] = summary_path

    return artifact_paths
