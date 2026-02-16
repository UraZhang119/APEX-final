from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import networkx as nx
import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_pt(file_path: str | Path) -> dict:
    """Load PyTorch objects handling the weights_only restriction."""

    file_path = Path(file_path)
    try:
        return torch.load(file_path)
    except Exception as exc:  # pragma: no cover - fallback path
        if "weights_only" in str(exc) or "UnpicklingError" in str(exc):
            return torch.load(file_path, weights_only=False)
        raise


def cmap_to_graph(
    all_features: torch.Tensor | np.ndarray,
    contact_map: torch.Tensor | np.ndarray,
    ratio: float,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, int]:
    """Convert a contact map into node features and an edge index list.

    Returns the node features, undirected edge index, the mapping from graph nodes
    to the original sequence indices and the original (pre-filtering) sequence length.
    """

    cmap = contact_map.detach().cpu().numpy() if isinstance(contact_map, torch.Tensor) else np.asarray(contact_map)
    features = all_features.detach().cpu() if isinstance(all_features, torch.Tensor) else torch.tensor(all_features)
    cmap_arr = cmap.flatten()
    threshold = np.quantile(cmap_arr, 1 - ratio)

    binary_cmap = np.zeros_like(cmap)
    binary_cmap[cmap > threshold] = 1
    np.fill_diagonal(binary_cmap, 0)

    row_sums = binary_cmap.sum(axis=1)
    nonzero_ids = np.where(row_sums != 0)[0]
    node_features = features[nonzero_ids]
    zero_ids = np.where(row_sums == 0)[0]

    if len(zero_ids) != 0:
        filtered = np.delete(binary_cmap, zero_ids, axis=0)
        binary_cmap = np.delete(filtered, zero_ids, axis=1)

    row_indices, col_indices = np.where(binary_cmap == 1)
    edges = list(zip(row_indices, col_indices))
    graph = nx.Graph()
    graph.add_edges_from(edges)
    if graph.number_of_edges() == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        edge_pairs = np.array(graph.edges, dtype=np.int64)
        undirected_pairs = np.concatenate((edge_pairs, edge_pairs[:, ::-1]), axis=0)
        edge_index = undirected_pairs.T
    original_length = cmap.shape[0]
    return node_features, edge_index, nonzero_ids, original_length


def compute_classification_metrics(
    real_score: Sequence[float] | np.ndarray,
    predict_score: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> Tuple[int, int, int, int, float, float, float, float, float, float, float, float]:
    """Return confusion matrix and classic metrics using a fixed threshold."""

    from sklearn.metrics import average_precision_score, roc_auc_score

    real_arr = np.asarray(real_score).flatten()
    pred_arr = np.asarray(predict_score).flatten()

    if real_arr.size == 0 or pred_arr.size == 0:
        LOGGER.warning("Empty predictions or labels. Returning zeros.")
        return 0, 0, 0, 0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    predict_binary = (pred_arr >= threshold).astype(int)

    tp = int(np.sum((real_arr == 1) & (predict_binary == 1)))
    fp = int(np.sum((real_arr == 0) & (predict_binary == 1)))
    tn = int(np.sum((real_arr == 0) & (predict_binary == 0)))
    fn = int(np.sum((real_arr == 1) & (predict_binary == 0)))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    try:
        auc = float(roc_auc_score(real_arr, pred_arr))
    except ValueError:
        auc = 0.5
    try:
        aupr = float(average_precision_score(real_arr, pred_arr))
    except ValueError:
        aupr = 0.5

    fpr_dummy = [0.0]
    tpr_dummy = [0.0]
    return tp, fp, fn, tn, fpr_dummy, tpr_dummy, auc, aupr, f1, accuracy, recall, specificity, precision
