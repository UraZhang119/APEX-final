from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import subgraph

from ..config import PipelineConfig, load_config
from ..data.esm import embed_sequence, load_esm_model
from ..inference.single import load_checkpoint_metadata
from ..models import GATNet
from ..utils import cmap_to_graph, ensure_dir


@dataclass
class ExplainerSummary:
    base_logit: float
    base_prob: float
    base_pred: int
    node_mask: np.ndarray
    top_indices: np.ndarray
    ks: Sequence[int]
    deletion_logits: Sequence[float]
    insertion_logits: Sequence[float]
    deletion_auc: float
    insertion_auc: float
    hard_remove_logit: float
    hard_keep_logit: float
    soft_remove_logit: float
    soft_keep_logit: float
    consistent_hard: bool
    consistent_soft: bool
    output_dir: Path


class ExplainerWrapper(nn.Module):
    """Thin wrapper so torch_geometric Explainer can call the model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch=None, **kwargs):  # type: ignore[override]
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        class _Wrapper:
            def __init__(self, x, edge_index, batch):
                self.x = x
                self.edge_index = edge_index
                self.batch = batch

        data = _Wrapper(x, edge_index, batch)
        return self.model(data)


def _set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _zero_out_nodes(data: Data, mask: torch.Tensor) -> Data:
    d = Data(x=data.x.clone(), edge_index=data.edge_index.clone(), batch=data.batch.clone())
    d.x[mask] = 0.0
    return d


def _subgraph_keep_nodes(data: Data, keep_mask: torch.Tensor) -> Data:
    keep_idx = torch.nonzero(keep_mask, as_tuple=False).view(-1)
    if keep_idx.numel() == 0:
        empty = Data(
            x=torch.empty((0, data.x.size(1)), device=data.x.device),
            edge_index=torch.empty((2, 0), dtype=torch.long, device=data.x.device),
            batch=torch.empty((0,), dtype=torch.long, device=data.x.device),
        )
        return empty
    new_edge_index, _ = subgraph(keep_idx, data.edge_index, relabel_nodes=True, num_nodes=data.x.size(0))
    new_x = data.x[keep_idx]
    new_batch = data.batch[keep_idx]
    return Data(x=new_x, edge_index=new_edge_index, batch=new_batch)


def _model_logit(model: nn.Module, data: Data) -> float:
    with torch.no_grad():
        logits = model(data)
    return float(logits.item())


def _insertion_deletion_curves(
    model: nn.Module,
    data: Data,
    order_idx: np.ndarray,
    steps: int,
    device: torch.device,
) -> tuple[list[int], list[float], list[float], float, float]:
    num_nodes = data.x.size(0)
    ks = np.linspace(0, len(order_idx), steps, dtype=int)

    deletion_vals: list[float] = []
    insertion_vals: list[float] = []

    for k in ks:
        del_mask = torch.zeros(num_nodes, dtype=torch.bool)
        if k > 0:
            del_mask[torch.from_numpy(order_idx[:k])] = True
        data_del = _zero_out_nodes(data.to(device), del_mask.to(device))
        deletion_vals.append(_model_logit(model, data_del))

        ins_mask = torch.ones(num_nodes, dtype=torch.bool)
        if k > 0:
            ins_mask[torch.from_numpy(order_idx[:k])] = False
        data_ins = _zero_out_nodes(data.to(device), ins_mask.to(device))
        insertion_vals.append(_model_logit(model, data_ins))

    def _auc(values: Sequence[float]) -> float:
        if len(values) <= 1:
            return 0.0
        x = np.arange(len(values))
        return float(np.trapz(values, x) / max(1, (len(values) - 1)))

    return ks.tolist(), deletion_vals, insertion_vals, _auc(deletion_vals), _auc(insertion_vals)


def _load_model(model_path: Path, device: torch.device, drop_prob: float) -> nn.Module:
    print(f"📥 Loading model from {model_path}")
    try:
        model_obj = torch.load(model_path, map_location=device)
        if isinstance(model_obj, nn.Module):
            model = model_obj
            if isinstance(model, nn.DataParallel):
                model = model.module
            model = model.to(device).eval()
            print("✅ Model loaded as full object")
            return model
    except Exception as exc:
        print(f"ℹ️ Could not load as full object: {exc}")

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model = GATNet(esm_embeds=1280, n_heads=2, drop_prob=drop_prob, n_output=1)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"⚠️ state_dict load warnings - missing: {missing}, unexpected: {unexpected}")
    model = model.to(device).eval()
    print("✅ Model loaded from state_dict")
    return model


def _build_graph(
    sequence: str,
    sequence_id: str,
    ratio: float,
    device: torch.device,
    esm_model_embeddings: str,
    esm_model_contacts: str,
) -> Data:
    emb_bundle = load_esm_model(esm_model_embeddings)
    _, representations, _ = embed_sequence(sequence_id, sequence, emb_bundle)
    contact_bundle = load_esm_model(esm_model_contacts)
    _, _, contact_map = embed_sequence(sequence_id, sequence, contact_bundle)
    node_features, edge_index, node_index_map, seq_length = cmap_to_graph(representations, contact_map, ratio=ratio)
    x = torch.as_tensor(node_features, dtype=torch.float32)
    if edge_index.size == 0:
        edge_tensor = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_tensor = torch.as_tensor(edge_index, dtype=torch.long)
    data = Data(x=x, edge_index=edge_tensor)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    data.node_index_map = torch.as_tensor(node_index_map, dtype=torch.long)
    data.sequence_length = torch.tensor(seq_length, dtype=torch.long)
    print(
        f"✅ Graph built: {data.x.size(0)} nodes (mapped from {seq_length}), {data.edge_index.size(1)} edges"
    )
    return data.to(device)


def run_node_explainer(
    sequence: str,
    model_path: Path,
    output_name: str,
    output_dir: Path,
    ratio: float = 0.2,
    top_fraction: float = 0.1,
    steps: int = 11,
    epochs: Optional[int] = None,
    seed: int = 42,
    device: Optional[torch.device] = None,
    drop_prob: float = 0.3,
    fold_number: Optional[int] = None,
    esm_model_embeddings: Optional[str] = None,
    esm_model_contacts: Optional[str] = None,
) -> ExplainerSummary:
    """Run the node-level GNNExplainer and persist artefacts to disk."""

    _set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    metadata = load_checkpoint_metadata(model_path)
    ratio = metadata.get("ratio", ratio)
    drop_prob = metadata.get("drop_prob", drop_prob)
    esm_model_embeddings = metadata.get("esm_model_embeddings", esm_model_embeddings or "facebook/esm2_t33_650M_UR50D")
    esm_model_contacts = metadata.get("esm_model_contacts", esm_model_contacts or "facebook/esm2_t33_650M_UR50D")

    model = _load_model(model_path, device, drop_prob=drop_prob)
    graph = _build_graph(
        sequence,
        output_name,
        ratio=ratio,
        device=device,
        esm_model_embeddings=esm_model_embeddings,
        esm_model_contacts=esm_model_contacts,
    )
    node_index_map = graph.node_index_map.detach().cpu().numpy()
    sequence_length = int(graph.sequence_length.item())
    sequence_slice = sequence[:sequence_length]

    with torch.no_grad():
        base_logits = model(graph)
        base_prob = torch.sigmoid(base_logits).item()
    base_logit = float(base_logits.item())
    base_pred = int(base_prob > 0.5)
    print(f"🎯 Base prediction: class={base_pred}, prob={base_prob:.6f}, logit={base_logit:.4f}")

    wrapped_model = ExplainerWrapper(model).to(device).eval()
    explainer_epochs = epochs or (400 if base_prob > 0.95 else 200)
    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=explainer_epochs, lr=0.01, beta1=0.9, beta2=0.999, log=True),
        explanation_type="phenomenon",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=dict(mode="binary_classification", task_level="graph", return_type="raw"),
    )

    target = torch.tensor([base_pred], dtype=torch.long, device=device)
    explanation = explainer(
        graph.x,
        graph.edge_index,
        target=target,
        batch=graph.batch,
        index=0,
    )
    node_mask = explanation.node_mask.detach().cpu().numpy()
    if node_mask.ndim == 2 and node_mask.shape[1] == 1:
        node_mask = node_mask.squeeze(1)
    print(f"✅ Node mask computed: shape={node_mask.shape}, min={node_mask.min():.6f}, max={node_mask.max():.6f}")

    full_node_mask = np.zeros(sequence_length, dtype=float)
    if node_index_map.size > 0:
        full_node_mask[node_index_map] = node_mask

    order = np.argsort(node_mask)[::-1].copy()
    top_k = max(1, int(top_fraction * len(node_mask)))
    top_indices = order[:top_k]

    keep_all = torch.ones(graph.x.size(0), dtype=torch.bool, device=device)
    top_mask = torch.zeros_like(keep_all)
    top_mask[top_indices] = True

    remove_mask = keep_all.clone()
    remove_mask[top_indices] = False
    keep_only_top = torch.zeros_like(keep_all)
    keep_only_top[top_indices] = True

    data_remove_hard = _subgraph_keep_nodes(graph, remove_mask)
    data_keep_hard = _subgraph_keep_nodes(graph, keep_only_top)

    remove_logit_hard = float("nan") if data_remove_hard.x.size(0) == 0 else _model_logit(model, data_remove_hard)
    keep_logit_hard = float("nan") if data_keep_hard.x.size(0) == 0 else _model_logit(model, data_keep_hard)

    data_remove_soft = _zero_out_nodes(graph, top_mask)
    data_keep_soft = _zero_out_nodes(graph, ~top_mask)
    remove_logit_soft = _model_logit(model, data_remove_soft)
    keep_logit_soft = _model_logit(model, data_keep_soft)

    consistent_hard = (
        (not np.isnan(remove_logit_hard))
        and (not np.isnan(keep_logit_hard))
        and (remove_logit_hard <= base_logit or keep_logit_hard >= base_logit)
    )
    consistent_soft = (remove_logit_soft <= base_logit) or (keep_logit_soft >= base_logit)

    ks, del_vals, ins_vals, del_auc, ins_auc = _insertion_deletion_curves(
        model, graph, top_indices, steps, device
    )

    ensure_dir(output_dir)
    ensure_dir(output_dir / output_name)
    target_dir = (output_dir / output_name).resolve()

    original_sequence_length = len(sequence)

    summary_dict = {
        "metadata": {
            "protein_name": output_name,
            "sequence_length": original_sequence_length,
            "sequence_graph_length": sequence_length,
            "timestamp": datetime.now().isoformat(),
            "model_path": str(model_path),
            "explainer": f"GNNExplainer(node_mask_type=object, explanation_type=phenomenon, epochs={explainer_epochs})",
            "ratio_contact_map": ratio,
            "drop_prob": drop_prob,
            "esm_model_embeddings": esm_model_embeddings,
            "esm_model_contacts": esm_model_contacts,
            "fold": fold_number,
            "edges_duplicated": False,
            "topk_nodes": int(top_k),
            "seed": seed,
        },
        "prediction": {
            "class": int(base_pred),
            "probability": float(base_prob),
            "logit": float(base_logit),
        },
        "sanity_checks": {
            "HARD": {
                "logit_remove_topk": float(remove_logit_hard),
                "logit_keep_only_topk": float(keep_logit_hard),
                "prob_remove_topk": float(torch.sigmoid(torch.tensor(remove_logit_hard)).item())
                if not np.isnan(remove_logit_hard)
                else float("nan"),
                "prob_keep_only_topk": float(torch.sigmoid(torch.tensor(keep_logit_hard)).item())
                if not np.isnan(keep_logit_hard)
                else float("nan"),
                "consistent": bool(consistent_hard),
            },
            "SOFT": {
                "logit_zero_remove_topk": float(remove_logit_soft),
                "logit_zero_keep_only_topk": float(keep_logit_soft),
                "prob_zero_remove_topk": float(torch.sigmoid(torch.tensor(remove_logit_soft)).item()),
                "prob_zero_keep_only_topk": float(torch.sigmoid(torch.tensor(keep_logit_soft)).item()),
                "consistent": bool(consistent_soft),
            },
            "insertion_deletion": {
                "k": ks,
                "deletion_logits": del_vals,
                "insertion_logits": ins_vals,
                "deletion_auc": float(del_auc),
                "insertion_auc": float(ins_auc),
            },
        },
        "top_nodes": [
            {
                "rank": i + 1,
                "position": int(node_index_map[idx] + 1) if node_index_map.size > 0 else int(idx + 1),
                "amino_acid": sequence_slice[node_index_map[idx]] if node_index_map.size > 0 and node_index_map[idx] < len(sequence_slice) else "X",
                "importance": float(node_mask[idx]),
            }
            for i, idx in enumerate(top_indices)
        ],
    }

    summary_path = target_dir / f"{output_name}_nodes_explainer.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_dict, fh, indent=2)

    positions = np.arange(1, sequence_length + 1)
    amino_acids = [sequence_slice[i] if i < len(sequence_slice) else "X" for i in range(sequence_length)]
    df = pd.DataFrame(
        {
            "position": positions,
            "amino_acid": amino_acids,
            "node_importance": full_node_mask,
        }
    )
    csv_path = target_dir / f"{output_name}_nodes_importance.csv"
    df.to_csv(csv_path, index=False)

    try:
        x_vals = positions
        order_full = np.argsort(node_mask)[::-1].copy()
        top20_idx = order_full[: min(20, len(node_mask))]
        top20_vals = node_mask[top20_idx]
        top20_positions = node_index_map[top20_idx] if node_index_map.size > 0 else top20_idx

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        axes[0].plot(positions, full_node_mask)
        axes[0].set_xlabel("Residue position")
        axes[0].set_ylabel("Node importance")
        axes[0].set_title(f"Node importance profile – {output_name}")
        axes[1].bar(range(len(top20_vals)), top20_vals)
        for i, (idx, val, pos) in enumerate(zip(top20_idx, top20_vals, top20_positions)):
            aa = sequence_slice[pos] if pos < len(sequence_slice) else "X"
            axes[1].text(i, val, f"{aa}{pos + 1}", ha="center", va="bottom", fontsize=8, rotation=90)
        plt.tight_layout()
        plt.savefig(target_dir / f"{output_name}_nodes_explainer.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        fig2 = plt.figure(figsize=(10, 6))
        plt.plot(ks, del_vals, label="Deletion (logits)")
        plt.plot(ks, ins_vals, label="Insertion (logits)")
        plt.xlabel("k (top nodes)")
        plt.ylabel("Predicted class logit")
        plt.title(f"Insertion/Deletion – {output_name}\nAUC_del={del_auc:.3f} | AUC_ins={ins_auc:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(target_dir / f"{output_name}_insertion_deletion.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
    except Exception as exc:  # pragma: no cover - visualisation best effort
        print(f"⚠️ Could not generate figures: {exc}")

    print(f"📁 Artefacts stored in {target_dir}")

    return ExplainerSummary(
        base_logit=base_logit,
        base_prob=base_prob,
        base_pred=base_pred,
        node_mask=node_mask,
        top_indices=top_indices,
        ks=ks,
        deletion_logits=del_vals,
        insertion_logits=ins_vals,
        deletion_auc=del_auc,
        insertion_auc=ins_auc,
        hard_remove_logit=remove_logit_hard,
        hard_keep_logit=keep_logit_hard,
        soft_remove_logit=remove_logit_soft,
        soft_keep_logit=keep_logit_soft,
        consistent_hard=consistent_hard,
        consistent_soft=consistent_soft,
        output_dir=target_dir,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run GNNExplainer for a single protein sequence.")
    parser.add_argument("--sequence", default=None, help="Amino acid sequence to analyse.")
    parser.add_argument("--sequence-file", default=None, help="Path to a file containing the sequence.")
    parser.add_argument("--name", required=True, help="Identifier used for outputs.")
    parser.add_argument("--model-checkpoint", required=True, type=Path, help="Path to the trained model checkpoint.")
    parser.add_argument("--output-dir", default="gnn_results", type=Path, help="Directory where outputs are stored.")
    parser.add_argument("--ratio", default=0.2, type=float, help="Quantile ratio used in cmap_to_graph.")
    parser.add_argument("--top-fraction", default=0.1, type=float, help="Fraction of nodes considered top-k.")
    parser.add_argument("--steps", default=11, type=int, help="Steps for insertion/deletion curves.")
    parser.add_argument("--epochs", default=None, type=int, help="Epochs for GNNExplainer (auto if unset).")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--config", default=None, help="Optional config file to reuse defaults (ratio).")
    args = parser.parse_args(argv)

    if args.config:
        config = load_config(args.config)
        default_ratio = config.ratio
    else:
        config = PipelineConfig()
        default_ratio = config.ratio

    if args.sequence_file:
        sequence = Path(args.sequence_file).read_text().strip()
    else:
        sequence = args.sequence

    if not sequence:
        raise ValueError("Provide --sequence or --sequence-file")

    ratio = args.ratio if args.ratio is not None else default_ratio

    run_node_explainer(
        sequence=sequence,
        model_path=args.model_checkpoint,
        output_name=args.name,
        output_dir=args.output_dir,
        ratio=ratio,
        top_fraction=args.top_fraction,
        steps=args.steps,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
