from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool as gmp

from ..config import PipelineConfig
from ..data.esm import embed_sequence, load_esm_model
from ..models import GATNet
from ..utils import cmap_to_graph


@dataclass
class InferenceResult:
    sequence_id: str
    probability: float
    prediction: int
    sequence_length: int


class GATNetWithAttention(torch.nn.Module):
    def __init__(self, trained_model: GATNet):
        super().__init__()
        self.gat_model = trained_model

    def forward_with_attention(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.gat_model.drop_prob, training=self.gat_model.training)
        x1, attn1 = self.gat_model.gcn1(x, edge_index, return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.gat_model.drop_prob, training=self.gat_model.training)
        x2, attn2 = self.gat_model.gcn2(x1, edge_index, return_attention_weights=True)
        x2 = self.gat_model.relu(x2)
        x2 = gmp(x2, batch)
        x2 = self.gat_model.fc_g1(x2)
        x2 = self.gat_model.relu(x2)
        out = self.gat_model.fc_g2(x2)
        return out, attn1, attn2


def _aggregate_node_attention(
    edge_index: torch.Tensor, attention_weights: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    if attention_weights.dim() > 1:
        attention = attention_weights.mean(dim=-1)
    else:
        attention = attention_weights
    attention = attention.detach()
    edge_index = edge_index.to(attention.device)
    dst_nodes = edge_index[1]
    scores = torch.zeros(num_nodes, device=attention.device)
    counts = torch.zeros(num_nodes, device=attention.device)
    scores.index_add_(0, dst_nodes, attention)
    counts.index_add_(0, dst_nodes, torch.ones_like(attention))
    counts = torch.clamp(counts, min=1)
    return (scores / counts).cpu()


def _build_graph(
    sequence_id: str,
    sequence: str,
    ratio: float,
    esm_model_embeddings: str,
    esm_model_contacts: str,
):
    emb_bundle = load_esm_model(esm_model_embeddings)
    _, features, _ = embed_sequence(sequence_id, sequence, emb_bundle)
    contact_bundle = load_esm_model(esm_model_contacts)
    _, _, contact_map = embed_sequence(sequence_id, sequence, contact_bundle)
    node_features, edge_index, node_index_map, seq_length = cmap_to_graph(features, contact_map, ratio=ratio)
    graph_data = Data(
        x=torch.as_tensor(node_features, dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        y=torch.tensor([0.0], dtype=torch.float32),
    )
    graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
    graph_data.node_index_map = torch.as_tensor(node_index_map, dtype=torch.long)
    graph_data.sequence_length = torch.tensor(seq_length, dtype=torch.long)
    return graph_data


def load_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, object]:
    meta_path = checkpoint_path.with_name(checkpoint_path.name + ".meta.json")
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def load_gat_model(checkpoint_path: Path, drop_prob: float, device: torch.device) -> GATNet:
    model = GATNet(esm_embeds=1280, n_heads=2, drop_prob=drop_prob, n_output=1)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference_with_outputs(
    sequence: str,
    sequence_id: str,
    checkpoint_path: Path,
    config: PipelineConfig,
    fold_number: int,
    output_base: Path,
) -> Dict[str, object]:
    device = torch.device(config.cuda_name if torch.cuda.is_available() else "cpu")
    metadata = load_checkpoint_metadata(checkpoint_path)
    ratio = metadata.get("ratio", config.ratio)
    drop_prob = metadata.get("drop_prob", config.drop_prob)
    esm_model_embeddings = metadata.get("esm_model_embeddings", config.esm_model_embeddings)
    esm_model_contacts = metadata.get("esm_model_contacts", config.esm_model_contacts)

    graph = _build_graph(sequence_id, sequence, ratio, esm_model_embeddings, esm_model_contacts)
    loader = DataLoader([graph], batch_size=1, shuffle=False)

    model = load_gat_model(checkpoint_path, drop_prob, device)
    model_wrapper = GATNetWithAttention(model)

    batch = next(iter(loader)).to(device)
    sequence_length_tensor = batch.sequence_length
    if isinstance(sequence_length_tensor, torch.Tensor):
        sequence_length = int(sequence_length_tensor.view(-1)[0].item())
    else:
        sequence_length = int(sequence_length_tensor)
    node_index_map = batch.node_index_map if hasattr(batch, "node_index_map") else None

    with torch.no_grad():
        logits, att1, att2 = model_wrapper.forward_with_attention(batch)
        probability = torch.sigmoid(logits).view(-1)[0].item()
        prediction = int(probability >= 0.5)

    edge_index1, weights1 = att1
    edge_index2, weights2 = att2
    num_nodes = batch.num_nodes
    att_layer1 = _aggregate_node_attention(edge_index1.cpu(), weights1.cpu(), num_nodes).numpy()
    att_layer2 = _aggregate_node_attention(edge_index2.cpu(), weights2.cpu(), num_nodes).numpy()

    full_layer1 = np.zeros(sequence_length, dtype=np.float32)
    full_layer2 = np.zeros(sequence_length, dtype=np.float32)
    if node_index_map is not None and node_index_map.numel() > 0:
        original_indices = node_index_map.detach().cpu().numpy()
        full_layer1[original_indices] = att_layer1
        full_layer2[original_indices] = att_layer2
    total_attention = full_layer1 + full_layer2

    sequence_slice = list(sequence[:sequence_length])
    positions = np.arange(1, sequence_length + 1)

    residue_df = pd.DataFrame(
        {
            "position": positions,
            "amino_acid": sequence_slice,
            "attention_layer1": full_layer1,
            "attention_layer2": full_layer2,
            "total_attention": total_attention,
        }
    )

    output_dir = output_base / sequence_id
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "protein_name": sequence_id,
        "sequence_length": len(sequence),
        "sequence_graph_length": sequence_length,
        "fold": fold_number,
        "checkpoint": str(checkpoint_path),
        "ratio": ratio,
        "drop_prob": drop_prob,
        "esm_model_embeddings": esm_model_embeddings,
        "esm_model_contacts": esm_model_contacts,
        "probability": float(probability),
        "prediction": prediction,
    }

    with (output_dir / f"{sequence_id}_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    residue_df.to_csv(output_dir / f"{sequence_id}_residue_attention.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(residue_df["position"], residue_df["total_attention"], color="tab:blue")
    plt.xlabel("Residue position")
    plt.ylabel("Attention score")
    plt.title(f"Attention profile – {sequence_id}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{sequence_id}_attention_profile.png", dpi=300, bbox_inches="tight")
    plt.close()

    return summary


def infer_sequence(
    sequence: str,
    sequence_id: str,
    checkpoint_path: Path,
    config: PipelineConfig,
    device: Optional[torch.device] = None,
) -> InferenceResult:
    device = device or torch.device(config.cuda_name if torch.cuda.is_available() else "cpu")
    metadata = load_checkpoint_metadata(checkpoint_path)
    ratio = metadata.get("ratio", config.ratio)
    drop_prob = metadata.get("drop_prob", config.drop_prob)
    esm_model_embeddings = metadata.get("esm_model_embeddings", config.esm_model_embeddings)
    esm_model_contacts = metadata.get("esm_model_contacts", config.esm_model_contacts)

    graph = _build_graph(sequence_id, sequence, ratio, esm_model_embeddings, esm_model_contacts)
    loader = DataLoader([graph], batch_size=1, shuffle=False)

    model = load_gat_model(checkpoint_path, drop_prob, device)
    model_wrapper = GATNetWithAttention(model)

    with torch.no_grad():
        batch = next(iter(loader)).to(device)
        logits, *_ = model_wrapper.forward_with_attention(batch)
        probability = torch.sigmoid(logits).view(-1)[0].item()
        prediction = int(probability >= 0.5)

    return InferenceResult(
        sequence_id=sequence_id,
        probability=float(probability),
        prediction=prediction,
        sequence_length=len(sequence),
    )
