#!/usr/bin/env python
"""
Script autónomo de inferencia para GAT (druggability + essentiality) 

Requisitos de instalación previos en el entorno :
  pip install torch torch-geometric fair-esm biopython numpy networkx tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_max_pool as gmp
from tqdm import tqdm


# ----------------------- Configuración de rutas por defecto -----------------------
# Aquí vamos a tener que hardcodear los checkpoints entrenados, en funcion de donde esten guardados

DRUGGABILITY_CHECKPOINT = Path("/mnt/home/users/agr_169_uma/luciajc/Repo_GAT/experiments/human/gat/fold_0/best_loss.pt")
ESSENTIALITY_CHECKPOINTS = {
    "fungi": Path("/mnt/home/users/agr_169_uma/luciajc/Repo_GAT/experiments/fungi/gat/fold_0/best_loss.pt"),
    "insect": Path("/mnt/home/users/agr_169_uma/luciajc/Repo_GAT/experiments/insect/gat/fold_0/best_loss.pt"),
    "bacteria": Path("/mnt/home/users/agr_169_uma/luciajc/Repo_GAT/experiments/bacteria/gat/fold_0/best_loss.pt"),
}

# Hiperparámetros por defecto si el .meta.json no existe
DEFAULT_RATIO = 0.2
DEFAULT_DROP_PROB = 0.3
DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"


# ----------------------- Módelo GAT -----------------------
class GATNet(torch.nn.Module):
    def __init__(self, esm_embeds: int, n_heads: int, drop_prob: float, n_output: int):
        super().__init__()
        self.drop_prob = drop_prob
        self.gcn1 = GATConv(esm_embeds, esm_embeds, heads=n_heads, dropout=drop_prob)
        self.gcn2 = GATConv(esm_embeds * n_heads, esm_embeds, dropout=drop_prob)
        self.fc_g1 = nn.Linear(esm_embeds, 16)
        self.fc_g2 = nn.Linear(16, n_output)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)
        x = self.fc_g1(x)
        x = self.relu(x)
        out = self.fc_g2(x)
        return out


# ----------------------- Utilidades de carga -----------------------
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


# ----------------------- ESM helpers ----------------------
# Parche para evitar dependencias de apex con ESM
import types  # noqa: E402
from importlib.machinery import ModuleSpec  # noqa: E402

apex_mock = types.ModuleType("apex")
apex_mock.__spec__ = ModuleSpec("apex", None)
apex_mock.normalization = types.ModuleType("apex.normalization")
apex_mock.normalization.__spec__ = ModuleSpec("apex.normalization", None)
apex_mock.normalization.FusedLayerNorm = torch.nn.LayerNorm
sys.modules["apex"] = apex_mock
sys.modules["apex.normalization"] = apex_mock.normalization


def load_esm_model(model_name: str = DEFAULT_ESM_MODEL):
    import os

    os.environ.setdefault("APEX_DISABLED", "1")
    os.environ.setdefault("APEX_FORCE_DISABLE_FUSED_LAYERNORM", "1")
    os.environ.setdefault("APEX_DISABLE_FUSED_LAYERNORM", "1")
    import esm  # type: ignore

    short_name = model_name.split("/")[-1]
    candidates = [short_name]
    alias_map = {
        "esm2_t33_650M_UR50S_2": "esm2_t33_650M_UR50S",
        "esm2_t33_650M_UR50D_2": "esm2_t33_650M_UR50D",
    }
    if short_name in alias_map:
        candidates.append(alias_map[short_name])
    base, _, suffix = short_name.rpartition("_")
    if suffix.isdigit():
        candidates.append(base)

    loader = None
    for candidate in candidates:
        loader = getattr(esm.pretrained, candidate, None)
        if loader is not None:
            break
    if loader is None:
        raise ValueError(f"Unsupported ESM model '{model_name}'")

    model, alphabet = loader()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, batch_converter


def embed_sequence(identifier: str, sequence: str, model_bundle):
    model, batch_converter = model_bundle
    device = next(model.parameters()).device
    trimmed_sequence = sequence[:1024] if len(sequence) > 1024 else sequence
    gene_ids, _, tokens = batch_converter([(identifier, trimmed_sequence)])
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[33], return_contacts=True)
        representations = results["representations"][33].squeeze(0)[1:-1, :].cpu()
        contact_map = results["contacts"].squeeze(0).cpu()

    return gene_ids[0], representations, contact_map


# ----------------------- Grafo a partir de la matriz de contactps de ESM2 -----------------------
def cmap_to_graph(
    all_features: torch.Tensor | np.ndarray,
    contact_map: torch.Tensor | np.ndarray,
    ratio: float,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, int]:
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


# ----------------------- Inference helpers -----------------------
@dataclass
class ModelRunner:
    name: str
    checkpoint: Path
    model: torch.nn.Module
    ratio: float
    drop_prob: float
    esm_model_embeddings: str
    esm_model_contacts: str
    device: torch.device


def _load_runner(name: str, checkpoint_path: Path, force_cpu: bool) -> ModelRunner:
    metadata = load_checkpoint_metadata(checkpoint_path)
    ratio = metadata.get("ratio", DEFAULT_RATIO)
    drop_prob = metadata.get("drop_prob", DEFAULT_DROP_PROB)
    esm_model_embeddings = metadata.get("esm_model_embeddings", DEFAULT_ESM_MODEL)
    esm_model_contacts = metadata.get("esm_model_contacts", DEFAULT_ESM_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    model = load_gat_model(checkpoint_path, drop_prob, device)
    return ModelRunner(
        name=name,
        checkpoint=checkpoint_path,
        model=model,
        ratio=ratio,
        drop_prob=drop_prob,
        esm_model_embeddings=esm_model_embeddings,
        esm_model_contacts=esm_model_contacts,
        device=device,
    )


def _get_esm_outputs(
    sequence_id: str,
    sequence: str,
    model_name: str,
    cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if model_name not in cache:
        bundle = load_esm_model(model_name)
        _, features, contact_map = embed_sequence(sequence_id, sequence, bundle)
        cache[model_name] = (features, contact_map)
    return cache[model_name]


def _build_graph(sequence_id: str, sequence: str, runner: ModelRunner, cache) -> Tuple[Data, int]:
    features, _ = _get_esm_outputs(sequence_id, sequence, runner.esm_model_embeddings, cache)
    _, contact_map = _get_esm_outputs(sequence_id, sequence, runner.esm_model_contacts, cache)

    node_features, edge_index, node_index_map, seq_length = cmap_to_graph(features, contact_map, ratio=runner.ratio)
    graph = Data(
        x=torch.as_tensor(node_features, dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        y=torch.tensor([0.0], dtype=torch.float32),
    )
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    graph.node_index_map = torch.as_tensor(node_index_map, dtype=torch.long)
    graph.sequence_length = torch.tensor(seq_length, dtype=torch.long)
    return graph, seq_length


def _predict(graph: Data, runner: ModelRunner) -> Tuple[float, int]:
    graph = graph.to(runner.device)
    with torch.no_grad():
        logits = runner.model(graph)
        probability = torch.sigmoid(logits).view(-1)[0].item()
    return float(probability), int(probability >= 0.5)


def _resolve_checkpoint(path_str: str | None, default_path: Path) -> Path:
    return Path(path_str).expanduser().resolve() if path_str else default_path


# ----------------------- CLI -----------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer druggability + essentiality sobre un FASTA con modelos GAT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fasta", required=True, help="FASTA con proteínas.")
    parser.add_argument(
        "--organism",
        choices=sorted(ESSENTIALITY_CHECKPOINTS.keys()),
        default="fungi",
        help="Organismo para el modelo de essentiality.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="TSV de salida (por defecto inference_results/production/<fasta>__predictions.tsv).",
    )
    parser.add_argument(
        "--druggability-checkpoint",
        default=None,
        help="Ruta al checkpoint de druggability (si se quiere sobreescribir el hardcodeado).",
    )
    parser.add_argument(
        "--essentiality-checkpoint",
        default=None,
        help="Ruta al checkpoint de essentiality (para el organismo elegido).",
    )
    parser.add_argument("--cpu", action="store_true", help="Forzar CPU aunque haya GPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fasta_path = Path(args.fasta).expanduser().resolve()
    if not fasta_path.exists():
        raise SystemExit(f"No se encontró el FASTA: {fasta_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path.cwd() / "inference_results" / "production" / f"{fasta_path.stem}__predictions.tsv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    druggability_checkpoint = _resolve_checkpoint(args.druggability_checkpoint, DRUGGABILITY_CHECKPOINT)
    essentiality_checkpoint = _resolve_checkpoint(args.essentiality_checkpoint, ESSENTIALITY_CHECKPOINTS[args.organism])
    if not druggability_checkpoint.exists():
        raise SystemExit(f"No se encontró el checkpoint de druggability: {druggability_checkpoint}")
    if not essentiality_checkpoint.exists():
        raise SystemExit(f"No se encontró el checkpoint de essentiality: {essentiality_checkpoint}")

    drug_runner = _load_runner("druggability", druggability_checkpoint, args.cpu)
    ess_runner = _load_runner(f"essentiality_{args.organism}", essentiality_checkpoint, args.cpu)

    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        raise SystemExit(f"FASTA vacío: {fasta_path}")

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "protein_id",
                "sequence_length",
                "druggability_prediction",
                "druggability_probability",
                "essentiality_prediction",
                "essentiality_probability",
                "target_score",
                "essentiality_organism",
                "druggability_checkpoint",
                "essentiality_checkpoint",
            ]
        )

        for record in tqdm(records, desc="Running inference"):
            sequence = str(record.seq)
            seq_id = record.id
            if not sequence:
                continue

            # Cachea ESM por modelo para reusar embeddings/contactos entre druggability y essentiality
            esm_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

            drug_graph, seq_length = _build_graph(seq_id, sequence, drug_runner, esm_cache)
            drug_prob, drug_pred = _predict(drug_graph, drug_runner)

            ess_graph, _ = _build_graph(seq_id, sequence, ess_runner, esm_cache)
            ess_prob, ess_pred = _predict(ess_graph, ess_runner)

            target_score = drug_prob * ess_prob

            writer.writerow(
                [
                    seq_id,
                    seq_length,
                    drug_pred,
                    drug_prob,
                    ess_pred,
                    ess_prob,
                    target_score,
                    args.organism,
                    str(druggability_checkpoint),
                    str(essentiality_checkpoint),
                ]
            )

    print(f"[OK] Predicciones guardadas en {output_path}")


if __name__ == "__main__":
    main()
