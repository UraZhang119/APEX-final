from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from ..config import PipelineConfig
from ..utils import ensure_dir
from .esm import embed_sequence, load_esm_model


def _trim_sequences(sequences: Iterable[str], trim_threshold: int) -> List[str]:
    trimmed = []
    for seq in sequences:
        trimmed.append(seq[:trim_threshold] if len(seq) > trim_threshold else seq)
    return trimmed


def generate_embeddings(config: PipelineConfig) -> None:
    """Generate ESM-2 embeddings and contact maps for the configured dataset."""

    ensure_dir(config.raw_data_path)

    gene_list_path = config.root_path / config.species / "orig_sample_list" / "gene_list.txt"
    if not gene_list_path.exists():
        raise FileNotFoundError(f"Missing gene list file at {gene_list_path}")

    orig_data = pd.read_csv(gene_list_path, sep="\t")
    genes = list(orig_data["Ensembl"].values)
    fastas = list(orig_data["Fasta"].values)
    targets = list(orig_data["Target"].values)
    trimmed_fastas = _trim_sequences(fastas, config.trim_thresh)
    packages: List[Tuple[str, str, int]] = list(zip(genes, trimmed_fastas, targets))

    model_bundle_embeddings = load_esm_model(config.esm_model_embeddings)
    model_bundle_contacts = load_esm_model(config.esm_model_contacts)

    for gene_id, sequence, target in tqdm(packages, desc="Embedding sequences"):
        _, representations, _ = embed_sequence(gene_id, sequence, model_bundle_embeddings)
        _, _, contact_map = embed_sequence(gene_id, sequence, model_bundle_contacts)
        gene_dict = {
            "gene_ensembl": gene_id,
            "feature_representation": representations,
            "cmap": contact_map,
            "target": target,
        }
        torch.save(gene_dict, config.raw_data_path / f"{gene_id}.pt")
