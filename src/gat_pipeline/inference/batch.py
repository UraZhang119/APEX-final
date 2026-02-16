from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import torch
from Bio import SeqIO
from tqdm import tqdm

from ..config import PipelineConfig
from .single import InferenceResult, infer_sequence


def infer_fasta(
    fasta_path: Path,
    checkpoint_path: Path,
    config: PipelineConfig,
    output_csv: Path,
) -> List[InferenceResult]:
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    device = torch.device(config.cuda_name if torch.cuda.is_available() else "cpu")

    results: List[InferenceResult] = []

    for record in tqdm(records, desc="Running inference"):
        sequence = str(record.seq)
        result = infer_sequence(sequence, record.id, checkpoint_path, config, device)
        results.append(result)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["protein_id", "sequence_length", "prediction", "probability"])
        for res in results:
            writer.writerow([res.sequence_id, res.sequence_length, res.prediction, res.probability])

    return results
