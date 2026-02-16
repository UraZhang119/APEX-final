from __future__ import annotations

from pathlib import Path

import pandas as pd
from Bio import SeqIO

from ..utils import ensure_dir


def convert_fasta_to_bingo_format(
    pathogenesis_fasta: str | Path,
    non_pathogenesis_fasta: str | Path,
    output_dir: str | Path,
    species: str | None = None,
) -> pd.DataFrame:
    """Convert FASTA files into the structure expected by the training pipeline."""

    output_dir = Path(output_dir)
    species_name = species or output_dir.name or "samples"
    ensure_dir(output_dir / "orig_sample_list")
    ensure_dir(output_dir / "raw")
    ensure_dir(output_dir / "kfold_splitted_data")

    all_sequences = []

    for record in SeqIO.parse(str(pathogenesis_fasta), "fasta"):
        all_sequences.append(
            {
                "Ensembl": record.id,
                "GeneSymbol": record.id,
                "Fasta": str(record.seq),
                "Target": 1,
            }
        )

    for record in SeqIO.parse(str(non_pathogenesis_fasta), "fasta"):
        all_sequences.append(
            {
                "Ensembl": record.id,
                "GeneSymbol": record.id,
                "Fasta": str(record.seq),
                "Target": 0,
            }
        )

    df = pd.DataFrame(all_sequences)
    gene_list_path = output_dir / "orig_sample_list" / "gene_list.txt"
    df.to_csv(gene_list_path, sep="\t", index=False)

    pathogenesis_df = df[df["Target"] == 1][["GeneSymbol"]]
    non_pathogenesis_df = df[df["Target"] == 0][["GeneSymbol"]]

    essential_name = f"{species_name}_Essential_Genes.xlsx"
    nonessential_name = f"{species_name}_NonEssential_Genes.xlsx"

    pathogenesis_df.to_excel(output_dir / "orig_sample_list" / essential_name, index=False)
    non_pathogenesis_df.to_excel(output_dir / "orig_sample_list" / nonessential_name, index=False)

    return df
