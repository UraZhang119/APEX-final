#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors as mcolors
import pandas as pd
import math


# Paleta y estilo inspirados en los t-SNE previos (naranja = malo, teal = candidatos)
TSNE_COLOR_OTHER = "#B17F0E"   # naranja
TSNE_COLOR_TARGET = "#36B7AA"  # teal
FIG_BG = "#FFFFFF"
# Gradiente simple de blanco a teal
WHITE_TEAL_CMAP = LinearSegmentedColormap.from_list("white_to_teal", ["#FFFFFF", TSNE_COLOR_TARGET], N=256)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge Botrytis pathogenesis + druggability predictions and plot a target landscape (t-SNE-like style)."
    )
    p.add_argument("--pathogenesis-csv",
                   default="inference_results/batch/botrytis_modelX/proteome_batch_predictions.csv")
    p.add_argument("--druggability-csv",
                   default="inference_results/batch/botrytis_modeldrugs/proteome_batch_predictions.csv")
    p.add_argument("--output-dir", default="metrics_analysis/botrytis_targets")

    p.add_argument("--path-threshold", type=float, default=0.6)
    p.add_argument("--drug-threshold", type=float, default=0.6)
    p.add_argument("--top-k", type=int, default=20)

    p.add_argument(
        "--fasta-filter",
        type=str,
        default="/mnt/home/users/agr_169_uma/luciajc/proteoma_botrytis_filtered_25_solanum.fasta",
        help="FASTA with protein_id entries to include in the plot.",
    )

    # Apariencia limpia (mismo look&feel que los t-SNE)
    p.add_argument("--point-size", type=float, default=48.0)
    p.add_argument("--alpha", type=float, default=0.78, help="Point transparency (0–1).")
    return p.parse_args()


def _load_predictions(csv_path: Path, prob_label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"protein_id", "probability"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns {required}")
    return (
        df[["protein_id", "probability"]]
        .rename(columns={"probability": prob_label})
        .drop_duplicates(subset="protein_id", keep="first")
    )


def _load_fasta_ids(fasta_path: Path) -> set[str]:
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    ids: set[str] = set()
    with fasta_path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                protein_id = line[1:].strip().split()[0]
                if protein_id:
                    ids.add(protein_id)

    if not ids:
        raise ValueError(f"No protein_id entries found in {fasta_path}")
    return ids


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    fasta_ids = _load_fasta_ids(Path(args.fasta_filter))

    path_df = _load_predictions(Path(args.pathogenesis_csv), "pathogenesis_prob")
    drug_df = _load_predictions(Path(args.druggability_csv), "druggability_prob")
    merged = path_df.merge(drug_df, on="protein_id", how="inner")

    merged["target_score"] = merged["pathogenesis_prob"] * merged["druggability_prob"]
    merged.sort_values("target_score", ascending=False, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    before_filter = len(merged)
    merged = merged[merged["protein_id"].isin(fasta_ids)].copy()
    if merged.empty:
        raise ValueError(
            f"Empty merge after filtering with IDs from {args.fasta_filter}. "
            f"Proteins before filter: {before_filter}"
        )

    mask_candidates = (merged["pathogenesis_prob"] >= args.path_threshold) & (
        merged["druggability_prob"] >= args.drug_threshold
    )
    candidates_df = merged[mask_candidates]
    background_df = merged[~mask_candidates]

    # Export
    merged_path = outdir / "botrytis_target_scores.csv"
    merged.to_csv(merged_path, index=False)
    top_k_path = outdir / f"botrytis_top_{args.top_k}_targets.csv"
    merged.head(args.top_k).to_csv(top_k_path, index=False)

    # ===== Plot estilo t-SNE =====
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(13, 10), dpi=200)
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(FIG_BG)

    cmap = WHITE_TEAL_CMAP
    # Centrar el rango del colorbar en la distribución real (reduce saltos bruscos)
    vmin = float(merged["target_score"].quantile(0.05))
    vmax = float(merged["target_score"].quantile(0.98))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Outside thresholds: colored by target_score (orange -> yellow -> teal)
    sc = ax.scatter(
        background_df["pathogenesis_prob"].values,
        background_df["druggability_prob"].values,
        c=background_df["target_score"].values,
        cmap=cmap,
        norm=norm,
        s=args.point_size * 1.1,
        alpha=max(args.alpha - 0.15, 0.35),
        edgecolor="none",
        rasterized=True,
        label=f"Outside thresholds (n={len(background_df)})",
    )

    # Inside thresholds: solid teal (same as t-SNE style)
    ax.scatter(
        candidates_df["pathogenesis_prob"].values,
        candidates_df["druggability_prob"].values,
        s=args.point_size * 1.25,
        color=TSNE_COLOR_TARGET,
        alpha=min(args.alpha + 0.12, 1.0),
        edgecolor="none",
        rasterized=True,
        label=f"Inside thresholds (n={len(candidates_df)})",
    )

    # Highlight top-5 by target_score (stars, no labels)
    top_n = min(5, len(merged))
    if top_n > 0:
        top_df = merged.head(top_n)
        star_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        ax.scatter(
            top_df["pathogenesis_prob"].values,
            top_df["druggability_prob"].values,
            s=args.point_size * 8.0,
            marker="*",
            c=star_colors[:top_n],
            edgecolor="#FFFFFF",
            linewidth=1.2,
            zorder=5,
            label=None,
        )
        legend_handles = []
        legend_labels = []
        for color, (_, row) in zip(star_colors, top_df.iterrows()):
            legend_handles.append(
                plt.Line2D([0], [0], marker="*", color="none", markerfacecolor=color,
                           markeredgecolor="#FFFFFF", markeredgewidth=1.2, markersize=18, linestyle="None")
            )
            legend_labels.append(row["protein_id"])
        ax.legend(
            legend_handles,
            legend_labels,
            title="Top targets",
            frameon=False,
            loc="lower right",
            bbox_to_anchor=(1.0, 0.02),
            fontsize=14,
            title_fontsize=15,
            handletextpad=0.4,
        )

    ax.axvline(args.path_threshold, color="#444444", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.axhline(args.drug_threshold, color="#444444", linestyle="--", linewidth=1.1, alpha=0.8)

    ax.set_xlabel("Pathogenesis probability", fontsize=18)
    ax.set_ylabel("Druggability probability", fontsize=18)
    ax.set_title("Botrytis candidate landscape (inside box = teal)", fontsize=24, pad=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="both", labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("Target score (low \u2192 high)", fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    plot_path = outdir / "botrytis_path_vs_drug_tsne_style.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=350)
    plt.close(fig)

    print(f"[INFO] Allowed IDs in FASTA: {len(fasta_ids)}")
    print(f"[INFO] Proteins after FASTA filter: {len(merged)} (before {before_filter})")
    print(f"[INFO] Saved merged scores to {merged_path}")
    print(f"[INFO] Saved top-{args.top_k} list to {top_k_path}")
    print(f"[INFO] Saved scatter plot to {plot_path}")


if __name__ == "__main__":
    main()
