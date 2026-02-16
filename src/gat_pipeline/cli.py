from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .config import PipelineConfig, load_config
from .data import (
    build_fold_graphs,
    convert_fasta_to_bingo_format,
    generate_embeddings,
    generate_kfold_splits,
)
from .explain import run_node_explainer
from .inference import infer_fasta, infer_sequence
from .training import train_fold
from .visualization import plot_attention_and_importance


def _parse_int_list(raw: Optional[str]) -> Optional[list[int]]:
    if not raw:
        return None
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            continue
    return values or None


def _load_config(path: Optional[str]) -> PipelineConfig:
    return load_config(path)


def _prepare_data(args: argparse.Namespace) -> None:
    config = _load_config(args.config)

    if args.pathogenesis_fasta and args.non_pathogenesis_fasta and not args.skip_dataset:
        convert_fasta_to_bingo_format(
            args.pathogenesis_fasta,
            args.non_pathogenesis_fasta,
            config.root_path / config.species,
            species=config.species,
        )

    if not args.skip_embeddings:
        generate_embeddings(config)

    if not args.skip_splits:
        generate_kfold_splits(config)

    if not args.skip_graphs:
        build_fold_graphs(config)


def _train_fold(args: argparse.Namespace) -> None:
    overrides = {}
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.drop_prob is not None:
        overrides["drop_prob"] = args.drop_prob
    if args.weight_decay is not None:
        overrides["weight_decay"] = args.weight_decay
    if args.ratio is not None:
        overrides["ratio"] = args.ratio
    if args.train_batch_size is not None:
        overrides["train_batch_size"] = args.train_batch_size
    if args.test_batch_size is not None:
        overrides["test_batch_size"] = args.test_batch_size

    config = load_config(args.config, overrides if overrides else None)
    if args.model and args.model not in {"gat", "gcn", "sage"}:
        raise ValueError("Model must be one of: gat, gcn, sage")
    summary = train_fold(
        config,
        fold=args.fold,
        model_name=args.model,
        use_wandb=False if args.no_wandb else None,
        wandb_run_name=args.wandb_run_name,
    )
    payload = {
        "best_aupr_path": str(summary.best_aupr_path) if summary.best_aupr_path else None,
        "best_loss_path": str(summary.best_loss_path) if summary.best_loss_path else None,
    }
    print(json.dumps(payload, indent=2))


def _infer_sequence(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    if args.sequence_file:
        sequence = Path(args.sequence_file).read_text().strip()
    else:
        sequence = args.sequence
    if not sequence:
        raise ValueError("Provide --sequence or --sequence-file")
    result = infer_sequence(
        sequence=sequence,
        sequence_id=args.name,
        checkpoint_path=Path(args.model_checkpoint),
        config=config,
    )
    print(json.dumps(result.__dict__, indent=2))


def _infer_fasta(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    results = infer_fasta(
        fasta_path=Path(args.fasta),
        checkpoint_path=Path(args.model_checkpoint),
        config=config,
        output_csv=Path(args.output),
    )
    print(f"Wrote {len(results)} predictions to {args.output}")


def _explain_nodes(args: argparse.Namespace) -> None:
    config = _load_config(args.config) if args.config else None
    if args.sequence_file:
        sequence = Path(args.sequence_file).read_text().strip()
    else:
        sequence = args.sequence
    if not sequence:
        raise ValueError("Provide --sequence or --sequence-file")
    ratio = args.ratio
    if ratio is None and config is not None:
        ratio = config.ratio
    ratio = ratio if ratio is not None else 0.2
    drop_prob = config.drop_prob if config is not None else 0.3
    run_node_explainer(
        sequence=sequence,
        model_path=Path(args.model_checkpoint),
        output_name=args.name,
        output_dir=Path(args.output_dir),
        ratio=ratio,
        top_fraction=args.top_fraction,
        steps=args.steps,
        epochs=args.epochs,
        seed=args.seed,
        drop_prob=drop_prob,
        esm_model_embeddings=config.esm_model_embeddings if config else "facebook/esm2_t33_650M_UR50D",
        esm_model_contacts=config.esm_model_contacts if config else "facebook/esm2_t33_650M_UR50D",
    )


def _plot_attention(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    if args.sequence_file:
        sequence = Path(args.sequence_file).read_text().strip()
    else:
        sequence = args.sequence
    if not sequence:
        raise ValueError("Provide --sequence or --sequence-file")
    pocket_list = _parse_int_list(args.active_site_pockets)
    line_path, contact_path, contact_alt_path = plot_attention_and_importance(
        sequence=sequence,
        protein_name=args.name,
        checkpoint_path=Path(args.model_checkpoint),
        config=config,
        fold_number=args.fold,
        inference_dir=Path(args.inference_dir),
        explain_dir=Path(args.explain_dir),
        output_dir=Path(args.output_dir),
        top_fraction=args.top_fraction,
        explainer_steps=args.steps,
        explainer_epochs=args.epochs,
        explainer_seed=args.seed,
        annotation_path=Path(args.annotation_json) if args.annotation_json else None,
        annotation_name=args.annotation_name,
        active_site_zip=Path(args.active_site_zip) if args.active_site_zip else None,
        active_site_pockets=pocket_list,
    )
    print(f"Saved line figure to {line_path}")
    print(f"Saved contact map (teal) to {contact_path}")
    print(f"Saved contact map (diverging) to {contact_alt_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gat-pipeline", description="Utilities for the GAT fungal pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-data", help="Prepare embeddings, splits and graphs")
    prepare_parser.add_argument("--config", default="configs/fungi.yaml")
    prepare_parser.add_argument("--pathogenesis-fasta", default=None)
    prepare_parser.add_argument("--non-pathogenesis-fasta", default=None)
    prepare_parser.add_argument("--skip-dataset", action="store_true")
    prepare_parser.add_argument("--skip-embeddings", action="store_true")
    prepare_parser.add_argument("--skip-splits", action="store_true")
    prepare_parser.add_argument("--skip-graphs", action="store_true")
    prepare_parser.set_defaults(func=_prepare_data)

    train_parser = subparsers.add_parser("train-fold", help="Train a specific fold")
    train_parser.add_argument("--config", default="configs/fungi.yaml")
    train_parser.add_argument("--fold", type=int, required=True)
    train_parser.add_argument("--model", default=None)
    train_parser.add_argument("--no-wandb", action="store_true")
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument("--drop-prob", type=float, default=None)
    train_parser.add_argument("--weight-decay", type=float, default=None)
    train_parser.add_argument("--ratio", type=float, default=None)
    train_parser.add_argument("--train-batch-size", type=int, default=None)
    train_parser.add_argument("--test-batch-size", type=int, default=None)
    train_parser.add_argument("--wandb-run-name", default=None)
    train_parser.set_defaults(func=_train_fold)

    single_parser = subparsers.add_parser("infer-sequence", help="Infer a single protein sequence")
    single_parser.add_argument("--config", default="configs/fungi.yaml")
    single_parser.add_argument("--sequence", default=None)
    single_parser.add_argument("--sequence-file", default=None)
    single_parser.add_argument("--name", required=True)
    single_parser.add_argument("--model-checkpoint", required=True)
    single_parser.set_defaults(func=_infer_sequence)

    batch_parser = subparsers.add_parser("infer-fasta", help="Infer all sequences in a FASTA file")
    batch_parser.add_argument("--config", default="configs/fungi.yaml")
    batch_parser.add_argument("--fasta", required=True)
    batch_parser.add_argument("--model-checkpoint", required=True)
    batch_parser.add_argument("--output", required=True)
    batch_parser.set_defaults(func=_infer_fasta)

    explain_parser = subparsers.add_parser("explain-nodes", help="Run GNNExplainer on a single sequence")
    explain_parser.add_argument("--config", default=None)
    explain_parser.add_argument("--sequence", default=None)
    explain_parser.add_argument("--sequence-file", default=None)
    explain_parser.add_argument("--name", required=True)
    explain_parser.add_argument("--model-checkpoint", required=True)
    explain_parser.add_argument("--output-dir", default="gnn_results")
    explain_parser.add_argument("--ratio", default=None, type=float)
    explain_parser.add_argument("--top-fraction", default=0.1, type=float)
    explain_parser.add_argument("--steps", default=11, type=int)
    explain_parser.add_argument("--epochs", default=None, type=int)
    explain_parser.add_argument("--seed", default=42, type=int)
    explain_parser.set_defaults(func=_explain_nodes)

    plot_parser = subparsers.add_parser(
        "plot-attention", help="Generate a dual-panel attention and importance figure for a protein sequence"
    )
    plot_parser.add_argument("--config", default="configs/fungi.yaml")
    plot_parser.add_argument("--sequence", default=None)
    plot_parser.add_argument("--sequence-file", default=None)
    plot_parser.add_argument("--name", required=True)
    plot_parser.add_argument("--model-checkpoint", required=True)
    plot_parser.add_argument("--fold", default=None, type=int)
    plot_parser.add_argument("--inference-dir", default="inference_results")
    plot_parser.add_argument("--explain-dir", default="gnn_results")
    plot_parser.add_argument("--output-dir", default="graficos")
    plot_parser.add_argument("--top-fraction", default=0.1, type=float)
    plot_parser.add_argument("--steps", default=11, type=int)
    plot_parser.add_argument("--epochs", default=None, type=int)
    plot_parser.add_argument("--seed", default=42, type=int)
    plot_parser.add_argument(
        "--annotation-json",
        default=None,
        help="InterProScan JSON file from which representative families/domains will be drawn",
    )
    plot_parser.add_argument(
        "--annotation-name",
        default=None,
        help="Specific sequence name/id inside the annotation JSON to use",
    )
    plot_parser.add_argument(
        "--active-site-zip",
        default=None,
        help="Zip file under asites/ containing structure.cif_residues.csv for active-site residues",
    )
    plot_parser.add_argument(
        "--active-site-pockets",
        default=None,
        help="Comma-separated pocket identifiers to highlight (default: 1)",
    )
    plot_parser.set_defaults(func=_plot_attention)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
