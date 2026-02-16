from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..data.esm import embed_sequence, load_esm_model
from ..explain.gnnexplainer import run_node_explainer
from ..inference.single import run_inference_with_outputs


def ensure_dir(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def prepare_sequence_artifacts(
    sequence: str,
    protein_name: str,
    checkpoint_path: Path,
    config: PipelineConfig,
    fold_number: Optional[int] = None,
    inference_dir: Path | str = Path("inference_results"),
    explain_dir: Path | str = Path("gnn_results"),
    top_fraction: float = 0.1,
    explainer_steps: int = 11,
    explainer_epochs: Optional[int] = None,
    explainer_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    sequence = sequence.strip()
    checkpoint_path = Path(checkpoint_path)
    fold_id = fold_number if fold_number is not None else 0

    inference_root = ensure_dir(inference_dir)
    explain_root = ensure_dir(explain_dir)

    run_inference_with_outputs(
        sequence=sequence,
        sequence_id=protein_name,
        checkpoint_path=checkpoint_path,
        config=config,
        fold_number=fold_id,
        output_base=inference_root,
    )

    run_node_explainer(
        sequence=sequence,
        model_path=checkpoint_path,
        output_name=protein_name,
        output_dir=explain_root,
        top_fraction=top_fraction,
        steps=explainer_steps,
        epochs=explainer_epochs,
        seed=explainer_seed,
        fold_number=fold_number,
    )

    attention_csv = inference_root / protein_name / f"{protein_name}_residue_attention.csv"
    if not attention_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo de atención en {attention_csv}")
    attention_df = pd.read_csv(attention_csv)

    importance_csv = explain_root / protein_name / f"{protein_name}_nodes_importance.csv"
    if not importance_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo de importancia en {importance_csv}")
    importance_df = pd.read_csv(importance_csv)

    contact_model_bundle = load_esm_model(config.esm_model_contacts)
    _, _, contact_map_tensor = embed_sequence(protein_name, sequence, contact_model_bundle)
    contact_map = contact_map_tensor.cpu().numpy()
    trimmed_length = min(contact_map.shape[0], len(sequence))
    contact_map = contact_map[:trimmed_length, :trimmed_length]

    return attention_df, importance_df, contact_map
