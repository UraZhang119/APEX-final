from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class PipelineConfig:
    """Container for data, training and runtime settings."""

    species: str = "fungi"
    root_path: Path = Path("./data")
    trim_thresh: int = 1000
    n_splits: int = 5
    ratio: float = 0.2

    pos_samples_path: Optional[Path] = None
    neg_samples_path: Optional[Path] = None
    kfold_root_path: Optional[Path] = None
    raw_data_path: Optional[Path] = None

    train_batch_size: int = 6
    test_batch_size: int = 5
    num_epochs: int = 400
    lr: float = 2.0e-6
    weight_decay: float = 5.0e-4
    drop_prob: float = 0.3
    model: str = "gat"
    model_saving_path: Path = Path("./experiments/fungi")
    cuda_name: str = "cuda:0"
    esm_model_embeddings: str = "facebook/esm2_t33_650M_UR50D"
    esm_model_contacts: str = "facebook/esm2_t33_650M_UR50D"
    use_fgm: bool = True

    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_prefix: str = "gat"
    use_wandb: bool = False

    default_checkpoint_type: str = "best_aupr"

    slurm: Dict[str, Any] = field(default_factory=dict)

    def resolve_paths(self, base_dir: Optional[Path] = None) -> None:
        """Resolve relative paths against the provided base directory."""

        base_dir = base_dir or Path.cwd()
        if not isinstance(self.root_path, Path):
            self.root_path = Path(self.root_path)
        self.root_path = (base_dir / self.root_path).resolve()

        def _resolve(value: Optional[Path], default: Path) -> Path:
            if value is None:
                candidate = default
            else:
                candidate = Path(value)
            return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()

        self.pos_samples_path = _resolve(
            self.pos_samples_path,
            self.root_path / self.species / "orig_sample_list" / "Fungi_Essential_Genes.xlsx",
        )
        self.neg_samples_path = _resolve(
            self.neg_samples_path,
            self.root_path / self.species / "orig_sample_list" / "Fungi_NonEssential_Genes.xlsx",
        )
        self.kfold_root_path = _resolve(
            self.kfold_root_path,
            self.root_path / self.species / "kfold_splitted_data",
        )
        self.raw_data_path = _resolve(
            self.raw_data_path,
            self.root_path / self.species / "raw",
        )
        if not isinstance(self.model_saving_path, Path):
            self.model_saving_path = Path(self.model_saving_path)
        self.model_saving_path = (
            self.model_saving_path
            if self.model_saving_path.is_absolute()
            else (base_dir / self.model_saving_path).resolve()
        )

    @property
    def experiments_dir(self) -> Path:
        return self.model_saving_path

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def load_config(config_path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    """Load a YAML config file and apply optional overrides."""

    data: Dict[str, Any] = {}
    if config_path is not None:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    if overrides:
        data.update(overrides)
    config = PipelineConfig(**data)
    config.resolve_paths(Path(config_path).parent if config_path else Path.cwd())
    return config
