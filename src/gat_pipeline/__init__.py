"""GAT-based fungal pathogenicity pipeline."""

from .config import PipelineConfig, load_config

__all__ = [
    "PipelineConfig",
    "load_config",
]
