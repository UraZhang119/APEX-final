from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from ..config import PipelineConfig
from ..utils import cmap_to_graph, load_pt


class ProteinGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root: str | Path,
        gene_list: List[str],
        mode: str,
        ratio: float,
        raw_data_path: Path,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        self.mode = mode
        self.gene_list = gene_list
        self.ratio = ratio
        self.raw_data_path = Path(raw_data_path)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = self._load_processed()

    @property
    def processed_file_names(self):
        return [f"{self.mode}.pt"]

    def _load_processed(self):
        processed_path = Path(self.processed_paths[0])
        if processed_path.exists():
            return load_pt(processed_path)
        self.process()
        return load_pt(processed_path)

    def process(self):  # type: ignore[override]
        data_list: List[Data] = []
        for gene in self.gene_list:
            node_features, edge_index, target = self._get_geometric_input(gene)
            data_list.append(
                Data(
                    x=node_features,
                    edge_index=torch.as_tensor(edge_index, dtype=torch.long),
                    y=torch.tensor([target], dtype=torch.float32),
                )
            )
        data, slices = self.collate(data_list)
        self._ensure_processed_dir()
        torch.save((data, slices), self.processed_paths[0])

    def _ensure_processed_dir(self) -> None:
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

    def _get_geometric_input(self, gene_id: str):
        raw_gene_path = self.raw_data_path / f"{gene_id}.pt"
        raw_data = load_pt(raw_gene_path)
        features = raw_data["feature_representation"]
        contact_map = raw_data["cmap"]
        target = raw_data["target"]
        node_features, edge_index, _, _ = cmap_to_graph(features, contact_map, ratio=self.ratio)
        return node_features, edge_index, target


def build_fold_graphs(config: PipelineConfig) -> None:
    for fold in range(config.n_splits):
        fold_path = config.kfold_root_path / f"fold{fold}"
        train_list = pd.read_csv(fold_path / "train_data.txt", sep="\t")
        test_list = pd.read_csv(fold_path / "test_data.txt", sep="\t")
        ProteinGraphDataset(
            root=fold_path,
            gene_list=train_list["GeneSymbol"].tolist(),
            mode="train",
            ratio=config.ratio,
            raw_data_path=config.raw_data_path,
        )
        ProteinGraphDataset(
            root=fold_path,
            gene_list=test_list["GeneSymbol"].tolist(),
            mode="test",
            ratio=config.ratio,
            raw_data_path=config.raw_data_path,
        )
