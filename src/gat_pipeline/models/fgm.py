"""Adversarial training helpers (FGM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class _FGMConfig:
    bias_one: str
    weight_one: str
    bias_two: str
    weight_two: str


class _BaseFGM:
    def __init__(self, model, config: _FGMConfig) -> None:
        self.model = model
        self.config = config
        self.backup = {}

    @property
    def _target_fragments(self) -> Iterable[str]:
        return (
            self.config.bias_one,
            self.config.weight_one,
            self.config.bias_two,
            self.config.weight_two,
        )

    def attack(self, epsilon: float = 1.0) -> None:
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if not any(fragment in name for fragment in self._target_fragments):
                continue
            self.backup[name] = param.data.clone()
            norm = torch.norm(param.grad)
            if norm != 0:
                r_at = epsilon * param.grad / norm
                param.data.add_(r_at)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class FGM_GCN(_BaseFGM):
    def __init__(self, model, bias_one="gcnconv1.bias", weight_one="gcnconv1.lin.weight", bias_two="gcnconv2.bias", weight_two="gcnconv2.lin.weight"):
        super().__init__(model, _FGMConfig(bias_one, weight_one, bias_two, weight_two))


class FGM_SAGE(_BaseFGM):
    def __init__(self, model, bias_one="gcnconv1.lin_l.bias", weight_one="gcnconv1.lin_l.weight", bias_two="gcnconv2.lin_l.bias", weight_two="gcnconv2.lin_r.weight"):
        super().__init__(model, _FGMConfig(bias_one, weight_one, bias_two, weight_two))


class FGM_GAT(_BaseFGM):
    def __init__(self, model, bias_one="gcn1.bias", weight_one="gcn1.lin_src.weight", bias_two="gcn2.bias", weight_two="gcn2.lin_src.weight"):
        super().__init__(model, _FGMConfig(bias_one, weight_one, bias_two, weight_two))
