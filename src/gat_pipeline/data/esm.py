from __future__ import annotations

# ===== PARCHE APEX - DEBE IR ANTES DE CUALQUIER IMPORT =====
import sys
import types
from importlib.machinery import ModuleSpec

# Crear mock completo de APEX
apex_mock = types.ModuleType('apex')
apex_mock.__spec__ = ModuleSpec('apex', None)
apex_mock.normalization = types.ModuleType('apex.normalization')
apex_mock.normalization.__spec__ = ModuleSpec('apex.normalization', None)

# Importar torch para usar LayerNorm
import torch.nn as nn
apex_mock.normalization.FusedLayerNorm = nn.LayerNorm

# Registrar en sys.modules ANTES de que ESM intente importarlo
sys.modules['apex'] = apex_mock
sys.modules['apex.normalization'] = apex_mock.normalization
# ===== FIN PARCHE APEX =====


from functools import lru_cache
from typing import Iterable, Iterator, Tuple

import torch


@lru_cache(maxsize=4)
def load_esm_model(model_name: str = "facebook/esm2_t33_650M_UR50D") -> Tuple[torch.nn.Module, callable]:
    """Load an ESM model from fair-esm using the native contact-head outputs."""

    import os

    os.environ.setdefault("APEX_DISABLED", "1")
    os.environ.setdefault("APEX_FORCE_DISABLE_FUSED_LAYERNORM", "1")
    os.environ.setdefault("APEX_DISABLE_FUSED_LAYERNORM", "1")
    import esm  # type: ignore

    short_name = model_name.split("/")[-1]
    candidates = [short_name]

    alias_map = {
        "esm2_t33_650M_UR50S_2": "esm2_t33_650M_UR50S",
        "esm2_t33_650M_UR50D_2": "esm2_t33_650M_UR50D",
    }
    if short_name in alias_map:
        candidates.append(alias_map[short_name])

    base, _, suffix = short_name.rpartition("_")
    if suffix.isdigit():
        candidates.append(base)

    loader = None
    for candidate in candidates:
        loader = getattr(esm.pretrained, candidate, None)
        if loader is not None:
            break

    if loader is None:
        raise ValueError(f"Unsupported ESM model '{model_name}'")

    model, alphabet = loader()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, batch_converter


def embed_sequence(identifier: str, sequence: str, model_bundle: Tuple[torch.nn.Module, callable]):
    """Return representations and the official ESM contact map for a single sequence."""

    model, batch_converter = model_bundle
    device = next(model.parameters()).device

    trimmed_sequence = sequence[:1024] if len(sequence) > 1024 else sequence
    gene_ids, _, tokens = batch_converter([(identifier, trimmed_sequence)])
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[33], return_contacts=True)
        representations = results["representations"][33].squeeze(0)[1:-1, :].cpu()
        contact_map = results["contacts"].squeeze(0).cpu()

    return gene_ids[0], representations, contact_map


def embed_sequences(bag: Iterable[Tuple[str, str]], model_bundle: Tuple[torch.nn.Module, callable]) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor]]:
    for identifier, sequence in bag:
        yield embed_sequence(identifier, sequence, model_bundle)
