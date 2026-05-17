"""Encodes a spec AST into a context vector + per-node embeddings used to condition graph generation."""
from __future__ import annotations

import torch
import torch.nn as nn

from spec.dsl import Spec


class SpecEncoder(nn.Module):
    """Small transformer over a linearized AST token stream. Returns
    (context: [d], node_embeds: [n_nodes, d]).
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        raise NotImplementedError

    def forward(self, spec: Spec) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
