"""Autoregressive GNN graph generator.

At each step:
  1. A GNN re-encodes the partial DAG of compute-graph nodes built so far.
  2. An op-type head, conditioned on (spec context, graph-state summary), picks the next op
     (or STOP).
  3. A pointer-net operand head picks each operand from {existing graph nodes, spec inputs}.
The terminal `return` node closes the graph.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GraphGenerator(nn.Module):
    def __init__(self, n_ops: int, d_model: int = 128, gnn_layers: int = 3):
        super().__init__()
        raise NotImplementedError

    def step(self, partial_graph, spec_context: torch.Tensor) -> dict:
        """One generation step. Returns {op_logits, operand_logits, stop_logit}."""
        raise NotImplementedError

    def generate(self, spec_context: torch.Tensor, max_nodes: int = 64) -> "ComputeGraph":  # noqa: F821
        raise NotImplementedError
