"""Autoregressive GNN graph generator.

At each generation step ``k`` the model:

  1. Builds candidate embeddings for the unified node space
     ``[spec_inputs ++ emitted_nodes[:k]]``.
  2. Runs a GNN (PyG ``GATConv``) over the partial DAG to refine those embeddings.
  3. Concatenates the spec context with a pooled graph-state vector to form a
     decision state.
  4. Emits four heads simultaneously:
     - ``op_logits``       — categorical over ``OpVocab``;
     - ``operand_logits``  — three pointer-net heads over the current candidate set;
     - ``const_logits``    — categorical over ``ConstVocab`` buckets (used iff op=CONST).

Training is teacher-forced via ``teacher_forced_loss``. Inference is via ``generate``
which rolls the model out step-by-step until it emits a ``RETURN`` op.

Why GAT specifically: short DAGs of size ≤ 20 with positional asymmetry between
operand slots. GAT's attention captures operand asymmetry better than mean-aggregation;
SAGE or simple SUM would also work and is a candidate ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
    _PYG_OK = True
except ImportError:
    _PYG_OK = False

from model.graph import ARITY, ComputeGraph, GraphNode, NodeOp
from model.vocab import ConstVocab, MAX_VARS, OpVocab
from spec.dsl import Param, Ty


_TYPES: tuple[Ty, ...] = tuple(Ty)
_MAX_INPUT_POS = MAX_VARS
_OPERAND_SLOTS = 3  # SELECT is the max-arity emitted op


class GraphGenerator(nn.Module):
    def __init__(
        self,
        op_vocab: OpVocab,
        const_vocab: ConstVocab,
        d_model: int = 128,
        gnn_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not _PYG_OK:
            raise ImportError("torch_geometric is required for GraphGenerator")
        self.op_vocab = op_vocab
        self.const_vocab = const_vocab
        self.d_model = d_model
        self.n_ops = op_vocab.size
        self.n_const_buckets = const_vocab.size

        # Candidate-side embeddings (used to seed candidate vectors).
        self.op_embed = nn.Embedding(self.n_ops, d_model)
        self.ty_embed = nn.Embedding(len(_TYPES), d_model)
        self.input_pos_embed = nn.Embedding(_MAX_INPUT_POS, d_model)
        self.const_bucket_embed = nn.Embedding(self.n_const_buckets, d_model)

        # GNN over the partial DAG. One layer is enough at this graph size.
        self.gnn = GATConv(d_model, d_model, heads=gnn_heads, concat=False, add_self_loops=True)

        # Decision-state combiner: spec ctx + pooled graph state.
        self.state_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Heads.
        self.op_head = nn.Linear(d_model, self.n_ops)
        self.const_head = nn.Linear(d_model, self.n_const_buckets)
        self.operand_query = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(_OPERAND_SLOTS)
        ])

    # ---- candidate construction ---------------------------------------------

    def _input_candidate(self, pos: int, ty: Ty, device: torch.device) -> torch.Tensor:
        """Seed embedding for a spec input at positional slot `pos`."""
        pos_t = torch.tensor(pos, device=device)
        ty_t = torch.tensor(_TYPES.index(ty), device=device)
        return self.input_pos_embed(pos_t) + self.ty_embed(ty_t)

    def _node_candidate(
        self,
        op_idx: int,
        out_ty: Ty,
        operand_embeds: list[torch.Tensor],
        const_bucket: int | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Seed embedding for a newly emitted node from its op + operand seeds."""
        op_t = torch.tensor(op_idx, device=device)
        ty_t = torch.tensor(_TYPES.index(out_ty), device=device)
        e = self.op_embed(op_t) + self.ty_embed(ty_t)
        if operand_embeds:
            e = e + torch.stack(operand_embeds).mean(0)
        if const_bucket is not None:
            cb_t = torch.tensor(const_bucket, device=device)
            e = e + self.const_bucket_embed(cb_t)
        return e

    # ---- forward pieces -----------------------------------------------------

    def _refine_with_gnn(
        self, candidate_embeds: torch.Tensor, edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Run one GAT layer; bypass when there are no edges (single-node graph)."""
        if edge_index.numel() == 0:
            return candidate_embeds
        return F.gelu(self.gnn(candidate_embeds, edge_index))

    def _decision_state(
        self, spec_ctx: torch.Tensor, candidate_embeds: torch.Tensor,
    ) -> torch.Tensor:
        pooled = candidate_embeds.mean(dim=0)
        return self.state_mlp(torch.cat([spec_ctx.squeeze(0), pooled]))

    def step_heads(
        self, state: torch.Tensor, node_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        op_logits = self.op_head(state)
        const_logits = self.const_head(state)
        operand_logits: list[torch.Tensor] = []
        for q_proj in self.operand_query:
            q = q_proj(state)
            operand_logits.append(node_embeds @ q)
        return op_logits, operand_logits, const_logits

    # ---- training: teacher-forced supervised loss ---------------------------

    def teacher_forced_loss(
        self,
        spec_ctx: torch.Tensor,
        sample: "model.data.Sample",  # noqa: F821 (forward reference)
    ) -> dict[str, torch.Tensor]:
        """Walk the reference graph step by step, computing the supervised CE loss
        at each generation step. Returns a dict of per-head losses and totals.
        """
        device = next(self.parameters()).device
        n_inputs = sample.n_inputs
        # Seed candidate embeddings for spec inputs.
        candidate_embeds: list[torch.Tensor] = [
            self._input_candidate(i, p.ty, device)
            for i, p in enumerate(sample.spec.inputs)
        ]
        edges_src: list[int] = []
        edges_tgt: list[int] = []

        total_op_loss = torch.zeros((), device=device)
        total_operand_loss = torch.zeros((), device=device)
        total_const_loss = torch.zeros((), device=device)
        n_op_steps = 0
        n_operand_terms = 0
        n_const_terms = 0

        for step_idx, (op_id, operand_list, const_bucket) in enumerate(
            zip(sample.op_ids, sample.operand_lists, sample.const_bucket_ids)
        ):
            stacked = torch.stack(candidate_embeds, dim=0)
            edge_index = (
                torch.tensor([edges_src, edges_tgt], dtype=torch.long, device=device)
                if edges_src else torch.zeros((2, 0), dtype=torch.long, device=device)
            )
            refined = self._refine_with_gnn(stacked, edge_index)
            state = self._decision_state(spec_ctx, refined)
            op_logits, operand_logits, const_logits = self.step_heads(state, refined)

            # Op loss
            total_op_loss = total_op_loss + F.cross_entropy(
                op_logits.unsqueeze(0), torch.tensor([op_id], device=device),
            )
            n_op_steps += 1

            # Operand losses — one per slot actually used by the target op.
            target_node_op = self.op_vocab.decode(op_id)
            arity = ARITY[target_node_op]
            for slot in range(arity):
                target_ptr = operand_list[slot]
                if target_ptr >= refined.shape[0]:
                    # Should never happen given correct data; skip defensively.
                    continue
                total_operand_loss = total_operand_loss + F.cross_entropy(
                    operand_logits[slot].unsqueeze(0),
                    torch.tensor([target_ptr], device=device),
                )
                n_operand_terms += 1

            # Const loss — only for CONST ops.
            if target_node_op is NodeOp.CONST and const_bucket is not None:
                total_const_loss = total_const_loss + F.cross_entropy(
                    const_logits.unsqueeze(0),
                    torch.tensor([const_bucket], device=device),
                )
                n_const_terms += 1

            # Update partial graph: teacher-force the emitted node into the candidate set.
            if target_node_op is NodeOp.RETURN:
                break
            operand_embeds = [candidate_embeds[p] for p in operand_list[:arity]]
            new_emb = self._node_candidate(
                op_id, sample.spec.ret_ty, operand_embeds, const_bucket, device,
            )
            new_idx = len(candidate_embeds)
            candidate_embeds.append(new_emb)
            for src in operand_list[:arity]:
                edges_src.append(src)
                edges_tgt.append(new_idx)

        # Mean-normalize across the term counts so step length doesn't dominate magnitude.
        op_l = total_op_loss / max(n_op_steps, 1)
        opd_l = total_operand_loss / max(n_operand_terms, 1)
        cst_l = total_const_loss / max(n_const_terms, 1)
        total = op_l + opd_l + cst_l
        return {
            "loss": total,
            "loss_op": op_l.detach(),
            "loss_operand": opd_l.detach(),
            "loss_const": cst_l.detach(),
            "n_op_steps": torch.tensor(n_op_steps),
            "n_operand_terms": torch.tensor(n_operand_terms),
            "n_const_terms": torch.tensor(n_const_terms),
        }

    # ---- inference ----------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        spec_ctx: torch.Tensor,
        inputs: tuple[Param, ...],
        ret_ty: Ty,
        max_steps: int = 32,
        greedy: bool = True,
    ) -> ComputeGraph | None:
        """Roll the model out to produce a ComputeGraph. Returns None if generation
        fails to terminate within ``max_steps`` or if it picks an invalid operand."""
        device = next(self.parameters()).device
        candidate_embeds: list[torch.Tensor] = [
            self._input_candidate(i, p.ty, device) for i, p in enumerate(inputs)
        ]
        edges_src: list[int] = []
        edges_tgt: list[int] = []
        emitted: list[GraphNode] = []
        n_inputs = len(inputs)

        for _ in range(max_steps):
            stacked = torch.stack(candidate_embeds, dim=0)
            edge_index = (
                torch.tensor([edges_src, edges_tgt], dtype=torch.long, device=device)
                if edges_src else torch.zeros((2, 0), dtype=torch.long, device=device)
            )
            refined = self._refine_with_gnn(stacked, edge_index)
            state = self._decision_state(spec_ctx, refined)
            op_logits, operand_logits, const_logits = self.step_heads(state, refined)

            if greedy:
                op_id = int(op_logits.argmax().item())
            else:
                op_id = int(torch.distributions.Categorical(logits=op_logits).sample().item())
            op = self.op_vocab.decode(op_id)
            arity = ARITY[op]

            # Pick operands
            operand_ptrs: list[int] = []
            for slot in range(arity):
                logits = operand_logits[slot]
                if greedy:
                    ptr = int(logits.argmax().item())
                else:
                    ptr = int(torch.distributions.Categorical(logits=logits).sample().item())
                if ptr >= refined.shape[0]:
                    return None  # bad pointer — model failure
                operand_ptrs.append(ptr)

            if op is NodeOp.RETURN:
                emitted.append(GraphNode(op=NodeOp.RETURN, operands=tuple(operand_ptrs), out_ty=ret_ty))
                return ComputeGraph(inputs=inputs, nodes=tuple(emitted))

            # Const value
            const_value: int | None = None
            if op is NodeOp.CONST:
                if greedy:
                    cb = int(const_logits.argmax().item())
                else:
                    cb = int(torch.distributions.Categorical(logits=const_logits).sample().item())
                v = self.const_vocab.decode(cb, ty=ret_ty)
                # OOV → fall back to 0; not a great guess but keeps generation going.
                const_value = v if v is not None else 0

            new_node = GraphNode(
                op=op, operands=tuple(operand_ptrs), out_ty=ret_ty, const_value=const_value,
            )
            emitted.append(new_node)

            operand_embeds = [candidate_embeds[p] for p in operand_ptrs]
            cb_for_seed = (
                self.const_vocab.encode(const_value, ty=ret_ty)
                if op is NodeOp.CONST else None
            )
            new_emb = self._node_candidate(op_id, ret_ty, operand_embeds, cb_for_seed, device)
            new_idx = len(candidate_embeds)
            candidate_embeds.append(new_emb)
            for src in operand_ptrs:
                edges_src.append(src)
                edges_tgt.append(new_idx)

        return None  # didn't terminate
