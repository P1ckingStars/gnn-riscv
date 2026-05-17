"""Dataset bridging the spec generator and the model.

Each sample is a dict carrying:
  - ``spec_tokens``  : List[int] of token IDs for the spec encoder (variable length).
  - ``op_ids``       : List[int] of OpVocab-encoded op indices, one per generation step.
  - ``operand_lists``: List[List[int]] of operand pointers per step (variable arity).
  - ``const_bucket_ids``: List[Optional[int]] — const-bucket id for CONST steps,
                          ``None`` otherwise (or OOV).
  - ``n_inputs``     : Number of spec inputs (== starting candidate set size).
  - ``spec``         : The Spec itself (for downstream eval / debugging).
  - ``graph``        : The ComputeGraph (ditto).

v1 keeps things deliberately small: pre-generates a fixed-size synthetic dataset by
sampling functional specs whose body falls inside the model's restricted vocab
(no SEXT/ZEXT/TRUNC). Specs whose Consts fall outside the bucket vocab are accepted
but their CONST steps train with the OOV bucket — fine for sanity-checking the loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from torch.utils.data import Dataset

from model.graph import ComputeGraph, UnsupportedAstOp, ast_to_graph, graph_to_steps
from model.vocab import ConstVocab, OpVocab, SpecVocab, tokenize_spec
from spec.dsl import Spec, Ty
from spec.generator import sample_spec


@dataclass
class Sample:
    spec_tokens: list[int]
    op_ids: list[int]
    operand_lists: list[list[int]]
    const_bucket_ids: list[Optional[int]]
    n_inputs: int
    spec: Spec
    graph: ComputeGraph


class SpecDataset(Dataset):
    """Pre-generates a fixed-size dataset of (spec, reference graph) pairs."""

    def __init__(
        self,
        n_specs: int,
        seed: int = 0,
        max_depth: int = 3,
        n_params: int = 2,
        ret_ty: Ty = Ty.I32,
        sv: Optional[SpecVocab] = None,
        ov: Optional[OpVocab] = None,
        max_attempts_per_sample: int = 8,
    ) -> None:
        self.sv = sv or SpecVocab()
        self.ov = ov or OpVocab()
        self.samples: list[Sample] = []
        attempts_budget = n_specs * max_attempts_per_sample
        seed_iter = seed
        while len(self.samples) < n_specs and attempts_budget > 0:
            attempts_budget -= 1
            try:
                spec = sample_spec(seed=seed_iter, max_depth=max_depth,
                                   n_params=n_params, ret_ty=ret_ty)
            except RuntimeError:
                seed_iter += 1
                continue
            seed_iter += 1
            try:
                g = ast_to_graph(spec)
            except UnsupportedAstOp:
                continue
            self.samples.append(self._encode(spec, g))
        if len(self.samples) < n_specs:
            raise RuntimeError(
                f"could only produce {len(self.samples)} samples after "
                f"{n_specs * max_attempts_per_sample} attempts"
            )

    def _encode(self, spec: Spec, graph: ComputeGraph) -> Sample:
        tokens = tokenize_spec(spec, self.sv)
        steps = graph_to_steps(graph)
        op_ids = [self.ov.encode(s.op) for s in steps]
        operand_lists = [list(s.operands) for s in steps]
        const_bucket_ids = [
            self.sv.consts.encode(s.const_value) if s.const_value is not None else None
            for s in steps
        ]
        return Sample(
            spec_tokens=tokens, op_ids=op_ids, operand_lists=operand_lists,
            const_bucket_ids=const_bucket_ids, n_inputs=len(spec.inputs),
            spec=spec, graph=graph,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def collate_list(batch: Iterable[Sample]) -> list[Sample]:
    """Trivial collate that returns the batch as a Python list. The training loop
    iterates per-sample; vectorizing across the batch is an optimization for later."""
    return list(batch)
