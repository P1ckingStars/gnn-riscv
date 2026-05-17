"""Dataset bridging the spec generator and the model.

Two construction modes:

  - ``SpecDataset.synthetic(n_specs, ...)`` — generate fresh specs on the fly. Best
    for tiny test/smoke runs.
  - ``SpecDataset.from_directory(path, ...)`` — load a pre-built dataset (as written
    by ``scripts/build_dataset.py``), keyed off the ``spec.json`` files. Best for any
    training run >100 samples; avoids re-sampling on every program start.

In both modes, specs whose body uses ops outside the v1 model vocab
(``SEXT/ZEXT/TRUNC``) are silently skipped — they can't be lowered to a ComputeGraph
that the model handles.

Each sample carries:
  - ``spec_tokens``     : List[int] of token IDs for the spec encoder.
  - ``op_ids``          : List[int] OpVocab-encoded ops per generation step.
  - ``operand_lists``   : List[List[int]] operand pointers per step.
  - ``const_bucket_ids``: List[Optional[int]] for CONST steps (OOV bucket when needed).
  - ``n_inputs``        : Number of spec inputs.
  - ``spec`` / ``graph``: The Spec and ComputeGraph (kept for eval / debugging).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from torch.utils.data import Dataset

from model.graph import ComputeGraph, UnsupportedAstOp, ast_to_graph, graph_to_steps
from model.vocab import ConstVocab, OpVocab, SpecVocab, tokenize_spec
from spec.dsl import Spec, Ty
from spec.generator import sample_spec
from spec.serialize import from_json


@dataclass
class Sample:
    spec_tokens: list[int]
    op_ids: list[int]
    operand_lists: list[list[int]]
    const_bucket_ids: list[Optional[int]]
    n_inputs: int
    spec: Spec
    graph: ComputeGraph


def _encode(spec: Spec, graph: ComputeGraph, sv: SpecVocab, ov: OpVocab) -> Sample:
    tokens = tokenize_spec(spec, sv)
    steps = graph_to_steps(graph)
    op_ids = [ov.encode(s.op) for s in steps]
    operand_lists = [list(s.operands) for s in steps]
    const_bucket_ids = [
        sv.consts.encode(s.const_value, ty=spec.ret_ty)
        if s.const_value is not None else None
        for s in steps
    ]
    return Sample(
        spec_tokens=tokens, op_ids=op_ids, operand_lists=operand_lists,
        const_bucket_ids=const_bucket_ids, n_inputs=len(spec.inputs),
        spec=spec, graph=graph,
    )


class SpecDataset(Dataset):
    """List-of-samples dataset, constructed via one of the factory methods."""

    def __init__(self, samples: list[Sample], sv: SpecVocab, ov: OpVocab) -> None:
        self.samples = samples
        self.sv = sv
        self.ov = ov

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    # -- factories ------------------------------------------------------------

    @classmethod
    def synthetic(
        cls,
        n_specs: int,
        seed: int = 0,
        max_depth: int = 3,
        n_params: int = 2,
        ret_ty: Ty = Ty.I32,
        sv: Optional[SpecVocab] = None,
        ov: Optional[OpVocab] = None,
        max_attempts_per_sample: int = 8,
    ) -> "SpecDataset":
        sv = sv or SpecVocab()
        ov = ov or OpVocab()
        samples: list[Sample] = []
        attempts_budget = n_specs * max_attempts_per_sample
        seed_iter = seed
        while len(samples) < n_specs and attempts_budget > 0:
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
            samples.append(_encode(spec, g, sv, ov))
        if len(samples) < n_specs:
            raise RuntimeError(
                f"could only produce {len(samples)} samples after "
                f"{n_specs * max_attempts_per_sample} attempts"
            )
        return cls(samples=samples, sv=sv, ov=ov)

    @classmethod
    def from_directory(
        cls,
        path: Path | str,
        sv: Optional[SpecVocab] = None,
        ov: Optional[OpVocab] = None,
        limit: Optional[int] = None,
        require_functional: bool = True,
    ) -> "SpecDataset":
        """Load a dataset written by ``scripts/build_dataset.py``. Each ``spec_*/spec.json``
        is parsed and (silently) skipped if it falls outside the model's supported subset
        (non-functional, or contains SEXT/ZEXT/TRUNC)."""
        sv = sv or SpecVocab()
        ov = ov or OpVocab()
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {root}")
        spec_dirs = sorted(
            p for p in root.iterdir() if p.is_dir() and p.name.startswith("spec_")
        )
        samples: list[Sample] = []
        n_skipped_relational = 0
        n_skipped_unsupported = 0
        for sd in spec_dirs:
            if limit is not None and len(samples) >= limit:
                break
            spec_file = sd / "spec.json"
            if not spec_file.exists():
                continue
            spec = from_json(spec_file.read_text())
            if require_functional and not spec.is_functional():
                n_skipped_relational += 1
                continue
            try:
                g = ast_to_graph(spec)
            except (UnsupportedAstOp, ValueError):
                n_skipped_unsupported += 1
                continue
            samples.append(_encode(spec, g, sv, ov))
        return cls(samples=samples, sv=sv, ov=ov)


def random_split(
    ds: SpecDataset, val_frac: float = 0.1, seed: int = 0,
) -> tuple[SpecDataset, SpecDataset]:
    """Deterministic shuffled split into (train, val). Vocabs are shared (not deep-copied);
    edits to one would affect the other — we treat vocabs as immutable post-construction."""
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1); got {val_frac}")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(ds) * val_frac)))
    val_set = set(indices[:n_val])
    train_samples = [ds.samples[i] for i in range(len(ds)) if i not in val_set]
    val_samples = [ds.samples[i] for i in range(len(ds)) if i in val_set]
    return (
        SpecDataset(samples=train_samples, sv=ds.sv, ov=ds.ov),
        SpecDataset(samples=val_samples, sv=ds.sv, ov=ds.ov),
    )


def collate_list(batch: Iterable[Sample]) -> list[Sample]:
    """Returns the batch as a Python list. Inner loop in train.py iterates per sample."""
    return list(batch)
