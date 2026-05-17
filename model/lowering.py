"""Deterministic graph → RV64GC asm lowering. v0 keeps this completely heuristic
so all failures are attributable to the generator, not the backend."""
from __future__ import annotations


def lower(graph) -> str:
    """Topologically sort, list-schedule, linear-scan allocate, emit RV64GC asm string."""
    raise NotImplementedError
