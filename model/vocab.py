"""Vocabularies for the spec encoder and graph generator.

Two distinct vocabs:
  - ``ConstVocab``: bucketed integer values. Used both for spec-side ``Const`` tokens
    and for the generator's CONST node value head. Unknown values → OOV bucket.
  - ``OpVocab``: emitted node operations (everything in ``NodeOp`` except ``INPUT``).
  - ``SpecVocab``: token IDs for linearized spec ASTs (types, vars, ops, structural).

Vocab sizes are constants — kept small so a tiny model can overfit a synthetic
dataset for sanity checking.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from model.graph import NodeOp
from spec.dsl import BinOp, CmpOp, Ty, UnOp, mask_to, signed


# ---- Const bucket vocab ------------------------------------------------------

_SMALL_INTS = list(range(-8, 9))                  # 17
_POW2_POS = [1 << i for i in range(4, 32)]        # 16,32,...,2**31 (28)
_POW2_NEG = [-(1 << i) for i in range(4, 32)]     # -16,-32,...,-2**31 (28)
_SPECIAL = [
    0x7FFF_FFFF,                                  # INT32_MAX
    0xFFFF_FFFF,                                  # -1 mod 2**32 / mask32
    0xFFFF,                                       # mask16
    0xFF,                                         # mask8
]
_CANON_VALUES: tuple[int, ...] = tuple(_SMALL_INTS + _POW2_POS + _POW2_NEG + _SPECIAL)


class ConstVocab:
    """Bucket of canonical integer values + an OOV bucket. Values are matched on the
    raw Python-int level (not modulo any width), so e.g. ``-1`` and ``0xFFFFFFFF`` are
    distinct buckets even though they're equal in i32."""

    OOV_INDEX = 0

    def __init__(self) -> None:
        # Bucket 0 is OOV; canonical values get buckets 1..N.
        self._val_to_idx: dict[int, int] = {v: i + 1 for i, v in enumerate(_CANON_VALUES)}
        self._idx_to_val: tuple[int | None, ...] = (None, *_CANON_VALUES)
        self.size = len(self._idx_to_val)

    def encode(self, value: int, ty: Ty | None = None) -> int:
        """Lookup the bucket for ``value``. If ``ty`` is given, also try the signed
        reinterpretation — so e.g. ``0xFFFFFFFF`` with ``ty=I32`` maps to the ``-1``
        bucket. Generator outputs are unsigned-canonical, so this re-check matters."""
        idx = self._val_to_idx.get(value)
        if idx is not None:
            return idx
        if ty is not None:
            s = signed(value, ty)
            if s != value:
                idx = self._val_to_idx.get(s)
                if idx is not None:
                    return idx
        return self.OOV_INDEX

    def decode(self, idx: int, ty: Ty | None = None) -> int | None:
        """Returns ``None`` for OOV. If ``ty`` is given, the decoded value is returned
        in unsigned-canonical form for that type — so model outputs slot directly into a
        ``Const`` node without further conversion at the call site."""
        if not (0 <= idx < self.size):
            raise IndexError(f"const bucket idx {idx} out of range [0, {self.size})")
        v = self._idx_to_val[idx]
        if v is None:
            return None
        if ty is not None:
            return mask_to(v, ty)
        return v


# ---- Op vocab (decoder side) -------------------------------------------------

_EMITTED_OPS: tuple[NodeOp, ...] = tuple(o for o in NodeOp if o is not NodeOp.INPUT)


class OpVocab:
    """Decoder-side: every op the generator can emit, indexed densely."""

    def __init__(self) -> None:
        self._op_to_idx: dict[NodeOp, int] = {o: i for i, o in enumerate(_EMITTED_OPS)}
        self._idx_to_op: tuple[NodeOp, ...] = _EMITTED_OPS
        self.size = len(self._idx_to_op)

    def encode(self, op: NodeOp) -> int:
        return self._op_to_idx[op]

    def decode(self, idx: int) -> NodeOp:
        return self._idx_to_op[idx]


# ---- Spec encoder vocab ------------------------------------------------------

class SpecTok(Enum):
    """Special tokens prepended to the spec-vocab id space."""
    PAD = 0
    CLS = 1     # sentence start, used for pooling
    SEP = 2     # separates inputs / pre / post sections
    UNK = 3
    INP_BEGIN = 4
    INP_END = 5
    PRE_BEGIN = 6
    PRE_END = 7
    POST_BEGIN = 8
    POST_END = 9
    LPAREN = 10
    RPAREN = 11
    COMMA = 12
    # AST node tags
    NODE_CONST = 13
    NODE_VAR = 14
    NODE_BIN = 15
    NODE_UN = 16
    NODE_SELECT = 17
    NODE_CMP = 18
    NODE_NOT = 19
    NODE_AND = 20
    NODE_OR = 21
    NODE_IMPLIES = 22
    NODE_IFF = 23
    NODE_FORALL = 24
    NODE_EXISTS = 25
    NODE_TRUE = 26
    NODE_FALSE = 27
    PARAM = 28


_TYPES: tuple[Ty, ...] = tuple(Ty)
_BIN_OPS: tuple[BinOp, ...] = tuple(BinOp)
_UN_OPS: tuple[UnOp, ...] = tuple(UnOp)
_CMP_OPS: tuple[CmpOp, ...] = tuple(CmpOp)


@dataclass(frozen=True)
class _VocabRanges:
    specials: int          # 0 .. N_specials-1
    types: int             # next 4
    bin_ops: int
    un_ops: int
    cmp_ops: int
    vars: int              # MAX_VARS positional slots
    consts: int            # ConstVocab.size buckets


MAX_VARS = 16              # positional var slots — VAR_0 .. VAR_{MAX_VARS-1}


class SpecVocab:
    """Tokenizes spec ASTs into a flat id sequence. Var names are mapped positionally
    (first var seen → VAR_0, second → VAR_1, ...) so the model never observes raw
    names — handles the generator's ``x0,x1`` and the parser's ``x,y`` uniformly.
    """

    def __init__(self) -> None:
        cv = ConstVocab()
        self.consts = cv
        # Compute offsets
        offset = max(t.value for t in SpecTok) + 1
        self._off_types = offset; offset += len(_TYPES)
        self._off_bin = offset; offset += len(_BIN_OPS)
        self._off_un = offset; offset += len(_UN_OPS)
        self._off_cmp = offset; offset += len(_CMP_OPS)
        self._off_vars = offset; offset += MAX_VARS
        self._off_consts = offset; offset += cv.size
        self.size = offset

    # -- token-id constructors
    def tok(self, t: SpecTok) -> int: return t.value
    def ty(self, t: Ty) -> int: return self._off_types + _TYPES.index(t)
    def bin_op(self, o: BinOp) -> int: return self._off_bin + _BIN_OPS.index(o)
    def un_op(self, o: UnOp) -> int: return self._off_un + _UN_OPS.index(o)
    def cmp_op(self, o: CmpOp) -> int: return self._off_cmp + _CMP_OPS.index(o)
    def var(self, idx: int) -> int:
        if not (0 <= idx < MAX_VARS):
            raise IndexError(f"var idx {idx} out of range")
        return self._off_vars + idx
    def const(self, value: int, ty: Ty | None = None) -> int:
        return self._off_consts + self.consts.encode(value, ty=ty)


# ---- Spec → token-id sequence -------------------------------------------------

def tokenize_spec(spec, sv: SpecVocab) -> list[int]:
    """Linearize a Spec into a flat token-id sequence. Pre-order traversal with
    structural delimiters. Variables are mapped positionally in their order of first
    appearance (with the spec's inputs+outputs binding the early slots)."""
    from spec.dsl import (
        And, Bin, BoolFalse, BoolTrue, Cmp, Const, Exists, Expr, Forall, Formula,
        Iff, Implies, LetTerm, Not, Or, Param, Select, Spec, Un, Var,
    )

    name_to_idx: dict[str, int] = {}
    def var_idx(name: str) -> int:
        if name not in name_to_idx:
            if len(name_to_idx) >= MAX_VARS:
                # Out of var slots — collapse to UNK so encoding still succeeds.
                return SpecTok.UNK.value
            name_to_idx[name] = len(name_to_idx)
        return name_to_idx[name]

    ids: list[int] = [sv.tok(SpecTok.CLS)]

    # Inputs section
    ids.append(sv.tok(SpecTok.INP_BEGIN))
    for i, p in enumerate(spec.inputs):
        idx = var_idx(p.name)
        if isinstance(idx, int) and idx != SpecTok.UNK.value:
            ids.append(sv.var(idx))
        else:
            ids.append(sv.tok(SpecTok.UNK))
        ids.append(sv.ty(p.ty))
    ids.append(sv.tok(SpecTok.INP_END))

    # Output(s) — same slot space as inputs.
    for p in spec.outputs:
        idx = var_idx(p.name)
        if isinstance(idx, int) and idx != SpecTok.UNK.value:
            ids.append(sv.var(idx))
        else:
            ids.append(sv.tok(SpecTok.UNK))
        ids.append(sv.ty(p.ty))

    # Pre (skipped if BoolTrue — frequent case, saves tokens).
    if not isinstance(spec.pre, BoolTrue):
        ids.append(sv.tok(SpecTok.PRE_BEGIN))
        _emit_formula(spec.pre, ids, sv, var_idx)
        ids.append(sv.tok(SpecTok.PRE_END))

    # Post — required.
    ids.append(sv.tok(SpecTok.POST_BEGIN))
    _emit_formula(spec.post, ids, sv, var_idx)
    ids.append(sv.tok(SpecTok.POST_END))

    return ids


def _emit_expr(e, ids: list[int], sv: SpecVocab, var_idx) -> None:
    from spec.dsl import Bin, Const, Select, Un, Var
    if isinstance(e, Const):
        ids.append(sv.tok(SpecTok.NODE_CONST))
        ids.append(sv.ty(e.ty))
        ids.append(sv.const(e.value, ty=e.ty))
        return
    if isinstance(e, Var):
        ids.append(sv.tok(SpecTok.NODE_VAR))
        ids.append(sv.ty(e.ty))
        slot = var_idx(e.name)
        ids.append(sv.var(slot) if isinstance(slot, int) and slot != SpecTok.UNK.value
                   else sv.tok(SpecTok.UNK))
        return
    if isinstance(e, Bin):
        ids.append(sv.tok(SpecTok.NODE_BIN))
        ids.append(sv.bin_op(e.op))
        ids.append(sv.ty(e.ty))
        _emit_expr(e.lhs, ids, sv, var_idx)
        _emit_expr(e.rhs, ids, sv, var_idx)
        return
    if isinstance(e, Un):
        ids.append(sv.tok(SpecTok.NODE_UN))
        ids.append(sv.un_op(e.op))
        ids.append(sv.ty(e.ty))
        _emit_expr(e.arg, ids, sv, var_idx)
        return
    if isinstance(e, Select):
        ids.append(sv.tok(SpecTok.NODE_SELECT))
        ids.append(sv.ty(e.ty))
        _emit_expr(e.cond, ids, sv, var_idx)
        _emit_expr(e.then, ids, sv, var_idx)
        _emit_expr(e.else_, ids, sv, var_idx)
        return
    raise TypeError(f"unhandled Expr: {type(e).__name__}")


def _emit_formula(f, ids: list[int], sv: SpecVocab, var_idx) -> None:
    from spec.dsl import (
        And, BoolFalse, BoolTrue, Cmp, Exists, Forall, Iff, Implies, Not, Or,
    )
    if isinstance(f, BoolTrue):
        ids.append(sv.tok(SpecTok.NODE_TRUE)); return
    if isinstance(f, BoolFalse):
        ids.append(sv.tok(SpecTok.NODE_FALSE)); return
    if isinstance(f, Cmp):
        ids.append(sv.tok(SpecTok.NODE_CMP))
        ids.append(sv.cmp_op(f.op))
        _emit_expr(f.lhs, ids, sv, var_idx)
        _emit_expr(f.rhs, ids, sv, var_idx)
        return
    if isinstance(f, Not):
        ids.append(sv.tok(SpecTok.NODE_NOT))
        _emit_formula(f.arg, ids, sv, var_idx); return
    if isinstance(f, And):
        ids.append(sv.tok(SpecTok.NODE_AND))
        _emit_formula(f.lhs, ids, sv, var_idx)
        _emit_formula(f.rhs, ids, sv, var_idx); return
    if isinstance(f, Or):
        ids.append(sv.tok(SpecTok.NODE_OR))
        _emit_formula(f.lhs, ids, sv, var_idx)
        _emit_formula(f.rhs, ids, sv, var_idx); return
    if isinstance(f, Implies):
        ids.append(sv.tok(SpecTok.NODE_IMPLIES))
        _emit_formula(f.ant, ids, sv, var_idx)
        _emit_formula(f.cons, ids, sv, var_idx); return
    if isinstance(f, Iff):
        ids.append(sv.tok(SpecTok.NODE_IFF))
        _emit_formula(f.lhs, ids, sv, var_idx)
        _emit_formula(f.rhs, ids, sv, var_idx); return
    if isinstance(f, (Forall, Exists)):
        ids.append(sv.tok(SpecTok.NODE_FORALL if isinstance(f, Forall) else SpecTok.NODE_EXISTS))
        slot = var_idx(f.var.name)
        ids.append(sv.var(slot) if isinstance(slot, int) and slot != SpecTok.UNK.value
                   else sv.tok(SpecTok.UNK))
        ids.append(sv.ty(f.var.ty))
        _emit_formula(f.body, ids, sv, var_idx); return
    raise TypeError(f"unhandled Formula: {type(f).__name__}")
