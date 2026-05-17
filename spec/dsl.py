"""DSL with first-order-logic expressiveness over fixed-width-integer bitvectors.

Two layers:

  - **Term layer**: arithmetic/bitwise expressions over fixed-width ints. Carry
    a ``Ty``. (``Const``, ``Var``, ``Bin``, ``Un``, ``Select``.)
  - **Formula layer**: propositional + first-order formulas over terms.
    (``BoolTrue/False``, ``Cmp``, ``Not/And/Or/Implies/Iff``, ``Forall/Exists``,
    ``LetTerm``.)

A ``Spec`` is a Hoare triple ``(inputs, outputs, pre, post)`` — see
``docs/problem-statement.md`` §2 for the formal definition. The ``Spec.functional``
constructor wraps an existing term as ``post: r == body``; relational specs are written
directly using the Formula API.

Inspired by the FOL formula vocabulary in github.com/<user>/fol-zfc (`forall x. φ`,
`exists x. φ`, `~ & | -> <->`), but specialized to a bitvector background theory — no
uninterpreted predicates, no ZFC axioms, no proof scripts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class Ty(Enum):
    I8 = 8
    I16 = 16
    I32 = 32
    I64 = 64

    @property
    def width(self) -> int:
        return self.value

    @property
    def mask(self) -> int:
        return (1 << self.value) - 1

    @property
    def sign_bit(self) -> int:
        return 1 << (self.value - 1)


XLEN = 64  # RV64 pointer width


@dataclass(frozen=True)
class PtrTy:
    """Typed pointer — a 64-bit address tagged with its element type.

    The element type can be any AnyTy (a scalar Ty or another PtrTy), so e.g.
    ``PtrTy(PtrTy(Ty.I32))`` is ``int**``. Arithmetic on pointers is *byte*-based
    (matches the hardware); element indexing is sugar built on top via PtrAdd with
    multiples of ``elem_size_bytes``.
    """
    elem_ty: "AnyTy"

    @property
    def width(self) -> int:
        return XLEN

    @property
    def mask(self) -> int:
        return (1 << XLEN) - 1

    @property
    def sign_bit(self) -> int:
        return 1 << (XLEN - 1)

    @property
    def elem_size_bytes(self) -> int:
        """Size in bytes of the pointee — used for canonical element-indexing offsets."""
        if isinstance(self.elem_ty, Ty):
            return self.elem_ty.width // 8
        if isinstance(self.elem_ty, PtrTy):
            return XLEN // 8
        raise TypeError(f"unhandled elem_ty: {self.elem_ty}")

    def __repr__(self) -> str:
        if isinstance(self.elem_ty, Ty):
            return f"Ptr<{self.elem_ty.name}>"
        return f"Ptr<{self.elem_ty}>"


AnyTy = Union[Ty, PtrTy]


# ---- term-layer operator enums ----------------------------------------------

class BinOp(Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    SDIV = "sdiv"
    UDIV = "udiv"
    SREM = "srem"
    UREM = "urem"
    AND = "and"
    OR = "or"
    XOR = "xor"
    SHL = "shl"
    LSHR = "lshr"
    ASHR = "ashr"


class UnOp(Enum):
    NEG = "neg"
    NOT = "not"
    SEXT = "sext"
    ZEXT = "zext"
    TRUNC = "trunc"


# ---- formula-layer operator enums -------------------------------------------

class CmpOp(Enum):
    EQ = "eq"
    NE = "ne"
    SLT = "slt"
    SLE = "sle"
    SGT = "sgt"
    SGE = "sge"
    ULT = "ult"
    ULE = "ule"
    UGT = "ugt"
    UGE = "uge"


# ---- term AST ---------------------------------------------------------------

@dataclass(frozen=True)
class Expr:
    ty: Ty


@dataclass(frozen=True)
class Const(Expr):
    value: int  # unsigned-canonical: 0 <= value < 2**ty.width


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class Bin(Expr):
    op: BinOp
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class Un(Expr):
    op: UnOp
    arg: Expr


@dataclass(frozen=True)
class Select(Expr):
    cond: Expr
    then: Expr
    else_: Expr


@dataclass(frozen=True)
class PtrAdd(Expr):
    """``base + offset_bytes``. ``base`` must have ``PtrTy``; ``offset`` is a scalar
    integer (sign-extended to 64 bits during interpretation/SMT). Result type equals
    ``base.ty`` (same pointee type as input).
    """
    base: Expr
    offset: Expr


@dataclass(frozen=True)
class Load(Expr):
    """``*ptr`` — read ``ty.width / 8`` bytes from the implicit memory state at address
    ``ptr``, little-endian. ``ty`` must equal ``ptr.ty.elem_ty``.
    """
    ptr: Expr


# ---- formula AST ------------------------------------------------------------

@dataclass(frozen=True)
class Formula:
    """Marker base for all formula nodes."""


@dataclass(frozen=True)
class BoolTrue(Formula):
    pass


@dataclass(frozen=True)
class BoolFalse(Formula):
    pass


@dataclass(frozen=True)
class Cmp(Formula):
    op: CmpOp
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class Not(Formula):
    arg: Formula


@dataclass(frozen=True)
class And(Formula):
    lhs: Formula
    rhs: Formula


@dataclass(frozen=True)
class Or(Formula):
    lhs: Formula
    rhs: Formula


@dataclass(frozen=True)
class Implies(Formula):
    ant: Formula
    cons: Formula


@dataclass(frozen=True)
class Iff(Formula):
    lhs: Formula
    rhs: Formula


@dataclass(frozen=True)
class Forall(Formula):
    var: "Param"
    body: Formula


@dataclass(frozen=True)
class Exists(Formula):
    var: "Param"
    body: Formula


@dataclass(frozen=True)
class LetTerm(Formula):
    """Bind a term `value` to `name:ty` within `body`. Term-level let; pure shorthand."""
    name: str
    ty: Ty
    value: Expr
    body: Formula


# ---- spec -------------------------------------------------------------------

@dataclass(frozen=True)
class Param:
    name: str
    ty: Ty


@dataclass(frozen=True)
class Spec:
    """Hoare-triple synthesis target.

    ``inputs``:  parameters the candidate function receives.
    ``outputs``: outputs the candidate must produce. v1 supports a single output.
    ``pre``:     precondition over inputs. ``BoolTrue()`` when omitted.
    ``post``:    postcondition over inputs ∪ outputs. The candidate ``f`` is correct iff
                 ``∀ inputs. pre(inputs) → post(inputs, f(inputs))``.
    """
    inputs: tuple[Param, ...]
    outputs: tuple[Param, ...]
    post: Formula
    pre: Formula = field(default_factory=lambda: BoolTrue())

    @property
    def ret_param(self) -> Param:
        if len(self.outputs) != 1:
            raise ValueError(f"v1 supports single-output specs only; got {len(self.outputs)}")
        return self.outputs[0]

    @property
    def ret_ty(self) -> Ty:
        return self.ret_param.ty

    @classmethod
    def functional(
        cls,
        inputs: tuple[Param, ...],
        body: Expr,
        ret_name: str = "r",
        pre: Formula | None = None,
    ) -> "Spec":
        """Wrap a term as a functional Hoare spec: ``post: r == body``."""
        out_param = Param(ret_name, body.ty)
        result_var = Var(ty=body.ty, name=ret_name)
        return cls(
            inputs=inputs,
            outputs=(out_param,),
            post=Cmp(op=CmpOp.EQ, lhs=result_var, rhs=body),
            pre=pre if pre is not None else BoolTrue(),
        )

    def is_functional(self) -> bool:
        """True iff post is exactly ``r == body`` with no other constraints."""
        return (
            isinstance(self.post, Cmp)
            and self.post.op is CmpOp.EQ
            and isinstance(self.post.lhs, Var)
            and len(self.outputs) == 1
            and self.post.lhs.name == self.outputs[0].name
        )

    def functional_body(self) -> Expr:
        """For a functional spec, return the term ``body`` from ``post: r == body``."""
        if not self.is_functional():
            raise ValueError("not a functional spec")
        assert isinstance(self.post, Cmp)
        return self.post.rhs


# ---- value helpers ----------------------------------------------------------

def mask_to(value: int, ty: Ty) -> int:
    """Wrap to ty's width, returning the unsigned-canonical representation."""
    return value & ty.mask


def signed(value: int, ty: Ty) -> int:
    """Interpret an unsigned-canonical value as a signed Python int (two's-complement)."""
    v = value & ty.mask
    if v & ty.sign_bit:
        v -= 1 << ty.width
    return v


# ---- type checker -----------------------------------------------------------

_PRESERVING_BIN_OPS = {
    BinOp.ADD, BinOp.SUB, BinOp.MUL,
    BinOp.SDIV, BinOp.UDIV, BinOp.SREM, BinOp.UREM,
    BinOp.AND, BinOp.OR, BinOp.XOR,
    BinOp.SHL, BinOp.LSHR, BinOp.ASHR,
}


def type_check(spec: Spec) -> None:
    """Verify the spec is well-typed. Raises TypeError on the first violation."""
    seen = set()
    for p in spec.inputs:
        if p.name in seen:
            raise TypeError(f"duplicate input param name: {p.name}")
        seen.add(p.name)
    for p in spec.outputs:
        if p.name in seen:
            raise TypeError(f"output name shadows input or another output: {p.name}")
        seen.add(p.name)
    env: dict[str, Ty] = {p.name: p.ty for p in (*spec.inputs, *spec.outputs)}
    _tc_formula(spec.pre, env)
    _tc_formula(spec.post, env)


def _tc_expr(e: Expr, env: dict[str, AnyTy]) -> None:
    if isinstance(e, Const):
        if not isinstance(e.ty, Ty):
            raise TypeError(f"Const must be scalar, got {e.ty}")
        if not (0 <= e.value <= e.ty.mask):
            raise TypeError(f"Const value {e.value} out of range for {e.ty}")
        return
    if isinstance(e, Var):
        if e.name not in env:
            raise TypeError(f"unbound var: {e.name}")
        if env[e.name] != e.ty:
            raise TypeError(f"var {e.name}: ty {e.ty} disagrees with binding {env[e.name]}")
        return
    if isinstance(e, Bin):
        _tc_expr(e.lhs, env)
        _tc_expr(e.rhs, env)
        # Scalar-only — pointer arithmetic goes through PtrAdd.
        if not (isinstance(e.lhs.ty, Ty) and isinstance(e.rhs.ty, Ty)):
            raise TypeError(f"{e.op.name}: pointer operands; use PtrAdd for ptr arithmetic")
        if e.op in _PRESERVING_BIN_OPS:
            if e.lhs.ty != e.rhs.ty:
                raise TypeError(f"{e.op.name}: operand tys {e.lhs.ty} vs {e.rhs.ty}")
            if e.ty != e.lhs.ty:
                raise TypeError(f"{e.op.name}: result ty {e.ty} != operand ty {e.lhs.ty}")
        return
    if isinstance(e, Un):
        _tc_expr(e.arg, env)
        if not isinstance(e.arg.ty, Ty):
            raise TypeError(f"{e.op.name}: pointer arg not allowed")
        if e.op in (UnOp.NEG, UnOp.NOT):
            if e.ty != e.arg.ty:
                raise TypeError(f"{e.op.name}: result ty {e.ty} != arg ty {e.arg.ty}")
        elif e.op in (UnOp.SEXT, UnOp.ZEXT):
            if not isinstance(e.ty, Ty):
                raise TypeError(f"{e.op.name}: result must be scalar, got {e.ty}")
            if e.ty.width <= e.arg.ty.width:
                raise TypeError(f"{e.op.name}: result width {e.ty.width} must exceed arg {e.arg.ty.width}")
        elif e.op is UnOp.TRUNC:
            if not isinstance(e.ty, Ty):
                raise TypeError(f"trunc: result must be scalar, got {e.ty}")
            if e.ty.width >= e.arg.ty.width:
                raise TypeError(f"trunc: result width {e.ty.width} must be less than arg {e.arg.ty.width}")
        return
    if isinstance(e, Select):
        _tc_expr(e.cond, env)
        _tc_expr(e.then, env)
        _tc_expr(e.else_, env)
        if e.then.ty != e.else_.ty or e.ty != e.then.ty:
            raise TypeError(f"select: result {e.ty}, then {e.then.ty}, else {e.else_.ty}")
        return
    if isinstance(e, PtrAdd):
        _tc_expr(e.base, env)
        _tc_expr(e.offset, env)
        if not isinstance(e.base.ty, PtrTy):
            raise TypeError(f"PtrAdd: base must be Ptr, got {e.base.ty}")
        if not isinstance(e.offset.ty, Ty):
            raise TypeError(f"PtrAdd: offset must be scalar int, got {e.offset.ty}")
        if e.ty != e.base.ty:
            raise TypeError(
                f"PtrAdd: result ty {e.ty} must equal base ty {e.base.ty}"
            )
        return
    if isinstance(e, Load):
        _tc_expr(e.ptr, env)
        if not isinstance(e.ptr.ty, PtrTy):
            raise TypeError(f"Load: ptr must be Ptr, got {e.ptr.ty}")
        if e.ty != e.ptr.ty.elem_ty:
            raise TypeError(
                f"Load: result ty {e.ty} != ptr elem ty {e.ptr.ty.elem_ty}"
            )
        return
    raise TypeError(f"unknown Expr node: {type(e).__name__}")


def _tc_formula(f: Formula, env: dict[str, Ty]) -> None:
    if isinstance(f, (BoolTrue, BoolFalse)):
        return
    if isinstance(f, Cmp):
        _tc_expr(f.lhs, env)
        _tc_expr(f.rhs, env)
        if f.lhs.ty != f.rhs.ty:
            raise TypeError(f"Cmp {f.op.name}: operand tys {f.lhs.ty} vs {f.rhs.ty}")
        if isinstance(f.lhs.ty, PtrTy) and f.op not in (CmpOp.EQ, CmpOp.NE):
            raise TypeError(
                f"Cmp {f.op.name}: only EQ/NE are defined on pointers"
            )
        return
    if isinstance(f, Not):
        _tc_formula(f.arg, env)
        return
    if isinstance(f, (And, Or)):
        _tc_formula(f.lhs, env)
        _tc_formula(f.rhs, env)
        return
    if isinstance(f, Implies):
        _tc_formula(f.ant, env)
        _tc_formula(f.cons, env)
        return
    if isinstance(f, Iff):
        _tc_formula(f.lhs, env)
        _tc_formula(f.rhs, env)
        return
    if isinstance(f, (Forall, Exists)):
        if f.var.name in env:
            raise TypeError(f"quantifier var {f.var.name} shadows outer binding")
        _tc_formula(f.body, {**env, f.var.name: f.var.ty})
        return
    if isinstance(f, LetTerm):
        _tc_expr(f.value, env)
        if f.value.ty != f.ty:
            raise TypeError(f"LetTerm {f.name}: value ty {f.value.ty} != declared {f.ty}")
        if f.name in env:
            raise TypeError(f"LetTerm var {f.name} shadows outer binding")
        _tc_formula(f.body, {**env, f.name: f.ty})
        return
    raise TypeError(f"unknown Formula node: {type(f).__name__}")
