"""Reference evaluator for the spec DSL.

Two evaluation surfaces:
  - ``evaluate(spec, args)`` — for **functional** specs, returns the unique
    output value implied by ``post: r == f(inputs)``.
  - ``satisfies(spec, inputs, outputs)`` — for **any** spec, returns whether the
    candidate ``outputs`` satisfy ``post`` (assuming ``pre`` holds).

Formula evaluation handles bounded quantifiers by enumeration. For practicality, only
quantifiers over types narrower than ``BOUNDED_ENUM_MAX_VALUES`` (default 256, i.e. i8)
are enumerated; wider quantifiers raise ``NeedsSMT`` — the caller routes to the SMT
verifier.
"""
from __future__ import annotations

import random

from spec.dsl import (
    And, Bin, BinOp, BoolFalse, BoolTrue, Cmp, CmpOp, Const, Exists, Expr, Forall,
    Formula, Iff, Implies, LetTerm, Not, Or, Param, Select, Spec, Ty, Un, UnOp, Var,
    mask_to, signed,
)


BOUNDED_ENUM_MAX_VALUES = 256  # enumerate quantifiers only when |Ty| <= this


class UndefinedBehavior(Exception):
    """Raised when evaluation would trigger DSL-undefined behavior."""


class NeedsSMT(Exception):
    """Raised when the interpreter can't decide a quantifier (too wide to enumerate)."""


# ---- term evaluation (unchanged from earlier) -------------------------------

def evaluate(spec: Spec, args: tuple[int, ...]) -> int:
    """Evaluate a **functional** spec. Raises ``ValueError`` if the spec is relational
    (no closed-form output). Use ``satisfies`` for relational specs."""
    if not spec.is_functional():
        raise ValueError("evaluate(): spec is not functional; use satisfies()")
    if len(args) != len(spec.inputs):
        raise ValueError(f"arity mismatch: spec expects {len(spec.inputs)}, got {len(args)}")
    env = {p.name: mask_to(v, p.ty) for p, v in zip(spec.inputs, args)}
    body = spec.functional_body()
    return mask_to(_eval_expr(body, env), body.ty)


def _eval_expr(e: Expr, env: dict[str, int]) -> int:
    if isinstance(e, Const):
        return e.value
    if isinstance(e, Var):
        return env[e.name]
    if isinstance(e, Bin):
        l = _eval_expr(e.lhs, env)
        r = _eval_expr(e.rhs, env)
        return _eval_bin(e.op, l, r, e.lhs.ty)
    if isinstance(e, Un):
        v = _eval_expr(e.arg, env)
        return _eval_un(e.op, v, e.arg.ty, e.ty)
    if isinstance(e, Select):
        c = _eval_expr(e.cond, env)
        return _eval_expr(e.then, env) if c != 0 else _eval_expr(e.else_, env)
    raise TypeError(f"unknown Expr: {type(e).__name__}")


def _eval_bin(op: BinOp, l: int, r: int, ty: Ty) -> int:
    if op is BinOp.ADD: return mask_to(l + r, ty)
    if op is BinOp.SUB: return mask_to(l - r, ty)
    if op is BinOp.MUL: return mask_to(l * r, ty)
    if op is BinOp.AND: return l & r
    if op is BinOp.OR:  return l | r
    if op is BinOp.XOR: return l ^ r
    if op is BinOp.UDIV:
        if r == 0: raise UndefinedBehavior("udiv by zero")
        return l // r
    if op is BinOp.UREM:
        if r == 0: raise UndefinedBehavior("urem by zero")
        return l % r
    if op is BinOp.SDIV:
        if r == 0: raise UndefinedBehavior("sdiv by zero")
        ls, rs = signed(l, ty), signed(r, ty)
        q = abs(ls) // abs(rs)
        if (ls < 0) ^ (rs < 0): q = -q
        return mask_to(q, ty)
    if op is BinOp.SREM:
        if r == 0: raise UndefinedBehavior("srem by zero")
        ls, rs = signed(l, ty), signed(r, ty)
        q = abs(ls) // abs(rs)
        if (ls < 0) ^ (rs < 0): q = -q
        return mask_to(ls - q * rs, ty)
    if op in (BinOp.SHL, BinOp.LSHR, BinOp.ASHR):
        if r >= ty.width:
            raise UndefinedBehavior(f"shift amount {r} >= width {ty.width}")
        if op is BinOp.SHL:  return mask_to(l << r, ty)
        if op is BinOp.LSHR: return l >> r
        return mask_to(signed(l, ty) >> r, ty)
    raise ValueError(f"unhandled BinOp: {op}")


def _eval_un(op: UnOp, v: int, arg_ty: Ty, res_ty: Ty) -> int:
    if op is UnOp.NEG:   return mask_to(-v, arg_ty)
    if op is UnOp.NOT:   return v ^ arg_ty.mask
    if op is UnOp.SEXT:  return mask_to(signed(v, arg_ty), res_ty)
    if op is UnOp.ZEXT:  return v
    if op is UnOp.TRUNC: return mask_to(v, res_ty)
    raise ValueError(f"unhandled UnOp: {op}")


# ---- formula evaluation -----------------------------------------------------

def eval_formula(f: Formula, env: dict[str, int]) -> bool:
    """Evaluate a closed-or-ground formula to bool. ``env`` binds free vars."""
    if isinstance(f, BoolTrue):  return True
    if isinstance(f, BoolFalse): return False
    if isinstance(f, Cmp):
        l = _eval_expr(f.lhs, env)
        r = _eval_expr(f.rhs, env)
        return _eval_cmp(f.op, l, r, f.lhs.ty)
    if isinstance(f, Not):
        return not eval_formula(f.arg, env)
    if isinstance(f, And):
        return eval_formula(f.lhs, env) and eval_formula(f.rhs, env)
    if isinstance(f, Or):
        return eval_formula(f.lhs, env) or eval_formula(f.rhs, env)
    if isinstance(f, Implies):
        return (not eval_formula(f.ant, env)) or eval_formula(f.cons, env)
    if isinstance(f, Iff):
        return eval_formula(f.lhs, env) == eval_formula(f.rhs, env)
    if isinstance(f, Forall):
        n_vals = 1 << f.var.ty.width
        if n_vals > BOUNDED_ENUM_MAX_VALUES:
            raise NeedsSMT(f"forall {f.var.name}: {f.var.ty.name} too wide to enumerate")
        return all(
            eval_formula(f.body, {**env, f.var.name: v})
            for v in range(n_vals)
        )
    if isinstance(f, Exists):
        n_vals = 1 << f.var.ty.width
        if n_vals > BOUNDED_ENUM_MAX_VALUES:
            raise NeedsSMT(f"exists {f.var.name}: {f.var.ty.name} too wide to enumerate")
        return any(
            eval_formula(f.body, {**env, f.var.name: v})
            for v in range(n_vals)
        )
    if isinstance(f, LetTerm):
        v = mask_to(_eval_expr(f.value, env), f.ty)
        return eval_formula(f.body, {**env, f.name: v})
    raise TypeError(f"unknown Formula: {type(f).__name__}")


def _eval_cmp(op: CmpOp, l: int, r: int, ty: Ty) -> bool:
    if op is CmpOp.EQ: return l == r
    if op is CmpOp.NE: return l != r
    if op is CmpOp.ULT: return l < r
    if op is CmpOp.ULE: return l <= r
    if op is CmpOp.UGT: return l > r
    if op is CmpOp.UGE: return l >= r
    ls, rs = signed(l, ty), signed(r, ty)
    if op is CmpOp.SLT: return ls < rs
    if op is CmpOp.SLE: return ls <= rs
    if op is CmpOp.SGT: return ls > rs
    if op is CmpOp.SGE: return ls >= rs
    raise ValueError(f"unhandled CmpOp: {op}")


# ---- spec-level predicates --------------------------------------------------

def precondition_holds(spec: Spec, inputs: tuple[int, ...]) -> bool:
    env = {p.name: mask_to(v, p.ty) for p, v in zip(spec.inputs, inputs)}
    return eval_formula(spec.pre, env)


def satisfies(
    spec: Spec, inputs: tuple[int, ...], outputs: tuple[int, ...],
) -> bool:
    """Return True iff (inputs, outputs) satisfies the spec's postcondition.
    Caller is responsible for checking ``precondition_holds`` separately."""
    env = {p.name: mask_to(v, p.ty) for p, v in zip(spec.inputs, inputs)}
    env.update({p.name: mask_to(v, p.ty) for p, v in zip(spec.outputs, outputs)})
    return eval_formula(spec.post, env)


# ---- input sampling ---------------------------------------------------------

def _boundary_values(ty: Ty) -> list[int]:
    return [0, 1, ty.mask, ty.sign_bit, ty.sign_bit - 1, ty.mask ^ 1]


def sample_inputs(spec: Spec, n: int, seed: int = 0) -> list[tuple[int, ...]]:
    """Return n input tuples that satisfy ``spec.pre`` and don't trigger UB during
    Pre evaluation. (UB inside Post is up to the caller; satisfies() will surface it.)"""
    rng = random.Random(seed)
    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def try_add(t: tuple[int, ...]) -> None:
        if t in seen:
            return
        try:
            if not precondition_holds(spec, t):
                return
        except UndefinedBehavior:
            return
        except NeedsSMT:
            # Pre is too rich to evaluate without SMT — accept the input pessimistically;
            # the SMT verifier will handle Pre symbolically.
            pass
        seen.add(t)
        out.append(t)

    from itertools import product
    per_param = [_boundary_values(p.ty) for p in spec.inputs]
    boundary_cap = min(64, max(1, n // 4))
    boundary_iter = iter(product(*per_param))
    while len(out) < boundary_cap:
        try:
            t = next(boundary_iter)
        except StopIteration:
            break
        try_add(t)

    max_attempts = n * 20
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        t = tuple(_rand_value(rng, p.ty) for p in spec.inputs)
        try_add(t)
    return out


def _rand_value(rng: random.Random, ty: Ty) -> int:
    r = rng.random()
    if r < 0.3:
        return rng.randint(0, 8)
    if r < 0.5:
        return mask_to(-rng.randint(1, 8), ty)
    if r < 0.7:
        return rng.randint(0, ty.mask >> 2)
    return rng.randint(0, ty.mask)
