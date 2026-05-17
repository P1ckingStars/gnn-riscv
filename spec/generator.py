"""Random spec sampler — produces well-typed Hoare-triple specs for the dataset.

v1 default mode is **functional**: sample a term ``body`` and wrap as
``Spec.functional(inputs, body)``. This keeps a clean teacher signal
(``post: r == body``) for supervised model training.

A ``relational=True`` mode is reserved for richer postcondition sampling — left as a
stub for milestone 2 since the supervised pipeline needs functional specs to start.
"""
from __future__ import annotations

import random
from collections.abc import Iterator

from spec.dsl import (
    Bin, BinOp, Const, Expr, Param, Select, Spec, Ty, Un, UnOp, Var,
    mask_to, type_check,
)
from spec.interpreter import UndefinedBehavior, evaluate, sample_inputs


_BIN_PRESERVING = (
    BinOp.ADD, BinOp.SUB, BinOp.MUL,
    BinOp.AND, BinOp.OR, BinOp.XOR,
    BinOp.SDIV, BinOp.UDIV, BinOp.SREM, BinOp.UREM,
)
_BIN_SHIFT = (BinOp.SHL, BinOp.LSHR, BinOp.ASHR)
_UN_SAMETYPE = (UnOp.NEG, UnOp.NOT)


def sample_spec(
    seed: int,
    max_depth: int = 4,
    n_params: int = 2,
    ret_ty: Ty = Ty.I32,
    relational: bool = False,
) -> Spec:
    if relational:
        raise NotImplementedError("relational spec sampling is reserved for milestone 2")
    rng = random.Random(seed)
    inputs = tuple(Param(f"x{i}", ret_ty) for i in range(n_params))
    for _ in range(64):
        body = _sample_expr(rng, ret_ty, [(p.name, p.ty) for p in inputs], max_depth)
        spec = Spec.functional(inputs=inputs, body=body, ret_name="r")
        type_check(spec)
        if _is_informative(spec, rng):
            return spec
    raise RuntimeError("sample_spec: could not produce an informative spec in 64 tries")


def stream(
    seed: int = 0,
    max_depth: int = 4,
    n_params: int = 2,
    ret_ty: Ty = Ty.I32,
) -> Iterator[Spec]:
    i = seed
    while True:
        try:
            yield sample_spec(i, max_depth=max_depth, n_params=n_params, ret_ty=ret_ty)
        except RuntimeError:
            pass
        i += 1


# ---- internals --------------------------------------------------------------

def _sample_expr(
    rng: random.Random,
    ty: Ty,
    env: list[tuple[str, Ty]],
    depth: int,
) -> Expr:
    stop_p = 1.0 if depth <= 0 else 0.15 + 0.2 * (4 - min(depth, 4))
    if rng.random() < stop_p:
        return _sample_leaf(rng, ty, env)

    choice = rng.choices(
        ["bin_pres", "bin_shift", "un_same", "ext", "trunc", "select"],
        weights=[6, 2, 2, 1, 1, 2],
    )[0]

    if choice == "bin_pres":
        op = rng.choice(_BIN_PRESERVING)
        lhs = _sample_expr(rng, ty, env, depth - 1)
        rhs = _sample_expr(rng, ty, env, depth - 1)
        return Bin(ty=ty, op=op, lhs=lhs, rhs=rhs)

    if choice == "bin_shift":
        op = rng.choice(_BIN_SHIFT)
        lhs = _sample_expr(rng, ty, env, depth - 1)
        amt = rng.randint(0, ty.width - 1)
        rhs = Const(ty=ty, value=amt)
        return Bin(ty=ty, op=op, lhs=lhs, rhs=rhs)

    if choice == "un_same":
        op = rng.choice(_UN_SAMETYPE)
        arg = _sample_expr(rng, ty, env, depth - 1)
        return Un(ty=ty, op=op, arg=arg)

    if choice == "ext":
        smaller = [t for t in Ty if t.width < ty.width]
        if not smaller:
            return _sample_leaf(rng, ty, env)
        arg_ty = rng.choice(smaller)
        arg = _sample_expr(rng, arg_ty, env, depth - 1)
        op = rng.choice((UnOp.SEXT, UnOp.ZEXT))
        return Un(ty=ty, op=op, arg=arg)

    if choice == "trunc":
        bigger = [t for t in Ty if t.width > ty.width]
        if not bigger:
            return _sample_leaf(rng, ty, env)
        arg_ty = rng.choice(bigger)
        arg = _sample_expr(rng, arg_ty, env, depth - 1)
        return Un(ty=ty, op=UnOp.TRUNC, arg=arg)

    cond = _sample_expr(rng, ty, env, depth - 1)
    then = _sample_expr(rng, ty, env, depth - 1)
    else_ = _sample_expr(rng, ty, env, depth - 1)
    return Select(ty=ty, cond=cond, then=then, else_=else_)


def _sample_leaf(rng: random.Random, ty: Ty, env: list[tuple[str, Ty]]) -> Expr:
    candidates_var = [name for name, t in env if t == ty]
    if candidates_var and rng.random() < 0.7:
        return Var(ty=ty, name=rng.choice(candidates_var))
    r = rng.random()
    if r < 0.4:
        v = rng.randint(0, 8)
    elif r < 0.7:
        v = mask_to(-rng.randint(1, 8), ty)
    elif r < 0.85:
        v = 1 << rng.randint(0, ty.width - 1)
    else:
        v = rng.randint(0, ty.mask)
    return Const(ty=ty, value=v)


def _is_informative(spec: Spec, rng: random.Random) -> bool:
    body = spec.functional_body()
    if isinstance(body, Const):
        return False
    try:
        probes = sample_inputs(spec, n=8, seed=rng.randint(0, 2**31 - 1))
    except Exception:
        return False
    if len(probes) < 2:
        return False
    outs = set()
    for p in probes:
        try:
            outs.add(evaluate(spec, p))
        except UndefinedBehavior:
            continue
        if len(outs) >= 2:
            return True
    return False
