"""JSON-round-trippable serialization for the FOL spec DSL.

Schema v2 (current): Hoare-triple specs with separate term + formula layers.
Schema v1 (deprecated): pre-FOL functional specs — not supported; regenerate datasets.
"""
from __future__ import annotations

import json

from spec.dsl import (
    And, Bin, BinOp, BoolFalse, BoolTrue, Cmp, CmpOp, Const, Exists, Expr, Forall,
    Formula, Iff, Implies, LetTerm, Not, Or, Param, Select, Spec, Ty, Un, UnOp, Var,
)


SCHEMA_VERSION = 2


def to_dict(spec: Spec) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "inputs": [_param_to_dict(p) for p in spec.inputs],
        "outputs": [_param_to_dict(p) for p in spec.outputs],
        "pre": _formula_to_dict(spec.pre),
        "post": _formula_to_dict(spec.post),
    }


def from_dict(d: dict) -> Spec:
    if d.get("schema") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema version: {d.get('schema')!r} (need {SCHEMA_VERSION})"
        )
    return Spec(
        inputs=tuple(_param_from_dict(p) for p in d["inputs"]),
        outputs=tuple(_param_from_dict(p) for p in d["outputs"]),
        pre=_formula_from_dict(d["pre"]),
        post=_formula_from_dict(d["post"]),
    )


def to_json(spec: Spec, indent: int | None = None) -> str:
    return json.dumps(to_dict(spec), indent=indent)


def from_json(text: str) -> Spec:
    return from_dict(json.loads(text))


def _param_to_dict(p: Param) -> dict:
    return {"name": p.name, "ty": p.ty.name}


def _param_from_dict(d: dict) -> Param:
    return Param(name=d["name"], ty=Ty[d["ty"]])


def _expr_to_dict(e: Expr) -> dict:
    if isinstance(e, Const):
        return {"node": "const", "ty": e.ty.name, "value": e.value}
    if isinstance(e, Var):
        return {"node": "var", "ty": e.ty.name, "name": e.name}
    if isinstance(e, Bin):
        return {"node": "bin", "ty": e.ty.name, "op": e.op.name,
                "lhs": _expr_to_dict(e.lhs), "rhs": _expr_to_dict(e.rhs)}
    if isinstance(e, Un):
        return {"node": "un", "ty": e.ty.name, "op": e.op.name,
                "arg": _expr_to_dict(e.arg)}
    if isinstance(e, Select):
        return {"node": "select", "ty": e.ty.name,
                "cond": _expr_to_dict(e.cond),
                "then": _expr_to_dict(e.then),
                "else": _expr_to_dict(e.else_)}
    raise TypeError(f"unhandled Expr: {type(e).__name__}")


def _expr_from_dict(d: dict) -> Expr:
    n = d["node"]
    ty = Ty[d["ty"]]
    if n == "const":
        return Const(ty=ty, value=d["value"])
    if n == "var":
        return Var(ty=ty, name=d["name"])
    if n == "bin":
        return Bin(ty=ty, op=BinOp[d["op"]],
                   lhs=_expr_from_dict(d["lhs"]), rhs=_expr_from_dict(d["rhs"]))
    if n == "un":
        return Un(ty=ty, op=UnOp[d["op"]], arg=_expr_from_dict(d["arg"]))
    if n == "select":
        return Select(ty=ty,
                      cond=_expr_from_dict(d["cond"]),
                      then=_expr_from_dict(d["then"]),
                      else_=_expr_from_dict(d["else"]))
    raise ValueError(f"unknown expr node: {n}")


def _formula_to_dict(f: Formula) -> dict:
    if isinstance(f, BoolTrue):
        return {"f": "true"}
    if isinstance(f, BoolFalse):
        return {"f": "false"}
    if isinstance(f, Cmp):
        return {"f": "cmp", "op": f.op.name,
                "lhs": _expr_to_dict(f.lhs), "rhs": _expr_to_dict(f.rhs)}
    if isinstance(f, Not):
        return {"f": "not", "arg": _formula_to_dict(f.arg)}
    if isinstance(f, And):
        return {"f": "and", "lhs": _formula_to_dict(f.lhs), "rhs": _formula_to_dict(f.rhs)}
    if isinstance(f, Or):
        return {"f": "or", "lhs": _formula_to_dict(f.lhs), "rhs": _formula_to_dict(f.rhs)}
    if isinstance(f, Implies):
        return {"f": "implies",
                "ant": _formula_to_dict(f.ant), "cons": _formula_to_dict(f.cons)}
    if isinstance(f, Iff):
        return {"f": "iff", "lhs": _formula_to_dict(f.lhs), "rhs": _formula_to_dict(f.rhs)}
    if isinstance(f, Forall):
        return {"f": "forall", "var": _param_to_dict(f.var),
                "body": _formula_to_dict(f.body)}
    if isinstance(f, Exists):
        return {"f": "exists", "var": _param_to_dict(f.var),
                "body": _formula_to_dict(f.body)}
    if isinstance(f, LetTerm):
        return {"f": "let", "name": f.name, "ty": f.ty.name,
                "value": _expr_to_dict(f.value), "body": _formula_to_dict(f.body)}
    raise TypeError(f"unhandled Formula: {type(f).__name__}")


def _formula_from_dict(d: dict) -> Formula:
    k = d["f"]
    if k == "true":  return BoolTrue()
    if k == "false": return BoolFalse()
    if k == "cmp":
        return Cmp(op=CmpOp[d["op"]],
                   lhs=_expr_from_dict(d["lhs"]), rhs=_expr_from_dict(d["rhs"]))
    if k == "not":
        return Not(arg=_formula_from_dict(d["arg"]))
    if k == "and":
        return And(lhs=_formula_from_dict(d["lhs"]), rhs=_formula_from_dict(d["rhs"]))
    if k == "or":
        return Or(lhs=_formula_from_dict(d["lhs"]), rhs=_formula_from_dict(d["rhs"]))
    if k == "implies":
        return Implies(ant=_formula_from_dict(d["ant"]),
                       cons=_formula_from_dict(d["cons"]))
    if k == "iff":
        return Iff(lhs=_formula_from_dict(d["lhs"]), rhs=_formula_from_dict(d["rhs"]))
    if k == "forall":
        return Forall(var=_param_from_dict(d["var"]),
                      body=_formula_from_dict(d["body"]))
    if k == "exists":
        return Exists(var=_param_from_dict(d["var"]),
                      body=_formula_from_dict(d["body"]))
    if k == "let":
        return LetTerm(name=d["name"], ty=Ty[d["ty"]],
                       value=_expr_from_dict(d["value"]),
                       body=_formula_from_dict(d["body"]))
    raise ValueError(f"unknown formula node: {k}")
