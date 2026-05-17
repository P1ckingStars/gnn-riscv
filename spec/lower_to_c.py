"""Lower a **functional** spec to a self-contained C source file for gcc/clang baseline
asm extraction. Raises ``ValueError`` on relational specs — only functional ones have a
closed-form computation to lower."""
from __future__ import annotations

from spec.dsl import (
    Bin, BinOp, Const, Expr, Select, Spec, Ty, Un, UnOp, Var,
)


_C_UINT = {Ty.I8: "uint8_t", Ty.I16: "uint16_t", Ty.I32: "uint32_t", Ty.I64: "uint64_t"}
_C_INT = {Ty.I8: "int8_t", Ty.I16: "int16_t", Ty.I32: "int32_t", Ty.I64: "int64_t"}


def to_c(spec: Spec, fn_name: str = "spec_fn") -> str:
    """Render the spec's functional body as a C function. Raises if non-functional."""
    if not spec.is_functional():
        raise ValueError("to_c(): spec is not functional; cannot lower a relational postcondition to C")
    body = spec.functional_body()
    param_list = ", ".join(f"{_C_UINT[p.ty]} {p.name}" for p in spec.inputs)
    body_str = _emit(body)
    return (
        "#include <stdint.h>\n\n"
        f"{_C_UINT[body.ty]} {fn_name}({param_list}) {{\n"
        f"    return ({_C_UINT[body.ty]})({body_str});\n"
        "}\n"
    )


def _emit(e: Expr) -> str:
    if isinstance(e, Const):
        suffix = "ULL" if e.ty is Ty.I64 else "U"
        return f"(({_C_UINT[e.ty]}){e.value}{suffix})"
    if isinstance(e, Var):
        return f"(({_C_UINT[e.ty]}){e.name})"
    if isinstance(e, Bin):
        return _emit_bin(e)
    if isinstance(e, Un):
        return _emit_un(e)
    if isinstance(e, Select):
        return f"(({_emit(e.cond)}) ? ({_emit(e.then)}) : ({_emit(e.else_)}))"
    raise TypeError(f"unknown Expr: {type(e).__name__}")


def _emit_bin(e: Bin) -> str:
    u = _C_UINT[e.ty]
    s = _C_INT[e.ty]
    l = _emit(e.lhs)
    r = _emit(e.rhs)
    op = e.op
    if op is BinOp.ADD:  return f"(({u})({l}) + ({u})({r}))"
    if op is BinOp.SUB:  return f"(({u})({l}) - ({u})({r}))"
    if op is BinOp.MUL:  return f"(({u})({l}) * ({u})({r}))"
    if op is BinOp.UDIV: return f"(({u})({l}) / ({u})({r}))"
    if op is BinOp.UREM: return f"(({u})({l}) % ({u})({r}))"
    if op is BinOp.SDIV: return f"(({u})(({s})({l}) / ({s})({r})))"
    if op is BinOp.SREM: return f"(({u})(({s})({l}) % ({s})({r})))"
    if op is BinOp.AND:  return f"(({u})({l}) & ({u})({r}))"
    if op is BinOp.OR:   return f"(({u})({l}) | ({u})({r}))"
    if op is BinOp.XOR:  return f"(({u})({l}) ^ ({u})({r}))"
    if op is BinOp.SHL:  return f"(({u})(({u})({l}) << ({u})({r})))"
    if op is BinOp.LSHR: return f"(({u})(({u})({l}) >> ({u})({r})))"
    if op is BinOp.ASHR: return f"(({u})(({s})({l}) >> ({u})({r})))"
    raise ValueError(f"unhandled BinOp: {op}")


def _emit_un(e: Un) -> str:
    u_arg = _C_UINT[e.arg.ty]
    u_res = _C_UINT[e.ty]
    s_arg = _C_INT[e.arg.ty]
    a = _emit(e.arg)
    op = e.op
    if op is UnOp.NEG:   return f"(({u_res})(-({u_arg})({a})))"
    if op is UnOp.NOT:   return f"(({u_res})(~({u_arg})({a})))"
    if op is UnOp.SEXT:  return f"(({u_res})({_C_INT[e.ty]})(({s_arg})({a})))"
    if op is UnOp.ZEXT:  return f"(({u_res})({u_arg})({a}))"
    if op is UnOp.TRUNC: return f"(({u_res})({a}))"
    raise ValueError(f"unhandled UnOp: {op}")
