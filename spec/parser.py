"""Text DSL parser for FOL Hoare specs.

Surface syntax (see examples/specs/):

    spec sq_sum(x: i32, y: i32) -> r: i32 {
        post: r == ((x * x) + (y * y)) >>u 1
    }

    spec isqrt(x: i32) -> r: i32 {
        pre:  x >=s 0
        post: r *u r <=u x  &  (r + 1) *u (r + 1) >u x
    }

Tokens
------
  - **Types**: ``i8 i16 i32 i64``.
  - **Term ops**: ``+ - * & | ^ ~`` plus signedness-tagged ``/s /u %s %u >>s >>u <<``.
  - **Comparisons** (term×term → formula): ``== != <s <=s >s >=s <u <=u >u >=u``.
  - **Formula connectives**: ``~ & | -> <->``, keywords ``true false``.
  - **Quantifiers**: ``forall x: i32. F``, ``exists x: i32. F``.
  - **Built-ins** (term-forming): ``sext(e, ty)``, ``zext(e, ty)``, ``trunc(e, ty)``,
    ``select(c, t, e)``.

The grammar separates term and formula rules; the boundary is ``cmp_atom: term CMP term``.
This makes ``&`` unambiguously bitwise inside terms and conjunction inside formulas.

Constants without an explicit ``:ty`` annotation default to ``i32``. To override, write
``123:i64``. Type-check is run after parsing — mismatches throw with the lark text span.
"""
from __future__ import annotations

from typing import Any

from lark import Lark, Transformer, v_args

from spec.dsl import (
    And, Bin, BinOp, BoolFalse, BoolTrue, Cmp, CmpOp, Const, Exists, Expr, Forall,
    Formula, Iff, Implies, Not, Or, Param, Select, Spec, Ty, Un, UnOp, Var,
    mask_to, type_check,
)


_GRAMMAR = r"""
start: spec+

spec: "spec" CNAME "(" [param ("," param)*] ")" "->" param "{" clause+ "}"

param: CNAME ":" ty

ty: "i8" -> ty_i8 | "i16" -> ty_i16 | "i32" -> ty_i32 | "i64" -> ty_i64

clause: "pre"  ":" formula                                                -> pre_clause
      | "post" ":" formula                                                -> post_clause

?formula: quant
        | iff_e
quant:    "forall" CNAME ":" ty "." formula                               -> qforall
        | "exists" CNAME ":" ty "." formula                               -> qexists
?iff_e:    implies_e ("<->" implies_e)*                                   -> iff_chain
?implies_e: or_e ("->" or_e)?                                             -> implies_chain
?or_e:     and_e ("|" and_e)*                                             -> or_chain
?and_e:    not_e ("&" not_e)*                                             -> and_chain
?not_e:    "~" not_e                                                      -> fnot
         | atom_f
?atom_f:  "true"                                                          -> btrue
        | "false"                                                         -> bfalse
        | "(" formula ")"
        | cmp_atom
cmp_atom: term CMP_OP term

CMP_OP: "<=s" | "<=u" | ">=s" | ">=u" | "<s" | "<u" | ">s" | ">u" | "==" | "!="

?term: shift_t
?shift_t: add_t (shift_op add_t)*                                         -> shift_chain
?add_t:   mul_t (add_op mul_t)*                                           -> add_chain
?mul_t:   bit_t (mul_op bit_t)*                                           -> mul_chain
?bit_t:   unary_t (bit_op unary_t)*                                       -> bit_chain
?unary_t: "-" unary_t                                                     -> tneg
        | "~" unary_t                                                     -> tnot
        | atom_t

shift_op: ">>s" -> ashr_op | ">>u" -> lshr_op | "<<" -> shl_op
add_op:   "+"   -> add_op  | "-"   -> sub_op
mul_op:   "*"   -> mul_op  | "/s"  -> sdiv_op | "/u" -> udiv_op | "%s" -> srem_op | "%u" -> urem_op
bit_op:   "&"   -> band_op | "|"   -> bor_op  | "^"  -> bxor_op

?atom_t: builtin_call
       | "(" term ")"
       | typed_int                                                         -> const_typed
       | INT                                                               -> const_default
       | CNAME                                                             -> var

typed_int: INT ":" ty

builtin_call: "sext"   "(" term "," ty ")"                                 -> b_sext
            | "zext"   "(" term "," ty ")"                                 -> b_zext
            | "trunc"  "(" term "," ty ")"                                 -> b_trunc
            | "select" "(" term "," term "," term ")"                      -> b_select

INT: /-?(0x[0-9a-fA-F]+|\d+)/

%import common.CNAME
%import common.WS
%ignore WS
COMMENT: "#" /[^\n]*/
%ignore COMMENT
"""


_BIN_TOK_TO_OP = {
    "add_op": BinOp.ADD, "sub_op": BinOp.SUB,
    "mul_op": BinOp.MUL,
    "sdiv_op": BinOp.SDIV, "udiv_op": BinOp.UDIV,
    "srem_op": BinOp.SREM, "urem_op": BinOp.UREM,
    "band_op": BinOp.AND, "bor_op": BinOp.OR, "bxor_op": BinOp.XOR,
    "shl_op": BinOp.SHL, "lshr_op": BinOp.LSHR, "ashr_op": BinOp.ASHR,
}


_CMP_TOK_TO_OP = {
    "==": CmpOp.EQ, "!=": CmpOp.NE,
    "<s": CmpOp.SLT, "<=s": CmpOp.SLE, ">s": CmpOp.SGT, ">=s": CmpOp.SGE,
    "<u": CmpOp.ULT, "<=u": CmpOp.ULE, ">u": CmpOp.UGT, ">=u": CmpOp.UGE,
}


DEFAULT_CONST_TY = Ty.I32


class _Build(Transformer):
    """Walks the lark tree, producing DSL nodes. Constants without explicit type take
    on ``DEFAULT_CONST_TY``; the type checker fires after construction if they don't
    match an operand."""

    # -- type rules
    def ty_i8(self, _):  return Ty.I8
    def ty_i16(self, _): return Ty.I16
    def ty_i32(self, _): return Ty.I32
    def ty_i64(self, _): return Ty.I64

    def ty(self, items):  return items[0]

    # -- param + spec
    @v_args(inline=True)
    def param(self, name, ty):
        return Param(name=str(name), ty=ty)

    @v_args(inline=True)
    def pre_clause(self, f):  return ("pre", f)
    @v_args(inline=True)
    def post_clause(self, f): return ("post", f)

    def spec(self, items):
        name = str(items[0])
        params_then_clauses = items[1:]
        clauses_start = None
        for i, x in enumerate(params_then_clauses):
            if isinstance(x, tuple) and x and x[0] in ("pre", "post"):
                clauses_start = i
                break
        if clauses_start is None or clauses_start < 1:
            raise SyntaxError(f"spec {name!r}: malformed body")
        inputs = tuple(params_then_clauses[:clauses_start - 1])
        output = params_then_clauses[clauses_start - 1]
        clause_pairs = params_then_clauses[clauses_start:]

        env: dict[str, Ty] = {p.name: p.ty for p in inputs}
        env[output.name] = output.ty

        pre: Formula = BoolTrue()
        post: Formula | None = None
        for kind, f in clause_pairs:
            if kind == "pre":
                pre = _retype(f, env)
            elif kind == "post":
                post = _retype(f, env)
        if post is None:
            raise SyntaxError(f"spec {name!r}: missing post clause")
        spec = Spec(inputs=inputs, outputs=(output,), pre=pre, post=post)
        type_check(spec)
        return (name, spec)

    def start(self, items):
        return dict(items)  # {name: Spec}

    # -- atoms
    @v_args(inline=True)
    def var(self, tok):
        # Type filled in by parent operation; default to i32 if used in isolation.
        # We can't infer here — emit a sentinel and let the operator-construction code
        # type the leaf based on the env at the spec-build call site. We don't have an
        # env here, so all Var nodes are emitted as i32; the type checker will catch
        # mismatches.
        return Var(ty=DEFAULT_CONST_TY, name=str(tok))

    @v_args(inline=True)
    def const_default(self, tok):
        return Const(ty=DEFAULT_CONST_TY, value=mask_to(int(str(tok), 0), DEFAULT_CONST_TY))

    @v_args(inline=True)
    def const_typed(self, typed):
        return typed  # typed_int builds the Const

    @v_args(inline=True)
    def typed_int(self, num, ty):
        return Const(ty=ty, value=mask_to(int(str(num), 0), ty))

    # -- builtins
    @v_args(inline=True)
    def b_sext(self, e, ty): return Un(ty=ty, op=UnOp.SEXT, arg=e)
    @v_args(inline=True)
    def b_zext(self, e, ty): return Un(ty=ty, op=UnOp.ZEXT, arg=e)
    @v_args(inline=True)
    def b_trunc(self, e, ty): return Un(ty=ty, op=UnOp.TRUNC, arg=e)
    @v_args(inline=True)
    def b_select(self, c, t, e): return Select(ty=t.ty, cond=c, then=t, else_=e)

    # -- unary term ops (fold unary minus/not on a literal so type coercion sees a Const)
    @v_args(inline=True)
    def tneg(self, x):
        if isinstance(x, Const):
            return Const(ty=x.ty, value=mask_to(-x.value, x.ty))
        return Un(ty=x.ty, op=UnOp.NEG, arg=x)

    @v_args(inline=True)
    def tnot(self, x):
        if isinstance(x, Const):
            return Const(ty=x.ty, value=mask_to(~x.value, x.ty))
        return Un(ty=x.ty, op=UnOp.NOT, arg=x)

    # -- chained binary term ops (left-assoc)
    def _binop_chain(self, items, op_extract=lambda x: x):
        # items = [lhs, op1, mid1, op2, ..., rhs]
        acc = items[0]
        for i in range(1, len(items), 2):
            op_tree = items[i]
            rhs = items[i + 1]
            op_name = op_tree.data if hasattr(op_tree, "data") else str(op_tree)
            op = _BIN_TOK_TO_OP[op_name]
            # Type follows lhs; type checker will enforce equality with rhs.
            acc = Bin(ty=self._coerce_const(acc, rhs).ty if isinstance(acc, Const) else acc.ty,
                      op=op, lhs=self._coerce_const(acc, rhs), rhs=self._coerce_const(rhs, acc))
        return acc

    def _coerce_const(self, e: Expr, other: Expr) -> Expr:
        """If `e` is a default-typed Const and `other` has a different type, retype `e`."""
        if isinstance(e, Const) and e.ty == DEFAULT_CONST_TY and isinstance(other, Expr) and other.ty != DEFAULT_CONST_TY:
            return Const(ty=other.ty, value=mask_to(e.value, other.ty))
        return e

    def shift_chain(self, items): return self._binop_chain(items)
    def add_chain(self, items):   return self._binop_chain(items)
    def mul_chain(self, items):   return self._binop_chain(items)
    def bit_chain(self, items):   return self._binop_chain(items)

    # -- formula leaves + connectives
    def btrue(self, _):  return BoolTrue()
    def bfalse(self, _): return BoolFalse()

    @v_args(inline=True)
    def cmp_atom(self, lhs, op_tok, rhs):
        op = _CMP_TOK_TO_OP[str(op_tok)]
        # Coerce one side if the other is typed
        lhs = self._coerce_const(lhs, rhs)
        rhs = self._coerce_const(rhs, lhs)
        return Cmp(op=op, lhs=lhs, rhs=rhs)

    def and_chain(self, items):
        acc = items[0]
        for x in items[1:]:
            acc = And(lhs=acc, rhs=x)
        return acc

    def or_chain(self, items):
        acc = items[0]
        for x in items[1:]:
            acc = Or(lhs=acc, rhs=x)
        return acc

    def implies_chain(self, items):
        if len(items) == 1:
            return items[0]
        return Implies(ant=items[0], cons=items[1])

    def iff_chain(self, items):
        acc = items[0]
        for x in items[1:]:
            acc = Iff(lhs=acc, rhs=x)
        return acc

    @v_args(inline=True)
    def fnot(self, x): return Not(arg=x)

    @v_args(inline=True)
    def qforall(self, name, ty, body):
        return Forall(var=Param(name=str(name), ty=ty), body=self._retag_var(body, str(name), ty))

    @v_args(inline=True)
    def qexists(self, name, ty, body):
        return Exists(var=Param(name=str(name), ty=ty), body=self._retag_var(body, str(name), ty))

    @v_args(inline=True)
    def qforall(self, name, ty, body):
        return Forall(var=Param(name=str(name), ty=ty), body=body)

    @v_args(inline=True)
    def qexists(self, name, ty, body):
        return Exists(var=Param(name=str(name), ty=ty), body=body)


# ---- post-parse retype pass -------------------------------------------------

def _retype(node, env: dict[str, Ty], hint: Ty | None = None):
    """Top-down typing pass: set Var types from env, coerce default-typed Consts toward
    their typed neighbors, propagate types through operators."""
    if isinstance(node, Var):
        if node.name in env:
            return Var(ty=env[node.name], name=node.name)
        if hint is not None and node.ty == DEFAULT_CONST_TY:
            return Var(ty=hint, name=node.name)
        return node
    if isinstance(node, Const):
        if hint is not None and node.ty == DEFAULT_CONST_TY:
            return Const(ty=hint, value=mask_to(node.value, hint))
        return node
    if isinstance(node, Bin):
        # First pass: retype with the parent's hint (may pick up types via the env).
        lhs = _retype(node.lhs, env, hint)
        rhs = _retype(node.rhs, env, hint)
        # Coerce default-typed consts to their non-const neighbor's type.
        if isinstance(lhs, Const) and lhs.ty == DEFAULT_CONST_TY and not isinstance(rhs, Const):
            lhs = _retype(node.lhs, env, rhs.ty)
        if isinstance(rhs, Const) and rhs.ty == DEFAULT_CONST_TY and not isinstance(lhs, Const):
            rhs = _retype(node.rhs, env, lhs.ty)
        return Bin(ty=lhs.ty, op=node.op, lhs=lhs, rhs=rhs)
    if isinstance(node, Un):
        if node.op in (UnOp.NEG, UnOp.NOT):
            arg = _retype(node.arg, env, hint)
            return Un(ty=arg.ty, op=node.op, arg=arg)
        # SEXT, ZEXT, TRUNC: result ty is explicitly set by the builtin call.
        arg = _retype(node.arg, env, None)
        return Un(ty=node.ty, op=node.op, arg=arg)
    if isinstance(node, Select):
        # then/else drive result ty; cond can be any int.
        then = _retype(node.then, env, hint)
        else_ = _retype(node.else_, env, then.ty if not isinstance(then, Const) or then.ty != DEFAULT_CONST_TY else hint)
        if isinstance(then, Const) and then.ty == DEFAULT_CONST_TY and not isinstance(else_, Const):
            then = _retype(node.then, env, else_.ty)
        cond = _retype(node.cond, env, None)
        return Select(ty=then.ty, cond=cond, then=then, else_=else_)
    if isinstance(node, (BoolTrue, BoolFalse)):
        return node
    if isinstance(node, Cmp):
        lhs = _retype(node.lhs, env, None)
        rhs = _retype(node.rhs, env, None)
        if isinstance(lhs, Const) and lhs.ty == DEFAULT_CONST_TY and not isinstance(rhs, Const):
            lhs = _retype(node.lhs, env, rhs.ty)
        if isinstance(rhs, Const) and rhs.ty == DEFAULT_CONST_TY and not isinstance(lhs, Const):
            rhs = _retype(node.rhs, env, lhs.ty)
        return Cmp(op=node.op, lhs=lhs, rhs=rhs)
    if isinstance(node, Not):
        return Not(arg=_retype(node.arg, env, None))
    if isinstance(node, And):
        return And(lhs=_retype(node.lhs, env, None), rhs=_retype(node.rhs, env, None))
    if isinstance(node, Or):
        return Or(lhs=_retype(node.lhs, env, None), rhs=_retype(node.rhs, env, None))
    if isinstance(node, Implies):
        return Implies(ant=_retype(node.ant, env, None), cons=_retype(node.cons, env, None))
    if isinstance(node, Iff):
        return Iff(lhs=_retype(node.lhs, env, None), rhs=_retype(node.rhs, env, None))
    if isinstance(node, (Forall, Exists)):
        inner_env = {**env, node.var.name: node.var.ty}
        new_body = _retype(node.body, inner_env, None)
        return type(node)(var=node.var, body=new_body)
    return node


_parser = Lark(_GRAMMAR, parser="earley", propagate_positions=True, ambiguity="resolve")


def parse(text: str) -> dict[str, Spec]:
    """Parse text containing one or more ``spec`` blocks; return ``{name: Spec}``.
    Type-check is run on each spec; raises ``TypeError`` on the first violation."""
    tree = _parser.parse(text)
    builder = _Build()
    return builder.transform(tree)


def parse_one(text: str) -> Spec:
    """Parse text containing exactly one spec; return the Spec."""
    specs = parse(text)
    if len(specs) != 1:
        raise ValueError(f"expected exactly one spec, got {len(specs)}")
    return next(iter(specs.values()))


def parse_file(path: str) -> dict[str, Spec]:
    with open(path, "r") as fh:
        return parse(fh.read())
