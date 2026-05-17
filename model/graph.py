"""ComputeGraph: the intermediate DAG the GNN generates between spec and asm.

A graph is a sequence of typed operation nodes, each with operand pointers into a
**unified node space**:

  - indices ``0 .. len(inputs)-1`` refer to spec inputs (positional);
  - indices ``len(inputs) .. len(inputs)+k-1`` refer to previously-emitted graph nodes.

Generation order is post-order on the source AST — children before parents. The last
node is always a ``RETURN`` whose single operand is the graph's output.

v1 vocabulary restriction: no SEXT/ZEXT/TRUNC (would require a per-node output-type
choice). Specs whose body uses these ops are filtered out of training data — see
``ast_to_graph`` for the exception path.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from spec.dsl import (
    Bin, BinOp, Const, Expr, Param, Select, Spec, Ty, Un, UnOp, Var,
)


class NodeOp(Enum):
    """Operations the graph generator can emit (plus INPUT as a non-emitted placeholder)."""
    INPUT = "input"     # placeholder for spec inputs — never emitted by the model
    CONST = "const"     # 0 operands, carries const_value
    # binary
    ADD = "add"; SUB = "sub"; MUL = "mul"
    SDIV = "sdiv"; UDIV = "udiv"; SREM = "srem"; UREM = "urem"
    AND = "and"; OR = "or"; XOR = "xor"
    SHL = "shl"; LSHR = "lshr"; ASHR = "ashr"
    # unary (same-type)
    NEG = "neg"; NOT = "not"
    # ternary
    SELECT = "select"
    # terminal
    RETURN = "return"


ARITY: dict[NodeOp, int] = {
    NodeOp.INPUT: 0,
    NodeOp.CONST: 0,
    NodeOp.ADD: 2, NodeOp.SUB: 2, NodeOp.MUL: 2,
    NodeOp.SDIV: 2, NodeOp.UDIV: 2, NodeOp.SREM: 2, NodeOp.UREM: 2,
    NodeOp.AND: 2, NodeOp.OR: 2, NodeOp.XOR: 2,
    NodeOp.SHL: 2, NodeOp.LSHR: 2, NodeOp.ASHR: 2,
    NodeOp.NEG: 1, NodeOp.NOT: 1,
    NodeOp.SELECT: 3,
    NodeOp.RETURN: 1,
}

# Map dsl BinOp/UnOp → NodeOp.
_BIN_OP_MAP: dict[BinOp, NodeOp] = {
    BinOp.ADD: NodeOp.ADD, BinOp.SUB: NodeOp.SUB, BinOp.MUL: NodeOp.MUL,
    BinOp.SDIV: NodeOp.SDIV, BinOp.UDIV: NodeOp.UDIV,
    BinOp.SREM: NodeOp.SREM, BinOp.UREM: NodeOp.UREM,
    BinOp.AND: NodeOp.AND, BinOp.OR: NodeOp.OR, BinOp.XOR: NodeOp.XOR,
    BinOp.SHL: NodeOp.SHL, BinOp.LSHR: NodeOp.LSHR, BinOp.ASHR: NodeOp.ASHR,
}
_UN_OP_MAP: dict[UnOp, NodeOp] = {UnOp.NEG: NodeOp.NEG, UnOp.NOT: NodeOp.NOT}


@dataclass(frozen=True)
class GraphNode:
    op: NodeOp
    operands: tuple[int, ...]  # indices into the unified [inputs ++ nodes] space
    out_ty: Ty
    const_value: int | None = None  # only for CONST


@dataclass(frozen=True)
class ComputeGraph:
    inputs: tuple[Param, ...]
    nodes: tuple[GraphNode, ...]  # last node is always RETURN

    @property
    def root_ref(self) -> int:
        """Index of the output node in the unified space."""
        return self.nodes[-1].operands[0]

    @property
    def n_emitted(self) -> int:
        """Number of generation steps (== number of emitted nodes, incl. RETURN)."""
        return len(self.nodes)

    def node_ty(self, ref: int) -> Ty:
        """Return the type of the value at `ref` (input or emitted node)."""
        n_in = len(self.inputs)
        if ref < n_in:
            return self.inputs[ref].ty
        return self.nodes[ref - n_in].out_ty


class UnsupportedAstOp(ValueError):
    """Raised when ast_to_graph hits a DSL op outside the v1 model vocabulary."""


def ast_to_graph(spec: Spec) -> ComputeGraph:
    """AST-as-graph reference target: walk the functional spec's body in post-order and
    emit one GraphNode per AST node. **No CSE** — identical sub-expressions become
    duplicate nodes. This keeps the teacher signal close to the source AST.

    Raises ``UnsupportedAstOp`` if the body contains SEXT/ZEXT/TRUNC (v1 vocab).
    """
    if not spec.is_functional():
        raise ValueError("ast_to_graph: requires a functional spec")
    inputs = spec.inputs
    n_in = len(inputs)
    body = spec.functional_body()
    nodes: list[GraphNode] = []
    name_to_input_idx = {p.name: i for i, p in enumerate(inputs)}

    def emit(e: Expr) -> int:
        if isinstance(e, Var):
            if e.name not in name_to_input_idx:
                raise UnsupportedAstOp(f"unbound var {e.name!r} in body")
            return name_to_input_idx[e.name]
        if isinstance(e, Const):
            nodes.append(GraphNode(
                op=NodeOp.CONST, operands=(), out_ty=e.ty, const_value=e.value,
            ))
            return n_in + len(nodes) - 1
        if isinstance(e, Bin):
            l = emit(e.lhs)
            r = emit(e.rhs)
            nodes.append(GraphNode(
                op=_BIN_OP_MAP[e.op], operands=(l, r), out_ty=e.ty,
            ))
            return n_in + len(nodes) - 1
        if isinstance(e, Un):
            if e.op not in _UN_OP_MAP:
                raise UnsupportedAstOp(f"v1 model vocab does not include {e.op.name}")
            a = emit(e.arg)
            nodes.append(GraphNode(op=_UN_OP_MAP[e.op], operands=(a,), out_ty=e.ty))
            return n_in + len(nodes) - 1
        if isinstance(e, Select):
            c = emit(e.cond)
            t = emit(e.then)
            f = emit(e.else_)
            nodes.append(GraphNode(
                op=NodeOp.SELECT, operands=(c, t, f), out_ty=e.ty,
            ))
            return n_in + len(nodes) - 1
        raise TypeError(f"unknown Expr: {type(e).__name__}")

    root = emit(body)
    nodes.append(GraphNode(op=NodeOp.RETURN, operands=(root,), out_ty=body.ty))
    return ComputeGraph(inputs=inputs, nodes=tuple(nodes))


# ---- generation-step view (teacher-forcing data) ---------------------------

@dataclass(frozen=True)
class GenStep:
    """One generation step: emit ``op`` with ``operands`` (and ``const_value`` for CONST).
    All fields are aligned with what the model's heads predict.
    """
    op: NodeOp
    operands: tuple[int, ...]
    const_value: int | None  # only for CONST


def graph_to_steps(g: ComputeGraph) -> list[GenStep]:
    """One step per emitted node (incl. terminal RETURN). At step k, the candidate
    operand-pointer set is {inputs ++ nodes[:k]} of size len(inputs) + k."""
    return [
        GenStep(op=n.op, operands=n.operands, const_value=n.const_value)
        for n in g.nodes
    ]


def evaluate_graph(g: ComputeGraph, input_values: tuple[int, ...]) -> int:
    """Reference evaluator over a ComputeGraph — used for sanity-checking that AST
    extraction round-trips through the graph form."""
    from spec.dsl import mask_to, signed
    n_in = len(g.inputs)
    vals: list[int] = [mask_to(v, p.ty) for v, p in zip(input_values, g.inputs)]
    for node in g.nodes:
        if node.op is NodeOp.CONST:
            vals.append(node.const_value & node.out_ty.mask)
            continue
        if node.op is NodeOp.RETURN:
            return vals[node.operands[0]]
        operand_vals = [vals[i] for i in node.operands]
        ty = node.out_ty
        vals.append(_apply_op(node.op, operand_vals, ty))
    raise RuntimeError("graph has no RETURN node")


def _apply_op(op: NodeOp, operands: list[int], ty: Ty) -> int:
    from spec.dsl import mask_to, signed
    if op is NodeOp.ADD: return mask_to(operands[0] + operands[1], ty)
    if op is NodeOp.SUB: return mask_to(operands[0] - operands[1], ty)
    if op is NodeOp.MUL: return mask_to(operands[0] * operands[1], ty)
    if op is NodeOp.AND: return operands[0] & operands[1]
    if op is NodeOp.OR:  return operands[0] | operands[1]
    if op is NodeOp.XOR: return operands[0] ^ operands[1]
    if op is NodeOp.UDIV:
        if operands[1] == 0:
            raise ZeroDivisionError
        return operands[0] // operands[1]
    if op is NodeOp.UREM:
        if operands[1] == 0:
            raise ZeroDivisionError
        return operands[0] % operands[1]
    if op is NodeOp.SDIV:
        if operands[1] == 0:
            raise ZeroDivisionError
        ls, rs = signed(operands[0], ty), signed(operands[1], ty)
        q = abs(ls) // abs(rs)
        if (ls < 0) ^ (rs < 0):
            q = -q
        return mask_to(q, ty)
    if op is NodeOp.SREM:
        if operands[1] == 0:
            raise ZeroDivisionError
        ls, rs = signed(operands[0], ty), signed(operands[1], ty)
        q = abs(ls) // abs(rs)
        if (ls < 0) ^ (rs < 0):
            q = -q
        return mask_to(ls - q * rs, ty)
    if op is NodeOp.SHL:  return mask_to(operands[0] << operands[1], ty)
    if op is NodeOp.LSHR: return operands[0] >> operands[1]
    if op is NodeOp.ASHR: return mask_to(signed(operands[0], ty) >> operands[1], ty)
    if op is NodeOp.NEG: return mask_to(-operands[0], ty)
    if op is NodeOp.NOT: return operands[0] ^ ty.mask
    if op is NodeOp.SELECT: return operands[1] if operands[0] != 0 else operands[2]
    raise ValueError(f"unhandled NodeOp: {op}")
