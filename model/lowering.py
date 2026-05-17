"""Deterministic ComputeGraph → RV64GC assembly lowering.

Calling convention: lp64 (parameters in a0..a7 sign-extended to 64 bits; result in a0
sign-extended). The lowering pipeline is:

  1. Liveness analysis — last-use step per value reference.
  2. Linear-scan register allocation over a pool of caller-saved registers
     (a0..a7 + t0..t6 = 15 slots; spec inputs already occupy a0..a_{N-1} at entry).
     No stack spills in v0; ``UnsupportedLowering`` is raised if the pool runs out.
  3. Per-node instruction emission (-w forms for the 32-bit DSL; branch sequence
     for SELECT).
  4. Prologue is empty (no callee-saved use), epilogue is ``ret`` after moving the
     result into a0.

v1 restriction: only ``Ty.I32`` graphs are supported. Wider types raise
``UnsupportedLowering`` until the allocator + per-op tables are extended.
"""
from __future__ import annotations

from dataclasses import dataclass

from model.graph import ARITY, ComputeGraph, GraphNode, NodeOp
from spec.dsl import Ty


class UnsupportedLowering(RuntimeError):
    """Raised when the v0 lowering can't handle a graph (e.g. ran out of registers
    or hit a non-i32 type)."""


# Caller-saved register pool. Inputs claim a0..a_{N-1} on entry; the remaining a-regs
# plus all t-regs are immediately available for temporaries.
_TEMP_POOL: tuple[str, ...] = ("t0", "t1", "t2", "t3", "t4", "t5", "t6",
                                "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7")


@dataclass
class _AsmState:
    lines: list[str]
    label_counter: int

    def emit(self, line: str) -> None:
        self.lines.append(line)

    def next_label(self, prefix: str) -> str:
        n = self.label_counter
        self.label_counter += 1
        return f".L{prefix}{n}"


def lower(graph: ComputeGraph, fn_name: str = "spec_fn") -> str:
    """Lower a ComputeGraph to RV64GC asm text (compatible with eval.verify_io)."""
    _check_supported(graph)

    # Liveness: last step that references each ref (input or emitted node).
    n_in = len(graph.inputs)
    last_use: dict[int, int] = {}
    for step, node in enumerate(graph.nodes):
        for op in node.operands:
            last_use[op] = step  # later occurrences overwrite earlier — keeps last

    state = _AsmState(lines=[], label_counter=0)
    state.emit("\t.text")
    state.emit(f"\t.globl\t{fn_name}")
    state.emit(f"\t.type\t{fn_name},@function")
    state.emit(f"{fn_name}:")

    # Initial register map: inputs in a0..a_{N-1}.
    ref_to_reg: dict[int, str] = {i: f"a{i}" for i in range(n_in)}
    # Free pool excludes regs currently in use by inputs.
    free_regs: list[str] = [r for r in _TEMP_POOL if r not in ref_to_reg.values()]

    def free_dead(after_step: int) -> None:
        for ref in list(ref_to_reg.keys()):
            if last_use.get(ref, -1) < after_step:
                reg = ref_to_reg.pop(ref)
                # Avoid double-free if the reg was already returned to pool.
                if reg not in free_regs:
                    free_regs.append(reg)

    def alloc_reg(node_label: str) -> str:
        if not free_regs:
            raise UnsupportedLowering(
                f"v0 lowering ran out of registers while emitting {node_label!r}; "
                "graph exceeds the 15-register caller-saved budget"
            )
        # Stable order: pop from the head so allocations are predictable.
        return free_regs.pop(0)

    for step, node in enumerate(graph.nodes):
        # Free anything whose last use was before this step.
        free_dead(after_step=step)

        if node.op is NodeOp.RETURN:
            src = ref_to_reg[node.operands[0]]
            if src != "a0":
                state.emit(f"\tmv\ta0,{src}")
            state.emit("\tret")
            break

        out_ref = n_in + step
        out_reg = alloc_reg(node.op.name)
        _emit_node(node, ref_to_reg, out_reg, state)
        ref_to_reg[out_ref] = out_reg
    else:
        raise UnsupportedLowering("graph has no RETURN node")

    state.emit(f"\t.size\t{fn_name}, .-{fn_name}")
    return "\n".join(state.lines) + "\n"


def _check_supported(graph: ComputeGraph) -> None:
    """v0 only handles single-type i32 graphs with ops in NodeOp (no SEXT/ZEXT/TRUNC,
    enforced by ``ast_to_graph``)."""
    for p in graph.inputs:
        if p.ty is not Ty.I32:
            raise UnsupportedLowering(
                f"v0 lowering supports i32 only; input {p.name} is {p.ty.name}"
            )
    for i, node in enumerate(graph.nodes):
        if node.out_ty is not Ty.I32:
            raise UnsupportedLowering(
                f"v0 lowering supports i32 only; node {i} ({node.op.name}) is "
                f"{node.out_ty.name}"
            )


def _emit_node(
    node: GraphNode,
    ref_to_reg: dict[int, str],
    rd: str,
    state: _AsmState,
) -> None:
    op = node.op

    if op is NodeOp.CONST:
        v = (node.const_value or 0) & 0xFFFFFFFF
        # Convert to a 32-bit signed literal so `li` picks the shortest sequence.
        if v >= 0x8000_0000:
            v -= 0x1_0000_0000
        state.emit(f"\tli\t{rd},{v}")
        return

    operand_regs = [ref_to_reg[p] for p in node.operands]
    a = operand_regs[0] if operand_regs else None
    b = operand_regs[1] if len(operand_regs) > 1 else None
    c = operand_regs[2] if len(operand_regs) > 2 else None

    if op is NodeOp.ADD:  state.emit(f"\taddw\t{rd},{a},{b}"); return
    if op is NodeOp.SUB:  state.emit(f"\tsubw\t{rd},{a},{b}"); return
    if op is NodeOp.MUL:  state.emit(f"\tmulw\t{rd},{a},{b}"); return
    if op is NodeOp.SDIV: state.emit(f"\tdivw\t{rd},{a},{b}"); return
    if op is NodeOp.UDIV: state.emit(f"\tdivuw\t{rd},{a},{b}"); return
    if op is NodeOp.SREM: state.emit(f"\tremw\t{rd},{a},{b}"); return
    if op is NodeOp.UREM: state.emit(f"\tremuw\t{rd},{a},{b}"); return
    if op is NodeOp.AND:  state.emit(f"\tand\t{rd},{a},{b}"); return
    if op is NodeOp.OR:   state.emit(f"\tor\t{rd},{a},{b}"); return
    if op is NodeOp.XOR:  state.emit(f"\txor\t{rd},{a},{b}"); return
    if op is NodeOp.SHL:  state.emit(f"\tsllw\t{rd},{a},{b}"); return
    if op is NodeOp.LSHR: state.emit(f"\tsrlw\t{rd},{a},{b}"); return
    if op is NodeOp.ASHR: state.emit(f"\tsraw\t{rd},{a},{b}"); return
    if op is NodeOp.NEG:  state.emit(f"\tnegw\t{rd},{a}"); return
    if op is NodeOp.NOT:
        # xori with sign-extended -1 on a sign-extended i32 yields the sign-extended
        # complement — no separate sext.w needed.
        state.emit(f"\txori\t{rd},{a},-1")
        return
    if op is NodeOp.SELECT:
        # rd = (cond != 0) ? then : else
        # Branch sequence with unique labels.
        else_lbl = state.next_label("sel_else_")
        end_lbl = state.next_label("sel_end_")
        state.emit(f"\tbeqz\t{a},{else_lbl}")
        state.emit(f"\tmv\t{rd},{b}")
        state.emit(f"\tj\t{end_lbl}")
        state.emit(f"{else_lbl}:")
        state.emit(f"\tmv\t{rd},{c}")
        state.emit(f"{end_lbl}:")
        return
    raise UnsupportedLowering(f"unhandled NodeOp: {op}")
