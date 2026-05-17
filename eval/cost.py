"""Static cost model for candidate asm — instruction count and an in-order RV64 cycle
estimate.

Only counts *instructions inside the named function*: skips directives (``.text``,
``.globl``, ``.size``, etc.), labels, comments, and the surrounding noise gcc emits.
The single-issue cycle model is per-op latency summed over the static schedule — no
bypass, no cache, no branch prediction. Sufficient for ranking short straight-line
sequences (v1 scope).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Cost:
    instruction_count: int
    estimated_cycles: int


# Latency table — covers the RV64GC scalar integer subset gcc/clang -O2 emits on the
# expression DSL. Unlisted instructions fall through to the default (1 cycle).
_LATENCY: dict[str, int] = {
    # ALU
    "add": 1, "addw": 1, "sub": 1, "subw": 1,
    "and": 1, "or": 1, "xor": 1,
    "sll": 1, "sllw": 1, "srl": 1, "srlw": 1, "sra": 1, "sraw": 1,
    "slt": 1, "sltu": 1,
    "addi": 1, "addiw": 1, "andi": 1, "ori": 1, "xori": 1,
    "slli": 1, "slliw": 1, "srli": 1, "srliw": 1, "srai": 1, "sraiw": 1,
    "slti": 1, "sltiu": 1,
    "lui": 1, "auipc": 1,
    # M-extension
    "mul": 3, "mulw": 3, "mulh": 3, "mulhsu": 3, "mulhu": 3,
    "div": 30, "divu": 30, "divw": 30, "divuw": 30,
    "rem": 30, "remu": 30, "remw": 30, "remuw": 30,
    # Memory
    "lb": 3, "lh": 3, "lw": 3, "ld": 3, "lbu": 3, "lhu": 3, "lwu": 3,
    "sb": 1, "sh": 1, "sw": 1, "sd": 1,
    # Branch / jump
    "beq": 1, "bne": 1, "blt": 1, "bge": 1, "bltu": 1, "bgeu": 1,
    "jal": 2, "jalr": 2,
    # Pseudo-ops gcc emits at -O2
    "mv": 1, "li": 1, "neg": 1, "negw": 1, "not": 1, "seqz": 1, "snez": 1,
    "ret": 2, "j": 2, "jr": 2, "nop": 1,
    "sext.w": 1, "zext.w": 1, "sext.b": 1, "sext.h": 1,
}

_INSTR_LINE_RE = re.compile(r"^\s+([a-z][a-z0-9_\.]*)(?:\s|$)")
_LABEL_RE = re.compile(r"^([A-Za-z_\.][\w\.]*):\s*$")
_FUNC_LABEL_RE = re.compile(r"^([A-Za-z_][\w]*):\s*$")


def estimate(candidate_asm: str, fn_name: str = "spec_fn") -> Cost:
    """Parse the asm and return the cost of the function body for ``fn_name``.

    Function body = everything between ``fn_name:`` (the function label) and the next
    blank line / next non-local label / explicit end-of-function marker (``.size``).
    """
    lines = candidate_asm.splitlines()
    inside = False
    n_instr = 0
    cycles = 0
    for raw in lines:
        line = raw.rstrip()
        if not inside:
            m = _FUNC_LABEL_RE.match(line)
            if m and m.group(1) == fn_name:
                inside = True
            continue
        # End-of-function markers
        if line.startswith("\t.size") and fn_name in line:
            break
        if line.strip().startswith(".cfi_endproc"):
            break
        # Skip empty, comments, directives, and local-label declarations
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        if stripped.startswith("."):
            continue
        m = _LABEL_RE.match(line)
        if m and m.group(1) != fn_name:
            # Another label inside the body (e.g. .L1:) — not an instruction
            continue
        m = _INSTR_LINE_RE.match(line)
        if not m:
            continue
        op = m.group(1)
        n_instr += 1
        cycles += _LATENCY.get(op, 1)
    if not inside:
        raise ValueError(f"function label {fn_name!r} not found in asm")
    return Cost(instruction_count=n_instr, estimated_cycles=cycles)
