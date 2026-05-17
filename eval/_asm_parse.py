"""Shared asm helpers: parse the body of a named function out of a -S output."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Insn:
    op: str
    operands: tuple[str, ...]
    raw: str


_INSTR_RE = re.compile(r"^\s+([a-z][a-z0-9_\.]*)(?:\s+(.+?))?\s*(?:#.*)?$")
_LABEL_RE = re.compile(r"^([A-Za-z_\.][\w\.]*):\s*(?:#.*)?$")
_FUNC_LABEL_RE = re.compile(r"^([A-Za-z_][\w]*):\s*$")


def parse_function(asm: str, fn_name: str) -> list[Insn]:
    """Extract the instruction list of ``fn_name`` from a gcc/clang -S output.

    Skips directives (lines starting with ``.``), comments, blank lines, and
    intermediate labels. The function body ends at ``.size fn_name, .-fn_name`` or
    ``.cfi_endproc``.
    """
    out: list[Insn] = []
    inside = False
    for raw in asm.splitlines():
        line = raw.rstrip()
        if not inside:
            m = _FUNC_LABEL_RE.match(line)
            if m and m.group(1) == fn_name:
                inside = True
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        if stripped.startswith(f".size\t{fn_name}") or stripped.startswith(f".size {fn_name}"):
            break
        if stripped.startswith(".cfi_endproc"):
            break
        if stripped.startswith("."):
            continue
        m = _LABEL_RE.match(line)
        if m and m.group(1) != fn_name:
            continue
        m = _INSTR_RE.match(line)
        if not m:
            continue
        op = m.group(1)
        rest = m.group(2) or ""
        operands = tuple(s.strip() for s in rest.split(",")) if rest else ()
        out.append(Insn(op=op, operands=operands, raw=line))
    if not inside:
        raise ValueError(f"function label {fn_name!r} not found in asm")
    return out


# RISC-V ABI register name → numeric index.
_REG_ALIASES = {
    "zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4,
    "t0": 5, "t1": 6, "t2": 7,
    "s0": 8, "fp": 8, "s1": 9,
    "a0": 10, "a1": 11, "a2": 12, "a3": 13, "a4": 14,
    "a5": 15, "a6": 16, "a7": 17,
    "s2": 18, "s3": 19, "s4": 20, "s5": 21, "s6": 22,
    "s7": 23, "s8": 24, "s9": 25, "s10": 26, "s11": 27,
    "t3": 28, "t4": 29, "t5": 30, "t6": 31,
}


def reg_index(name: str) -> int:
    """Resolve an ABI register name (a0, sp, ...) or x<N> to an integer 0..31."""
    n = name.strip()
    if n in _REG_ALIASES:
        return _REG_ALIASES[n]
    if n.startswith("x") and n[1:].isdigit():
        i = int(n[1:])
        if 0 <= i < 32:
            return i
    raise ValueError(f"unknown register: {name!r}")


def parse_imm(tok: str) -> int:
    """Parse a RISC-V immediate (decimal, hex, or negative)."""
    tok = tok.strip()
    sign = 1
    if tok.startswith("-"):
        sign = -1
        tok = tok[1:]
    if tok.startswith("0x") or tok.startswith("0X"):
        return sign * int(tok, 16)
    return sign * int(tok, 10)
