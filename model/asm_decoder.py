"""Asm tokenizer/detokenizer + transformer decoder for RV64GC function bodies.

Vocabulary (one flat int space):
  0       PAD
  1       BOS
  2       EOS
  3       EOL   — end of an instruction line
  4       COMMA
  5       LPAREN
  6       RPAREN
  7       COLON — for label definitions like ``.L3:``
  8..NO   OPCODES (~120, common RV64GC -O2 ops)
  NO..NR  REGISTERS — 32 ABI names (zero, ra, sp, gp, tp, t0..t6, s0..s11, a0..a7)
  NR..NL1 LABEL_REF_*   — positional label references (e.g. branch targets)
  NL1..NL2 LABEL_DEF_*  — positional label definitions
  NL2..NI IMM buckets   — canonical immediate values + OOV
  NI      OOV_IMM

Per-function label normalization: ``.L3, .L4, .L10`` are renumbered positionally so
the same source produces the same token stream regardless of gcc's internal naming.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ---- Static asm vocab -------------------------------------------------------

PAD = 0
BOS = 1
EOS = 2
EOL = 3
COMMA = 4
LPAREN = 5
RPAREN = 6
COLON = 7

_OPCODES = [
    # RV64I + M + common pseudos seen in gcc -O2 on integer code.
    "add", "addi", "addw", "addiw", "sub", "subw", "neg", "negw",
    "and", "andi", "or", "ori", "xor", "xori", "not",
    "sll", "slli", "sllw", "slliw", "srl", "srli", "srlw", "srliw",
    "sra", "srai", "sraw", "sraiw",
    "slt", "sltu", "slti", "sltiu", "seqz", "snez", "sgtz", "sltz",
    "mul", "mulw", "mulh", "mulhsu", "mulhu",
    "div", "divu", "divw", "divuw", "rem", "remu", "remw", "remuw",
    "lui", "auipc", "li", "mv",
    "sext.w", "sext.b", "sext.h", "zext.w", "zext.b", "zext.h",
    "beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez",
    "bgt", "ble", "bgtu", "bleu", "bgtz", "blez", "bltz", "bgez",
    "j", "jr", "jal", "jalr", "ret", "call", "tail", "nop", "la",
    "lb", "lh", "lw", "ld", "lbu", "lhu", "lwu",
    "sb", "sh", "sw", "sd",
    "fence",
]

_REGS = [
    "zero", "ra", "sp", "gp", "tp",
    "t0", "t1", "t2", "s0", "s1",
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
    "t3", "t4", "t5", "t6",
]
# Aliases for parser: gcc sometimes emits "x10" instead of "a0"; treat as same idx.
_REG_ALIAS = {f"x{i}": _REGS[i] for i in range(32)}
_REG_ALIAS["fp"] = "s0"

_LABEL_MAX = 32  # positional slots per function

# Canonical immediates, hand-picked to cover what gcc -O2 emits frequently.
_IMM_BUCKETS = [
    -2147483648, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 20, 24, 28, 31, 32, 48, 56, 60, 63, 64,
    96, 100, 128, 255, 256, 511, 512, 1000, 1023, 1024, 2048, 4096,
    8192, 0xFFFF, 0x10000, 0x7FFFFFFF, -32, -16, -8, -4, -2,
    0x80000000,
]


@dataclass
class AsmVocab:
    """Single flat int vocab with the offsets pre-computed for fast tokenize/detokenize."""

    def __init__(self) -> None:
        self.opcodes = list(_OPCODES)
        self.regs = list(_REGS)
        self.imms = list(_IMM_BUCKETS)

        n = 8
        self.OP_OFFSET = n; n += len(self.opcodes)
        self.REG_OFFSET = n; n += len(self.regs)
        self.LBL_REF_OFFSET = n; n += _LABEL_MAX
        self.LBL_DEF_OFFSET = n; n += _LABEL_MAX
        self.IMM_OFFSET = n; n += len(self.imms)
        self.OOV_IMM = n; n += 1
        self.size = n

        self._op_to_idx = {o: i for i, o in enumerate(self.opcodes)}
        self._reg_to_idx = {r: i for i, r in enumerate(self.regs)}
        self._imm_to_bucket = {v: i for i, v in enumerate(self.imms)}

    # -- encoders
    def op(self, name: str) -> int | None:
        i = self._op_to_idx.get(name)
        return None if i is None else self.OP_OFFSET + i

    def reg(self, name: str) -> int | None:
        name = _REG_ALIAS.get(name, name)
        i = self._reg_to_idx.get(name)
        return None if i is None else self.REG_OFFSET + i

    def lbl_ref(self, slot: int) -> int:
        if not (0 <= slot < _LABEL_MAX):
            return self.OOV_IMM
        return self.LBL_REF_OFFSET + slot

    def lbl_def(self, slot: int) -> int:
        if not (0 <= slot < _LABEL_MAX):
            return self.OOV_IMM
        return self.LBL_DEF_OFFSET + slot

    def imm(self, value: int) -> int:
        i = self._imm_to_bucket.get(value)
        return self.OOV_IMM if i is None else self.IMM_OFFSET + i

    # -- decoders
    def what(self, tok: int) -> tuple[str, object]:
        """Return ('opcode'|'reg'|'lbl_ref'|'lbl_def'|'imm'|'special', value)."""
        if tok < 8:
            return ("special", {
                PAD: "PAD", BOS: "BOS", EOS: "EOS", EOL: "EOL",
                COMMA: "COMMA", LPAREN: "LPAREN", RPAREN: "RPAREN", COLON: "COLON",
            }[tok])
        if tok < self.REG_OFFSET:
            return ("opcode", self.opcodes[tok - self.OP_OFFSET])
        if tok < self.LBL_REF_OFFSET:
            return ("reg", self.regs[tok - self.REG_OFFSET])
        if tok < self.LBL_DEF_OFFSET:
            return ("lbl_ref", tok - self.LBL_REF_OFFSET)
        if tok < self.IMM_OFFSET:
            return ("lbl_def", tok - self.LBL_DEF_OFFSET)
        if tok < self.OOV_IMM:
            return ("imm", self.imms[tok - self.IMM_OFFSET])
        return ("imm", None)  # OOV


# ---- Tokenizer --------------------------------------------------------------

_INSTR_RE = re.compile(r"^\s+([a-z][a-z0-9_\.]*)(?:\s+(.+?))?\s*(?:#.*)?$")
_LABEL_DEF_RE = re.compile(r"^(\.?[A-Za-z_][\w\.]*):\s*$")
_FUNC_LABEL_RE = re.compile(r"^([A-Za-z_][\w]*):\s*$")
_INT_LIT_RE = re.compile(r"^-?(?:0x[0-9a-fA-F]+|\d+)$")


class TokenizeError(ValueError):
    pass


def tokenize_asm(asm: str, fn_name: str, vocab: AsmVocab) -> list[int]:
    """Tokenize the body of ``fn_name`` from a -S output. Returns a flat list of token
    IDs bracketed by BOS / EOS. Labels are renumbered positionally so the stream is
    invariant to gcc's internal naming.

    Skips: directives, .LFB*/.LFE* anonymous markers, .cfi_*, .size, the file/option
    preamble.
    """
    ids: list[int] = [BOS]
    inside = False
    label_map: dict[str, int] = {}

    def label_slot(name: str) -> int:
        if name not in label_map:
            label_map[name] = len(label_map)
        return label_map[name]

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
        # End-of-function markers.
        if stripped.startswith(".cfi_endproc") or stripped.startswith(f".size\t{fn_name}") \
                or stripped.startswith(f".size {fn_name}"):
            break
        # Skip directives and anonymous gcc markers.
        if stripped.startswith(".") and not _LABEL_DEF_RE.match(stripped):
            continue
        # User-visible label definitions (e.g. ``.L3:``). Skip ``.LFB*`` / ``.LFE*``
        # — these are DWARF markers, not branch targets.
        m = _LABEL_DEF_RE.match(stripped)
        if m:
            name = m.group(1)
            if name.startswith(".LFB") or name.startswith(".LFE") or name == fn_name:
                continue
            slot = label_slot(name)
            ids.append(vocab.lbl_def(slot))
            ids.append(EOL)
            continue
        m = _INSTR_RE.match(line)
        if not m:
            continue
        opcode = m.group(1)
        op_id = vocab.op(opcode)
        if op_id is None:
            raise TokenizeError(f"unknown opcode {opcode!r} in {fn_name}")
        ids.append(op_id)
        operands_str = (m.group(2) or "").strip()
        if operands_str:
            _emit_operands(operands_str, vocab, label_slot, ids)
        ids.append(EOL)

    if not inside:
        raise TokenizeError(f"function {fn_name!r} not found in asm")
    ids.append(EOS)
    return ids


def _emit_operands(operands: str, vocab: AsmVocab, label_slot, ids: list[int]) -> None:
    """Split comma-separated operands and emit tokens for each."""
    pieces = [p.strip() for p in operands.split(",")]
    for i, p in enumerate(pieces):
        if i > 0:
            ids.append(COMMA)
        _emit_operand(p, vocab, label_slot, ids)


def _emit_operand(operand: str, vocab: AsmVocab, label_slot, ids: list[int]) -> None:
    # Memory ref: ``offset(reg)`` — split into imm LPAREN reg RPAREN.
    mem = re.match(r"^(-?\d+|0x[0-9a-fA-F]+)\(([a-z0-9]+)\)$", operand)
    if mem:
        ids.append(vocab.imm(int(mem.group(1), 0)))
        ids.append(LPAREN)
        r = vocab.reg(mem.group(2))
        if r is None:
            raise TokenizeError(f"unknown register in memref: {operand!r}")
        ids.append(r)
        ids.append(RPAREN)
        return
    # Register
    r = vocab.reg(operand)
    if r is not None:
        ids.append(r)
        return
    # Integer literal
    if _INT_LIT_RE.match(operand):
        ids.append(vocab.imm(int(operand, 0)))
        return
    # Label reference (anything starting with .L that's not a directive).
    if operand.startswith(".L"):
        ids.append(vocab.lbl_ref(label_slot(operand)))
        return
    # Function symbol reference (call target).
    if re.match(r"^[A-Za-z_][\w]*$", operand):
        # Out-of-corpus symbol — represent as OOV_IMM for now.
        ids.append(vocab.OOV_IMM)
        return
    raise TokenizeError(f"unparseable operand {operand!r}")


# ---- Detokenizer (best-effort: reconstructs legal asm for our valid vocab) --

def detokenize_asm(ids: list[int], vocab: AsmVocab, fn_name: str = "spec_fn") -> str:
    out: list[str] = [
        "\t.text",
        f"\t.globl\t{fn_name}",
        f"\t.type\t{fn_name},@function",
        f"{fn_name}:",
    ]
    line: list[str] = []
    operand_buf: list[str] = []
    pending_op: str | None = None

    def flush_line() -> None:
        nonlocal pending_op
        if pending_op is None:
            return
        if operand_buf:
            out.append(f"\t{pending_op}\t{''.join(operand_buf)}")
        else:
            out.append(f"\t{pending_op}")
        pending_op = None
        operand_buf.clear()

    for tok in ids:
        kind, val = vocab.what(tok)
        if kind == "special":
            if val in ("BOS", "PAD"):
                continue
            if val == "EOS":
                break
            if val == "EOL":
                flush_line()
                continue
            if val == "COMMA":
                operand_buf.append(",")
                continue
            if val == "LPAREN":
                operand_buf.append("(")
                continue
            if val == "RPAREN":
                operand_buf.append(")")
                continue
            if val == "COLON":
                operand_buf.append(":")
                continue
        elif kind == "opcode":
            if pending_op is not None:
                flush_line()
            pending_op = val  # type: ignore[assignment]
        elif kind == "reg":
            operand_buf.append(val)  # type: ignore[arg-type]
        elif kind == "imm":
            operand_buf.append(str(val if val is not None else 0))
        elif kind == "lbl_ref":
            operand_buf.append(f".L{val}")
        elif kind == "lbl_def":
            flush_line()
            out.append(f".L{val}:")
    flush_line()
    out.append(f"\t.size\t{fn_name}, .-{fn_name}")
    return "\n".join(out) + "\n"
