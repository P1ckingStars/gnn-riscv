"""Sound equivalence gate via Z3 bitvector symbolic execution, for Hoare-triple specs.

Query: ``∃ inputs. pre(inputs) ∧ no_UB(inputs) ∧ ¬post(inputs, f(inputs))``.
  - UNSAT → candidate satisfies the spec on all valid inputs.
  - SAT   → counterexample.
  - UNKNOWN → timeout.

Supported asm subset: see module docstring section below; unsupported ops raise
``UnsupportedAsm`` so callers can route to verify_io only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import z3

from spec.dsl import (
    And, Bin, BinOp, BoolFalse, BoolTrue, Cmp, CmpOp, Const, Exists, Expr, Forall,
    Formula, Iff, Implies, LetTerm, Not, Or, Select, Spec, Ty, Un, UnOp, Var,
)
from eval._asm_parse import Insn, parse_function, parse_imm, reg_index


class UnsupportedAsm(RuntimeError):
    pass


@dataclass
class SMTVerdict:
    equivalent: bool
    counterexample: tuple[int, ...] | None
    solver_time_s: float
    timed_out: bool


# ---- term/formula translation to Z3 ----------------------------------------

def _term_to_z3(
    e: Expr, env: dict[str, z3.BitVecRef], ub: list[z3.BoolRef]
) -> z3.BitVecRef:
    if isinstance(e, Const):
        return z3.BitVecVal(e.value, e.ty.width)
    if isinstance(e, Var):
        return env[e.name]
    if isinstance(e, Bin):
        l = _term_to_z3(e.lhs, env, ub)
        r = _term_to_z3(e.rhs, env, ub)
        op = e.op
        w = e.ty.width
        if op is BinOp.ADD: return l + r
        if op is BinOp.SUB: return l - r
        if op is BinOp.MUL: return l * r
        if op is BinOp.AND: return l & r
        if op is BinOp.OR:  return l | r
        if op is BinOp.XOR: return l ^ r
        if op is BinOp.UDIV:
            ub.append(r != 0); return z3.UDiv(l, r)
        if op is BinOp.UREM:
            ub.append(r != 0); return z3.URem(l, r)
        if op is BinOp.SDIV:
            ub.append(r != 0); return l / r
        if op is BinOp.SREM:
            ub.append(r != 0); return z3.SRem(l, r)
        if op in (BinOp.SHL, BinOp.LSHR, BinOp.ASHR):
            ub.append(z3.ULT(r, w))
            if op is BinOp.SHL:  return l << r
            if op is BinOp.LSHR: return z3.LShR(l, r)
            return l >> r
    if isinstance(e, Un):
        a = _term_to_z3(e.arg, env, ub)
        op = e.op
        if op is UnOp.NEG: return -a
        if op is UnOp.NOT: return ~a
        if op is UnOp.SEXT: return z3.SignExt(e.ty.width - e.arg.ty.width, a)
        if op is UnOp.ZEXT: return z3.ZeroExt(e.ty.width - e.arg.ty.width, a)
        if op is UnOp.TRUNC: return z3.Extract(e.ty.width - 1, 0, a)
    if isinstance(e, Select):
        c = _term_to_z3(e.cond, env, ub)
        t = _term_to_z3(e.then, env, ub)
        f = _term_to_z3(e.else_, env, ub)
        return z3.If(c != 0, t, f)
    raise TypeError(f"unhandled term: {type(e).__name__}")


def _formula_to_z3(
    f: Formula, env: dict[str, z3.BitVecRef], ub: list[z3.BoolRef]
) -> z3.BoolRef:
    if isinstance(f, BoolTrue):  return z3.BoolVal(True)
    if isinstance(f, BoolFalse): return z3.BoolVal(False)
    if isinstance(f, Cmp):
        l = _term_to_z3(f.lhs, env, ub)
        r = _term_to_z3(f.rhs, env, ub)
        op = f.op
        if op is CmpOp.EQ:  return l == r
        if op is CmpOp.NE:  return l != r
        if op is CmpOp.SLT: return l < r        # Z3 BitVec < is signed
        if op is CmpOp.SLE: return l <= r
        if op is CmpOp.SGT: return l > r
        if op is CmpOp.SGE: return l >= r
        if op is CmpOp.ULT: return z3.ULT(l, r)
        if op is CmpOp.ULE: return z3.ULE(l, r)
        if op is CmpOp.UGT: return z3.UGT(l, r)
        if op is CmpOp.UGE: return z3.UGE(l, r)
    if isinstance(f, Not):
        return z3.Not(_formula_to_z3(f.arg, env, ub))
    if isinstance(f, And):
        return z3.And(_formula_to_z3(f.lhs, env, ub), _formula_to_z3(f.rhs, env, ub))
    if isinstance(f, Or):
        return z3.Or(_formula_to_z3(f.lhs, env, ub), _formula_to_z3(f.rhs, env, ub))
    if isinstance(f, Implies):
        return z3.Implies(_formula_to_z3(f.ant, env, ub),
                          _formula_to_z3(f.cons, env, ub))
    if isinstance(f, Iff):
        return _formula_to_z3(f.lhs, env, ub) == _formula_to_z3(f.rhs, env, ub)
    if isinstance(f, (Forall, Exists)):
        bound = z3.BitVec(f"_b_{f.var.name}", f.var.ty.width)
        inner_ub: list[z3.BoolRef] = []
        body = _formula_to_z3(f.body, {**env, f.var.name: bound}, inner_ub)
        # Within a quantifier, UB conditions are guarded by the bound variable;
        # treat the quantifier as "for all (resp. exists) bound such that no UB ∧ body".
        if inner_ub:
            ub_pred = z3.And(*inner_ub)
            body = z3.Implies(ub_pred, body) if isinstance(f, Forall) else z3.And(ub_pred, body)
        if isinstance(f, Forall):
            return z3.ForAll([bound], body)
        return z3.Exists([bound], body)
    if isinstance(f, LetTerm):
        v = _term_to_z3(f.value, env, ub)
        # Bind the let-name to the value; pure substitution.
        return _formula_to_z3(f.body, {**env, f.name: v}, ub)
    raise TypeError(f"unhandled formula: {type(f).__name__}")


# ---- asm symbolic executor (unchanged from earlier) ------------------------

_SEXT32 = lambda x: z3.SignExt(32, z3.Extract(31, 0, x))  # noqa: E731


def _exec_asm(insns: list[Insn], arg_regs: list[z3.BitVecRef]) -> z3.BitVecRef:
    regs: list[z3.BitVecRef] = [z3.BitVecVal(0, 64) for _ in range(32)]
    regs[0] = z3.BitVecVal(0, 64)
    for i, av in enumerate(arg_regs):
        if i >= 8:
            raise UnsupportedAsm(f"too many args ({i+1}); a0..a7 only")
        regs[10 + i] = av

    def set_reg(i: int, val: z3.BitVecRef) -> None:
        if i == 0:
            return
        regs[i] = val

    for ins in insns:
        op = ins.op
        ops = ins.operands

        if op == "ret" or (op == "jr" and ops == ("ra",)) or (
            op == "jalr" and ops in (("x0", "ra", "0"), ("zero", "ra", "0"))
        ):
            break

        if op == "mv":
            rd, rs = ops; set_reg(reg_index(rd), regs[reg_index(rs)]); continue
        if op == "li":
            rd, imm = ops
            set_reg(reg_index(rd), z3.BitVecVal(parse_imm(imm) & ((1 << 64) - 1), 64))
            continue
        if op == "neg":
            rd, rs = ops; set_reg(reg_index(rd), -regs[reg_index(rs)]); continue
        if op == "negw":
            rd, rs = ops; set_reg(reg_index(rd), _SEXT32(-regs[reg_index(rs)])); continue
        if op == "not":
            rd, rs = ops; set_reg(reg_index(rd), ~regs[reg_index(rs)]); continue
        if op == "seqz":
            rd, rs = ops
            set_reg(reg_index(rd), z3.If(regs[reg_index(rs)] == 0,
                                         z3.BitVecVal(1, 64), z3.BitVecVal(0, 64)))
            continue
        if op == "snez":
            rd, rs = ops
            set_reg(reg_index(rd), z3.If(regs[reg_index(rs)] != 0,
                                         z3.BitVecVal(1, 64), z3.BitVecVal(0, 64)))
            continue
        if op == "sext.w":
            rd, rs = ops; set_reg(reg_index(rd), _SEXT32(regs[reg_index(rs)])); continue
        if op == "zext.w":
            rd, rs = ops
            set_reg(reg_index(rd), z3.ZeroExt(32, z3.Extract(31, 0, regs[reg_index(rs)])))
            continue
        if op == "lui":
            rd, imm = ops
            v = (parse_imm(imm) & 0xFFFFF) << 12
            set_reg(reg_index(rd), _SEXT32(z3.BitVecVal(v & 0xFFFFFFFF, 64)))
            continue

        if op in _R_TYPE_64:
            rd, rs1, rs2 = ops
            a = regs[reg_index(rs1)]; b = regs[reg_index(rs2)]
            set_reg(reg_index(rd), _R_TYPE_64[op](a, b)); continue
        if op in _R_TYPE_32:
            rd, rs1, rs2 = ops
            a = z3.Extract(31, 0, regs[reg_index(rs1)])
            b = z3.Extract(31, 0, regs[reg_index(rs2)])
            set_reg(reg_index(rd), z3.SignExt(32, _R_TYPE_32[op](a, b))); continue
        if op in _I_TYPE_64:
            rd, rs1, imm = ops
            a = regs[reg_index(rs1)]
            iv = z3.BitVecVal(parse_imm(imm) & ((1 << 64) - 1), 64)
            set_reg(reg_index(rd), _I_TYPE_64[op](a, iv)); continue
        if op in _I_TYPE_32:
            rd, rs1, imm = ops
            a = z3.Extract(31, 0, regs[reg_index(rs1)])
            iv = z3.BitVecVal(parse_imm(imm) & ((1 << 32) - 1), 32)
            set_reg(reg_index(rd), z3.SignExt(32, _I_TYPE_32[op](a, iv))); continue
        if op in _I_SHIFT_64:
            rd, rs1, sh = ops
            a = regs[reg_index(rs1)]
            shamt = parse_imm(sh) & 0x3F
            set_reg(reg_index(rd), _I_SHIFT_64[op](a, shamt)); continue
        if op in _I_SHIFT_32:
            rd, rs1, sh = ops
            a = z3.Extract(31, 0, regs[reg_index(rs1)])
            shamt = parse_imm(sh) & 0x1F
            set_reg(reg_index(rd), z3.SignExt(32, _I_SHIFT_32[op](a, shamt))); continue

        raise UnsupportedAsm(f"unsupported asm op: {ins.raw!r}")

    return regs[10]


_R_TYPE_64: dict[str, callable] = {  # type: ignore[type-arg]
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
    "and": lambda a, b: a & b, "or": lambda a, b: a | b, "xor": lambda a, b: a ^ b,
    "sll": lambda a, b: a << (b & 0x3F),
    "srl": lambda a, b: z3.LShR(a, b & 0x3F),
    "sra": lambda a, b: a >> (b & 0x3F),
    "slt":  lambda a, b: z3.If(a < b, z3.BitVecVal(1, 64), z3.BitVecVal(0, 64)),
    "sltu": lambda a, b: z3.If(z3.ULT(a, b), z3.BitVecVal(1, 64), z3.BitVecVal(0, 64)),
    "mul": lambda a, b: a * b,
    "mulh":   lambda a, b: z3.Extract(127, 64, z3.SignExt(64, a) * z3.SignExt(64, b)),
    "mulhsu": lambda a, b: z3.Extract(127, 64, z3.SignExt(64, a) * z3.ZeroExt(64, b)),
    "mulhu":  lambda a, b: z3.Extract(127, 64, z3.ZeroExt(64, a) * z3.ZeroExt(64, b)),
    "div":  lambda a, b: a / b,        "divu": lambda a, b: z3.UDiv(a, b),
    "rem":  lambda a, b: z3.SRem(a, b), "remu": lambda a, b: z3.URem(a, b),
}

_R_TYPE_32: dict[str, callable] = {  # type: ignore[type-arg]
    "addw": lambda a, b: a + b, "subw": lambda a, b: a - b,
    "sllw": lambda a, b: a << (b & 0x1F),
    "srlw": lambda a, b: z3.LShR(a, b & 0x1F),
    "sraw": lambda a, b: a >> (b & 0x1F),
    "mulw": lambda a, b: a * b,
    "divw":  lambda a, b: a / b,        "divuw": lambda a, b: z3.UDiv(a, b),
    "remw":  lambda a, b: z3.SRem(a, b), "remuw": lambda a, b: z3.URem(a, b),
}

_I_TYPE_64: dict[str, callable] = {  # type: ignore[type-arg]
    "addi": lambda a, i: a + i, "andi": lambda a, i: a & i,
    "ori":  lambda a, i: a | i, "xori": lambda a, i: a ^ i,
    "slti":  lambda a, i: z3.If(a < i, z3.BitVecVal(1, 64), z3.BitVecVal(0, 64)),
    "sltiu": lambda a, i: z3.If(z3.ULT(a, i), z3.BitVecVal(1, 64), z3.BitVecVal(0, 64)),
}

_I_TYPE_32: dict[str, callable] = {  # type: ignore[type-arg]
    "addiw": lambda a, i: a + i,
}

_I_SHIFT_64: dict[str, callable] = {  # type: ignore[type-arg]
    "slli": lambda a, s: a << s,
    "srli": lambda a, s: z3.LShR(a, s),
    "srai": lambda a, s: a >> s,
}

_I_SHIFT_32: dict[str, callable] = {  # type: ignore[type-arg]
    "slliw": lambda a, s: a << s,
    "srliw": lambda a, s: z3.LShR(a, s),
    "sraiw": lambda a, s: a >> s,
}


# ---- entry point ------------------------------------------------------------

def _input_z3_vars(spec: Spec) -> tuple[list[z3.BitVecRef], dict[str, z3.BitVecRef]]:
    """Allocate a typed BitVec per spec input, plus the sign-extended-to-64 form that
    goes into a<i> per the lp64 calling convention."""
    typed: dict[str, z3.BitVecRef] = {}
    arg_regs: list[z3.BitVecRef] = []
    for p in spec.inputs:
        bv = z3.BitVec(p.name, p.ty.width)
        typed[p.name] = bv
        arg_regs.append(bv if p.ty.width == 64 else z3.SignExt(64 - p.ty.width, bv))
    return arg_regs, typed


def verify_smt(
    spec: Spec,
    candidate_asm: str,
    timeout_s: float = 30.0,
    fn_name: str = "spec_fn",
) -> SMTVerdict:
    insns = parse_function(candidate_asm, fn_name)
    arg_regs, typed_inputs = _input_z3_vars(spec)
    a0_after = _exec_asm(insns, arg_regs)

    # Bind the single output var to the appropriate slice of a0.
    out = spec.ret_param
    if out.ty.width == 64:
        out_bv = a0_after
    else:
        out_bv = z3.Extract(out.ty.width - 1, 0, a0_after)
    env: dict[str, z3.BitVecRef] = {**typed_inputs, out.name: out_bv}

    ub_pre: list[z3.BoolRef] = []
    ub_post: list[z3.BoolRef] = []
    pre_z3 = _formula_to_z3(spec.pre, env, ub_pre)
    post_z3 = _formula_to_z3(spec.post, env, ub_post)

    # "Valid input" = pre holds AND no UB during pre/post evaluation.
    valid = z3.And(pre_z3, *ub_pre, *ub_post) if (ub_pre or ub_post) else pre_z3

    solver = z3.Solver()
    solver.set("timeout", int(timeout_s * 1000))
    solver.add(valid)
    solver.add(z3.Not(post_z3))

    t0 = time.time()
    result = solver.check()
    elapsed = time.time() - t0

    if result == z3.unsat:
        return SMTVerdict(True, None, elapsed, False)
    if result == z3.sat:
        m = solver.model()
        cex = tuple(m.eval(typed_inputs[p.name]).as_long() for p in spec.inputs)
        return SMTVerdict(False, cex, elapsed, False)
    return SMTVerdict(False, None, elapsed, True)
