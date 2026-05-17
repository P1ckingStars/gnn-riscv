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
    And, AnyTy, Bin, BinOp, BoolFalse, BoolTrue, Cmp, CmpOp, Const, Exists, Expr,
    Forall, Formula, Iff, Implies, LetTerm, Load, Not, Or, PtrAdd, PtrTy, Select,
    Spec, Ty, Un, UnOp, Var, XLEN,
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

def _z3_load(mem: z3.ArrayRef, addr: z3.BitVecRef, ty: Ty) -> z3.BitVecRef:
    """Read ``ty.width / 8`` bytes from ``mem`` starting at ``addr``, little-endian
    (low byte at ``addr``, high byte at ``addr + n - 1``). Returns a BitVec of
    width ``ty.width``."""
    nbytes = ty.width // 8
    parts = [z3.Select(mem, addr + i) for i in range(nbytes)]
    # Concat is MSB-to-LSB, so reverse the LE byte list.
    return z3.Concat(list(reversed(parts)))


def _term_to_z3(
    e: Expr, env: dict[str, z3.BitVecRef], ub: list[z3.BoolRef],
    mem: z3.ArrayRef | None = None,
) -> z3.BitVecRef:
    if isinstance(e, Const):
        return z3.BitVecVal(e.value, e.ty.width)
    if isinstance(e, Var):
        return env[e.name]
    if isinstance(e, Bin):
        l = _term_to_z3(e.lhs, env, ub, mem)
        r = _term_to_z3(e.rhs, env, ub, mem)
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
        a = _term_to_z3(e.arg, env, ub, mem)
        op = e.op
        if op is UnOp.NEG: return -a
        if op is UnOp.NOT: return ~a
        if op is UnOp.SEXT: return z3.SignExt(e.ty.width - e.arg.ty.width, a)
        if op is UnOp.ZEXT: return z3.ZeroExt(e.ty.width - e.arg.ty.width, a)
        if op is UnOp.TRUNC: return z3.Extract(e.ty.width - 1, 0, a)
    if isinstance(e, Select):
        c = _term_to_z3(e.cond, env, ub, mem)
        t = _term_to_z3(e.then, env, ub, mem)
        f = _term_to_z3(e.else_, env, ub, mem)
        return z3.If(c != 0, t, f)
    if isinstance(e, PtrAdd):
        base = _term_to_z3(e.base, env, ub, mem)  # bv64
        off = _term_to_z3(e.offset, env, ub, mem)
        if off.size() < XLEN:
            # Sign-extend so negative offsets behave like in C.
            off = z3.SignExt(XLEN - off.size(), off)
        elif off.size() > XLEN:
            off = z3.Extract(XLEN - 1, 0, off)
        return base + off
    if isinstance(e, Load):
        if mem is None:
            raise ValueError("Load encountered but no memory Z3 array was provided")
        addr = _term_to_z3(e.ptr, env, ub, mem)
        if not isinstance(e.ty, Ty):
            raise TypeError(f"Load result must be scalar; got {e.ty}")
        return _z3_load(mem, addr, e.ty)
    raise TypeError(f"unhandled term: {type(e).__name__}")


def _formula_to_z3(
    f: Formula, env: dict[str, z3.BitVecRef], ub: list[z3.BoolRef],
    mem: z3.ArrayRef | None = None,
) -> z3.BoolRef:
    if isinstance(f, BoolTrue):  return z3.BoolVal(True)
    if isinstance(f, BoolFalse): return z3.BoolVal(False)
    if isinstance(f, Cmp):
        l = _term_to_z3(f.lhs, env, ub, mem)
        r = _term_to_z3(f.rhs, env, ub, mem)
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
        return z3.Not(_formula_to_z3(f.arg, env, ub, mem))
    if isinstance(f, And):
        return z3.And(_formula_to_z3(f.lhs, env, ub, mem), _formula_to_z3(f.rhs, env, ub, mem))
    if isinstance(f, Or):
        return z3.Or(_formula_to_z3(f.lhs, env, ub, mem), _formula_to_z3(f.rhs, env, ub, mem))
    if isinstance(f, Implies):
        return z3.Implies(_formula_to_z3(f.ant, env, ub, mem),
                          _formula_to_z3(f.cons, env, ub, mem))
    if isinstance(f, Iff):
        return _formula_to_z3(f.lhs, env, ub, mem) == _formula_to_z3(f.rhs, env, ub, mem)
    if isinstance(f, (Forall, Exists)):
        bound = z3.BitVec(f"_b_{f.var.name}", f.var.ty.width)
        inner_ub: list[z3.BoolRef] = []
        body = _formula_to_z3(f.body, {**env, f.var.name: bound}, inner_ub, mem)
        if inner_ub:
            ub_pred = z3.And(*inner_ub)
            body = z3.Implies(ub_pred, body) if isinstance(f, Forall) else z3.And(ub_pred, body)
        if isinstance(f, Forall):
            return z3.ForAll([bound], body)
        return z3.Exists([bound], body)
    if isinstance(f, LetTerm):
        v = _term_to_z3(f.value, env, ub, mem)
        return _formula_to_z3(f.body, {**env, f.name: v}, ub, mem)
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
    """Allocate a typed BitVec per spec input. For scalar params the BitVec is
    ``p.ty.width`` bits and a sign-extended 64-bit version is what would go into ``a<i>``
    under the lp64 ABI. For pointer params the value is a 64-bit address that lands in
    ``a<i>`` directly."""
    typed: dict[str, z3.BitVecRef] = {}
    arg_regs: list[z3.BitVecRef] = []
    for p in spec.inputs:
        if isinstance(p.ty, PtrTy):
            bv = z3.BitVec(p.name, XLEN)
            typed[p.name] = bv
            arg_regs.append(bv)
        else:
            bv = z3.BitVec(p.name, p.ty.width)
            typed[p.name] = bv
            arg_regs.append(bv if p.ty.width == 64 else z3.SignExt(64 - p.ty.width, bv))
    return arg_regs, typed


def _make_memory() -> z3.ArrayRef:
    """Allocate a fresh symbolic memory array (addr: bv64 → byte: bv8)."""
    return z3.Array("mem", z3.BitVecSort(XLEN), z3.BitVecSort(8))


def _spec_uses_memory(spec: Spec) -> bool:
    """Return True if any Load appears in pre or post — saves a Z3 array allocation when
    the spec doesn't actually touch memory."""
    return _has_load(spec.pre) or _has_load(spec.post)


def _has_load(node) -> bool:
    if isinstance(node, Load):
        return True
    if hasattr(node, "__dataclass_fields__"):
        for fname in node.__dataclass_fields__:
            v = getattr(node, fname)
            if isinstance(v, (Expr, Formula)) and _has_load(v):
                return True
    return False


def verify_smt(
    spec: Spec,
    candidate_asm: str,
    timeout_s: float = 30.0,
    fn_name: str = "spec_fn",
) -> SMTVerdict:
    """Spec-vs-asm equivalence. The asm-side symbolic executor (v1 MVP) doesn't model
    memory loads/stores yet — if the candidate asm contains lw/sw, parse_function will
    surface them via ``UnsupportedAsm`` in ``_exec_asm``. Pure-arithmetic asm + a
    Load-free spec works fully."""
    insns = parse_function(candidate_asm, fn_name)
    arg_regs, typed_inputs = _input_z3_vars(spec)
    a0_after = _exec_asm(insns, arg_regs)

    out = spec.ret_param
    if isinstance(out.ty, PtrTy):
        out_bv = a0_after  # already bv64
    elif out.ty.width == 64:
        out_bv = a0_after
    else:
        out_bv = z3.Extract(out.ty.width - 1, 0, a0_after)
    env: dict[str, z3.BitVecRef] = {**typed_inputs, out.name: out_bv}

    mem = _make_memory() if _spec_uses_memory(spec) else None

    ub_pre: list[z3.BoolRef] = []
    ub_post: list[z3.BoolRef] = []
    pre_z3 = _formula_to_z3(spec.pre, env, ub_pre, mem)
    post_z3 = _formula_to_z3(spec.post, env, ub_post, mem)

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


def prove_functional_equiv(
    spec_a: Spec, spec_b: Spec, timeout_s: float = 10.0,
) -> SMTVerdict:
    """Prove two functional specs always compute the same value given matching inputs
    + a shared memory state. Useful for spec-vs-spec equivalence checks (e.g. testing
    that a rewrite is sound) without going through asm.

    Both specs must be functional and share the same input + output signature. Returns
    UNSAT (equivalent=True) iff the post-conditions agree under any inputs satisfying
    both pre-conditions and no-UB."""
    if not (spec_a.is_functional() and spec_b.is_functional()):
        raise ValueError("prove_functional_equiv: both specs must be functional")
    if [p.ty for p in spec_a.inputs] != [p.ty for p in spec_b.inputs]:
        raise ValueError("input type signatures differ")
    if spec_a.ret_ty != spec_b.ret_ty:
        raise ValueError("return types differ")

    # Allocate Z3 vars from spec_a's input signature (matched by spec_b).
    typed: dict[str, z3.BitVecRef] = {}
    for p in spec_a.inputs:
        if isinstance(p.ty, PtrTy):
            typed[p.name] = z3.BitVec(p.name, XLEN)
        else:
            typed[p.name] = z3.BitVec(p.name, p.ty.width)
    # spec_b may use different param names; build a parallel mapping by position.
    env_b = dict(typed)
    for pa, pb in zip(spec_a.inputs, spec_b.inputs):
        env_b[pb.name] = typed[pa.name]

    mem = _make_memory() if (_spec_uses_memory(spec_a) or _spec_uses_memory(spec_b)) else None

    ub: list[z3.BoolRef] = []
    body_a = _term_to_z3(spec_a.functional_body(), typed, ub, mem)
    body_b = _term_to_z3(spec_b.functional_body(), env_b, ub, mem)

    pre_ub: list[z3.BoolRef] = []
    pre_a = _formula_to_z3(spec_a.pre, typed, pre_ub, mem)
    pre_b = _formula_to_z3(spec_b.pre, env_b, pre_ub, mem)

    solver = z3.Solver()
    solver.set("timeout", int(timeout_s * 1000))
    solver.add(z3.And(pre_a, pre_b, *pre_ub, *ub))
    solver.add(body_a != body_b)

    t0 = time.time()
    result = solver.check()
    elapsed = time.time() - t0

    if result == z3.unsat:
        return SMTVerdict(True, None, elapsed, False)
    if result == z3.sat:
        m = solver.model()
        cex = tuple(m.eval(typed[p.name]).as_long() for p in spec_a.inputs)
        return SMTVerdict(False, cex, elapsed, False)
    return SMTVerdict(False, None, elapsed, True)
