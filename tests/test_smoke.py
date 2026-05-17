"""End-to-end smoke test for the gnn-riscv research infrastructure.

Splits cleanly into:
  - Pure-Python tests (always run): DSL + types, interpreter, formulas, generator,
    C lowering, serializer, parser, cost model.
  - Toolchain-gated tests (skipped if riscv64-linux-gnu-gcc + qemu-riscv64-static aren't
    on PATH): full pipeline through gcc -O2 → cost → verify_io → verify_smt on simple
    specs.
"""
from __future__ import annotations

import shutil

import pytest

from spec.dsl import (
    And, Bin, BinOp, BoolTrue, Cmp, CmpOp, Const, Exists, Forall, Implies, Not, Or,
    Param, Spec, Ty, Un, UnOp, Var, mask_to, signed, type_check,
)
from spec.interpreter import (
    NeedsSMT, UndefinedBehavior, eval_formula, evaluate, precondition_holds,
    sample_inputs, satisfies,
)
from spec.generator import sample_spec
from spec.lower_to_c import to_c
from spec.parser import parse_one
from spec.serialize import from_dict, to_dict
from eval.cost import estimate
from eval._asm_parse import parse_imm, reg_index


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _sq_sum_spec() -> Spec:
    """λ(x, y : i32). (x*x + y*y) >> 1 — as a functional Hoare spec."""
    x = Var(ty=Ty.I32, name="x")
    y = Var(ty=Ty.I32, name="y")
    xsq = Bin(ty=Ty.I32, op=BinOp.MUL, lhs=x, rhs=x)
    ysq = Bin(ty=Ty.I32, op=BinOp.MUL, lhs=y, rhs=y)
    s = Bin(ty=Ty.I32, op=BinOp.ADD, lhs=xsq, rhs=ysq)
    body = Bin(ty=Ty.I32, op=BinOp.LSHR, lhs=s, rhs=Const(ty=Ty.I32, value=1))
    return Spec.functional(
        inputs=(Param("x", Ty.I32), Param("y", Ty.I32)),
        body=body, ret_name="r",
    )


# ----------------------------------------------------------------------------
# Term layer
# ----------------------------------------------------------------------------

def test_value_helpers():
    assert mask_to(-1, Ty.I32) == 0xFFFF_FFFF
    assert signed(0xFFFF_FFFF, Ty.I32) == -1
    assert signed(0x8000_0000, Ty.I32) == -(2**31)


def test_type_check_ok():
    type_check(_sq_sum_spec())


def test_type_check_rejects_bad_op_tys():
    bad = Spec.functional(
        inputs=(Param("x", Ty.I32),),
        body=Bin(ty=Ty.I32, op=BinOp.ADD,
                 lhs=Var(ty=Ty.I32, name="x"),
                 rhs=Const(ty=Ty.I64, value=1)),
    )
    with pytest.raises(TypeError):
        type_check(bad)


def test_interpreter_known_values():
    spec = _sq_sum_spec()
    assert evaluate(spec, (3, 4)) == (9 + 16) >> 1
    assert evaluate(spec, (0, 0)) == 0
    assert evaluate(spec, (-2 & 0xFFFFFFFF, 0)) == 2


def test_interpreter_sdiv_truncates_toward_zero():
    spec = Spec.functional(
        inputs=(Param("x", Ty.I32), Param("y", Ty.I32)),
        body=Bin(ty=Ty.I32, op=BinOp.SDIV,
                 lhs=Var(ty=Ty.I32, name="x"),
                 rhs=Var(ty=Ty.I32, name="y")),
    )
    res = evaluate(spec, (mask_to(-7, Ty.I32), 2))
    assert signed(res, Ty.I32) == -3
    with pytest.raises(UndefinedBehavior):
        evaluate(spec, (1, 0))


def test_sample_inputs_avoids_ub_via_pre():
    """Pre-condition x != 0 should be respected by sample_inputs."""
    x = Var(ty=Ty.I32, name="x")
    y = Var(ty=Ty.I32, name="y")
    spec = Spec(
        inputs=(Param("x", Ty.I32), Param("y", Ty.I32)),
        outputs=(Param("r", Ty.I32),),
        pre=Cmp(op=CmpOp.NE, lhs=y, rhs=Const(ty=Ty.I32, value=0)),
        post=Cmp(op=CmpOp.EQ, lhs=Var(ty=Ty.I32, name="r"),
                 rhs=Bin(ty=Ty.I32, op=BinOp.UDIV, lhs=x, rhs=y)),
    )
    inputs = sample_inputs(spec, n=64, seed=1)
    assert inputs
    for t in inputs:
        assert precondition_holds(spec, t)
        evaluate(spec, t)  # functional + pre-filtered → no UB


# ----------------------------------------------------------------------------
# Formula layer
# ----------------------------------------------------------------------------

def test_eval_formula_propositional():
    f_true = BoolTrue()
    assert eval_formula(f_true, {})
    f = And(BoolTrue(), Not(arg=BoolTrue()))
    assert not eval_formula(f, {})


def test_eval_formula_cmp_signedness():
    x = Var(ty=Ty.I32, name="x")
    cst = Const(ty=Ty.I32, value=0)
    assert not eval_formula(Cmp(op=CmpOp.SLT, lhs=x, rhs=cst), {"x": 0})
    # -1 is < 0 signed, > 0 unsigned
    assert eval_formula(Cmp(op=CmpOp.SLT, lhs=x, rhs=cst), {"x": 0xFFFFFFFF})
    assert eval_formula(Cmp(op=CmpOp.UGT, lhs=x, rhs=cst), {"x": 0xFFFFFFFF})


def test_eval_formula_bounded_quantifier():
    # ∀ x:i8. x + 1 != x   (true)
    x = Var(ty=Ty.I8, name="x")
    body = Cmp(op=CmpOp.NE,
               lhs=Bin(ty=Ty.I8, op=BinOp.ADD, lhs=x, rhs=Const(ty=Ty.I8, value=1)),
               rhs=x)
    f = Forall(var=Param("x", Ty.I8), body=body)
    assert eval_formula(f, {})


def test_eval_formula_wide_quantifier_raises():
    x = Var(ty=Ty.I32, name="x")
    f = Forall(var=Param("x", Ty.I32),
               body=Cmp(op=CmpOp.EQ, lhs=x, rhs=Const(ty=Ty.I32, value=0)))
    with pytest.raises(NeedsSMT):
        eval_formula(f, {})


def test_satisfies_relational_spec():
    # Spec: r ∈ {x-y, y-x}  (absolute difference, two valid outputs)
    x = Var(ty=Ty.I8, name="x")
    y = Var(ty=Ty.I8, name="y")
    r = Var(ty=Ty.I8, name="r")
    spec = Spec(
        inputs=(Param("x", Ty.I8), Param("y", Ty.I8)),
        outputs=(Param("r", Ty.I8),),
        pre=BoolTrue(),
        post=Or(
            lhs=Cmp(op=CmpOp.EQ, lhs=r, rhs=Bin(ty=Ty.I8, op=BinOp.SUB, lhs=x, rhs=y)),
            rhs=Cmp(op=CmpOp.EQ, lhs=r, rhs=Bin(ty=Ty.I8, op=BinOp.SUB, lhs=y, rhs=x)),
        ),
    )
    assert satisfies(spec, (5, 3), (2,))             # 5-3
    assert satisfies(spec, (5, 3), (mask_to(-2, Ty.I8),))  # 3-5 = -2 (=254)
    assert not satisfies(spec, (5, 3), (7,))


# ----------------------------------------------------------------------------
# Generator + serializer + lowering + cost
# ----------------------------------------------------------------------------

def test_generator_produces_typed_specs():
    for s in range(5):
        spec = sample_spec(seed=s, max_depth=4, n_params=2)
        type_check(spec)
        assert spec.is_functional()
        inputs = sample_inputs(spec, n=8, seed=s)
        assert inputs


def test_serialize_roundtrip_functional():
    spec = _sq_sum_spec()
    d = to_dict(spec)
    spec2 = from_dict(d)
    type_check(spec2)
    assert spec == spec2


def test_serialize_roundtrip_relational_with_quantifier():
    src = """
        spec absdiff(x: i8, y: i8) -> r: i8 {
            pre: true
            post: (r == x - y) | (r == y - x)
        }
    """
    spec = parse_one(src)
    spec2 = from_dict(to_dict(spec))
    assert spec == spec2


def test_lower_to_c_renders_non_empty_for_functional():
    src = to_c(_sq_sum_spec(), fn_name="spec_fn")
    assert "#include <stdint.h>" in src
    assert "spec_fn" in src
    assert "uint32_t" in src


def test_lower_to_c_rejects_relational():
    src = """
        spec isqrt(x: i32) -> r: i32 {
            post: r * r <=u x  &  (r + 1) * (r + 1) >u x
        }
    """
    spec = parse_one(src)
    with pytest.raises(ValueError):
        to_c(spec)


def test_cost_model_on_synthetic_asm():
    asm = (
        "\t.text\n"
        "\t.globl\tspec_fn\n"
        "spec_fn:\n"
        "\tmulw\ta0,a0,a0\n"
        "\tmulw\ta1,a1,a1\n"
        "\taddw\ta0,a0,a1\n"
        "\tsrliw\ta0,a0,1\n"
        "\tret\n"
        "\t.size\tspec_fn, .-spec_fn\n"
    )
    cost = estimate(asm, fn_name="spec_fn")
    assert cost.instruction_count == 5
    assert cost.estimated_cycles == 10


def test_asm_parser_helpers():
    assert reg_index("a0") == 10
    assert reg_index("x10") == 10
    assert parse_imm("0x20") == 32
    assert parse_imm("-7") == -7


# ----------------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------------

def test_parser_functional_spec():
    src = """
        spec sq_sum(x: i32, y: i32) -> r: i32 {
            post: r == ((x * x) + (y * y)) >>u 1
        }
    """
    spec = parse_one(src)
    type_check(spec)
    assert spec.is_functional()
    assert evaluate(spec, (3, 4)) == (9 + 16) >> 1


def test_parser_relational_spec():
    src = """
        spec isqrt(x: i32) -> r: i32 {
            pre:  x >=s 0
            post: r * r <=u x  &  (r + 1) * (r + 1) >u x
        }
    """
    spec = parse_one(src)
    type_check(spec)
    assert not spec.is_functional()
    assert satisfies(spec, (25,), (5,))
    assert not satisfies(spec, (25,), (6,))


def test_parser_quantifier():
    src = """
        spec all_positive_succ(x: i8) -> r: i8 {
            post: forall y: i8. (y >=s 0) -> ((y + 1) >s y) | ((y + 1) == -128)
        }
    """
    spec = parse_one(src)
    type_check(spec)
    # The post body is independent of x and r — should hold under enumeration over i8.
    assert satisfies(spec, (0,), (0,))


def test_parser_typed_const():
    src = """
        spec mask_low(x: i32) -> r: i32 {
            post: r == x & 255:i32
        }
    """
    spec = parse_one(src)
    type_check(spec)
    assert evaluate(spec, (0xDEADBEEF,)) == 0xBEEF & 0xFF


def test_parser_select_builtin_term_cond():
    """select(cond, then, else_) takes a *term* cond (nonzero/zero); for Formula
    conditions, use implications. Here we test the term-cond form via the sign bit."""
    src = """
        spec sign(x: i32) -> r: i32 {
            post: r == select(x, 1:i32, 0:i32)
        }
    """
    spec = parse_one(src)
    type_check(spec)
    assert evaluate(spec, (5,)) == 1
    assert evaluate(spec, (0,)) == 0
    assert evaluate(spec, (mask_to(-5, Ty.I32),)) == 1


def test_parser_implications_for_branching_post():
    """Branching specs are written with implications, not select."""
    src = """
        spec absval(x: i32) -> r: i32 {
            pre:  x != -2147483648:i32
            post: ((x >=s 0) -> (r == x)) & ((x <s 0) -> (r == 0 - x))
        }
    """
    spec = parse_one(src)
    type_check(spec)
    assert satisfies(spec, (5,), (5,))
    assert satisfies(spec, (mask_to(-5, Ty.I32),), (5,))
    assert not satisfies(spec, (mask_to(-5, Ty.I32),), (mask_to(-5, Ty.I32),))


# ----------------------------------------------------------------------------
# Toolchain-gated end-to-end pipeline
# ----------------------------------------------------------------------------

requires_toolchain = pytest.mark.skipif(
    not (shutil.which("riscv64-linux-gnu-gcc") and shutil.which("qemu-riscv64-static")),
    reason="riscv64-linux-gnu-gcc + qemu-riscv64-static required; run sudo bash scripts/install_system_deps.sh",
)


@requires_toolchain
def test_pipeline_functional_self_equivalent_io():
    from eval.baselines import gcc_o2
    from eval.verify_io import verify_io
    spec = _sq_sum_spec()
    asm = gcc_o2(spec)
    verdict = verify_io(spec, asm, n_inputs=128, seed=0)
    assert verdict.equivalent, (
        f"gcc -O2 output failed verify_io: reason={verdict.failure_reason} "
        f"input={verdict.first_failing_input} actual={verdict.actual_output}"
    )


@requires_toolchain
def test_pipeline_functional_cost_is_reasonable():
    from eval.baselines import gcc_o2
    spec = _sq_sum_spec()
    asm = gcc_o2(spec)
    cost = estimate(asm)
    assert cost.instruction_count > 0
    assert cost.estimated_cycles >= cost.instruction_count


@requires_toolchain
def test_pipeline_functional_smt_equivalent():
    from eval.baselines import gcc_o2
    from eval.verify_smt import verify_smt, UnsupportedAsm
    spec = _sq_sum_spec()
    asm = gcc_o2(spec)
    try:
        verdict = verify_smt(spec, asm, timeout_s=15.0)
    except UnsupportedAsm as e:
        pytest.skip(f"SMT MVP doesn't cover this asm: {e}")
    assert verdict.equivalent, f"SMT counterexample: {verdict.counterexample}"


@requires_toolchain
def test_pipeline_smt_catches_known_mismatch():
    from eval.verify_smt import verify_smt
    spec = _sq_sum_spec()
    wrong_asm = (
        "\t.text\n\t.globl\tspec_fn\nspec_fn:\n"
        "\taddw\ta0,a0,a1\n\tret\n\t.size\tspec_fn, .-spec_fn\n"
    )
    verdict = verify_smt(spec, wrong_asm, timeout_s=15.0)
    assert not verdict.equivalent
    assert verdict.counterexample is not None
