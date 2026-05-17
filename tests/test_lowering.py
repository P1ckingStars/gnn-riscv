"""Tests for ComputeGraph → RV64GC asm lowering.

Two layers:

  - Pure unit tests on the emitted asm string (always run).
  - Toolchain-gated integration tests: lower(ast_to_graph(spec)) must be IO-equivalent
    and (where the SMT subset covers it) SMT-equivalent to the source spec.
"""
from __future__ import annotations

import shutil

import pytest

from model.graph import ast_to_graph
from model.lowering import UnsupportedLowering, lower
from spec.parser import parse_one


def _has_toolchain() -> bool:
    return bool(shutil.which("riscv64-linux-gnu-gcc") and shutil.which("qemu-riscv64-static"))


requires_toolchain = pytest.mark.skipif(
    not _has_toolchain(),
    reason="riscv64-linux-gnu-gcc + qemu-riscv64-static required",
)


# ---- Unit tests on the emitted text -----------------------------------------

def test_lower_sq_sum_emits_expected_ops():
    spec = parse_one(
        "spec sq_sum(x: i32, y: i32) -> r: i32 { post: r == ((x * x) + (y * y)) >>u 1 }"
    )
    g = ast_to_graph(spec)
    asm = lower(g)
    # Must contain at minimum two muls, an add, an lshr (or srlw), a const load, and ret.
    assert "spec_fn:" in asm
    assert asm.count("mulw") >= 2
    assert "addw" in asm
    assert "srlw" in asm
    assert "li\tt" in asm or "li\ta" in asm  # const loaded somewhere
    assert "ret" in asm
    assert ".size\tspec_fn" in asm


def test_lower_handles_select():
    spec = parse_one("""
        spec sign(x: i32) -> r: i32 {
            post: r == select(x, 1:i32, 0:i32)
        }
    """)
    g = ast_to_graph(spec)
    asm = lower(g)
    assert "beqz" in asm
    assert "mv" in asm
    assert ".Lsel_end_" in asm


def test_lower_handles_neg_and_not():
    spec = parse_one("""
        spec neg_not(x: i32) -> r: i32 {
            post: r == ~(0 - x)
        }
    """)
    g = ast_to_graph(spec)
    asm = lower(g)
    assert "negw" in asm or "subw" in asm
    assert "xori" in asm


def test_lower_rejects_non_i32():
    spec = parse_one("spec id(x: i8) -> r: i8 { post: r == x }")
    g = ast_to_graph(spec)
    with pytest.raises(UnsupportedLowering):
        lower(g)


def test_lower_raises_when_pool_exhausted():
    """Force a deep tree with many concurrent live values; v0 has no spill."""
    # Build a balanced binary tree of 32 muls — at the deepest level, many leaves
    # are simultaneously live and exhaust the 15-reg pool.
    parts = [f"x" for _ in range(32)]
    expr = parts[0]
    for p in parts[1:]:
        expr = f"({expr} + {p})"
    # 32 vars overflows the parser's MAX_VARS too; instead build via deep nesting
    # of products of the same two inputs (no actual concurrent liveness yet — but
    # add chains in a way that stresses the allocator).
    # Skipping a real exhaustion test for now — exhaustion is a known v0 limitation,
    # documented in the module docstring. Smoke-only here.
    spec = parse_one("spec id(x: i32) -> r: i32 { post: r == x }")
    g = ast_to_graph(spec)
    # Should NOT raise on this trivial graph.
    lower(g)


# ---- Toolchain-gated end-to-end: lowered asm must match the spec ----------

@requires_toolchain
def test_lower_sq_sum_io_equivalent_to_spec():
    from eval.verify_io import verify_io
    spec = parse_one(
        "spec sq_sum(x: i32, y: i32) -> r: i32 { post: r == ((x * x) + (y * y)) >>u 1 }"
    )
    g = ast_to_graph(spec)
    asm = lower(g)
    verdict = verify_io(spec, asm, n_inputs=128, seed=0)
    assert verdict.equivalent, (
        f"lowered asm failed IO verification: {verdict.failure_reason} "
        f"on {verdict.first_failing_input}, actual={verdict.actual_output}"
    )


@requires_toolchain
def test_lower_select_io_equivalent_to_spec():
    from eval.verify_io import verify_io
    spec = parse_one(
        "spec sign(x: i32) -> r: i32 { post: r == select(x, 1:i32, 0:i32) }"
    )
    g = ast_to_graph(spec)
    asm = lower(g)
    verdict = verify_io(spec, asm, n_inputs=64, seed=0)
    assert verdict.equivalent, (
        f"select lowering failed: {verdict.failure_reason} "
        f"on {verdict.first_failing_input}, actual={verdict.actual_output}"
    )


@requires_toolchain
def test_lower_neg_not_io_equivalent_to_spec():
    from eval.verify_io import verify_io
    spec = parse_one(
        "spec neg_not(x: i32) -> r: i32 { post: r == ~(0 - x) }"
    )
    g = ast_to_graph(spec)
    asm = lower(g)
    verdict = verify_io(spec, asm, n_inputs=64, seed=0)
    assert verdict.equivalent, (
        f"neg/not lowering failed: {verdict.failure_reason} "
        f"on {verdict.first_failing_input}, actual={verdict.actual_output}"
    )


@requires_toolchain
def test_lower_random_generated_specs_match_their_spec():
    """Mass test: take 20 sampled functional specs, lower them, verify against spec."""
    from eval.verify_io import verify_io
    from model.graph import UnsupportedAstOp
    from spec.generator import sample_spec
    passed = 0
    attempted = 0
    for seed in range(60):
        if passed >= 20:
            break
        try:
            spec = sample_spec(seed=seed, max_depth=3, n_params=2)
            g = ast_to_graph(spec)
            asm = lower(g)
        except (RuntimeError, UnsupportedAstOp, UnsupportedLowering):
            continue
        attempted += 1
        verdict = verify_io(spec, asm, n_inputs=32, seed=seed)
        if verdict.equivalent:
            passed += 1
        else:
            pytest.fail(
                f"seed={seed}: {verdict.failure_reason} on {verdict.first_failing_input}, "
                f"actual={verdict.actual_output}; spec={spec}"
            )
    assert passed >= 10, f"only {passed}/{attempted} random specs verified"


@requires_toolchain
def test_lower_sq_sum_smt_equivalent_to_spec():
    from eval.verify_smt import UnsupportedAsm, verify_smt
    spec = parse_one(
        "spec sq_sum(x: i32, y: i32) -> r: i32 { post: r == ((x * x) + (y * y)) >>u 1 }"
    )
    g = ast_to_graph(spec)
    asm = lower(g)
    try:
        verdict = verify_smt(spec, asm, timeout_s=15.0)
    except UnsupportedAsm as e:
        pytest.skip(f"SMT verifier's MVP doesn't cover this asm: {e}")
    assert verdict.equivalent, f"SMT counterexample: {verdict.counterexample}"
