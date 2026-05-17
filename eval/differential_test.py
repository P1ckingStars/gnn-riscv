"""Differential testing: run two RV64GC asm functions on identical inputs and verify
they produce the same output. Used in the LeetCode pipeline where the reference is
``gcc -O2`` of the human C solution and the candidate is whatever the model emits.

The harness is generated per-signature: each int parameter becomes a stdin field
parsed with ``%d``; the function's int return value is printed with ``%d\\n``.
Multi-input testing happens in one qemu invocation (amortises startup).

v1 restricts to **all-int signatures** (~95% of our LeetCode corpus). Pointer / struct
parameters require buffer allocation and a typed harness — deferred.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_INT_TYPES = {
    "int", "unsigned int", "long", "unsigned long", "long long",
    "unsigned long long", "int32_t", "uint32_t", "int64_t", "uint64_t",
    "bool",
}


@dataclass
class DifferentialVerdict:
    passed: bool
    n_inputs_tested: int
    first_failing_input: tuple[int, ...] | None
    reference_output: str | None
    candidate_output: str | None
    failure_reason: str | None


class UnsupportedSignature(ValueError):
    """The function uses argument/return types this v1 harness can't generate for."""


def signature_is_int_only(sig: dict) -> bool:
    """Return True iff the function takes only int-like scalars and returns one."""
    if sig["return_ty"].strip() not in _INT_TYPES:
        return False
    for p in sig["params"]:
        if p["ty"].strip() not in _INT_TYPES:
            return False
    return True


def _require(tool: str) -> str:
    p = shutil.which(tool)
    if not p:
        raise RuntimeError(f"required tool {tool!r} not on PATH")
    return p


def _harness_for_int_signature(sig: dict) -> str:
    """Generate a C harness that reads param tuples from stdin and prints the result."""
    fn = sig["fn_name"]
    ret = sig["return_ty"].strip()
    params = sig["params"]
    extern_args = ", ".join(f"{p['ty']} {p['name']}" for p in params) or "void"
    var_decls = "".join(f"    {p['ty']} {p['name']};\n" for p in params)
    scn_fmt = " ".join("%lld" for _ in params)
    # Use long long as the universal scanf intermediate; cast per-param.
    scn_decls = "".join(f"    long long _tmp_{p['name']};\n" for p in params)
    scn_args = ", ".join(f"&_tmp_{p['name']}" for p in params)
    cast_assigns = "".join(
        f"        {p['name']} = ({p['ty']}) _tmp_{p['name']};\n" for p in params
    )
    call_args = ", ".join(p["name"] for p in params)
    if ret == "bool":
        printf_fmt = "%d"
        print_expr = f"(int)({fn}({call_args}))"
    else:
        # Print as long long to cover signed/unsigned 32/64.
        printf_fmt = "%lld"
        print_expr = f"(long long)({fn}({call_args}))"
    return f"""\
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

extern {ret} {fn}({extern_args});

int main(void) {{
    char line[2048];
    while (fgets(line, sizeof(line), stdin)) {{
{scn_decls}        if (sscanf(line, "{scn_fmt}", {scn_args}) != {len(params)}) continue;
{var_decls}{cast_assigns}        printf("{printf_fmt}\\n", {print_expr});
    }}
    return 0;
}}
"""


_BOUNDARY = [0, 1, -1, 2, -2, 7, -7, 0x7FFFFFFF, -0x7FFFFFFF - 1, 0x80000000, 0xFFFFFFFF]


def sample_int_inputs(n_params: int, n_inputs: int, seed: int = 0) -> list[tuple[int, ...]]:
    """Random + boundary int tuples for differential testing."""
    rng = random.Random(seed)
    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    # Boundary cross-product, capped.
    from itertools import product
    cap = min(n_inputs // 4, 64) if n_inputs >= 4 else n_inputs
    for combo in product(_BOUNDARY, repeat=n_params):
        if len(out) >= cap:
            break
        if combo in seen:
            continue
        seen.add(combo)
        out.append(combo)
    # Random fill
    attempts = 0
    max_attempts = n_inputs * 8
    while len(out) < n_inputs and attempts < max_attempts:
        attempts += 1
        r = rng.random()
        t = tuple(_rand_int(rng, r) for _ in range(n_params))
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _rand_int(rng: random.Random, bias: float) -> int:
    if bias < 0.4:
        return rng.randint(-32, 32)
    if bias < 0.7:
        return rng.randint(-(1 << 16), (1 << 16) - 1)
    if bias < 0.9:
        return rng.randint(-(1 << 30), (1 << 30) - 1)
    return rng.randint(-(1 << 31), (1 << 31) - 1)


def _build_binary(
    asm_text: str, harness_c: str, out_path: Path, gcc: str,
) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="diff-build-") as d:
        dp = Path(d)
        (dp / "asm.s").write_text(asm_text)
        (dp / "main.c").write_text(harness_c)
        try:
            subprocess.run(
                [gcc, "-march=rv64gc", "-mabi=lp64d", "-O0", "-static",
                 "-Wno-implicit-function-declaration",
                 "-o", str(out_path), str(dp / "main.c"), str(dp / "asm.s")],
                check=True, capture_output=True, timeout=30,
            )
            return True, ""
        except subprocess.CalledProcessError as e:
            return False, e.stderr.decode(errors="replace")[:400]
        except subprocess.TimeoutExpired:
            return False, "compile_timeout"


def _run_under_qemu(
    binary: Path, qemu: str, stdin_data: str, timeout_s: float = 60.0,
) -> tuple[bool, str, str]:
    try:
        proc = subprocess.run(
            [qemu, str(binary)], input=stdin_data, text=True,
            check=True, capture_output=True, timeout=timeout_s,
        )
        return True, proc.stdout, ""
    except subprocess.CalledProcessError as e:
        return False, e.stdout or "", f"rc={e.returncode} stderr={e.stderr[:300]}"
    except subprocess.TimeoutExpired:
        return False, "", "qemu_timeout"


def differential_test(
    candidate_asm: str,
    reference_asm: str,
    signature: dict,
    n_inputs: int = 256,
    seed: int = 0,
) -> DifferentialVerdict:
    """Run candidate vs reference on N int tuples; first mismatch fails the test."""
    if not signature_is_int_only(signature):
        raise UnsupportedSignature(
            f"v1 harness handles all-int signatures only; got "
            f"return={signature['return_ty']} params="
            f"{[p['ty'] for p in signature['params']]}"
        )
    gcc = _require("riscv64-linux-gnu-gcc")
    qemu = _require("qemu-riscv64-static")

    harness = _harness_for_int_signature(signature)
    inputs = sample_int_inputs(len(signature["params"]), n_inputs, seed=seed)
    stdin_data = "\n".join(" ".join(str(v) for v in t) for t in inputs) + "\n"

    with tempfile.TemporaryDirectory(prefix="diff-run-") as d:
        dp = Path(d)
        cand_bin = dp / "cand"
        ref_bin = dp / "ref"
        ok_c, err_c = _build_binary(candidate_asm, harness, cand_bin, gcc)
        if not ok_c:
            return DifferentialVerdict(False, 0, None, None, None,
                                       f"candidate_compile: {err_c}")
        ok_r, err_r = _build_binary(reference_asm, harness, ref_bin, gcc)
        if not ok_r:
            return DifferentialVerdict(False, 0, None, None, None,
                                       f"reference_compile: {err_r}")

        ok_cr, cand_out, err_cr = _run_under_qemu(cand_bin, qemu, stdin_data)
        if not ok_cr:
            return DifferentialVerdict(False, 0, None, None, None,
                                       f"candidate_run: {err_cr}")
        ok_rr, ref_out, err_rr = _run_under_qemu(ref_bin, qemu, stdin_data)
        if not ok_rr:
            return DifferentialVerdict(False, 0, None, None, None,
                                       f"reference_run: {err_rr}")

        cand_lines = cand_out.split("\n")
        ref_lines = ref_out.split("\n")
        for idx, t in enumerate(inputs):
            if idx >= len(cand_lines) or idx >= len(ref_lines):
                return DifferentialVerdict(
                    False, idx, t, _safe(ref_lines, idx), _safe(cand_lines, idx),
                    "missing_output",
                )
            c_line = cand_lines[idx].strip()
            r_line = ref_lines[idx].strip()
            if c_line != r_line:
                return DifferentialVerdict(
                    False, idx, t, r_line, c_line, "output_mismatch",
                )
        return DifferentialVerdict(True, len(inputs), None, None, None, None)


def _safe(lines: list[str], idx: int) -> str | None:
    return lines[idx].strip() if 0 <= idx < len(lines) else None


def load_problem(problem_dir: Path) -> tuple[str, str, dict]:
    """Load (reference_asm, source.c, signature) from an ingested problem directory."""
    sig = json.loads((problem_dir / "signature.json").read_text())
    return (
        (problem_dir / "ref_o2.s").read_text(),
        (problem_dir / "source.c").read_text(),
        sig,
    )
