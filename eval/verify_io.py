"""Fast equivalence filter for Hoare-spec synthesis.

Pipeline per spec:
  1. Sample inputs (already Pre-filtered).
  2. Generate a C harness, link with candidate asm, run under qemu-riscv64-static.
  3. For each input row, check ``satisfies(spec, input, [actual_output])``.

A spec may admit multiple correct outputs (relational post); ``satisfies`` does the
right thing — the verifier doesn't compare to a single "expected" value.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from spec.dsl import Spec, Ty, mask_to
from spec.interpreter import (
    NeedsSMT, UndefinedBehavior, precondition_holds, sample_inputs, satisfies,
)


@dataclass
class IOVerdict:
    equivalent: bool
    n_inputs_tested: int
    first_failing_input: tuple[int, ...] | None
    actual_output: int | None
    failure_reason: str | None  # "post_failed" | "assemble_error" | "qemu_crash" | "missing_output" | "parse_error" | "qemu_timeout"


_C_UINT = {Ty.I8: "uint8_t", Ty.I16: "uint16_t", Ty.I32: "uint32_t", Ty.I64: "uint64_t"}
_SCN = {Ty.I8: "SCNu8", Ty.I16: "SCNu16", Ty.I32: "SCNu32", Ty.I64: "SCNu64"}
_PRI = {Ty.I8: "PRIu8", Ty.I16: "PRIu16", Ty.I32: "PRIu32", Ty.I64: "PRIu64"}


def _require(tool: str) -> str:
    p = shutil.which(tool)
    if not p:
        raise RuntimeError(f"required tool '{tool}' not found on PATH")
    return p


def _make_harness(spec: Spec, fn_name: str = "spec_fn") -> str:
    inputs = spec.inputs
    ret_ty = spec.ret_ty
    extern_args = ", ".join(f"{_C_UINT[p.ty]} {p.name}" for p in inputs)
    var_decls = "\n    ".join(f"{_C_UINT[p.ty]} {p.name};" for p in inputs)
    scn_fmt = " ".join(f"%\" {_SCN[p.ty]} \"" for p in inputs)
    scn_args = ", ".join(f"&{p.name}" for p in inputs)
    call_args = ", ".join(p.name for p in inputs)
    return f"""\
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

extern {_C_UINT[ret_ty]} {fn_name}({extern_args});

int main(void) {{
    char line[2048];
    while (fgets(line, sizeof(line), stdin)) {{
        {var_decls}
        if (sscanf(line, "{scn_fmt}", {scn_args}) != {len(inputs)}) continue;
        {_C_UINT[ret_ty]} r = {fn_name}({call_args});
        printf("%" {_PRI[ret_ty]} "\\n", r);
    }}
    return 0;
}}
"""


def _format_input(spec: Spec, args: tuple[int, ...]) -> str:
    return " ".join(str(mask_to(v, p.ty)) for p, v in zip(spec.inputs, args))


def verify_io(
    spec: Spec,
    candidate_asm: str,
    n_inputs: int = 1000,
    seed: int = 0,
    fn_name: str = "spec_fn",
) -> IOVerdict:
    gcc = _require("riscv64-linux-gnu-gcc")
    qemu = _require("qemu-riscv64-static")

    inputs = sample_inputs(spec, n=n_inputs, seed=seed)
    if not inputs:
        return IOVerdict(True, 0, None, None, None)

    harness = _make_harness(spec, fn_name)
    with tempfile.TemporaryDirectory(prefix="gnnriscv-verify-") as d:
        dp = Path(d)
        (dp / "asm.s").write_text(candidate_asm)
        (dp / "main.c").write_text(harness)
        binary = dp / "prog"
        try:
            subprocess.run(
                [gcc, "-march=rv64gc", "-mabi=lp64d", "-O0", "-static",
                 "-o", str(binary), str(dp / "main.c"), str(dp / "asm.s")],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            return IOVerdict(False, 0, None, None,
                             f"assemble_error: {e.stderr.decode(errors='replace')[:400]}")

        stdin_data = "\n".join(_format_input(spec, t) for t in inputs) + "\n"
        try:
            proc = subprocess.run(
                [qemu, str(binary)],
                input=stdin_data, text=True,
                check=True, capture_output=True, timeout=60,
            )
        except subprocess.CalledProcessError as e:
            return IOVerdict(False, 0, None, None,
                             f"qemu_crash: rc={e.returncode} stderr={e.stderr[:400]}")
        except subprocess.TimeoutExpired:
            return IOVerdict(False, 0, None, None, "qemu_timeout")

        outputs = proc.stdout.split("\n")
        tested = 0
        for idx, t in enumerate(inputs):
            try:
                if not precondition_holds(spec, t):
                    continue
            except (UndefinedBehavior, NeedsSMT):
                continue
            if idx >= len(outputs) or not outputs[idx].strip():
                return IOVerdict(False, tested, t, None, "missing_output")
            try:
                actual = mask_to(int(outputs[idx].strip()), spec.ret_ty)
            except ValueError:
                return IOVerdict(False, tested, t, None, f"parse_error: {outputs[idx]!r}")
            try:
                ok = satisfies(spec, t, (actual,))
            except UndefinedBehavior:
                # Post evaluation hit UB for this (input, output) — skip; spec is silent here.
                continue
            except NeedsSMT:
                # Post has unbounded quantifier; cannot decide via interpreter.
                # Skip and let SMT verifier handle this case.
                continue
            tested += 1
            if not ok:
                return IOVerdict(False, tested, t, actual, "post_failed")
        return IOVerdict(True, tested, None, None, None)
