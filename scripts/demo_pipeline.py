#!/usr/bin/env python3
"""End-to-end demo of the research substrate.

  python scripts/demo_pipeline.py [--n-specs 5] [--seed 0]

For each sampled spec, this script:
  1. type-checks it,
  2. prints a few sample inputs and reference outputs,
  3. lowers it to C,
  4. (toolchain-gated) compiles to RV64GC asm via gcc -O2,
  5. (toolchain-gated) measures cost,
  6. (toolchain-gated) verifies the baseline asm against the spec with verify_io,
  7. (toolchain-gated) verifies with verify_smt.

If the RISC-V toolchain isn't installed, steps 4-7 are skipped with a note.
"""
from __future__ import annotations

import argparse
import shutil

from spec.dsl import Ty, type_check
from spec.generator import sample_spec
from spec.interpreter import evaluate, sample_inputs
from spec.lower_to_c import to_c


def _has_toolchain() -> bool:
    return bool(shutil.which("riscv64-linux-gnu-gcc") and shutil.which("qemu-riscv64-static"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-specs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--n-params", type=int, default=2)
    args = ap.parse_args()

    toolchain_ok = _has_toolchain()
    if not toolchain_ok:
        print("[toolchain] riscv64-linux-gnu-gcc + qemu-riscv64-static not found "
              "— skipping compile / verify steps.")
        print("[toolchain] To enable: sudo bash scripts/install_system_deps.sh\n")

    if toolchain_ok:
        from eval.baselines import gcc_o2
        from eval.cost import estimate
        from eval.verify_io import verify_io
        from eval.verify_smt import verify_smt, UnsupportedAsm

    for i in range(args.n_specs):
        seed = args.seed + i
        print(f"=== spec #{i} (seed={seed}) ===")
        spec = sample_spec(seed=seed, max_depth=args.max_depth,
                          n_params=args.n_params, ret_ty=Ty.I32)
        type_check(spec)
        print(f"  inputs : {[(p.name, p.ty.name) for p in spec.inputs]}")
        print(f"  outputs: {[(p.name, p.ty.name) for p in spec.outputs]}")
        if spec.is_functional():
            print(f"  body   : {spec.functional_body()}")
        else:
            print(f"  post   : {spec.post}")

        probes = sample_inputs(spec, n=4, seed=seed)
        for t in probes[:4]:
            print(f"    eval({t}) = {evaluate(spec, t)}")

        c_src = to_c(spec)
        print(f"  C lowering: {len(c_src)} bytes")

        if not toolchain_ok:
            print()
            continue

        try:
            asm = gcc_o2(spec)
        except Exception as e:
            print(f"  gcc -O2 FAILED: {e}")
            print()
            continue
        cost = estimate(asm)
        print(f"  gcc -O2 cost: {cost.instruction_count} insns, ~{cost.estimated_cycles} cycles")

        iov = verify_io(spec, asm, n_inputs=128, seed=seed)
        print(f"  verify_io: equivalent={iov.equivalent} "
              f"(tested {iov.n_inputs_tested} inputs)")
        if not iov.equivalent:
            print(f"    failure : {iov.failure_reason} on {iov.first_failing_input} "
                  f"expected={iov.expected} actual={iov.actual}")

        try:
            smt = verify_smt(spec, asm, timeout_s=15.0)
            print(f"  verify_smt: equivalent={smt.equivalent} "
                  f"(solver_time={smt.solver_time_s:.2f}s, timed_out={smt.timed_out})")
            if not smt.equivalent and smt.counterexample is not None:
                print(f"    counterexample: {smt.counterexample}")
        except UnsupportedAsm as e:
            print(f"  verify_smt: SKIP ({e})")
        print()


if __name__ == "__main__":
    main()
