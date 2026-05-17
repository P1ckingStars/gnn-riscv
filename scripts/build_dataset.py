#!/usr/bin/env python3
"""Build a (spec, io-examples, reference-asm) dataset under data/<name>/.

  python scripts/build_dataset.py --n 100 --out data/v1 --seed 0

Per spec, writes:
  data/v1/spec_000000/spec.json
  data/v1/spec_000000/io.json         # {"inputs": [[a,b],...], "outputs": [r,...]}
  data/v1/spec_000000/ref_gcc_o2.s    # (toolchain-gated)
  data/v1/spec_000000/ref_clang_o2.s  # (toolchain-gated, if clang present)

A top-level data/<name>/manifest.json records counts and toolchain status so a model
trainer can audit what it's consuming without rescanning subdirectories.

Use this as the dataset stage of any experiment — model code consumes the cached
artifacts so training doesn't need to re-run gcc per epoch.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from spec.dsl import Ty, mask_to, type_check
from spec.generator import sample_spec
from spec.interpreter import UndefinedBehavior, evaluate, sample_inputs
from spec.serialize import to_dict


def _has(tool: str) -> bool:
    return bool(shutil.which(tool))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of specs to build")
    ap.add_argument("--out", type=Path, default=Path("data/v1"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--n-params", type=int, default=2)
    ap.add_argument("--n-inputs", type=int, default=1000)
    ap.add_argument("--ret-ty", choices=[t.name for t in Ty], default="I32")
    ap.add_argument("--no-baselines", action="store_true",
                    help="skip running gcc/clang even if available")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    ret_ty = Ty[args.ret_ty]

    gcc_ok = (not args.no_baselines) and _has("riscv64-linux-gnu-gcc")
    clang_ok = (not args.no_baselines) and _has("clang")
    if gcc_ok:
        from eval.baselines import gcc_o2
    if clang_ok:
        from eval.baselines import clang_o2

    t0 = time.time()
    written = 0
    skipped = 0
    seed_iter = args.seed
    while written < args.n:
        try:
            spec = sample_spec(seed=seed_iter, max_depth=args.max_depth,
                               n_params=args.n_params, ret_ty=ret_ty)
        except RuntimeError:
            seed_iter += 1
            continue
        type_check(spec)
        inputs = sample_inputs(spec, n=args.n_inputs, seed=seed_iter)
        if not inputs:
            seed_iter += 1
            skipped += 1
            continue

        # Evaluate; drop any UB-tainted inputs.
        ios: list[tuple[tuple[int, ...], int]] = []
        for t in inputs:
            try:
                r = evaluate(spec, t)
            except UndefinedBehavior:
                continue
            ios.append((tuple(mask_to(v, p.ty) for v, p in zip(t, spec.inputs)), r))
        if not ios:
            seed_iter += 1
            skipped += 1
            continue

        spec_id = f"spec_{written:06d}"
        sd = out / spec_id
        sd.mkdir(exist_ok=True)
        (sd / "spec.json").write_text(json.dumps(to_dict(spec), indent=2))
        (sd / "io.json").write_text(json.dumps({
            "inputs": [list(t) for t, _ in ios],
            "outputs": [r for _, r in ios],
        }))
        if gcc_ok:
            try:
                (sd / "ref_gcc_o2.s").write_text(gcc_o2(spec))
            except Exception as e:
                (sd / "ref_gcc_o2.error").write_text(repr(e))
        if clang_ok:
            try:
                (sd / "ref_clang_o2.s").write_text(clang_o2(spec))
            except Exception as e:
                (sd / "ref_clang_o2.error").write_text(repr(e))
        written += 1
        seed_iter += 1
        if written % 10 == 0:
            print(f"  built {written}/{args.n} ({time.time() - t0:.1f}s)")

    manifest = {
        "n_specs": written,
        "n_skipped_during_gen": skipped,
        "seed_range": [args.seed, seed_iter],
        "n_inputs_per_spec_target": args.n_inputs,
        "max_depth": args.max_depth,
        "n_params": args.n_params,
        "ret_ty": ret_ty.name,
        "baselines": {"gcc": gcc_ok, "clang": clang_ok},
        "wall_time_s": time.time() - t0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone — {written} specs in {out}/ "
          f"({time.time() - t0:.1f}s, baselines: gcc={gcc_ok} clang={clang_ok})")


if __name__ == "__main__":
    main()
