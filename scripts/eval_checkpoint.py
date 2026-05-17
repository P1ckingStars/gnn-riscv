#!/usr/bin/env python3
"""Load a trained checkpoint, run the model end-to-end on a sample of specs, and
report both model-level accuracy (greedy match against the reference graph) and
downstream metrics (lowered asm passes verify_io, beats gcc -O2 on cost).

Usage:
    python scripts/eval_checkpoint.py experiments/001_scale1k/best.pt \
        --dataset-path data/v1 --n-eval 50
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch

from model.data import SpecDataset
from model.graph import graph_to_steps
from model.graph_gen import GraphGenerator
from model.lowering import UnsupportedLowering, lower
from model.spec_encoder import SpecEncoder
from model.train import greedy_match
from model.vocab import OpVocab, SpecVocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--dataset-path", type=Path, required=True)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    sv = SpecVocab()
    ov = OpVocab()
    ds = SpecDataset.from_directory(args.dataset_path, sv=sv, ov=ov, limit=args.n_eval)
    print(f"loaded {len(ds)} samples for eval")

    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    enc = SpecEncoder(
        vocab_size=sv.size,
        d_model=cfg.get("d_model", 128),
        n_layers=cfg.get("n_layers", 2),
        n_heads=cfg.get("n_heads", 4),
    ).to(device)
    gen = GraphGenerator(
        op_vocab=ov, const_vocab=sv.consts,
        d_model=cfg.get("d_model", 128), gnn_heads=cfg.get("n_heads", 4),
    ).to(device)
    enc.load_state_dict(blob["encoder"])
    gen.load_state_dict(blob["generator"])
    enc.eval(); gen.eval()
    print(f"loaded checkpoint  epoch={blob.get('epoch', '?')}  "
          f"val_match={blob.get('val_match', '?')}")

    has_toolchain = bool(
        shutil.which("riscv64-linux-gnu-gcc") and shutil.which("qemu-riscv64-static")
    )
    if has_toolchain:
        from eval.verify_io import verify_io
        from eval.cost import estimate

    t0 = time.time()
    full_match = 0
    lowered_ok = 0
    lowered_attempted = 0
    io_verified = 0
    cost_vs_gcc_wins = 0
    cost_vs_gcc_ties = 0
    n_with_cost = 0

    for i, sample in enumerate(ds):
        ok, _correct, _total, gen_graph = greedy_match(enc, gen, sample, device)
        if ok:
            full_match += 1

        if gen_graph is None:
            continue
        lowered_attempted += 1
        try:
            asm = lower(gen_graph)
        except UnsupportedLowering:
            continue
        lowered_ok += 1

        if not has_toolchain:
            continue
        verdict = verify_io(sample.spec, asm, n_inputs=64, seed=i)
        if not verdict.equivalent:
            continue
        io_verified += 1

        # Cost comparison vs gcc -O2 (only when a gcc baseline exists in the dataset dir).
        gcc_path = args.dataset_path / f"spec_{i:06d}" / "ref_gcc_o2.s"
        if gcc_path.exists():
            try:
                model_cost = estimate(asm).instruction_count
                gcc_cost = estimate(gcc_path.read_text()).instruction_count
                n_with_cost += 1
                if model_cost < gcc_cost:
                    cost_vs_gcc_wins += 1
                elif model_cost == gcc_cost:
                    cost_vs_gcc_ties += 1
            except Exception:
                pass

    n = len(ds)
    print()
    print(f"=== eval over {n} samples ({time.time() - t0:.1f}s) ===")
    print(f"  greedy_match         : {full_match}/{n} = {full_match / n:.1%}")
    print(f"  graphs lowered       : {lowered_ok}/{lowered_attempted}")
    if has_toolchain:
        print(f"  io_verified          : {io_verified}/{lowered_ok}")
        if n_with_cost:
            print(f"  cost vs gcc -O2      : wins {cost_vs_gcc_wins}, "
                  f"ties {cost_vs_gcc_ties}, losses "
                  f"{n_with_cost - cost_vs_gcc_wins - cost_vs_gcc_ties} "
                  f"(of {n_with_cost} compared)")
    else:
        print("  (toolchain not present — skipped io_verify + gcc cost comparison)")


if __name__ == "__main__":
    main()
