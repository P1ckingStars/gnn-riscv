#!/usr/bin/env python3
"""Diagnostic: take a trained LeetCode-pipeline checkpoint, greedy-generate asm for
every diff-testable LeetCode problem, and report:

  - syntactic validity (parses + assembles + runs under qemu)
  - correctness vs the gcc -O2 reference (full differential-test pass)
  - average per-token loss (teacher-forced)

The point is to surface softer signals than "passes 16/16 of differential testing,"
which is currently 0%. If syntactic-validity is high but correctness is low, the
model is producing well-formed asm that just computes a different function — that's
useful diagnostic context for the val_pass=0 numbers.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch


def _try_assemble(asm_text: str, fn_name: str, gcc: str) -> str | None:
    """Try to assemble + link the asm into a valid object. Returns None on success,
    error string on failure."""
    with tempfile.TemporaryDirectory(prefix="lc-asm-") as d:
        dp = Path(d)
        (dp / "asm.s").write_text(asm_text)
        (dp / "main.c").write_text(
            f"int {fn_name}(void); int main(void){{return 0;}}\n"
        )
        try:
            subprocess.run(
                [gcc, "-march=rv64gc", "-mabi=lp64d", "-O0", "-static",
                 "-Wno-implicit-function-declaration", "-o", str(dp / "prog"),
                 str(dp / "main.c"), str(dp / "asm.s")],
                check=True, capture_output=True, timeout=15,
            )
            return None
        except subprocess.CalledProcessError as e:
            return e.stderr.decode(errors="replace")[:200]
        except subprocess.TimeoutExpired:
            return "assemble_timeout"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/leetcode"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"device: {device}")

    from model.c_encoder import (
        CursorKindVocab, TypeClassVocab, ValueBucketVocab, CEncoder,
    )
    from model.asm_decoder import AsmVocab, detokenize_asm
    from model.asm_decoder import BOS, EOS
    from model.asm_decoder_module import AsmDecoder
    from model.train_leetcode import load_corpus, teacher_forced_loss

    kv, tv, vv = CursorKindVocab(), TypeClassVocab(), ValueBucketVocab()
    av = AsmVocab()

    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = blob["config"]
    enc = CEncoder.build(
        kv, tv, vv, d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
    ).to(device)
    dec = AsmDecoder(
        vocab_size=av.size, d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
    ).to(device)
    enc.load_state_dict(blob["encoder"]); dec.load_state_dict(blob["decoder"])
    enc.eval(); dec.eval()
    print(f"loaded ckpt  epoch={blob.get('epoch','?')}  "
          f"reported val_pass={blob.get('val_pass_rate','-')}")

    samples = load_corpus(
        args.data, kv, tv, vv, av, limit=args.limit, int_only=True,
    )
    print(f"diff-testable samples: {len(samples)}")

    gcc = shutil.which("riscv64-linux-gnu-gcc")
    if not gcc:
        raise SystemExit("riscv64-linux-gnu-gcc required")

    from eval.differential_test import differential_test, UnsupportedSignature

    t0 = time.time()
    n_generate_ok = 0
    n_detok_ok = 0
    n_assemble_ok = 0
    n_io_passed = 0
    losses: list[float] = []
    for i, s in enumerate(samples):
        # 1. teacher-forced per-sample loss
        with torch.no_grad():
            losses.append(float(
                teacher_forced_loss(enc, dec, s, device).item()
            ))
        # 2. greedy generate
        try:
            pyg = s.pyg_data
            pyg.x = pyg.x.to(device); pyg.edge_index = pyg.edge_index.to(device)
            pyg.edge_type = pyg.edge_type.to(device)
            with torch.no_grad():
                ctx, per_node = enc(pyg)
                ids = dec.generate(
                    bos_id=BOS, eos_id=EOS, encoder_node_embeds=per_node,
                    encoder_ctx=ctx, max_len=min(2 * len(s.asm_ids) + 32, 2048),
                    greedy=True,
                )
            n_generate_ok += 1
        except Exception:
            continue
        # 3. detokenize
        try:
            gen_asm = detokenize_asm(ids, av, fn_name=s.signature["fn_name"])
            n_detok_ok += 1
        except Exception:
            continue
        # 4. assemble check (parses + links)
        if _try_assemble(gen_asm, s.signature["fn_name"], gcc) is None:
            n_assemble_ok += 1
        else:
            continue
        # 5. differential test
        try:
            v = differential_test(gen_asm, s.ref_o2_asm, s.signature, n_inputs=32, seed=0)
            if v.passed:
                n_io_passed += 1
        except (UnsupportedSignature, Exception):
            continue

    n = len(samples)
    avg_loss = sum(losses) / max(len(losses), 1)
    print()
    print(f"=== eval over {n} diff-testable LeetCode samples ({time.time()-t0:.1f}s) ===")
    print(f"  avg teacher-forced loss : {avg_loss:.3f}")
    print(f"  greedy completes        : {n_generate_ok}/{n} = {n_generate_ok/n:.1%}")
    print(f"  detokenize ok           : {n_detok_ok}/{n} = {n_detok_ok/n:.1%}")
    print(f"  assembles + links       : {n_assemble_ok}/{n} = {n_assemble_ok/n:.1%}")
    print(f"  passes differential test: {n_io_passed}/{n} = {n_io_passed/n:.1%}")


if __name__ == "__main__":
    main()
