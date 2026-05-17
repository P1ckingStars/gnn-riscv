"""Supervised training entry point for the GNN graph generator.

Sample-by-sample loop (no batch padding yet). Each step:
  1. Tokenize spec → ids → ``SpecEncoder`` → spec context.
  2. Walk the reference graph step-by-step inside ``GraphGenerator.teacher_forced_loss``,
     accumulating per-head CE losses.
  3. Optimizer step on the summed loss.

Usage:
    python -m model.train --n-specs 32 --epochs 50 --device cpu

Includes an ``--overfit`` flag that uses a fixed tiny dataset and prints per-step
greedy-generation accuracy at the end — the v0 sanity check that the loop is wired
correctly is "loss goes down, and the model reproduces the reference graph on its own
training set."
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model.data import SpecDataset, Sample, collate_list
from model.graph import ARITY, ComputeGraph, NodeOp, graph_to_steps
from model.graph_gen import GraphGenerator
from model.spec_encoder import SpecEncoder
from model.vocab import OpVocab, SpecVocab


def build_models(
    sv: SpecVocab, ov: OpVocab, d_model: int, n_layers: int, n_heads: int, device: torch.device,
) -> tuple[SpecEncoder, GraphGenerator]:
    encoder = SpecEncoder(
        vocab_size=sv.size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
    ).to(device)
    generator = GraphGenerator(
        op_vocab=ov, const_vocab=sv.consts, d_model=d_model, gnn_heads=n_heads,
    ).to(device)
    return encoder, generator


def train_one_step(
    encoder: SpecEncoder, generator: GraphGenerator, sample: Sample, device: torch.device,
) -> dict[str, torch.Tensor]:
    ids = torch.tensor(sample.spec_tokens, dtype=torch.long, device=device).unsqueeze(0)
    ctx, _per_tok = encoder(ids)
    losses = generator.teacher_forced_loss(ctx, sample)
    return losses


def greedy_match(
    encoder: SpecEncoder, generator: GraphGenerator, sample: Sample, device: torch.device,
) -> tuple[bool, ComputeGraph | None]:
    encoder.eval()
    generator.eval()
    with torch.no_grad():
        ids = torch.tensor(sample.spec_tokens, dtype=torch.long, device=device).unsqueeze(0)
        ctx, _ = encoder(ids)
        gen_graph = generator.generate(
            ctx, sample.spec.inputs, sample.spec.ret_ty,
            max_steps=max(16, len(sample.op_ids) + 4), greedy=True,
        )
    encoder.train()
    generator.train()
    if gen_graph is None:
        return False, None
    ref_steps = graph_to_steps(sample.graph)
    gen_steps = graph_to_steps(gen_graph)
    if len(ref_steps) != len(gen_steps):
        return False, gen_graph
    for r, g in zip(ref_steps, gen_steps):
        if r.op is not g.op:
            return False, gen_graph
        arity = ARITY[r.op]
        if r.operands[:arity] != g.operands[:arity]:
            return False, gen_graph
        if r.op is NodeOp.CONST and r.const_value != g.const_value:
            return False, gen_graph
    return True, gen_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-specs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--n-params", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--ckpt", type=Path, default=Path("experiments/000_smoke/ckpt.pt"))
    ap.add_argument("--overfit", action="store_true",
                    help="after training, eval greedy match rate on the train set")
    args = ap.parse_args()

    device = torch.device(args.device)
    sv = SpecVocab()
    ov = OpVocab()
    ds = SpecDataset(
        n_specs=args.n_specs, seed=args.seed, max_depth=args.max_depth,
        n_params=args.n_params, sv=sv, ov=ov,
    )
    print(f"dataset: {len(ds)} samples")
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_list)

    encoder, generator = build_models(
        sv, ov, args.d_model, args.n_layers, args.n_heads, device,
    )
    n_params = sum(p.numel() for p in encoder.parameters()) + sum(
        p.numel() for p in generator.parameters()
    )
    print(f"model params: {n_params / 1e6:.2f}M")
    opt = AdamW(list(encoder.parameters()) + list(generator.parameters()), lr=args.lr)

    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n = 0
        for batch in loader:
            for sample in batch:
                opt.zero_grad()
                losses = train_one_step(encoder, generator, sample, device)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(generator.parameters()), 1.0,
                )
                opt.step()
                epoch_loss += float(losses["loss"].item())
                n += 1
        avg = epoch_loss / max(n, 1)
        print(f"epoch {epoch:3d}  loss={avg:.4f}  elapsed={time.time() - t0:.1f}s")

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "generator": generator.state_dict(),
        "config": vars(args),
    }, args.ckpt)
    print(f"saved checkpoint to {args.ckpt}")

    if args.overfit:
        print("\n[overfit eval] greedy-generate on the training set:")
        matches = 0
        for i, sample in enumerate(ds):
            ok, gen = greedy_match(encoder, generator, sample, device)
            matches += int(ok)
            if i < 5:
                marker = "OK " if ok else "MISS"
                print(f"  [{marker}] sample {i}: ref={len(sample.op_ids)} steps"
                      f"  gen={len(gen.nodes) if gen else 'None'} steps")
        print(f"greedy match rate: {matches}/{len(ds)} = {matches / len(ds):.1%}")


if __name__ == "__main__":
    main()
