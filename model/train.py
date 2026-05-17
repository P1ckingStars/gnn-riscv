"""Supervised training entry point for the GNN graph generator.

Loads either a synthetic dataset (``--n-specs N``) or a pre-built one
(``--dataset-path data/v1``), splits into train/val, runs sample-by-sample teacher-forced
CE, and logs per-epoch metrics to ``<exp-dir>/log.csv``.

Usage:

    # Quick smoke run on synthetic data
    python -m model.train --n-specs 64 --epochs 30

    # Full run from a pre-built dataset
    python scripts/build_dataset.py --n 1000 --out data/v1 --seed 0 --no-baselines
    python -m model.train --dataset-path data/v1 --epochs 60 \
        --exp-dir experiments/001_scale1k --val-frac 0.1
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model.data import Sample, SpecDataset, collate_list, random_split
from model.graph import ARITY, ComputeGraph, NodeOp, graph_to_steps
from model.graph_gen import GraphGenerator
from model.spec_encoder import SpecEncoder
from model.vocab import OpVocab, SpecVocab


def build_models(
    sv: SpecVocab, ov: OpVocab, d_model: int, n_layers: int, n_heads: int,
    device: torch.device,
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
    ctx, _ = encoder(ids)
    return generator.teacher_forced_loss(ctx, sample)


@torch.no_grad()
def eval_val_loss(
    encoder: SpecEncoder, generator: GraphGenerator, ds: SpecDataset, device: torch.device,
) -> float:
    encoder.eval(); generator.eval()
    total = 0.0
    n = 0
    for sample in ds:
        ids = torch.tensor(sample.spec_tokens, dtype=torch.long, device=device).unsqueeze(0)
        ctx, _ = encoder(ids)
        losses = generator.teacher_forced_loss(ctx, sample)
        total += float(losses["loss"].item())
        n += 1
    encoder.train(); generator.train()
    return total / max(n, 1)


def greedy_match(
    encoder: SpecEncoder, generator: GraphGenerator, sample: Sample, device: torch.device,
) -> tuple[bool, int, int, ComputeGraph | None]:
    """Returns (full_match, op_steps_correct, op_steps_total, generated_graph)."""
    encoder.eval(); generator.eval()
    with torch.no_grad():
        ids = torch.tensor(sample.spec_tokens, dtype=torch.long, device=device).unsqueeze(0)
        ctx, _ = encoder(ids)
        gen_graph = generator.generate(
            ctx, sample.spec.inputs, sample.spec.ret_ty,
            max_steps=max(16, len(sample.op_ids) + 4), greedy=True,
        )
    encoder.train(); generator.train()
    ref_steps = graph_to_steps(sample.graph)
    ref_n = len(ref_steps)
    if gen_graph is None:
        return False, 0, ref_n, None
    gen_steps = graph_to_steps(gen_graph)
    correct_ops = sum(
        1 for r, g in zip(ref_steps, gen_steps) if r.op is g.op
    )
    if len(ref_steps) != len(gen_steps):
        return False, correct_ops, ref_n, gen_graph
    for r, g in zip(ref_steps, gen_steps):
        if r.op is not g.op:
            return False, correct_ops, ref_n, gen_graph
        arity = ARITY[r.op]
        if r.operands[:arity] != g.operands[:arity]:
            return False, correct_ops, ref_n, gen_graph
        if r.op is NodeOp.CONST and r.const_value != g.const_value:
            return False, correct_ops, ref_n, gen_graph
    return True, correct_ops, ref_n, gen_graph


def eval_val_match(
    encoder: SpecEncoder, generator: GraphGenerator, ds: SpecDataset, device: torch.device,
) -> dict[str, float]:
    matches = 0
    op_correct = 0
    op_total = 0
    for sample in ds:
        full, correct, total, _ = greedy_match(encoder, generator, sample, device)
        if full:
            matches += 1
        op_correct += correct
        op_total += total
    return {
        "val_match": matches / max(len(ds), 1),
        "val_op_acc": op_correct / max(op_total, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    # Dataset
    ap.add_argument("--dataset-path", type=Path, default=None,
                    help="load from disk (scripts/build_dataset.py output); "
                         "otherwise synthesize on the fly")
    ap.add_argument("--n-specs", type=int, default=64,
                    help="used only when --dataset-path is not provided")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--n-params", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the loaded dataset size (for quick iteration)")
    # Training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--val-every", type=int, default=5,
                    help="run greedy-generate eval every K epochs (slower than val-loss)")
    # Output
    ap.add_argument("--exp-dir", type=Path, default=Path("experiments/000_smoke"))
    args = ap.parse_args()

    device = torch.device(args.device)
    sv = SpecVocab()
    ov = OpVocab()
    t_load = time.time()
    if args.dataset_path is not None:
        full_ds = SpecDataset.from_directory(
            args.dataset_path, sv=sv, ov=ov, limit=args.limit,
        )
        print(f"dataset: loaded {len(full_ds)} samples from {args.dataset_path} "
              f"in {time.time() - t_load:.1f}s")
    else:
        full_ds = SpecDataset.synthetic(
            n_specs=args.n_specs, seed=args.seed, max_depth=args.max_depth,
            n_params=args.n_params, sv=sv, ov=ov,
        )
        print(f"dataset: synthesized {len(full_ds)} samples in {time.time() - t_load:.1f}s")

    train_ds, val_ds = random_split(full_ds, val_frac=args.val_frac, seed=args.split_seed)
    print(f"split: train={len(train_ds)}  val={len(val_ds)}")

    loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_list)
    encoder, generator = build_models(
        sv, ov, args.d_model, args.n_layers, args.n_heads, device,
    )
    n_params = sum(p.numel() for p in encoder.parameters()) + sum(
        p.numel() for p in generator.parameters()
    )
    print(f"model params: {n_params / 1e6:.2f}M")
    opt = AdamW(list(encoder.parameters()) + list(generator.parameters()), lr=args.lr)

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    (args.exp_dir / "config.json").write_text(json.dumps({
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_params": n_params,
    }, indent=2))
    log_path = args.exp_dir / "log.csv"
    with log_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "train_loss", "val_loss", "val_match",
                    "val_op_acc", "wall_s"])

    t0 = time.time()
    best_val_match = -1.0
    for epoch in range(args.epochs):
        encoder.train(); generator.train()
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
        avg_train = epoch_loss / max(n, 1)

        val_loss = eval_val_loss(encoder, generator, val_ds, device)

        val_match = ""
        val_op_acc = ""
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            metrics = eval_val_match(encoder, generator, val_ds, device)
            val_match = f"{metrics['val_match']:.4f}"
            val_op_acc = f"{metrics['val_op_acc']:.4f}"
            if metrics["val_match"] > best_val_match:
                best_val_match = metrics["val_match"]
                torch.save({
                    "encoder": encoder.state_dict(),
                    "generator": generator.state_dict(),
                    "config": vars(args),
                    "epoch": epoch,
                    "val_match": metrics["val_match"],
                }, args.exp_dir / "best.pt")

        wall = time.time() - t0
        with log_path.open("a", newline="") as fh:
            csv.writer(fh).writerow(
                [epoch, f"{avg_train:.4f}", f"{val_loss:.4f}",
                 val_match, val_op_acc, f"{wall:.1f}"]
            )
        extra = f"  val_match={val_match}  val_op_acc={val_op_acc}" if val_match else ""
        print(f"epoch {epoch:3d}  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}"
              f"{extra}  wall={wall:.1f}s")

    torch.save({
        "encoder": encoder.state_dict(),
        "generator": generator.state_dict(),
        "config": vars(args),
        "epoch": args.epochs - 1,
    }, args.exp_dir / "final.pt")
    print(f"\nsaved: {args.exp_dir / 'final.pt'}  (best val_match: {best_val_match:.4f})")


if __name__ == "__main__":
    main()
