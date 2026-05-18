"""Training loop for the LeetCode-style learned-compilation model.

Pipeline:
    parsed C function (graph)  ──►  CEncoder (GNN)  ──►  (ctx, per-node)
                                                            │
    asm token ids (teacher forced)  ──►  AsmDecoder  ──►  next-token logits
                                                            │
                                                            ▼
                                                      cross-entropy

Eval (slow — runs every --val-every epochs):
    greedy generate → detokenize → differential_test vs gcc -O2 reference.

Usage:
    python -m model.train_leetcode --data data/leetcode --epochs 30 \
        --exp-dir experiments/leetcode_001 --val-every 5
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from eval.differential_test import (
    DifferentialVerdict, UnsupportedSignature, differential_test, signature_is_int_only,
)
from model.asm_decoder import AsmVocab, TokenizeError, detokenize_asm, tokenize_asm
from model.asm_decoder_module import AsmDecoder, lm_loss, shift_for_teacher_forcing
from model.c_encoder import (
    CEncoder, CursorKindVocab, TypeClassVocab, ValueBucketVocab, graph_to_pyg,
    parse_c_function,
)


@dataclass
class LCSample:
    """One LeetCode training example."""
    problem_id: str
    pyg_data: object       # PyG Data
    asm_ids: torch.Tensor  # (L,) long
    signature: dict
    ref_o2_asm: str        # kept for differential eval


def load_corpus(
    data_dir: Path, kv: CursorKindVocab, tv: TypeClassVocab, vv: ValueBucketVocab,
    av: AsmVocab, limit: int | None = None, int_only: bool = False,
) -> list[LCSample]:
    """Load (source.c, ref_o2.s, signature.json) triples from a dataset directory.

    ``int_only`` filters to diff-testable signatures (needed for val_pass eval). When
    False, accepts any sample whose asm tokenizes — appropriate for pretraining
    on AnghaBench-style real C where most signatures touch pointers/structs.
    """
    samples: list[LCSample] = []
    n_skipped = 0
    skip_reasons: dict[str, int] = {}

    def bump(r: str) -> None:
        skip_reasons[r] = skip_reasons.get(r, 0) + 1

    candidates = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        # AnghaBench dirs start with 'ang_'; LeetCode dirs start with a digit.
        if not (d.name[0].isdigit() or d.name.startswith("ang_")):
            continue
        candidates.append(d)

    for d in candidates:
        if limit is not None and len(samples) >= limit:
            break
        try:
            sig = json.loads((d / "signature.json").read_text())
            src = (d / "source.c").read_text()
            asm = (d / "ref_o2.s").read_text()
        except FileNotFoundError:
            n_skipped += 1; bump("missing_file"); continue
        if int_only:
            ok = sig.get("diff_testable", signature_is_int_only(sig))
            if not ok:
                n_skipped += 1; bump("non_int_signature"); continue
        try:
            g = parse_c_function(src, sig["fn_name"], kv, tv, vv)
        except Exception as e:
            n_skipped += 1; bump(f"parse_c: {type(e).__name__}"); continue
        try:
            ids = torch.tensor(
                tokenize_asm(asm, sig["fn_name"], av), dtype=torch.long,
            )
        except TokenizeError as e:
            n_skipped += 1; bump(f"tokenize: {str(e)[:40]}"); continue
        pyg = graph_to_pyg(g, kv, tv, vv)
        samples.append(LCSample(
            problem_id=d.name, pyg_data=pyg, asm_ids=ids, signature=sig,
            ref_o2_asm=asm,
        ))
    print(f"loaded {len(samples)} samples from {data_dir}  (skipped {n_skipped})")
    if skip_reasons:
        for r, n in sorted(skip_reasons.items(), key=lambda kv: -kv[1])[:5]:
            print(f"    {n:3d}  {r}")
    return samples


def teacher_forced_loss(
    encoder, decoder: AsmDecoder, sample: LCSample, device: torch.device,
) -> torch.Tensor:
    pyg = sample.pyg_data
    pyg.x = pyg.x.to(device)
    pyg.edge_index = pyg.edge_index.to(device)
    pyg.edge_type = pyg.edge_type.to(device)
    ctx, per_node = encoder(pyg)
    inp, tgt = shift_for_teacher_forcing(sample.asm_ids.to(device))
    logits = decoder(inp.unsqueeze(0), per_node, ctx)
    return lm_loss(logits, tgt.unsqueeze(0))


def greedy_eval_one(
    encoder, decoder: AsmDecoder, sample: LCSample, av: AsmVocab,
    device: torch.device, n_inputs: int,
) -> tuple[bool, str | None, str]:
    """Greedy-generate, detokenize, differential-test against ref_o2. Returns
    (passed, failure_reason_or_None, generated_asm)."""
    from model.asm_decoder import BOS, EOS
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        pyg = sample.pyg_data
        pyg.x = pyg.x.to(device); pyg.edge_index = pyg.edge_index.to(device)
        pyg.edge_type = pyg.edge_type.to(device)
        ctx, per_node = encoder(pyg)
        # Cap generation length at 2x the reference length.
        max_len = min(2 * len(sample.asm_ids) + 32, 2048)
        ids = decoder.generate(
            bos_id=BOS, eos_id=EOS, encoder_node_embeds=per_node, encoder_ctx=ctx,
            max_len=max_len, greedy=True,
        )
    encoder.train(); decoder.train()
    try:
        gen_asm = detokenize_asm(ids, av, fn_name=sample.signature["fn_name"])
    except Exception as e:
        return False, f"detokenize: {e}", ""
    try:
        v = differential_test(
            gen_asm, sample.ref_o2_asm, sample.signature, n_inputs=n_inputs, seed=0,
        )
    except UnsupportedSignature as e:
        return False, f"unsupported: {e}", gen_asm
    except Exception as e:
        return False, f"diff_test: {type(e).__name__}: {e}", gen_asm
    if v.passed:
        return True, None, gen_asm
    return False, v.failure_reason or "unknown", gen_asm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/leetcode"),
                    help="training corpus (single-source mode)")
    ap.add_argument("--train-data", type=Path, default=None,
                    help="training corpus; overrides --data when used with --eval-data")
    ap.add_argument("--eval-data", type=Path, default=None,
                    help="val_pass eval corpus (held-out from training). When set,"
                         " --train-data is the training corpus and val_pass evals on"
                         " --eval-data's diff-testable subset.")
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="single-source mode only; ignored when --eval-data is set")
    ap.add_argument("--val-every", type=int, default=5,
                    help="run greedy + differential-test eval every K epochs (slow)")
    ap.add_argument("--val-n-inputs", type=int, default=32)
    ap.add_argument("--val-max-samples", type=int, default=32,
                    help="cap how many eval samples we run val_pass on per round")
    ap.add_argument("--device", type=str, default="auto",
                    help="'auto' uses cuda if available")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp-dir", type=Path, default=Path("experiments/leetcode_001"))
    ap.add_argument("--init-ckpt", type=Path, default=None,
                    help="warm-start encoder+decoder weights from a checkpoint "
                         "(arch must match)")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"device: {device}")
    torch.manual_seed(args.seed)
    import random
    random.seed(args.seed)

    kv, tv, vv = CursorKindVocab(), TypeClassVocab(), ValueBucketVocab()
    av = AsmVocab()
    print(f"vocabs — kind: {kv.size}  type: {tv.size}  value: {vv.size}  asm: {av.size}")

    t0 = time.time()
    if args.eval_data is not None:
        train_dir = args.train_data or args.data
        train_samples = load_corpus(
            train_dir, kv, tv, vv, av, limit=args.train_limit, int_only=False,
        )
        eval_pool = load_corpus(
            args.eval_data, kv, tv, vv, av, limit=args.eval_limit, int_only=True,
        )
        # Split eval_pool into a val_loss subset (small) + val_pass subset (capped).
        random.Random(args.seed).shuffle(eval_pool)
        val_samples = eval_pool[: max(8, len(eval_pool) // 5)]
        val_pass_samples = eval_pool[: args.val_max_samples]
        print(f"split: train={len(train_samples)}  val_loss={len(val_samples)}  "
              f"val_pass={len(val_pass_samples)}")
    else:
        samples = load_corpus(
            args.data, kv, tv, vv, av, limit=args.train_limit, int_only=True,
        )
        indices = list(range(len(samples)))
        random.Random(args.seed).shuffle(indices)
        n_val = max(1, int(round(len(samples) * args.val_frac)))
        val_idx = set(indices[:n_val])
        train_samples = [s for i, s in enumerate(samples) if i not in val_idx]
        val_samples = [s for i, s in enumerate(samples) if i in val_idx]
        val_pass_samples = val_samples[: args.val_max_samples]
        print(f"split: train={len(train_samples)}  val={len(val_samples)}")
    print(f"loaded corpora in {time.time() - t0:.1f}s")

    encoder = CEncoder.build(
        kv, tv, vv, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
    ).to(device)
    decoder = AsmDecoder(
        vocab_size=av.size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(device)
    n_params = sum(p.numel() for p in encoder.parameters()) + \
               sum(p.numel() for p in decoder.parameters())
    print(f"model params: {n_params / 1e6:.2f}M")

    if args.init_ckpt is not None:
        blob = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        enc_state = blob["encoder"]; dec_state = blob["decoder"]
        encoder.load_state_dict(enc_state)
        decoder.load_state_dict(dec_state)
        print(f"warm-start from {args.init_ckpt}  (pretrain epoch {blob.get('epoch','?')})")
    opt = AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr,
    )

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    (args.exp_dir / "config.json").write_text(json.dumps({
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_params": n_params,
        "n_train": len(train_samples),
        "n_val_loss": len(val_samples),
        "n_val_pass": len(val_pass_samples),
    }, indent=2))
    log_path = args.exp_dir / "log.csv"
    with log_path.open("w", newline="") as fh:
        csv.writer(fh).writerow(["epoch", "train_loss", "val_loss",
                                  "val_pass_rate", "wall_s"])

    best_val_pass = -1.0
    t0 = time.time()
    for epoch in range(args.epochs):
        encoder.train(); decoder.train()
        random.Random(args.seed + epoch).shuffle(train_samples)
        ep_loss = 0.0; n = 0
        for s in train_samples:
            opt.zero_grad()
            loss = teacher_forced_loss(encoder, decoder, s, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0,
            )
            opt.step()
            ep_loss += float(loss.item()); n += 1
        avg_train = ep_loss / max(n, 1)

        val_loss_total = 0.0; vn = 0
        encoder.eval(); decoder.eval()
        with torch.no_grad():
            for s in val_samples:
                val_loss_total += float(
                    teacher_forced_loss(encoder, decoder, s, device).item()
                )
                vn += 1
        encoder.train(); decoder.train()
        val_loss = val_loss_total / max(vn, 1)

        val_pass_str = ""
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            n_pass = 0
            for s in val_pass_samples:
                ok, _why, _gen = greedy_eval_one(
                    encoder, decoder, s, av, device, args.val_n_inputs,
                )
                if ok:
                    n_pass += 1
            val_pass_rate = n_pass / max(len(val_pass_samples), 1)
            val_pass_str = f"{val_pass_rate:.4f}"
            if val_pass_rate > best_val_pass:
                best_val_pass = val_pass_rate
                torch.save({
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "config": vars(args),
                    "epoch": epoch, "val_pass_rate": val_pass_rate,
                }, args.exp_dir / "best.pt")

        wall = time.time() - t0
        with log_path.open("a", newline="") as fh:
            csv.writer(fh).writerow(
                [epoch, f"{avg_train:.4f}", f"{val_loss:.4f}",
                 val_pass_str, f"{wall:.1f}"]
            )
        extra = f"  val_pass={val_pass_str}" if val_pass_str else ""
        print(f"epoch {epoch:3d}  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}"
              f"{extra}  wall={wall:.1f}s")

    torch.save({
        "encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
        "config": vars(args), "epoch": args.epochs - 1,
    }, args.exp_dir / "final.pt")
    print(f"\nsaved {args.exp_dir/'final.pt'}  (best val_pass: {best_val_pass:.4f})")


if __name__ == "__main__":
    main()
