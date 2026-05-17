"""Smoke tests for the GNN model stack.

Covers: graph round-trip, vocab coverage, dataset materialization, encoder forward,
generator forward + greedy generate, single training step decreases loss on a tiny
overfit batch.

PyTorch + torch_geometric required; tests skip if not installed.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from model.data import SpecDataset
from model.graph import (
    NodeOp, UnsupportedAstOp, ast_to_graph, evaluate_graph, graph_to_steps,
)
from model.graph_gen import GraphGenerator
from model.spec_encoder import SpecEncoder
from model.vocab import ConstVocab, OpVocab, SpecVocab, tokenize_spec
from spec.dsl import Param, Ty
from spec.generator import sample_spec
from spec.interpreter import evaluate
from spec.parser import parse_one


def _sq_sum_spec():
    return parse_one(
        "spec sq_sum(x: i32, y: i32) -> r: i32 { post: r == ((x * x) + (y * y)) >>u 1 }"
    )


# ---- Graph + vocab ----------------------------------------------------------

def test_ast_to_graph_roundtrips_interpreter():
    spec = _sq_sum_spec()
    g = ast_to_graph(spec)
    for inp in [(0, 0), (3, 4), (7, 11), (123, 456)]:
        assert evaluate(spec, inp) == evaluate_graph(g, inp)


def test_graph_to_steps_terminates_with_return():
    spec = _sq_sum_spec()
    g = ast_to_graph(spec)
    steps = graph_to_steps(g)
    assert steps[-1].op is NodeOp.RETURN
    assert all(s.op is not NodeOp.RETURN for s in steps[:-1])


def test_const_vocab_round_trips_canonical_values():
    cv = ConstVocab()
    for v in (0, 1, -1, 8, -8, 16, 32, 0xFFFFFFFF, 0x7FFF_FFFF):
        idx = cv.encode(v)
        assert idx != cv.OOV_INDEX, f"value {v} should be in vocab"
        assert cv.decode(idx) == v
    # OOV
    assert cv.encode(123456789) == cv.OOV_INDEX
    assert cv.decode(cv.OOV_INDEX) is None


def test_op_vocab_covers_all_emitted_ops():
    ov = OpVocab()
    # Every NodeOp except INPUT should be encodable.
    for op in NodeOp:
        if op is NodeOp.INPUT:
            continue
        idx = ov.encode(op)
        assert ov.decode(idx) is op


def test_spec_tokenize_uses_positional_var_slots():
    sv = SpecVocab()
    ids_xy = tokenize_spec(
        parse_one("spec f(x: i32, y: i32) -> r: i32 { post: r == x + y }"), sv,
    )
    ids_ab = tokenize_spec(
        parse_one("spec f(a: i32, b: i32) -> r: i32 { post: r == a + b }"), sv,
    )
    # Different names but same structure → identical token streams.
    assert ids_xy == ids_ab


# ---- Dataset ----------------------------------------------------------------

def test_dataset_builds_small_set():
    sv = SpecVocab(); ov = OpVocab()
    ds = SpecDataset.synthetic(n_specs=4, seed=42, max_depth=2, n_params=2, sv=sv, ov=ov)
    assert len(ds) == 4
    s = ds[0]
    assert s.spec_tokens, "non-empty token list"
    assert len(s.op_ids) == len(s.operand_lists) == len(s.const_bucket_ids)
    assert s.op_ids[-1] == ov.encode(NodeOp.RETURN)


def test_dataset_from_directory(tmp_path):
    """Build a tiny disk dataset via the public script's API, then load it back."""
    import subprocess, sys
    out = tmp_path / "ds"
    subprocess.run(
        [sys.executable, "scripts/build_dataset.py",
         "--n", "6", "--out", str(out), "--seed", "0",
         "--max-depth", "2", "--n-params", "2",
         "--n-inputs", "8", "--no-baselines"],
        check=True, capture_output=True,
    )
    sv = SpecVocab(); ov = OpVocab()
    ds = SpecDataset.from_directory(out, sv=sv, ov=ov)
    assert len(ds) >= 1   # some specs may be filtered for SEXT/ZEXT/TRUNC
    for s in ds.samples:
        assert s.op_ids[-1] == ov.encode(NodeOp.RETURN)


def test_random_split_partition_and_determinism():
    from model.data import random_split
    sv = SpecVocab(); ov = OpVocab()
    ds = SpecDataset.synthetic(n_specs=10, seed=0, max_depth=2, n_params=2, sv=sv, ov=ov)
    a, b = random_split(ds, val_frac=0.2, seed=7)
    a2, b2 = random_split(ds, val_frac=0.2, seed=7)
    assert len(a) + len(b) == len(ds)
    assert len(b) == 2
    # Determinism: same seed → same split.
    assert [s.spec_tokens for s in b.samples] == [s.spec_tokens for s in b2.samples]


def test_const_vocab_encodes_wrapped_negative():
    """Generator emits negatives in unsigned-canonical form; vocab must round-trip
    them via the ty hint, not put them in OOV."""
    cv = ConstVocab()
    from spec.dsl import Ty, mask_to
    wrapped = mask_to(-1, Ty.I32)  # 0xFFFFFFFF
    idx = cv.encode(wrapped, ty=Ty.I32)
    assert idx != cv.OOV_INDEX, "wrapped -1 should not be OOV when ty given"
    assert cv.decode(idx, ty=Ty.I32) == wrapped


# ---- Model forward + generate ----------------------------------------------

def test_encoder_forward_shapes():
    sv = SpecVocab()
    enc = SpecEncoder(vocab_size=sv.size, d_model=64, n_layers=1, n_heads=4)
    ids = torch.tensor(tokenize_spec(_sq_sum_spec(), sv), dtype=torch.long).unsqueeze(0)
    ctx, per_tok = enc(ids)
    assert ctx.shape == (1, 64)
    assert per_tok.shape == (1, ids.shape[1], 64)


def test_generator_teacher_forced_loss_is_finite():
    sv = SpecVocab(); ov = OpVocab()
    ds = SpecDataset.synthetic(n_specs=2, seed=7, max_depth=2, n_params=2, sv=sv, ov=ov)
    enc = SpecEncoder(vocab_size=sv.size, d_model=64, n_layers=1, n_heads=4)
    gen = GraphGenerator(op_vocab=ov, const_vocab=sv.consts, d_model=64, gnn_heads=4)
    sample = ds[0]
    ids = torch.tensor(sample.spec_tokens, dtype=torch.long).unsqueeze(0)
    ctx, _ = enc(ids)
    losses = gen.teacher_forced_loss(ctx, sample)
    assert torch.isfinite(losses["loss"])
    assert losses["loss"].item() > 0


def test_generator_generate_terminates_on_random_init():
    sv = SpecVocab(); ov = OpVocab()
    enc = SpecEncoder(vocab_size=sv.size, d_model=64, n_layers=1, n_heads=4)
    gen = GraphGenerator(op_vocab=ov, const_vocab=sv.consts, d_model=64, gnn_heads=4)
    spec = _sq_sum_spec()
    ids = torch.tensor(tokenize_spec(spec, sv), dtype=torch.long).unsqueeze(0)
    ctx, _ = enc(ids)
    # Random init may or may not emit RETURN within budget; both outcomes are valid.
    g = gen.generate(ctx, spec.inputs, spec.ret_ty, max_steps=16, greedy=True)
    assert g is None or g.nodes[-1].op is NodeOp.RETURN


# ---- Training step actually moves loss --------------------------------------

def test_overfit_one_sample_decreases_loss():
    """Sanity: 50 steps on a single sample should monotonically decrease loss."""
    sv = SpecVocab(); ov = OpVocab()
    ds = SpecDataset.synthetic(n_specs=1, seed=11, max_depth=2, n_params=2, sv=sv, ov=ov)
    enc = SpecEncoder(vocab_size=sv.size, d_model=64, n_layers=1, n_heads=4)
    gen = GraphGenerator(op_vocab=ov, const_vocab=sv.consts, d_model=64, gnn_heads=4)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(gen.parameters()), lr=5e-3)
    sample = ds[0]
    initial = None
    final = None
    for step in range(60):
        opt.zero_grad()
        ids = torch.tensor(sample.spec_tokens, dtype=torch.long).unsqueeze(0)
        ctx, _ = enc(ids)
        losses = gen.teacher_forced_loss(ctx, sample)
        losses["loss"].backward()
        opt.step()
        if step == 0:
            initial = float(losses["loss"].item())
        final = float(losses["loss"].item())
    assert initial is not None and final is not None
    assert final < initial * 0.6, f"loss did not drop enough: {initial:.3f} → {final:.3f}"
