"""Smoke tests for the LeetCode-style learned-compilation pipeline.

Covers: C parser → graph, asm tokenizer round-trip, encoder forward, decoder forward
+ teacher-forced backward, single-sample overfit decreases loss. Skips toolchain-gated
tests if the LeetCode corpus or RV64 toolchain isn't present.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from model.c_encoder import (
    CursorKindVocab, EdgeType, TypeClassVocab, ValueBucketVocab, parse_c_function,
    graph_to_pyg, CEncoder,
)
from model.asm_decoder import AsmVocab, detokenize_asm, tokenize_asm
from model.asm_decoder_module import AsmDecoder, lm_loss, shift_for_teacher_forcing


HAS_CORPUS = Path("data/leetcode/0007_reverse-integer").exists()
HAS_TOOLCHAIN = bool(
    shutil.which("riscv64-linux-gnu-gcc") and shutil.which("qemu-riscv64-static")
)
requires_corpus = pytest.mark.skipif(
    not HAS_CORPUS, reason="run scripts/ingest_leetcode.py first",
)


_REVERSE_C = """
#include <stdint.h>
int reverse(int x) {
    long long res = 0;
    while (x != 0) { res = res * 10 + x % 10; x /= 10; }
    return res < -2147483648LL || res > 2147483647LL ? 0 : (int)res;
}
"""


def test_parse_c_function_basic():
    kv, tv, vv = CursorKindVocab(), TypeClassVocab(), ValueBucketVocab()
    g = parse_c_function(_REVERSE_C, "reverse", kv, tv, vv)
    assert g.n_nodes() > 5
    assert g.n_edges() > 5
    kinds = {n.kind_name for n in g.nodes}
    assert "FUNCTION_DECL" in kinds and "WHILE_STMT" in kinds


def test_graph_has_three_edge_types():
    kv, tv, vv = CursorKindVocab(), TypeClassVocab(), ValueBucketVocab()
    g = parse_c_function(_REVERSE_C, "reverse", kv, tv, vv)
    for t in EdgeType:
        # Every edge type should appear at least once on this non-trivial function.
        assert any(e.type is t for e in g.edges), f"missing edge type {t.name}"


def test_graph_to_pyg_shapes():
    kv, tv, vv = CursorKindVocab(), TypeClassVocab(), ValueBucketVocab()
    g = parse_c_function(_REVERSE_C, "reverse", kv, tv, vv)
    data = graph_to_pyg(g, kv, tv, vv)
    assert data.x.shape == (g.n_nodes(), 4)
    assert data.edge_index.shape == (2, g.n_edges())
    assert data.edge_type.shape == (g.n_edges(),)


def test_encoder_forward_finite():
    kv, tv, vv = CursorKindVocab(), TypeClassVocab(), ValueBucketVocab()
    g = parse_c_function(_REVERSE_C, "reverse", kv, tv, vv)
    data = graph_to_pyg(g, kv, tv, vv)
    enc = CEncoder.build(kv, tv, vv, d_model=32, n_layers=1, n_heads=2)
    ctx, per_node = enc(data)
    assert torch.isfinite(ctx).all() and torch.isfinite(per_node).all()
    assert ctx.shape == (32,)
    assert per_node.shape == (g.n_nodes(), 32)


# ---- asm tokenizer / decoder -----------------------------------------------

_TINY_ASM = """\
\t.text
\t.globl\treverse
\t.type\treverse, @function
reverse:
\tbeq\ta0,zero,.L4
\tli\ta3,10
.L3:
\tremw\ta2,a0,a3
\tdivw\ta0,a0,a3
\tbne\ta0,zero,.L3
.L4:
\tret
\t.size\treverse, .-reverse
"""


def test_tokenizer_round_trip_static_asm():
    av = AsmVocab()
    ids = tokenize_asm(_TINY_ASM, "reverse", av)
    back = detokenize_asm(ids, av, fn_name="reverse")
    ids2 = tokenize_asm(back, "reverse", av)
    assert ids == ids2


def test_decoder_forward_backward():
    kv, tv, vv = CursorKindVocab(), TypeClassVocab(), ValueBucketVocab()
    av = AsmVocab()
    g = parse_c_function(_REVERSE_C, "reverse", kv, tv, vv)
    data = graph_to_pyg(g, kv, tv, vv)
    enc = CEncoder.build(kv, tv, vv, d_model=32, n_layers=1, n_heads=2)
    dec = AsmDecoder(vocab_size=av.size, d_model=32, n_layers=1, n_heads=2)
    ids = torch.tensor(tokenize_asm(_TINY_ASM, "reverse", av), dtype=torch.long)
    ctx, per_node = enc(data)
    inp, tgt = shift_for_teacher_forcing(ids)
    logits = dec(inp.unsqueeze(0), per_node, ctx)
    loss = lm_loss(logits, tgt.unsqueeze(0))
    loss.backward()
    assert torch.isfinite(loss)


def test_overfit_one_sample_decreases_loss():
    """50 steps on one sample should drive teacher-forced loss substantially below
    the random-init baseline."""
    kv, tv, vv = CursorKindVocab(), TypeClassVocab(), ValueBucketVocab()
    av = AsmVocab()
    g = parse_c_function(_REVERSE_C, "reverse", kv, tv, vv)
    data = graph_to_pyg(g, kv, tv, vv)
    enc = CEncoder.build(kv, tv, vv, d_model=64, n_layers=2, n_heads=4)
    dec = AsmDecoder(vocab_size=av.size, d_model=64, n_layers=2, n_heads=4)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=3e-3)
    ids = torch.tensor(tokenize_asm(_TINY_ASM, "reverse", av), dtype=torch.long)
    initial = None
    final = None
    for step in range(60):
        opt.zero_grad()
        ctx, per_node = enc(data)
        inp, tgt = shift_for_teacher_forcing(ids)
        logits = dec(inp.unsqueeze(0), per_node, ctx)
        loss = lm_loss(logits, tgt.unsqueeze(0))
        loss.backward()
        opt.step()
        if step == 0: initial = float(loss.item())
        final = float(loss.item())
    assert final < initial * 0.5, f"loss did not drop enough: {initial:.3f} → {final:.3f}"


# ---- end-to-end via corpus (skip if not present) ---------------------------

@requires_corpus
def test_load_one_problem_from_corpus_and_train_one_step():
    from model.train_leetcode import LCSample, teacher_forced_loss
    kv, tv, vv = CursorKindVocab(), TypeClassVocab(), ValueBucketVocab()
    av = AsmVocab()
    d = Path("data/leetcode/0007_reverse-integer")
    sig = json.loads((d / "signature.json").read_text())
    src = (d / "source.c").read_text()
    asm = (d / "ref_o2.s").read_text()
    g = parse_c_function(src, sig["fn_name"], kv, tv, vv)
    ids = torch.tensor(tokenize_asm(asm, sig["fn_name"], av), dtype=torch.long)
    pyg = graph_to_pyg(g, kv, tv, vv)
    enc = CEncoder.build(kv, tv, vv, d_model=32, n_layers=1, n_heads=2)
    dec = AsmDecoder(vocab_size=av.size, d_model=32, n_layers=1, n_heads=2)
    sample = LCSample(problem_id=d.name, pyg_data=pyg, asm_ids=ids,
                      signature=sig, ref_o2_asm=asm)
    loss = teacher_forced_loss(enc, dec, sample, torch.device("cpu"))
    assert torch.isfinite(loss)
    loss.backward()
