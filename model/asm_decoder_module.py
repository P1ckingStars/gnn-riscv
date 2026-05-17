"""Transformer decoder over asm token streams with cross-attention to encoder
per-node embeddings.

Architecture:
  - Token embedding + sinusoidal positional embedding.
  - N causal-self-attention + cross-attention transformer layers.
  - LM head over the asm vocab.

Training is teacher-forced via the standard "shift-by-one + CE" loss. Inference is
greedy or sampled rollout, stopping on EOS or a max-length cap.

Cross-attention attends to the C encoder's per-node embeddings (shape ``[N_nodes, D]``).
The function-level context vector ``ctx`` is broadcast as a prefix token so the decoder
gets a global summary cheaply.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsmDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        max_len: int = 4096,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            _DecoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,             # (B, L) or (L,)
        encoder_node_embeds: torch.Tensor,   # (N, D) or (B, N, D)
        encoder_ctx: torch.Tensor | None = None,  # (D,) or (B, D), prepended as prefix
    ) -> torch.Tensor:
        """Returns logits (B, L, V)."""
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        B, L = token_ids.shape
        if L > self.max_len:
            raise ValueError(f"asm seq len {L} > max_len {self.max_len}")
        if encoder_node_embeds.dim() == 2:
            encoder_node_embeds = encoder_node_embeds.unsqueeze(0).expand(B, -1, -1)
        if encoder_ctx is not None and encoder_ctx.dim() == 1:
            encoder_ctx = encoder_ctx.unsqueeze(0).expand(B, -1)

        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        if encoder_ctx is not None:
            # Add context to position 0 so it always conditions the rest.
            x[:, 0, :] = x[:, 0, :] + encoder_ctx

        # Causal mask
        causal = torch.triu(
            torch.ones(L, L, device=token_ids.device, dtype=torch.bool), diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, encoder_node_embeds, causal)
        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        bos_id: int,
        eos_id: int,
        encoder_node_embeds: torch.Tensor,
        encoder_ctx: torch.Tensor | None = None,
        max_len: int = 1024,
        greedy: bool = True,
        temperature: float = 1.0,
    ) -> list[int]:
        device = next(self.parameters()).device
        seq = torch.tensor([bos_id], dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(max_len - 1):
            logits = self.forward(seq, encoder_node_embeds, encoder_ctx)
            next_logits = logits[0, -1, :] / max(temperature, 1e-6)
            if greedy:
                nxt = int(next_logits.argmax().item())
            else:
                nxt = int(torch.distributions.Categorical(logits=next_logits).sample().item())
            seq = torch.cat([seq, torch.tensor([[nxt]], device=device)], dim=1)
            if nxt == eos_id:
                break
        return seq[0].tolist()


class _DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, enc: torch.Tensor, causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.n1(x)
        sa, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.dropout(sa)
        h = self.n2(x)
        ca, _ = self.cross_attn(h, enc, enc, need_weights=False)
        x = x + self.dropout(ca)
        h = self.n3(x)
        x = x + self.dropout(self.ff(h))
        return x


def shift_for_teacher_forcing(
    target_ids: torch.Tensor, pad_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a full token sequence (incl. BOS and EOS), return (input, target) where
    input = sequence[:-1] and target = sequence[1:] (next-token prediction)."""
    if target_ids.dim() == 1:
        return target_ids[:-1], target_ids[1:]
    return target_ids[:, :-1], target_ids[:, 1:]


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Standard CE next-token-prediction loss; ignores PAD positions."""
    V = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), ignore_index=pad_idx,
    )
