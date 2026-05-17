"""Small transformer encoder over the linearized spec AST.

Input is a 1-D tensor of token IDs from ``model.vocab.tokenize_spec``. Output is
(context, per_token_embeddings). The context is the embedding of the leading ``CLS``
token; per-token embeddings are used downstream as cross-attention keys/values for the
graph generator.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.vocab import SpecTok


class SpecEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = SpecTok.PAD.value,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ids: (B, L). Returns (context: (B, D), per_token: (B, L, D))."""
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, L = ids.shape
        if L > self.max_len:
            raise ValueError(f"spec_tokens length {L} > max_len {self.max_len}")
        positions = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.token_embed(ids) + self.pos_embed(positions)
        pad_mask = ids == self.pad_idx
        out = self.transformer(x, src_key_padding_mask=pad_mask)
        out = self.norm(out)
        ctx = out[:, 0, :]  # CLS pooling
        return ctx, out
