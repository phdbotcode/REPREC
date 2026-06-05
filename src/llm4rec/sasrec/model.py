"""SASRec – Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).

Right-padded implementation: sequences are ``[i1, i2, …, iN, 0, 0, …]``.
A combined causal + padding attention mask ensures correct behaviour.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class PointWiseFeedForward(nn.Module):
    """Two-layer position-wise FFN (same dim, following original SASRec)."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class SASRecBlock(nn.Module):
    """One transformer block: self-attention → residual+LN → FFN → residual+LN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = PointWiseFeedForward(d_model, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention + residual + layer norm
        attn_out, _ = self.attn(
            x, x, x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
        )
        x = self.ln1(x + self.drop1(attn_out))

        # Feed-forward + residual + layer norm
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.drop2(ffn_out))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation model.

    Parameters
    ----------
    num_items : vocabulary size (item IDs are 1 … num_items; 0 = pad).
    emb_dim : embedding / hidden dimension.
    max_len : maximum sequence length.
    n_heads : number of attention heads.
    n_blocks : number of transformer blocks.
    dropout : dropout rate applied throughout.
    """

    def __init__(
        self,
        num_items: int,
        emb_dim: int = 64,
        max_len: int = 50,
        n_heads: int = 2,
        n_blocks: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.max_len = max_len

        # Embeddings (padding_idx=0 keeps the pad vector at zero)
        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [SASRecBlock(emb_dim, n_heads, dropout) for _ in range(n_blocks)]
        )
        self.final_ln = nn.LayerNorm(emb_dim)

        self._init_weights()

    # ── weight init ──────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── masks ────────────────────────────────────────────────────────────
    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (float): 0 = attend, -inf = block."""
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode a right-padded item sequence.

        Parameters
        ----------
        seq : ``(B, L)`` tensor of item IDs (right-padded with 0).

        Returns
        -------
        ``(B, L, D)`` hidden states at every position.
        """
        B, L = seq.shape

        # Masks
        causal_mask = self._make_causal_mask(L, seq.device)        # (L, L)
        key_padding_mask = (seq == 0)                               # (B, L)

        # Embeddings
        positions = torch.arange(L, device=seq.device).unsqueeze(0)  # (1, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask, key_padding_mask)

        x = self.final_ln(x)
        return x  # (B, L, D)

    # ── helpers used by trainer / inference ───────────────────────────────
    def get_last_hidden(self, seq: torch.Tensor) -> torch.Tensor:
        """Return the hidden state at the last *non-padded* position.

        Parameters
        ----------
        seq : ``(B, L)`` right-padded item IDs.

        Returns
        -------
        ``(B, D)`` user representations.
        """
        hidden = self.forward(seq)                       # (B, L, D)
        lengths = (seq != 0).sum(dim=1) - 1              # (B,) last valid idx
        return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]

    def score_candidates(
        self,
        user_emb: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Dot-product score between user embeddings and candidate items.

        Parameters
        ----------
        user_emb : ``(B, D)`` or ``(D,)`` user representations.
        candidate_ids : ``(B, N)`` or ``(N,)`` item IDs to score.

        Returns
        -------
        ``(B, N)`` or ``(N,)`` scores.
        """
        cand_emb = self.item_emb(candidate_ids)
        if user_emb.dim() == 1:
            return (cand_emb * user_emb.unsqueeze(0)).sum(-1)
        return (cand_emb * user_emb.unsqueeze(-2)).sum(-1)
