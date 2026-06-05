"""Frozen (non-trainable) projector: SASRec user embedding → soft prompt tokens.

``FrozenSoftPromptProjector`` maps a user embedding (B, d_in) into *m*
soft prompt token embeddings (B, m, d_out) using a **fixed random
projection** — no trainable parameters.  The projection matrix is stored
as a buffer so it is saved/loaded with ``state_dict`` and follows
``.to(device)`` calls, but never receives gradients.

This enables "Model C": frozen SASRec + frozen base LLaMA + frozen
projector → only LoRA adapters are trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm4rec.utils.logging import get_logger

logger = get_logger(__name__)


class FrozenSoftPromptProjector(nn.Module):
    """Fixed random projection from user-embedding space to LLM hidden space.

    All weights are registered as **buffers** (not ``Parameter``), so
    ``.parameters()`` is empty and no gradients are computed.

    Parameters
    ----------
    d_in : dimensionality of the input user embedding (e.g. 64).
    d_out : hidden size of the target LLM (e.g. 3072 for LLaMA-3.2-3B).
    n_soft_tokens : number of virtual prefix tokens to produce.
    per_token : if ``True``, each soft token has its own projection matrix
        ``(n_soft_tokens, d_in, d_out)``; if ``False``, a single
        ``(d_in, d_out)`` matrix is shared and the result is repeated.
    scale : scalar multiplier applied after projection (default 0.1).
    normalize : if ``True``, L2-normalize the input embedding before
        projection so the output magnitude is controlled by *scale* alone.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_soft_tokens: int = 8,
        per_token: bool = True,
        scale: float = 0.1,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_soft_tokens = n_soft_tokens
        self.per_token = per_token
        self.scale = scale
        self.normalize = normalize

        # ── Build fixed projection weights (buffers, NOT parameters) ─────
        if per_token:
            W = torch.randn(n_soft_tokens, d_in, d_out)
            # Kaiming-style scaling so output variance ≈ 1
            W.mul_(1.0 / (d_in ** 0.5))
            self.register_buffer("W", W)   # (m, d_in, d_out)
        else:
            W = torch.randn(d_in, d_out)
            W.mul_(1.0 / (d_in ** 0.5))
            self.register_buffer("W", W)   # (d_in, d_out)

        logger.info(
            "FrozenSoftPromptProjector: d_in=%d → d_out=%d, "
            "n_soft_tokens=%d, per_token=%s, scale=%.4f, normalize=%s, "
            "W shape=%s",
            d_in, d_out, n_soft_tokens, per_token, scale, normalize,
            tuple(self.W.shape),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project user embeddings into soft prompt token embeddings.

        Parameters
        ----------
        x : ``(B, d_in)`` user representations from SASRec.

        Returns
        -------
        soft_prompt : ``(B, n_soft_tokens, d_out)`` virtual token embeddings.
        """
        # Optional L2 normalisation
        if self.normalize:
            x = F.normalize(x, dim=-1)

        if self.per_token:
            # W: (m, d_in, d_out),  x: (B, d_in)
            # einsum: for each token m,  out_{b,m} = x_b @ W_m
            soft_prompt = torch.einsum("bi,mid->bmd", x, self.W)  # (B, m, d_out)
        else:
            # W: (d_in, d_out),  x: (B, d_in)
            projected = x @ self.W                                # (B, d_out)
            soft_prompt = projected.unsqueeze(1).expand(
                -1, self.n_soft_tokens, -1,
            )                                                     # (B, m, d_out)

        soft_prompt = soft_prompt * self.scale
        return soft_prompt
