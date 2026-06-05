"""Soft-prompt injector: SASRec user embedding → virtual prefix tokens.

The ``SoftPromptInjector`` is a small MLP that projects a user embedding
from SASRec (B, d_user) into *m* soft prompt token embeddings
(B, m, d_model) that live in the same space as LLaMA's hidden states.

These virtual tokens are **prepended** to the text-token embeddings
before the first transformer layer, giving the frozen (or LoRA-adapted)
LLM a user-personalised context without modifying any of its weights.

Typical forward flow
--------------------
::

    user_emb        = sasrec.get_last_hidden(seq)        # (B, d_user)
    soft_prompt, _  = injector(user_emb)                 # (B, m, d_model)
    text_embeds     = llm.get_input_embeddings()(ids)    # (B, L, d_model)
    merged          = prepend_soft_prompt(text_embeds, attn_mask, soft_prompt)
    out             = llm(**merged)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class SoftPromptInjector(nn.Module):
    """MLP projector: user embedding → soft prompt tokens.

    Parameters
    ----------
    user_dim : dimensionality of the SASRec user embedding.
    llm_dim : hidden size of the target LLM (e.g. 4096 for LLaMA-7B).
    n_soft_tokens : number of virtual prefix tokens to generate.
    hidden_dim : width of the MLP hidden layer.
    dropout : dropout rate applied after the activation.
    activation : activation function name (``"gelu"`` | ``"relu"`` |
        ``"silu"`` | ``"tanh"``).
    """

    def __init__(
        self,
        user_dim: int = 64,
        llm_dim: int = 4096,
        n_soft_tokens: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.n_soft_tokens = n_soft_tokens
        self.llm_dim = llm_dim

        act_cls = _ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(
                f"Unknown activation {activation!r}. "
                f"Choose from {list(_ACTIVATIONS)}"
            )

        self.projector = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            act_cls(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_soft_tokens * llm_dim),
        )
        self.ln = nn.LayerNorm(llm_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, user_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project user embeddings into soft prompt token embeddings.

        Parameters
        ----------
        user_emb : ``(B, d_user)`` user representations from SASRec.

        Returns
        -------
        soft_prompt : ``(B, m, d_model)`` virtual token embeddings.
        attn_extension : ``(B, m)`` ones — attention mask entries for the
            prepended tokens.
        """
        B = user_emb.size(0)

        projected = self.projector(user_emb)                     # (B, m * D)
        soft_prompt = projected.view(B, self.n_soft_tokens, self.llm_dim)
        soft_prompt = self.ln(soft_prompt)                       # (B, m, D)

        attn_extension = torch.ones(
            B, self.n_soft_tokens,
            device=user_emb.device, dtype=torch.long,
        )
        return soft_prompt, attn_extension


# ─────────────────────────────────────────────────────────────────────────────
# Helper: merge soft prompt with text embeddings
# ─────────────────────────────────────────────────────────────────────────────

def prepend_soft_prompt(
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    soft_prompt_embeds: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Concatenate soft prompt tokens in front of text token embeddings.

    After this call the sequence layout is::

        [soft_1, …, soft_m, text_1, …, text_L, PAD, …]

    The returned dict can be unpacked directly into a HuggingFace
    ``model(**merged)`` call (uses ``inputs_embeds`` instead of
    ``input_ids``).

    Parameters
    ----------
    input_embeds : ``(B, L, D)`` text token embeddings from
        ``model.get_input_embeddings()(input_ids)``.
    attention_mask : ``(B, L)`` text attention mask (1 = real, 0 = pad).
    soft_prompt_embeds : ``(B, m, D)`` output of ``SoftPromptInjector``.
    labels : ``(B, L)`` optional training labels.  If provided, ``-100``
        entries are prepended for the soft prompt positions so the LM loss
        is **not** computed over the virtual tokens.

    Returns
    -------
    dict with keys ``inputs_embeds``, ``attention_mask``, and optionally
    ``labels`` — all with sequence length ``m + L``.
    """
    B, m, D = soft_prompt_embeds.shape

    # Cast soft prompt to match LLM embedding dtype (e.g. fp32 → bf16)
    # to avoid silent upcast in torch.cat
    if soft_prompt_embeds.dtype != input_embeds.dtype:
        soft_prompt_embeds = soft_prompt_embeds.to(dtype=input_embeds.dtype)

    # Embeddings: [soft | text]
    combined_embeds = torch.cat(
        [soft_prompt_embeds, input_embeds], dim=1,
    )  # (B, m + L, D)

    # Attention mask: [1…1 | original mask]
    soft_mask = torch.ones(
        B, m, device=attention_mask.device, dtype=attention_mask.dtype,
    )
    combined_mask = torch.cat([soft_mask, attention_mask], dim=1)  # (B, m + L)

    result: Dict[str, torch.Tensor] = {
        "inputs_embeds": combined_embeds,
        "attention_mask": combined_mask,
    }

    # Labels: [-100…-100 | original labels]
    if labels is not None:
        soft_labels = torch.full(
            (B, m), fill_value=-100,
            device=labels.device, dtype=labels.dtype,
        )
        result["labels"] = torch.cat([soft_labels, labels], dim=1)  # (B, m + L)

    return result
