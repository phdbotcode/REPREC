"""Compute ranking scores from LLaMA logits for YES/NO recommendation.

Two scoring paths are provided:

1. **Prompt-only** (:func:`score_from_logits`) — fast, single-token.
   Forward-pass on the prompt alone; extract ``logP("Yes")`` at the last
   non-pad position.  Works when "Yes" / "No" are each a single token
   (the common case for LLaMA).

2. **Sequence-level** (:func:`compute_sequence_logprob`) — robust,
   multi-token.  Forward-pass on ``prompt + " Yes"`` with labels set via
   :func:`~llm4rec.llm.tokenizer_utils.build_training_input`.  Sums
   ``logP`` over every answer-token position, correctly handling answers
   that span more than one BPE token.

For evaluation ranking (1 positive + 1000 negatives) path 1 is
recommended for speed; path 2 is useful for monitoring training loss
or when using a tokenizer where "Yes" is multi-token.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Token-ID resolution
# ─────────────────────────────────────────────────────────────────────────────

def get_yes_no_token_ids(
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, List[int]]:
    """Return the **full** token-ID lists for "Yes" and "No".

    Unlike ``tokenizer_utils.get_answer_token_ids`` (which returns only the
    first ID), this function returns the complete list so that multi-token
    scoring is possible.

    Returns
    -------
    ``{"yes": [id, ...], "no": [id, ...]}``
    """
    result: Dict[str, List[int]] = {}

    for key, candidates in [
        ("yes", [" Yes", "Yes", " yes", "yes"]),
        ("no",  [" No",  "No",  " no",  "no"]),
    ]:
        for text in candidates:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                result[key] = ids
                break
        if key not in result:
            raise ValueError(
                f"Could not resolve token IDs for answer '{key}'"
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Path 1 — prompt-only scoring (fast, single-token)
# ─────────────────────────────────────────────────────────────────────────────

def score_from_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    yes_id: int,
    no_id: Optional[int] = None,
    mode: str = "logprob_yes",
) -> torch.Tensor:
    """Extract a ranking score from the logits at the last prompt position.

    Parameters
    ----------
    logits : ``(B, L, V)`` raw logits from a causal-LM forward pass on
        the **prompt only** (no answer tokens appended).
    attention_mask : ``(B, L)`` — 1 for real tokens, 0 for padding.
    yes_id : single token ID for "Yes".
    no_id : single token ID for "No"  (required when *mode* is
        ``"logprob_diff"`` or ``"prob_ratio"``).
    mode : scoring strategy:

        * ``"logprob_yes"``  — ``logP(Yes)``  (default, simplest).
        * ``"logprob_diff"`` — ``logP(Yes) − logP(No)``.
        * ``"prob_ratio"``   — ``P(Yes) / (P(Yes) + P(No))``.

    Returns
    -------
    ``(B,)`` scalar scores — higher means more likely to interact.
    """
    B = logits.size(0)

    # Last non-pad position (logits there predict the *next* token = answer)
    last_pos = attention_mask.sum(dim=1) - 1                  # (B,)
    last_logits = logits[torch.arange(B, device=logits.device), last_pos]  # (B, V)

    log_probs = F.log_softmax(last_logits, dim=-1)            # (B, V)

    yes_lp = log_probs[:, yes_id]                             # (B,)

    if mode == "logprob_yes":
        return yes_lp

    if no_id is None:
        raise ValueError(f"mode={mode!r} requires no_id to be set")

    no_lp = log_probs[:, no_id]                               # (B,)

    if mode == "logprob_diff":
        return yes_lp - no_lp

    if mode == "prob_ratio":
        yes_p = yes_lp.exp()
        no_p = no_lp.exp()
        return yes_p / (yes_p + no_p + 1e-10)

    raise ValueError(f"Unknown scoring mode: {mode!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Path 2 — sequence-level scoring (robust, multi-token)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sequence_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Sum of token log-probabilities over answer positions.

    Designed for inputs built with
    :func:`~llm4rec.llm.tokenizer_utils.build_training_input`, where
    ``labels`` is ``-100`` over the prompt and the actual token IDs over
    the answer.  Handles multi-token answers automatically.

    Parameters
    ----------
    logits : ``(B, L, V)`` causal-LM logits (answer tokens appended).
    labels : ``(B, L)`` with ``-100`` for prompt / pad positions.

    Returns
    -------
    ``(B,)`` sum of log-probs for the answer tokens in each example.
    """
    # Causal LM convention: logits[i] predicts token at position i+1
    shift_logits = logits[:, :-1, :].contiguous()              # (B, L-1, V)
    shift_labels = labels[:, 1:].contiguous()                  # (B, L-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)            # (B, L-1, V)

    answer_mask = shift_labels != -100                         # (B, L-1)

    # Replace -100 with 0 for safe gather (masked out afterward)
    gather_ids = shift_labels.clamp(min=0).unsqueeze(-1)       # (B, L-1, 1)
    token_lp = log_probs.gather(2, gather_ids).squeeze(-1)     # (B, L-1)

    token_lp = token_lp * answer_mask.float()                  # zero non-answer
    return token_lp.sum(dim=-1)                                # (B,)


def compute_yes_logprob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    yes_ids: List[int],
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``logP(Yes)`` from logits where "Yes" tokens are appended.

    This is a convenience wrapper that locates the answer tokens at the
    **end** of each (non-padded) sequence and sums their log-probs.

    Parameters
    ----------
    logits : ``(B, L, V)`` logits from ``prompt + " Yes"`` forward pass.
    input_ids : ``(B, L)`` corresponding token IDs.
    yes_ids : list of token IDs comprising "Yes" (length ≥ 1).
    attention_mask : ``(B, L)``; if ``None``, no padding is assumed.

    Returns
    -------
    ``(B,)`` log-probabilities.
    """
    B, L, V = logits.shape
    m = len(yes_ids)

    if attention_mask is None:
        attention_mask = torch.ones(B, L, device=logits.device, dtype=torch.long)

    log_probs = F.log_softmax(logits, dim=-1)                  # (B, L, V)

    # Last non-pad position index
    seq_lens = attention_mask.sum(dim=1)                       # (B,)

    total_lp = torch.zeros(B, device=logits.device)
    for k, tok_id in enumerate(yes_ids):
        # Answer token k sits at position (seq_len - m + k)
        # Logit that predicts it is one position earlier
        logit_pos = seq_lens - m + k - 1                       # (B,)
        lp = log_probs[torch.arange(B, device=logits.device), logit_pos, tok_id]
        total_lp = total_lp + lp

    return total_lp                                            # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# High-level batch scorer
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    mode: str = "logprob_yes",
    inputs_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward-pass + score in one call (prompt-only, path 1).

    Accepts either ``input_ids`` (standard) **or** ``inputs_embeds``
    (when soft-prompt tokens have been prepended via
    :func:`~llm4rec.llm.injector.prepend_soft_prompt`).

    Parameters
    ----------
    model : HuggingFace causal LM (possibly PEFT-wrapped).
    input_ids : ``(B, L)`` — ignored when *inputs_embeds* is provided.
    attention_mask : ``(B, L)`` (or ``(B, m+L)`` if soft-prompt prepended).
    tokenizer : used to resolve Yes/No token IDs.
    mode : ``"logprob_yes"`` | ``"logprob_diff"`` | ``"prob_ratio"``.
    inputs_embeds : ``(B, L', D)`` — pass this instead of *input_ids*
        when using the soft-prompt injector.

    Returns
    -------
    ``(B,)`` ranking scores.
    """
    token_ids = get_yes_no_token_ids(tokenizer)
    yes_id = token_ids["yes"][0]
    no_id = token_ids["no"][0]

    model.eval()
    if inputs_embeds is not None:
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    return score_from_logits(
        outputs.logits, attention_mask,
        yes_id=yes_id, no_id=no_id, mode=mode,
    )
