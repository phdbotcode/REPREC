"""Tokenizer helpers for LLaMA-based recommendation scoring.

Key conventions
---------------
* **Right padding** is used for training (loss is masked on pad positions).
* **"Yes" / "No"** are the binary answer tokens.  We resolve their token IDs
  at init time so scoring is fast.
* ``build_training_input`` creates (input_ids, attention_mask, labels) where
  *labels* are ``-100`` over the prompt and real token IDs over the answer,
  matching HuggingFace's ``CausalLM`` internal label-shift convention.

Leakage note
------------
The prompt contains only the *prefix* history available at a given split.
The target / candidate item is mentioned in the prompt question but is
**not** part of the user's history.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from transformers import PreTrainedTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_tokenizer(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """Configure a LLaMA tokenizer for this project.

    * Sets ``pad_token`` to ``eos_token`` if missing.
    * Forces **right** padding (prompt on the left, pad on the right).
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Answer-token resolution
# ─────────────────────────────────────────────────────────────────────────────

def get_answer_token_ids(
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, int]:
    """Find the **first** token ID for "Yes" and "No" answer words.

    Tries the space-prefixed variant first (``" Yes"``) because LLaMA's
    tokenizer normally prepends a space to non-BOS tokens.  Falls back to
    the bare word and to lower-case.

    Returns
    -------
    dict with keys ``"yes"`` and ``"no"``, each mapping to an ``int``
    token ID suitable for log-probability extraction.
    """
    result: Dict[str, int] = {}

    for key, candidates in [
        ("yes", [" Yes", "Yes", " yes", "yes"]),
        ("no",  [" No",  "No",  " no",  "no"]),
    ]:
        for text in candidates:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                result[key] = ids[0]
                break
        if key not in result:
            raise ValueError(
                f"Could not resolve a token ID for answer '{key}' "
                f"with tokenizer {type(tokenizer).__name__}"
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Single-prompt tokenization (inference / scoring)
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_prompt(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """Tokenize one prompt for inference (no labels).

    Returns
    -------
    dict with ``input_ids`` and ``attention_mask``, each ``(1, L)``.
    """
    encoded = tokenizer(
        prompt,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],           # (1, L)
        "attention_mask": encoded["attention_mask"],  # (1, L)
    }


def tokenize_prompts(
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """Tokenize a list of prompts for batched inference.

    Returns
    -------
    dict with ``input_ids`` and ``attention_mask``, each ``(B, L)``.
    """
    encoded = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training-input construction (prompt + answer → masked labels)
# ─────────────────────────────────────────────────────────────────────────────

_IGNORE_INDEX = -100


def build_training_input(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    target: str,
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """Create a single training example with properly masked labels.

    The full text is ``prompt + " " + target`` (e.g. ``"… buy X next? Yes"``).
    Labels are set to ``-100`` over the prompt tokens so that the loss is
    computed **only** on the answer token(s).

    Parameters
    ----------
    tokenizer : configured tokenizer (see :func:`setup_tokenizer`).
    prompt : the natural-language prompt (no answer).
    target : answer string, typically ``"Yes"`` or ``"No"``.
    max_length : maximum total sequence length.

    Returns
    -------
    dict with ``input_ids``, ``attention_mask``, ``labels`` — each ``(L,)``
    (un-batched; stack in a collate function).
    """
    # Tokenize prompt (with BOS) and answer (no special tokens) separately,
    # then concatenate — avoids BPE boundary misalignment from double-encode.
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    answer_ids = tokenizer.encode(" " + target, add_special_tokens=False)

    # Right-truncate prompt to fit answer within max_length (preserves BOS)
    max_prompt_len = max_length - len(answer_ids)
    if max_prompt_len < 1:
        max_prompt_len = 1  # keep at least BOS
    prompt_ids = prompt_ids[:max_prompt_len]

    full_ids = prompt_ids + answer_ids
    seq_len = len(full_ids)
    prompt_len = len(prompt_ids)

    # Labels: -100 over prompt, real IDs over answer
    labels = [_IGNORE_INDEX] * prompt_len + answer_ids

    # Pad to max_length on the right
    pad_len = max_length - seq_len
    input_ids = full_ids + [tokenizer.pad_token_id] * pad_len
    attention_mask = [1] * seq_len + [0] * pad_len
    labels = labels + [_IGNORE_INDEX] * pad_len

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_training_batch(
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    targets: List[str],
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """Batch version of :func:`build_training_input`.

    Pads to the longest sequence in the batch (capped at *max_length*).

    Returns
    -------
    dict with ``input_ids``, ``attention_mask``, ``labels`` — each ``(B, L)``.
    """
    batch_input_ids: List[List[int]] = []
    batch_attn: List[List[int]] = []
    batch_labels: List[List[int]] = []

    for prompt, target in zip(prompts, targets):
        # Tokenize prompt (with BOS) and answer (no special tokens) separately,
        # then concatenate — avoids BPE boundary misalignment.
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        answer_ids = tokenizer.encode(" " + target, add_special_tokens=False)

        # Right-truncate prompt to fit answer within max_length (preserves BOS)
        max_prompt_len = max_length - len(answer_ids)
        if max_prompt_len < 1:
            max_prompt_len = 1
        prompt_ids = prompt_ids[:max_prompt_len]

        full_ids = prompt_ids + answer_ids
        prompt_len = len(prompt_ids)
        labels = [_IGNORE_INDEX] * prompt_len + answer_ids

        batch_input_ids.append(full_ids)
        batch_attn.append([1] * len(full_ids))
        batch_labels.append(labels)

    # Pad to longest in batch
    max_len = min(max(len(ids) for ids in batch_input_ids), max_length)

    for i in range(len(batch_input_ids)):
        pad_len = max_len - len(batch_input_ids[i])
        batch_input_ids[i] += [tokenizer.pad_token_id] * pad_len
        batch_attn[i] += [0] * pad_len
        batch_labels[i] += [_IGNORE_INDEX] * pad_len

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }
