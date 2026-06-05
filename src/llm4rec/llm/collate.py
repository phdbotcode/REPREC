"""Collate functions for batching variable-length LLM recommendation prompts.

Two modes:

* **Training** — each sample carries a ``label_text`` (``"Yes"`` / ``"No"``).
  The collator builds ``(input_ids, attention_mask, labels)`` with ``-100``
  masking over the prompt so the LM loss applies only to the answer tokens.

* **Inference** — no answer is appended.  The collator returns
  ``(input_ids, attention_mask)`` plus integer labels and metadata for
  downstream metric computation.

Usage with ``DataLoader``::

    from llm4rec.llm.collate import LLMCollator

    collator = LLMCollator(tokenizer, max_length=512, mode="train")
    loader   = DataLoader(dataset, batch_size=16, collate_fn=collator)
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import PreTrainedTokenizer

from llm4rec.llm.tokenizer_utils import build_training_batch, tokenize_prompts


class LLMCollator:
    """Callable collate function for :class:`~torch.utils.data.DataLoader`.

    Parameters
    ----------
    tokenizer : a configured HuggingFace tokenizer (see
        :func:`~llm4rec.llm.tokenizer_utils.setup_tokenizer`).
    max_length : maximum token-sequence length (prompt + answer).
    mode : ``"train"`` (with YES/NO supervision) or ``"eval"``
        (prompt-only for scoring / ranking).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        mode: str = "train",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    # ─────────────────────────────────────────────────────────────────────
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a list of samples from :class:`LLMPairDataset`.

        Expected sample keys (from ``LLMPairDataset.__getitem__``):
            * ``prompt``         — str
            * ``label``          — int (0 or 1)
            * ``label_text``     — str (``"Yes"`` / ``"No"``)
            * ``user_idx``       — int
            * ``candidate_item`` — int
        """
        prompts: List[str] = [s["prompt"] for s in batch]
        user_idxs = torch.tensor(
            [s["user_idx"] for s in batch], dtype=torch.long,
        )
        candidate_items = torch.tensor(
            [s["candidate_item"] for s in batch], dtype=torch.long,
        )
        labels_int = torch.tensor(
            [s["label"] for s in batch], dtype=torch.long,
        )

        if self.mode == "train":
            return self._collate_train(
                prompts, batch, user_idxs, candidate_items, labels_int,
            )
        return self._collate_eval(
            prompts, user_idxs, candidate_items, labels_int,
        )

    # ── Training ─────────────────────────────────────────────────────────
    def _collate_train(
        self,
        prompts: List[str],
        batch: List[Dict[str, Any]],
        user_idxs: torch.Tensor,
        candidate_items: torch.Tensor,
        labels_int: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        targets: List[str] = [s["label_text"] for s in batch]

        encoded = build_training_batch(
            self.tokenizer, prompts, targets, self.max_length,
        )

        encoded["user_idx"] = user_idxs
        encoded["candidate_item"] = candidate_items
        encoded["labels_int"] = labels_int          # 0/1 for convenience

        # Pass through pair_idx if present (from PrebuiltPairDataset)
        if "pair_idx" in batch[0]:
            encoded["pair_idx"] = torch.tensor(
                [s["pair_idx"] for s in batch], dtype=torch.long,
            )

        return encoded

    # ── Inference / ranking ──────────────────────────────────────────────
    def _collate_eval(
        self,
        prompts: List[str],
        user_idxs: torch.Tensor,
        candidate_items: torch.Tensor,
        labels_int: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        encoded = tokenize_prompts(
            self.tokenizer, prompts, self.max_length,
        )

        encoded["user_idx"] = user_idxs
        encoded["candidate_item"] = candidate_items
        encoded["labels_int"] = labels_int          # ground-truth for metrics
        return encoded
