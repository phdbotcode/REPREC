"""PyTorch Dataset classes for SASRec / BERT4Rec training and LLM pair generation.

Four datasets:
    1. ``SASRecDataset``       – returns (sequence, positive, negative) for BPR
       training of the sequential recommender.
    2. ``BERT4RecDataset``     – returns right-padded sequences for masked item
       modelling; masking itself is applied inside the trainer each epoch.
    3. ``LLMPairDataset``      – builds (prompt_text, label) pairs on the fly
       for binary YES/NO scoring with a language model.
    4. ``PrebuiltPairDataset`` – loads pre-built pairs from JSONL (produced by
       ``03_build_llm_pairs.py``) and returns ``pair_idx`` for embedding lookup.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from llm4rec.data.prompts import format_prompt


# ─────────────────────────────────────────────────────────────────────────────
# SASRec Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SASRecDataset(Dataset):
    """Dataset for SASRec training with BPR-style negative sampling.

    Each sample corresponds to **one user** and returns:
        - ``seq``  : padded item-id sequence  (length = max_len)
        - ``pos``  : positive target item ids  (length = max_len)
        - ``neg``  : sampled negative item ids  (length = max_len)

    During training the model predicts the *next* item at each time-step, so
    ``pos[t]`` is the ground-truth item following ``seq[t]`` and ``neg[t]`` is
    a random item the user has not interacted with.
    """

    def __init__(
        self,
        user_sequences: Dict[int, List[int]],
        num_items: int,
        max_len: int = 50,
        pad_value: int = 0,
    ) -> None:
        self.num_items = num_items
        self.max_len = max_len
        self.pad_value = pad_value

        # Store as list of (uid, seq) for indexing
        self.users: List[int] = []
        self.sequences: List[List[int]] = []
        for uid, seq in user_sequences.items():
            if len(seq) >= 2:  # need at least one input + one target
                self.users.append(uid)
                self.sequences.append(seq)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[index]
        user_set = set(seq)

        # Input sequence = all items except the last
        input_ids = seq[:-1]
        # Target (positive) = all items except the first
        target_ids = seq[1:]

        # Truncate / pad to max_len
        n = len(input_ids)
        if n > self.max_len:
            input_ids = input_ids[-self.max_len :]
            target_ids = target_ids[-self.max_len :]
            n = self.max_len

        pad_len = self.max_len - n

        # Right padding: real tokens first, pad tokens on the right
        input_padded = input_ids + [self.pad_value] * pad_len
        pos_padded = target_ids + [self.pad_value] * pad_len

        # Sample negatives (one per time-step), then pad on the right
        neg_sampled: list[int] = []
        for _ in range(n):
            neg = random.randint(1, self.num_items)
            while neg in user_set:
                neg = random.randint(1, self.num_items)
            neg_sampled.append(neg)
        neg_padded = neg_sampled + [self.pad_value] * pad_len

        return {
            "seq": torch.tensor(input_padded, dtype=torch.long),
            "pos": torch.tensor(pos_padded, dtype=torch.long),
            "neg": torch.tensor(neg_padded, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# LLM Pair Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LLMPairDataset(Dataset):
    """Dataset that yields (prompt_text, label) pairs for LLM binary scoring.

    For **training** we create both positive and negative pairs:
        - positive pair: user history + true next item  → label = 1 ("Yes")
        - negative pair: user history + sampled neg item → label = 0 ("No")

    For **evaluation** we only build candidate pairs (no labels needed during
    ranking; labels are used post-hoc to compute metrics).

    Parameters
    ----------
    user_sequences : train sequences (list of item ids per user).
    targets : single target item per user (valid or test item).
    negatives : sampled negatives per user (only used for training or eval
        candidate construction).
    item_meta : optional dict ``item_idx → {title, category, …}``.
    max_history : how many recent items to include in the prompt.
    prompt_id : which prompt template to use (see ``prompts.py``).
    mode : ``"train"`` generates pos + neg pairs; ``"eval"`` generates
        1 pos + N neg candidate pairs per user.
    neg_per_pos : number of negatives per positive during training.
    """

    def __init__(
        self,
        user_sequences: Dict[int, List[int]],
        targets: Dict[int, int],
        negatives: Optional[Dict[int, List[int]]] = None,
        item_meta: Optional[Dict[int, dict]] = None,
        max_history: int = 20,
        prompt_id: str = "2-11",
        mode: str = "train",
        neg_per_pos: int = 1,
    ) -> None:
        self.item_meta = item_meta or {}
        self.max_history = max_history
        self.prompt_id = prompt_id
        self.mode = mode

        self.samples: List[Dict[str, Any]] = []

        for uid, seq in user_sequences.items():
            if uid not in targets:
                continue

            history = seq[-max_history:] if max_history > 0 else []
            target_item = targets[uid]

            if mode == "train":
                # Positive pair
                self.samples.append(
                    {
                        "user_idx": uid,
                        "history": history,
                        "candidate": target_item,
                        "label": 1,
                    }
                )
                # Negative pair(s)
                if negatives and uid in negatives:
                    negs = negatives[uid][:neg_per_pos]
                else:
                    negs = []
                for neg_item in negs:
                    self.samples.append(
                        {
                            "user_idx": uid,
                            "history": history,
                            "candidate": neg_item,
                            "label": 0,
                        }
                    )

            elif mode == "eval":
                # 1 positive + negatives → candidate list
                self.samples.append(
                    {
                        "user_idx": uid,
                        "history": history,
                        "candidate": target_item,
                        "label": 1,
                    }
                )
                if negatives and uid in negatives:
                    eval_negs = negatives[uid][:neg_per_pos] if neg_per_pos > 0 else negatives[uid]
                    for neg_item in eval_negs:
                        self.samples.append(
                            {
                                "user_idx": uid,
                                "history": history,
                                "candidate": neg_item,
                                "label": 0,
                            }
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def _item_name(self, item_idx: int) -> str:
        """Return a human-readable item name (title or fallback to id)."""
        meta = self.item_meta.get(item_idx, {})
        title = meta.get("title", "").strip()
        return title if title else f"item_{item_idx}"

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]

        history_names = [self._item_name(i) for i in sample["history"]]
        candidate_name = self._item_name(sample["candidate"])

        prompt_text = format_prompt(
            prompt_id=self.prompt_id,
            user_id=sample["user_idx"],
            history=history_names,
            candidate=candidate_name,
        )

        label_text = "Yes" if sample["label"] == 1 else "No"

        return {
            "user_idx": sample["user_idx"],
            "prompt": prompt_text,
            "label": sample["label"],
            "label_text": label_text,
            "candidate_item": sample["candidate"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pre-built Pair Dataset (loads from JSONL produced by 03_build_llm_pairs.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_pairs_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load pairs from a JSONL file produced by ``03_build_llm_pairs.py``."""
    pairs: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


class PrebuiltPairDataset(Dataset):
    """Dataset that loads pre-built pairs from JSONL and returns ``pair_idx``.

    Unlike :class:`LLMPairDataset` which rebuilds pairs on the fly, this class
    uses the pairs produced by ``03_build_llm_pairs.py``.  Each sample carries
    a ``pair_idx`` that maps directly into ``train_pair_embeddings.pt`` for
    correct SASRec embedding lookup.

    Parameters
    ----------
    pairs : list of pair dicts loaded from JSONL.
    prompt_id : which prompt template to use.
    neg_per_pos : max negatives to keep per positive (filters from JSONL).
    """

    def __init__(
        self,
        pairs: List[Dict[str, Any]],
        prompt_id: str = "2-11",
        neg_per_pos: int = 2,
        max_history: int = 0,
    ) -> None:
        self.prompt_id = prompt_id
        self.max_history = max_history
        self.samples: List[Dict[str, Any]] = []

        # Filter: keep 1 pos + neg_per_pos negatives per group.
        # Groups are consecutive: each positive is followed by its negatives.
        neg_count = 0
        for p in pairs:
            if p["label"] == 1:
                self.samples.append(p)
                neg_count = 0
            else:
                if neg_count < neg_per_pos:
                    self.samples.append(p)
                    neg_count += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        p = self.samples[index]

        if self.max_history == 0:
            history = []
        elif self.max_history > 0:
            history = p["history_text"][-self.max_history:]
        else:
            history = p["history_text"]

        prompt_text = format_prompt(
            prompt_id=self.prompt_id,
            user_id=p["user_idx"],
            history=history,
            candidate=p["candidate_text"],
        )

        label_text = "Yes" if p["label"] == 1 else "No"

        return {
            "pair_idx": p["pair_idx"],
            "user_idx": p["user_idx"],
            "prompt": prompt_text,
            "label": p["label"],
            "label_text": label_text,
            "candidate_item": p["candidate_item"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# BERT4Rec Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BERT4RecDataset(Dataset):
    """Dataset for BERT4Rec training with masked item modelling.

    Returns right-padded item sequences.  Masking (replacing tokens with
    [MASK]) is intentionally *not* done here — it is applied fresh each
    epoch inside ``BERT4RecTrainer._apply_mask`` so every epoch sees
    different mask positions.

    For users whose sequence exceeds ``max_len``, a sliding window is applied
    to generate multiple overlapping training examples, matching the data
    augmentation used in the original BERT4Rec paper.

    Each sample returns:
        - ``seq`` : right-padded item-id sequence  (length = max_len)
        - ``uid`` : user index (int)

    Parameters
    ----------
    sliding_step : step size for the sliding window over long sequences.
        Defaults to ``max_len // 2``.  Set to ``None`` to disable (take only
        the most recent ``max_len`` items, original behaviour).
    """

    def __init__(
        self,
        user_sequences: Dict[int, List[int]],
        max_len: int = 50,
        pad_value: int = 0,
        sliding_step: Optional[int] = None,
    ) -> None:
        self.max_len = max_len
        self.pad_value = pad_value

        if sliding_step is None:
            sliding_step = max(1, max_len // 2)

        self.users: List[int] = []
        self.sequences: List[List[int]] = []
        for uid, seq in user_sequences.items():
            if len(seq) < 1:
                continue
            if len(seq) <= max_len:
                self.users.append(uid)
                self.sequences.append(seq)
            else:
                # Sliding window from the end of the sequence backwards.
                # Each window is at most max_len items long.
                end = len(seq)
                while end > 0:
                    start = max(0, end - max_len)
                    window = seq[start:end]
                    if len(window) >= 2:
                        self.users.append(uid)
                        self.sequences.append(window)
                    end -= sliding_step

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[index]

        pad_len = self.max_len - len(seq)
        padded = list(seq) + [self.pad_value] * pad_len

        return {
            "seq": torch.tensor(padded, dtype=torch.long),
            "uid": torch.tensor(self.users[index], dtype=torch.long),
        }
