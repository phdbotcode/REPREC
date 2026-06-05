"""Preprocess Amazon reviews → remapped IDs, user sequences, and splits.

Split strategy (leave-one-out, standard for sequential recommendation):
    - Per user, sort interactions chronologically.
    - Last item           → **test**
    - Second-to-last item → **valid**
    - Everything before   → **train**

For SASRec the train set is the only set used during model training; the
valid split is used for early-stopping.  The test split is reserved for
final evaluation with the LLM ranker.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ── k-core filtering ────────────────────────────────────────────────────────

def _kcore_filter(
    df: pd.DataFrame,
    min_user: int = 5,
    min_item: int = 5,
    max_iters: int = 100,
) -> pd.DataFrame:
    """Iteratively remove users/items with fewer than *k* interactions."""
    for _ in range(max_iters):
        prev_len = len(df)
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_user].index)]

        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= min_item].index)]

        if len(df) == prev_len:
            break
    return df.reset_index(drop=True)


# ── ID remapping ─────────────────────────────────────────────────────────────

def remap_ids(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """Remap raw string user/item IDs to contiguous integers starting at 1.

    ID 0 is reserved for padding.

    Returns
    -------
    df : DataFrame with ``user_idx`` and ``item_idx`` columns added.
    user2idx : mapping raw_user_id → int.
    item2idx : mapping raw_item_id → int.
    """
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user2idx = {uid: i + 1 for i, uid in enumerate(unique_users)}
    item2idx = {iid: i + 1 for i, iid in enumerate(unique_items)}

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user2idx)
    df["item_idx"] = df["item_id"].map(item2idx)
    return df, user2idx, item2idx


# ── Sequence building ────────────────────────────────────────────────────────

def build_user_sequences(df: pd.DataFrame) -> Dict[int, List[int]]:
    """Build chronologically-sorted item sequences per user.

    Parameters
    ----------
    df : DataFrame with columns ``user_idx``, ``item_idx``, ``timestamp``.

    Returns
    -------
    dict mapping ``user_idx`` → list of ``item_idx`` in time order.
    """
    df_sorted = df.sort_values(["user_idx", "timestamp"])
    sequences: Dict[int, List[int]] = {}
    for uid, group in df_sorted.groupby("user_idx"):
        sequences[int(uid)] = group["item_idx"].tolist()
    return sequences


# ── Leave-one-out split ──────────────────────────────────────────────────────

def leave_one_out_split(
    sequences: Dict[int, List[int]],
) -> Dict[str, Dict[int, Any]]:
    """Split each user sequence into train / valid / test.

    * test  = last item
    * valid = second-to-last item
    * train = all preceding items

    Users with fewer than 3 interactions are dropped.

    Returns
    -------
    dict with keys ``train``, ``valid``, ``test``.
        - ``train[uid]`` : list[int]   (the training subsequence)
        - ``valid[uid]`` : int         (single held-out item)
        - ``test[uid]``  : int         (single held-out item)
    """
    splits: Dict[str, Dict[int, Any]] = {
        "train": {},
        "valid": {},
        "test": {},
    }

    for uid, seq in sequences.items():
        if len(seq) < 3:
            continue
        splits["test"][uid] = seq[-1]
        splits["valid"][uid] = seq[-2]
        splits["train"][uid] = seq[:-2]

    return splits


# ── Public entry point ───────────────────────────────────────────────────────

def preprocess_category(
    reviews: List[dict],
    metadata: Dict[str, dict] | None = None,
    min_user: int = 5,
    min_item: int = 5,
    max_seq_len: int = 50,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    """Full preprocessing pipeline for one Amazon category.

    Parameters
    ----------
    reviews : list of review dicts (from ``amazon_download.load_reviews``).
    metadata : optional item metadata dict (keyed by raw ASIN).
    min_user, min_item : k-core thresholds.
    max_seq_len : truncate training sequences to this length (most recent).
    output_dir : if given, persist artefacts to disk.

    Returns
    -------
    dict with keys: ``splits``, ``sequences``, ``user2idx``, ``item2idx``,
    ``num_users``, ``num_items``, ``item_meta`` (if metadata provided).
    """
    # 1. DataFrame
    df = pd.DataFrame(reviews)

    # 2. k-core filtering
    df = _kcore_filter(df, min_user=min_user, min_item=min_item)

    # 3. Remap IDs (0 = pad)
    df, user2idx, item2idx = remap_ids(df)
    num_users = len(user2idx)
    num_items = len(item2idx)

    # 4. Build sequences
    sequences = build_user_sequences(df)

    # 5. Truncate to max_seq_len (keep most recent)
    for uid in sequences:
        if len(sequences[uid]) > max_seq_len:
            sequences[uid] = sequences[uid][-max_seq_len:]

    # 6. Leave-one-out split
    splits = leave_one_out_split(sequences)

    # 7. Remap metadata if available
    item_meta = None
    if metadata is not None:
        idx2item = {v: k for k, v in item2idx.items()}
        item_meta = {}
        for idx, raw_id in idx2item.items():
            item_meta[idx] = metadata.get(raw_id, {})

    result = {
        "splits": splits,
        "sequences": sequences,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "num_users": num_users,
        "num_items": num_items,
        "item_meta": item_meta,
    }

    # 8. Persist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # Mappings
        mappings = {
            "user2idx": user2idx,
            "item2idx": item2idx,
            "num_users": num_users,
            "num_items": num_items,
        }
        with open(os.path.join(output_dir, "mappings.json"), "w") as f:
            json.dump(mappings, f)

        # Splits  (convert numpy ints to plain ints for JSON)
        def _to_python(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, dict):
                return {str(k): _to_python(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_python(v) for v in obj]
            return obj

        with open(os.path.join(output_dir, "splits.json"), "w") as f:
            json.dump(_to_python(splits), f)

        # Sequences as parquet
        rows = [
            {"user_idx": uid, "item_sequence": seq}
            for uid, seq in sequences.items()
        ]
        pd.DataFrame(rows).to_parquet(
            os.path.join(output_dir, "sequences.parquet"), index=False
        )

        # Item metadata
        if item_meta is not None:
            with open(os.path.join(output_dir, "item_meta.json"), "w") as f:
                json.dump(_to_python(item_meta), f)

        print(
            f"Saved preprocessed data to {output_dir}  "
            f"({num_users} users, {num_items} items)"
        )

    return result
