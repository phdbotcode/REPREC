"""Fixed negative sampling for reproducible evaluation.

Hybrid strategy: Random + Popularity-based Negative Sampling (PNS).
A fraction of negatives are sampled uniformly at random from the full item
pool, and the rest are sampled proportionally to item interaction frequency
(popularity).  Popular non-purchased items act as informative hard negatives.
Negatives are sampled from the *entire* item pool (no exclusion of
user-interacted items).
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from typing import Any, Dict, List


def _build_item_popularity(
    splits: Dict[str, Dict[int, Any]],
    num_items: int,
) -> tuple[list[int], list[float]]:
    """Compute item popularity weights from training interactions.

    Returns (items, weights) where weights are proportional to interaction
    frequency, suitable for use with ``random.choices``.
    """
    counts: Counter = Counter()
    for uid, seq in splits["train"].items():
        counts.update(seq)

    items = list(range(1, num_items + 1))
    # Laplace smoothing (+1) so every item has non-zero probability
    weights = [float(counts.get(iid, 0) + 1) for iid in items]
    return items, weights


def sample_negatives(
    splits: Dict[str, Dict[int, Any]],
    sequences: Dict[int, List[int]],
    num_items: int,
    num_negatives: int = 500,
    popularity_ratio: float = 0.5,
    seed: int = 2024,
) -> Dict[int, List[int]]:
    """Sample fixed negatives for each test user (Random + PNS hybrid).

    Parameters
    ----------
    splits : dict with keys ``train``, ``valid``, ``test`` from
        :func:`preprocess.leave_one_out_split`.
    sequences : full user sequences (kept for API compatibility).
    num_items : total number of items (item IDs range from 1 .. num_items).
    num_negatives : how many negatives to sample per user.
    popularity_ratio : fraction of negatives drawn via popularity weighting.
        The remaining ``1 - popularity_ratio`` are uniform random.
        Default 0.5 (50/50 split).
    seed : random seed for reproducibility.

    Returns
    -------
    dict mapping ``user_idx`` → list of ``num_negatives`` negative item IDs.
    """
    rng = random.Random(seed)
    all_items = list(range(1, num_items + 1))
    items_pop, weights_pop = _build_item_popularity(splits, num_items)

    num_popular = int(num_negatives * popularity_ratio)
    num_random = num_negatives - num_popular

    negatives: Dict[int, List[int]] = {}

    for uid in splits["test"]:
        random_negs = rng.choices(all_items, k=num_random)
        popular_negs = rng.choices(items_pop, weights=weights_pop, k=num_popular)
        sampled = random_negs + popular_negs
        rng.shuffle(sampled)
        negatives[uid] = sampled

    return negatives


def save_negatives(
    negatives: Dict[int, List[int]],
    output_path: str,
) -> None:
    """Save negatives as JSONL (one line per user).

    Format per line: ``{"user_idx": <int>, "negatives": [<int>, ...]}``
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for uid in sorted(negatives):
            record = {"user_idx": uid, "negatives": negatives[uid]}
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(negatives)} negative sets → {output_path}")


def load_negatives(path: str) -> Dict[int, List[int]]:
    """Load negatives from a JSONL file produced by :func:`save_negatives`."""
    negatives: Dict[int, List[int]] = {}
    with open(path, "r") as f:
        for line in f:
            record = json.loads(line)
            negatives[int(record["user_idx"])] = record["negatives"]
    return negatives
