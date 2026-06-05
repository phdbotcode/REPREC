"""Ranking metrics optimised for the single-positive evaluation protocol.

In our setup each test user has exactly **1 relevant item** ranked against
``N`` negatives (e.g. 1001 candidates total).  The functions below exploit
this structure for clarity and speed — no need for a full multi-label NDCG.

Metrics
-------
* **HR@K**   (Hit Rate): 1 if the positive item appears in the top-K, else 0.
* **NDCG@K** (Normalised DCG): ``1 / log2(rank + 1)`` if the positive item
  is in the top-K, else 0  (since IDCG = 1 for a single relevant item).
* **MRR**    (Mean Reciprocal Rank): ``1 / rank`` of the positive item.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Union

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Per-user metrics (operate on a single ranked list)
# ─────────────────────────────────────────────────────────────────────────────

def _positive_rank(scores: np.ndarray, positive_idx: int = 0) -> int:
    """Return the 1-based rank of the positive item.

    Parameters
    ----------
    scores : 1-D array of scores where ``scores[positive_idx]`` is the
        positive item's score.
    positive_idx : index of the positive item in *scores* (default 0,
        matching the convention that the positive is placed first).

    Returns
    -------
    int : 1-based rank (1 = best).
    """
    pos_score = scores[positive_idx]
    # Count how many items score strictly higher
    rank = int((scores > pos_score).sum()) + 1
    return rank


def hit_rate_at_k(
    scores: np.ndarray,
    k: int = 10,
    positive_idx: int = 0,
) -> float:
    """HR@K for a single user (1 positive item).

    Returns 1.0 if the positive item is in the top-K, else 0.0.
    """
    rank = _positive_rank(scores, positive_idx)
    return 1.0 if rank <= k else 0.0


def ndcg_at_k(
    scores: np.ndarray,
    k: int = 10,
    positive_idx: int = 0,
) -> float:
    """NDCG@K for a single user (1 positive item).

    With a single relevant item, IDCG = 1, so
    NDCG@K = 1 / log2(rank + 1) if rank <= K, else 0.
    """
    rank = _positive_rank(scores, positive_idx)
    if rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def mrr(
    scores: np.ndarray,
    positive_idx: int = 0,
) -> float:
    """Mean Reciprocal Rank for a single user."""
    rank = _positive_rank(scores, positive_idx)
    return 1.0 / rank


# ─────────────────────────────────────────────────────────────────────────────
# Batch / aggregate evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ranking(
    all_scores: List[np.ndarray],
    ks: Sequence[int] = (1, 5, 10),
    positive_idx: int = 0,
) -> Dict[str, float]:
    """Compute aggregated metrics over many users.

    Parameters
    ----------
    all_scores : list of 1-D score arrays, one per user.  In each array
        ``all_scores[i][positive_idx]`` is the positive item's score and
        the remaining entries are negatives.
    ks : tuple of K values for HR@K and NDCG@K.
    positive_idx : position of the positive item in each score array.

    Returns
    -------
    dict mapping metric name (e.g. ``"HR@10"``, ``"NDCG@5"``, ``"MRR"``)
    to its mean value across all users.
    """
    n_users = len(all_scores)
    if n_users == 0:
        return {}

    results: Dict[str, float] = {}

    hr_accum = {k: 0.0 for k in ks}
    ndcg_accum = {k: 0.0 for k in ks}
    mrr_accum = 0.0

    for scores in all_scores:
        scores = np.asarray(scores, dtype=np.float64)
        for k in ks:
            hr_accum[k] += hit_rate_at_k(scores, k=k, positive_idx=positive_idx)
            ndcg_accum[k] += ndcg_at_k(scores, k=k, positive_idx=positive_idx)
        mrr_accum += mrr(scores, positive_idx=positive_idx)

    for k in ks:
        results[f"HR@{k}"] = hr_accum[k] / n_users
        results[f"NDCG@{k}"] = ndcg_accum[k] / n_users
    results["MRR"] = mrr_accum / n_users

    return results
