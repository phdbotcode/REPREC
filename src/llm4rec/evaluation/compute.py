"""Trainable-parameter counting and training-compute (FLOP) estimation.

The FLOP proxy uses the standard approximation for transformer training:

    FLOPs ≈ C × P_trainable × T

where:
    * **C** = 6  (≈ 2 for forward + 4 for backward per parameter per token,
      widely used in Kaplan et al. / Chinchilla scaling papers).
    * **P_trainable** = number of trainable parameters.
    * **T** = total tokens processed during training
              (``steps × batch_size × avg_seq_len``).

Because we use the **same constant C for every method**, the resulting
numbers are directly comparable across SASRec, Injector, and LoRA
even though absolute FLOP counts are approximate.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

# Constant factor: 2 (fwd) + 4 (bwd) per param per token
_C_FACTOR = 6


# ─────────────────────────────────────────────────────────────────────────────
# Parameter counting
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters.

    Returns
    -------
    ``{"total": int, "trainable": int, "frozen": int}``
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def count_params_multi(**named_modules: nn.Module) -> Dict[str, Dict[str, int]]:
    """Count parameters for multiple named modules at once.

    Example::

        count_params_multi(sasrec=sasrec, injector=injector, llm=llm)

    Returns
    -------
    ``{"sasrec": {"total":…, "trainable":…}, …, "_combined": {…}}``
    """
    result: Dict[str, Dict[str, int]] = {}
    combined_total = 0
    combined_trainable = 0
    for name, module in named_modules.items():
        counts = count_params(module)
        result[name] = counts
        combined_total += counts["total"]
        combined_trainable += counts["trainable"]
    result["_combined"] = {
        "total": combined_total,
        "trainable": combined_trainable,
        "frozen": combined_total - combined_trainable,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Token counting
# ─────────────────────────────────────────────────────────────────────────────

def compute_tokens_processed(
    num_steps: int,
    batch_size: int,
    avg_seq_len: int,
) -> int:
    """Total tokens seen during training: ``steps × batch × seq_len``."""
    return num_steps * batch_size * avg_seq_len


def compute_tokens_from_config(
    cfg: Dict[str, Any],
    num_train_samples: int,
) -> int:
    """Derive total tokens from a training config + dataset size.

    ``num_steps = ceil(num_train_samples / batch_size) × num_epochs``
    """
    batch_size = cfg.get("batch_size", 16)
    num_epochs = cfg.get("num_epochs", 3)
    avg_seq_len = cfg.get("max_seq_length", cfg.get("max_seq_len", 512))

    steps_per_epoch = max(1, (num_train_samples + batch_size - 1) // batch_size)
    total_steps = steps_per_epoch * num_epochs

    return compute_tokens_processed(total_steps, batch_size, avg_seq_len)


# ─────────────────────────────────────────────────────────────────────────────
# FLOP estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_train_flops(
    trainable_params: int,
    total_tokens: int,
    c_factor: int = _C_FACTOR,
) -> int:
    """Approximate training FLOPs: ``C × P_trainable × T``.

    Parameters
    ----------
    trainable_params : number of parameters updated during training.
    total_tokens : total tokens processed (see :func:`compute_tokens_processed`).
    c_factor : constant multiplier (default 6).

    Returns
    -------
    int : estimated FLOPs.
    """
    return c_factor * trainable_params * total_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Unified record builder
# ─────────────────────────────────────────────────────────────────────────────

def build_compute_record(
    method: str,
    model: nn.Module,
    cfg: Dict[str, Any],
    num_train_samples: int,
    extra_trainable: int = 0,
    metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build a single compute-vs-performance record for tables / plots.

    Parameters
    ----------
    method : label for this method (e.g. ``"injector"``, ``"lora_r8"``).
    model : the model (for parameter counting).
    cfg : training config dict.
    num_train_samples : number of training examples.
    extra_trainable : additional trainable params not in *model*
        (e.g. injector params when only the LLM is passed).
    metrics : optional metrics dict (e.g. from :func:`evaluate_ranker`).

    Returns
    -------
    dict with keys: ``method``, ``trainable_params``, ``total_params``,
    ``tokens_processed``, ``estimated_flops``, and any metric keys.
    """
    params = count_params(model)
    trainable = params["trainable"] + extra_trainable
    tokens = compute_tokens_from_config(cfg, num_train_samples)
    flops = estimate_train_flops(trainable, tokens)

    record: Dict[str, Any] = {
        "method": method,
        "trainable_params": trainable,
        "total_params": params["total"],
        "trainable_pct": 100.0 * trainable / max(params["total"], 1),
        "tokens_processed": tokens,
        "estimated_flops": flops,
    }
    if metrics:
        record.update(metrics)
    return record
