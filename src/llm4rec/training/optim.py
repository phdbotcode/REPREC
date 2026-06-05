"""Optimizer, scheduler, and gradient utilities for LLM training."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Union

import torch
import torch.nn as nn
from transformers import get_scheduler as _hf_get_scheduler


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(
    model_or_params: Union[nn.Module, Iterable[torch.nn.Parameter]],
    cfg: Dict[str, Any],
) -> torch.optim.AdamW:
    """Build an AdamW optimizer over **trainable** parameters only.

    Parameters
    ----------
    model_or_params : an ``nn.Module`` (trainable params are auto-filtered)
        or an explicit iterable of parameters.
    cfg : dict with keys ``learning_rate``, ``weight_decay`` (optional,
        default 0.01).
    """
    if isinstance(model_or_params, nn.Module):
        params = [p for p in model_or_params.parameters() if p.requires_grad]
    else:
        params = list(model_or_params)

    return torch.optim.AdamW(
        params,
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a warmup + decay learning-rate scheduler.

    Parameters
    ----------
    optimizer : the optimizer to schedule.
    cfg : dict with keys ``warmup_ratio`` (float, default 0.05) and
        ``scheduler`` (``"cosine"`` | ``"linear"``, default ``"cosine"``).
    num_training_steps : total optimiser steps across all epochs.

    Returns
    -------
    A HuggingFace ``LambdaLR`` scheduler.
    """
    warmup_ratio = cfg.get("warmup_ratio", 0.05)
    warmup_steps = int(warmup_ratio * num_training_steps)
    sched_type = cfg.get("scheduler", "cosine")

    return _hf_get_scheduler(
        name=sched_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradient helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_requires_grad(module: nn.Module, flag: bool) -> None:
    """Set ``requires_grad`` for every parameter in *module*."""
    for p in module.parameters():
        p.requires_grad = flag


def grad_clip(
    model_or_params: Union[nn.Module, List[torch.nn.Parameter]],
    max_norm: float,
) -> torch.Tensor:
    """Clip gradients of trainable parameters by global norm.

    Returns the total gradient norm **before** clipping.
    """
    if isinstance(model_or_params, nn.Module):
        params = [p for p in model_or_params.parameters() if p.requires_grad]
    else:
        params = list(model_or_params)
    return nn.utils.clip_grad_norm_(params, max_norm)
