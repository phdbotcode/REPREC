"""Load a HuggingFace LLaMA model and tokenizer.

Handles dtype selection (fp16 / bf16 / fp32), device mapping, gradient
checkpointing, and provides helpers to freeze the base model before
attaching parameter-efficient adapters (LoRA, soft-prompt injector).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm4rec.llm.tokenizer_utils import setup_tokenizer
from llm4rec.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.lower().strip()
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown dtype {dtype_str!r}. Choose from {list(_DTYPE_MAP)}"
        )
    return _DTYPE_MAP[dtype_str]


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_llama(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    dtype: str = "bf16",
    device_map: str = "auto",
    gradient_checkpointing: bool = False,
    train_mode: bool = False,
    token: str | None = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a LLaMA (or compatible) causal-LM and its tokenizer.

    Parameters
    ----------
    model_name : HuggingFace model identifier or local path.
    dtype : ``"fp16"`` | ``"bf16"`` | ``"fp32"``.
    device_map : ``"auto"`` for multi-GPU, ``"cuda:0"`` for single-GPU, etc.
    gradient_checkpointing : enable to reduce VRAM at the cost of speed.
    train_mode : if ``True`` the model is returned in ``.train()`` mode with
        gradients enabled; otherwise ``.eval()`` with ``torch.no_grad`` expected.
    token : optional HuggingFace access token (for gated models).

    Returns
    -------
    (model, tokenizer) ready for downstream use.
    """
    torch_dtype = _resolve_dtype(dtype)

    logger.info(
        "Loading %s  dtype=%s  device_map=%s", model_name, dtype, device_map,
    )

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=token,
    )
    tokenizer = setup_tokenizer(tokenizer)

    # ── Model ────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        token=token,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Required for gradient checkpointing when some inputs don't need grad
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        logger.info("Gradient checkpointing enabled")

    if train_mode:
        model.train()
    else:
        model.eval()

    total, _ = count_parameters(model)
    logger.info("Model loaded — %.2fB total params", total / 1e9)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Parameter utilities
# ─────────────────────────────────────────────────────────────────────────────

def freeze_model(model: nn.Module) -> None:
    """Freeze every parameter in *model* (set ``requires_grad = False``)."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module) -> None:
    """Unfreeze every parameter in *model*."""
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_trainable_params(model: nn.Module) -> Tuple[int, int]:
    """Log and return (trainable, total) parameter counts."""
    total, trainable = count_parameters(model)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    logger.info(
        "Trainable params: %s / %s  (%.4f%%)",
        f"{trainable:,}", f"{total:,}", pct,
    )
    return trainable, total
