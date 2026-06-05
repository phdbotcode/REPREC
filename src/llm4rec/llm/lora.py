"""LoRA adapter wrapper around HuggingFace PEFT for LLaMA.

Provides a single entry point :func:`build_lora_model` that takes a base
causal-LM and a config dict, attaches LoRA adapters via PEFT, and returns
the wrapped model ready for training.  Only the adapter parameters are
trainable; the base model weights stay frozen.

Typical usage
-------------
::

    from llm4rec.llm.llama_backbone import load_llama
    from llm4rec.llm.lora import build_lora_model, save_lora, load_lora

    base, tok = load_llama("meta-llama/Llama-2-7b-hf", train_mode=True)
    model = build_lora_model(base, cfg["lora"])

    # … train …

    save_lora(model, "outputs/checkpoints/lora_beauty")
    model = load_lora(base, "outputs/checkpoints/lora_beauty")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.nn as nn
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)

from llm4rec.utils.logging import get_logger

logger = get_logger(__name__)

# Default target modules covering attention + MLP in LLaMA
_DEFAULT_TARGET_MODULES: List[str] = [
    "q_proj",
    "v_proj",
]

_TASK_TYPE_MAP = {
    "CAUSAL_LM": TaskType.CAUSAL_LM,
    "SEQ_CLS": TaskType.SEQ_CLS,
}


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────

def build_lora_model(
    base_model: nn.Module,
    lora_cfg: Dict[str, Any],
) -> PeftModel:
    """Attach LoRA adapters to *base_model* and return the PEFT wrapper.

    Parameters
    ----------
    base_model : a HuggingFace ``AutoModelForCausalLM`` (or compatible).
    lora_cfg : dict with LoRA hyper-parameters.  Recognised keys:

        ============== ============================== ===========
        Key            Description                    Default
        ============== ============================== ===========
        r              rank                           8
        lora_alpha     scaling factor                 16
        lora_dropout   dropout on adapter layers      0.05
        target_modules list of linear layer names     [q,v]_proj
        bias           "none" | "lora_only" | "all"   "none"
        task_type      "CAUSAL_LM" | "SEQ_CLS"        "CAUSAL_LM"
        ============== ============================== ===========

    Returns
    -------
    ``PeftModel`` with only the LoRA parameters marked as trainable.
    """
    target_modules = lora_cfg.get("target_modules", _DEFAULT_TARGET_MODULES)
    task_str = lora_cfg.get("task_type", "CAUSAL_LM")
    task_type = _TASK_TYPE_MAP.get(task_str)
    if task_type is None:
        raise ValueError(
            f"Unknown task_type {task_str!r}. Choose from {list(_TASK_TYPE_MAP)}"
        )

    config = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias=lora_cfg.get("bias", "none"),
        task_type=task_type,
    )

    model = get_peft_model(base_model, config)

    trainable, total = print_trainable_params(model)
    logger.info(
        "LoRA attached  r=%d  alpha=%d  targets=%s",
        config.r, config.lora_alpha, target_modules,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Parameter counting
# ─────────────────────────────────────────────────────────────────────────────

def print_trainable_params(model: nn.Module) -> Tuple[int, int]:
    """Log and return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    logger.info(
        "Trainable params: %s / %s  (%.4f%%)",
        f"{trainable:,}", f"{total:,}", pct,
    )
    return trainable, total


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def save_lora(
    model: PeftModel,
    out_dir: Union[str, Path],
) -> None:
    """Save only the LoRA adapter weights to *out_dir*.

    Creates the directory if it does not exist.  The base model is **not**
    saved — only the small adapter delta.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    logger.info("LoRA adapter saved → %s", out_dir)


def load_lora(
    base_model: nn.Module,
    adapter_dir: Union[str, Path],
    is_trainable: bool = False,
) -> PeftModel:
    """Load a previously saved LoRA adapter onto *base_model*.

    Parameters
    ----------
    base_model : the same (or equivalent) base model used during training.
    adapter_dir : directory produced by :func:`save_lora`.
    is_trainable : if ``True`` the adapter weights remain trainable
        (for continued fine-tuning); otherwise they are frozen for inference.

    Returns
    -------
    ``PeftModel`` with the loaded adapter.
    """
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        is_trainable=is_trainable,
    )
    logger.info("LoRA adapter loaded ← %s  (trainable=%s)", adapter_dir, is_trainable)
    return model
