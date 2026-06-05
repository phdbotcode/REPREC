"""I/O helpers: YAML configs, JSON, and PyTorch checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml


# ── YAML config loading ─────────────────────────────────────────────────────

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file and return a nested dict.

    Parameters
    ----------
    path : path to ``.yaml`` / ``.yml`` file.

    Returns
    -------
    dict with the parsed configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}


def merge_configs(*cfgs: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge multiple config dicts (later overrides earlier)."""
    merged: Dict[str, Any] = {}
    for cfg in cfgs:
        for key, val in cfg.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(val, dict)
            ):
                merged[key] = merge_configs(merged[key], val)
            else:
                merged[key] = val
    return merged


# ── JSON I/O ─────────────────────────────────────────────────────────────────

def save_json(data: Any, path: Union[str, Path]) -> None:
    """Serialize *data* to a JSON file, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Union[str, Path]) -> Any:
    """Load and return a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


# ── Checkpoint I/O ───────────────────────────────────────────────────────────

def save_checkpoint(
    state: Dict[str, Any],
    path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """Save a training checkpoint.

    Parameters
    ----------
    state : arbitrary dict (epoch, metrics, config, …).
    path : destination file path.
    model : if given, ``model.state_dict()`` is stored under key ``model``.
    optimizer : if given, ``optimizer.state_dict()`` is stored under key
        ``optimizer``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if model is not None:
        state["model"] = model.state_dict()
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()

    torch.save(state, path)


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Parameters
    ----------
    path : checkpoint file path.
    model : if given, loads weights from the ``model`` key.
    optimizer : if given, loads state from the ``optimizer`` key.
    device : map location for ``torch.load``.

    Returns
    -------
    The full state dict (minus ``model`` / ``optimizer`` keys which are
    loaded into the provided objects instead).
    """
    state = torch.load(path, map_location=device, weights_only=False)

    if model is not None and "model" in state:
        model.load_state_dict(state.pop("model"))
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state.pop("optimizer"))

    return state
