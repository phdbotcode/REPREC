"""Global reproducibility seed for all RNGs."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set seed for ``random``, ``numpy``, ``torch`` (CPU + CUDA).

    Also configures PyTorch for deterministic behaviour where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic convolutions (slight perf cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
