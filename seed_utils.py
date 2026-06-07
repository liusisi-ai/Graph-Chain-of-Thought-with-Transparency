"""Centralised seeding so every run is reproducible.

Call ``seed_everything(seed)`` once at the very top of ``main.py`` (before any
model / dataloader / RNG-using helper is constructed).  Every subsequent
``random.*``, ``numpy.random.*``, ``torch.*`` operation will then be
deterministic and identical across runs with the same seed.

To run a seed sweep::

    python main.py --seed 42
    python main.py --seed 1
    python main.py --seed 7
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed Python, NumPy, PyTorch (CPU + CUDA) and the hash function."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"🎲 GLOBAL_SEED = {seed}  (deterministic={deterministic})")
    return seed


def get_global_seed(default: int = 42) -> int:
    """Read the active seed from the env (set by ``seed_everything``)."""
    try:
        return int(os.environ.get("PYTHONHASHSEED", default))
    except ValueError:
        return default
