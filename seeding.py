"""Deterministic seeding across Python, NumPy, and PyTorch.

Replaces the one use of lightning.seed_everything so the project does not depend
on all of Lightning just to seed three libraries.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
