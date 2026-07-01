import random
import os

import numpy as np
import torch


def set_reproducibility(seed, deterministic=False):
    seed = int(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    return {
        'seed': seed,
        'deterministic': bool(deterministic),
    }
