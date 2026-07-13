import random
import numpy as np
import torch
import hashlib

def deep_set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _deep_seed_from_path(path: str, base_seed: int) -> int:
    digest = hashlib.md5(path.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2 ** 32 - 1)
