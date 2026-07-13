import os
from pathlib import Path

def is_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

import torch
DEEP_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
