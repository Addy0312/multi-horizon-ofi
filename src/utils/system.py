import gc
import psutil
import torch

def _avail_ram_gb() -> float:
    return psutil.virtual_memory().available / 1e9

def _deep_cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
