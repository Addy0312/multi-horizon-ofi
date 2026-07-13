# ═══════════════════════════════════════════════════════════════════════════
# Cell 6 — Training Utilities
# ═══════════════════════════════════════════════════════════════════════════
import psutil
import os
import glob
import hashlib
import random
import numpy as np
import torch
import gc
from typing import Dict, List, Tuple


def deep_set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _avail_ram_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def _deep_cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _deep_choose_subsample(cfg: dict) -> int:
    free = _avail_ram_gb()
    if free < 2.0:
        return int(cfg.get("critical_pressure_subsample", 16))
    if free < 4.0:
        return int(cfg.get("high_pressure_subsample", 8))
    return int(cfg.get("base_subsample", 4))


def _deep_choose_max_rows(cfg: dict, is_train: bool) -> int:
    key = "max_rows_per_day_train" if is_train else "max_rows_per_day_eval"
    default = 8000 if is_train else 10000
    return int(max(1, cfg.get(key, default)))


def _deep_seed_from_path(path: str, base_seed: int) -> int:
    digest = hashlib.md5(path.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) % (2 ** 32 - 1)


def _deep_resolve_tickers(cfg: dict) -> list:
    if cfg.get("tickers"):
        return list(cfg["tickers"])
    data_dir = cfg["data_dir"]
    return sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and glob.glob(os.path.join(data_dir, d, "*.parquet"))
    ])


def _deep_collect_files_by_ticker(
    data_dir: str,
    tickers: list,
    max_files: int = 0,
) -> dict:
    files_by_ticker: Dict[str, List[str]] = {}
    for ticker in tickers:
        files = sorted(glob.glob(os.path.join(data_dir, ticker, "*.parquet")))
        if max_files > 0:
            files = files[:max_files]
        if files:
            files_by_ticker[ticker] = files
    return files_by_ticker


def _deep_split_train_eval_files(
    files_by_ticker: dict,
    train_frac: float = 0.8,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    train_files: List[Tuple[str, str]] = []
    eval_files:  List[Tuple[str, str]] = []
    for ticker, files in files_by_ticker.items():
        n = len(files)
        n_train = max(1, min(n - 1, int(np.floor(n * train_frac)))) if n > 1 else 1
        for i, p in enumerate(files):
            (train_files if i < n_train else eval_files).append((ticker, p))
    return train_files, eval_files


def _deep_update_confusion(confusion: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    mask = (y_true >= 0) & (y_true < 3) & (y_pred >= 0) & (y_pred < 3)
    yt = y_true[mask].astype(np.int64)
    yp = y_pred[mask].astype(np.int64)
    np.add.at(confusion, (yt, yp), 1)


def _deep_metrics_from_confusion(confusion: np.ndarray, h: int) -> dict:
    metrics: Dict[str, float] = {}
    n_cls = confusion.shape[0]
    total = float(confusion.sum())
    metrics[f"h{h}_accuracy"] = float(np.trace(confusion)) / max(total, 1.0)
    f1s = []
    for c in range(n_cls):
        tp = float(confusion[c, c])
        fp = float(confusion[:, c].sum()) - tp
        fn = float(confusion[c, :].sum()) - tp
        prec = tp / max(tp + fp, 1e-9)
        rec  = tp / max(tp + fn, 1e-9)
        f1   = 2.0 * prec * rec / max(prec + rec, 1e-9)
        f1s.append(f1)
        metrics[f"h{h}_f1_c{c}"] = f1
    metrics[f"h{h}_f1_macro"]    = float(np.mean(f1s))
    metrics[f"h{h}_f1_weighted"] = float(
        sum(f1s[c] * float(confusion[c, :].sum()) for c in range(n_cls)) / max(total, 1.0)
    )
    return metrics


def _mean_macro_f1(metrics: dict, horizons: list) -> float:
    vals = [float(metrics.get(f"h{h}_f1_macro", 0.0)) for h in horizons]
    return float(np.mean(vals)) if vals else 0.0


print("Utilities ready.")