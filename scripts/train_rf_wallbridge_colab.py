"""
Train a paper-faithful Random Forest baseline using Wallbridge/FI-2010 inputs.

Final Colab cell:
- No argparse
- Uses DATA_DIR / WEIGHTS_DIR / RESULTS_DIR already defined in previous cells
- RAM-aware partial loading
- No StandardScaler (RF does not need scaling)
- Detailed RAM logging to diagnose OOM causes
"""

import gc
import glob
import json
import logging
import os
import time
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


CONFIG = {
    "data_dir": str(DATA_DIR),
    "weights_dir": str(WEIGHTS_DIR),
    "results_dir": str(RESULTS_DIR),
    "ticker": None,
    "tickers": None,
    "seq_len": 100,
    "alpha": 0.00005,
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_leaf": 20,
    "n_jobs": -1,
    "random_state": 42,
    "max_total_samples": None,            # None => auto from available RAM
    "ram_fraction_for_dataset": 0.2,
    "min_free_ram_gb": 4.0,
    "base_subsample": 4,
    "high_pressure_subsample": 8,
    "critical_pressure_subsample": 16,

    # Logging / diagnostics
    "log_level": "INFO",                  # DEBUG for very verbose
    "warn_if_avail_below_gb": 2.0,
    "warn_if_proc_tree_above_gb": 12.0,
    "window_log_every": 50000,
    "fail_if_single_class_split": True,
    "max_files_per_ticker": 5,            # process only first N parquet files per ticker
}


RAW_LOB_10_COLS = [
    f"{side}_{field}_{lvl}"
    for lvl in range(1, 11)
    for side, field in (("ask", "price"), ("ask", "size"), ("bid", "price"), ("bid", "size"))
]


def _setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("rf_ram_diag")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(h)
    logger.propagate = False
    return logger


LOGGER = _setup_logger(CONFIG.get("log_level", "INFO"))


def _mem_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def _avail_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def _proc_tree_rss_gb() -> tuple[float, float, float]:
    """
    Returns:
      (parent_rss_gb, children_rss_gb, total_tree_rss_gb)
    """
    p = psutil.Process(os.getpid())
    parent = p.memory_info().rss
    children = 0
    for c in p.children(recursive=True):
        try:
            children += c.memory_info().rss
        except Exception:
            pass
    parent_gb = parent / 1e9
    child_gb = children / 1e9
    return parent_gb, child_gb, (parent + children) / 1e9


def _fmt_gb(nbytes: int) -> str:
    return f"{nbytes / 1e9:.3f} GB"


def _array_nbytes(arr: np.ndarray | None) -> int:
    if arr is None:
        return 0
    return int(getattr(arr, "nbytes", 0))


def _log_mem(tag: str, arrays: dict | None = None) -> None:
    parent_gb, child_gb, total_tree_gb = _proc_tree_rss_gb()
    avail = _avail_gb()
    vm = psutil.virtual_memory()
    msg = (
        f"[{tag}] parent_rss={parent_gb:.2f} GB | child_rss={child_gb:.2f} GB "
        f"| proc_tree={total_tree_gb:.2f} GB | avail={avail:.2f} GB | sys_used={vm.percent:.1f}%"
    )

    if arrays:
        parts = []
        for name, arr in arrays.items():
            if arr is None:
                parts.append(f"{name}=None")
            else:
                parts.append(f"{name}.shape={arr.shape}, {name}.bytes={_fmt_gb(_array_nbytes(arr))}")
        msg += " | " + " ; ".join(parts)

    LOGGER.info(msg)

    if avail < CONFIG["warn_if_avail_below_gb"]:
        LOGGER.warning(
            f"[{tag}] Low available RAM: {avail:.2f} GB < {CONFIG['warn_if_avail_below_gb']:.2f} GB"
        )
    if total_tree_gb > CONFIG["warn_if_proc_tree_above_gb"]:
        LOGGER.warning(
            f"[{tag}] High process tree RAM: {total_tree_gb:.2f} GB > {CONFIG['warn_if_proc_tree_above_gb']:.2f} GB"
        )


@contextmanager
def _stage(name: str, arrays_before: dict | None = None):
    t0 = time.time()
    _log_mem(f"{name} START", arrays_before)
    try:
        yield
    finally:
        _log_mem(f"{name} END")
        LOGGER.info(f"[{name}] elapsed={time.time() - t0:.2f}s")


def _discover_tickers(data_dir: str) -> list[str]:
    tickers = []
    for d in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, d)
        if os.path.isdir(p) and glob.glob(os.path.join(p, "*.parquet")):
            tickers.append(d)
    return tickers


def _build_day_samples(
    parquet_path: str,
    horizons: list[int],
    seq_len: int,
    alpha: float,
    step: int = 1,
    max_rows: int | None = None,
    log_every: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert one day of parquet to windowed RF samples.

    X_day: (n_samples, seq_len * 40)
    y_day: (n_samples, len(horizons))
    """
    with _stage(f"build_day {os.path.basename(parquet_path)}"):
        df = pd.read_parquet(parquet_path, columns=RAW_LOB_10_COLS)
        _log_mem("after read_parquet", {"df_values": df.values})

        cls_df = make_fixed_threshold_classification_labels(
            df,
            horizons=horizons,
            alpha=alpha,
            use_smoothing=True,
        )
        _log_mem("after label generation", {"cls_values": cls_df.values})

        raw = df.values.astype(np.float32)
        y_all = cls_df.values.astype(np.float32)

        max_h = max(horizons)
        valid_end = len(df) - max_h
        if valid_end <= seq_len:
            return np.empty((0, seq_len * raw.shape[1]), dtype=np.float32), np.empty((0, len(horizons)), dtype=np.int32)

        # Critical fix 1/3:
        # Apply subsampling while constructing windows, not after allocation.
        # Critical fix 2/3:
        # Stop as soon as day-level budget is reached.
        start_t = seq_len - 1
        if step < 1:
            step = 1

        max_possible = (valid_end - start_t + step - 1) // step
        if max_possible <= 0:
            return np.empty((0, seq_len * raw.shape[1]), dtype=np.float32), np.empty((0, len(horizons)), dtype=np.int32)

        target_rows = max_possible if max_rows is None else min(max_possible, max_rows)
        if target_rows <= 0:
            return np.empty((0, seq_len * raw.shape[1]), dtype=np.float32), np.empty((0, len(horizons)), dtype=np.int32)

        x_day = np.empty((target_rows, seq_len * raw.shape[1]), dtype=np.float32)
        y_day = np.empty((target_rows, len(horizons)), dtype=np.int32)

        written = 0
        for t in range(start_t, valid_end, step):
            if written >= target_rows:
                break
            window = raw[t - seq_len + 1 : t + 1]
            y_t = y_all[t]
            if np.isnan(y_t).any() or np.isnan(window).any():
                continue
            x_day[written] = window.reshape(-1)
            y_day[written] = y_t.astype(np.int32)
            written += 1

            if log_every > 0 and written % log_every == 0:
                _log_mem(
                    f"build progress {os.path.basename(parquet_path)}",
                    {
                        "x_day_partial": x_day[:written],
                        "y_day_partial": y_day[:written],
                    },
                )

        if written == 0:
            return np.empty((0, seq_len * raw.shape[1]), dtype=np.float32), np.empty((0, len(horizons)), dtype=np.int32)

        if written < target_rows:
            x_day = x_day[:written]
            y_day = y_day[:written]

        _log_mem("after vstack day", {"x_day": x_day, "y_day": y_day})

        del df, cls_df, raw, y_all
        gc.collect()
        _log_mem("after gc day")

        return x_day, y_day


def _estimate_auto_max_samples(seq_len: int, n_horizons: int, ram_fraction: float) -> int:
    bytes_per_sample = (seq_len * 40 * 4) + (n_horizons * 4)
    avail_bytes = psutil.virtual_memory().available
    budget_bytes = int(avail_bytes * ram_fraction)
    est = max(50_000, budget_bytes // max(1, bytes_per_sample))
    LOGGER.info(
        "auto sample cap estimate | bytes_per_sample=%d | avail=%.2f GB | budget=%.2f GB | est=%d",
        bytes_per_sample,
        avail_bytes / 1e9,
        budget_bytes / 1e9,
        est,
    )
    return int(est)


def _choose_subsample(cfg: dict) -> int:
    free = _avail_gb()
    if free < max(1.0, cfg["min_free_ram_gb"] * 0.5):
        return max(cfg["critical_pressure_subsample"], cfg["base_subsample"])
    if free < cfg["min_free_ram_gb"]:
        return max(cfg["high_pressure_subsample"], cfg["base_subsample"])
    return cfg["base_subsample"]


def load_dataset_partial(
    data_dir: str,
    tickers: list[str],
    horizons: list[int],
    seq_len: int,
    alpha: float,
    max_total_samples: int,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    feat_dim = seq_len * len(RAW_LOB_10_COLS)
    n_h = len(horizons)

    # Discover and cap files first so we can allocate a fair sample budget per ticker.
    ticker_files: dict[str, list[str]] = {}
    max_files = int(cfg.get("max_files_per_ticker", 0) or 0)
    for ticker in tickers:
        files = sorted(glob.glob(os.path.join(data_dir, ticker, "*.parquet")))
        if max_files > 0:
            files = files[:max_files]
        if files:
            ticker_files[ticker] = files

    if not ticker_files:
        raise FileNotFoundError(
            "No parquet files found for requested tickers. "
            "Ensure data/processed/<TICKER>/*.parquet exists."
        )

    active_tickers = list(ticker_files.keys())
    n_active = len(active_tickers)
    base_budget = max_total_samples // n_active
    budget_remainder = max_total_samples % n_active
    ticker_budget = {
        ticker: base_budget + (1 if i < budget_remainder else 0)
        for i, ticker in enumerate(active_tickers)
    }

    LOGGER.info(
        "Budget plan | active_tickers=%d | max_total_samples=%d | base_budget=%d | remainder=%d",
        n_active,
        max_total_samples,
        base_budget,
        budget_remainder,
    )

    # Critical fix 3/3: avoid list-of-arrays + concatenate double memory.
    x_out = np.empty((max_total_samples, feat_dim), dtype=np.float32)
    y_out = np.empty((max_total_samples, n_h), dtype=np.int32)

    total = 0
    files_seen = 0

    _log_mem("load_dataset_partial begin", {"x_out": x_out, "y_out": y_out})

    for ticker in active_tickers:
        files = ticker_files[ticker]
        ticker_target = ticker_budget[ticker]
        ticker_written = 0

        LOGGER.info(
            "[%s] files=%d | ticker_budget=%d | rss=%.2f GB | avail=%.2f GB",
            ticker,
            len(files),
            ticker_target,
            _mem_gb(),
            _avail_gb(),
        )

        for file_idx, pf in enumerate(files):
            if total >= max_total_samples:
                break

            ticker_remaining = ticker_target - ticker_written
            if ticker_remaining <= 0:
                LOGGER.info("[%s] reached ticker budget (%d).", ticker, ticker_target)
                break

            files_seen += 1
            sub = _choose_subsample(cfg)
            files_left = len(files) - file_idx
            file_budget = max(1, (ticker_remaining + files_left - 1) // files_left)

            LOGGER.info(
                "processing file=%s | subsample=%d | ticker_remaining=%d | file_budget=%d | files_left=%d",
                os.path.basename(pf),
                sub,
                ticker_remaining,
                file_budget,
                files_left,
            )

            global_remaining = max_total_samples - total
            if global_remaining <= 0:
                break

            max_rows_for_file = min(global_remaining, ticker_remaining, file_budget)
            if max_rows_for_file <= 0:
                continue

            x_day, y_day = _build_day_samples(
                pf,
                horizons,
                seq_len,
                alpha,
                step=sub,
                max_rows=max_rows_for_file,
                log_every=cfg.get("window_log_every", 50000),
            )

            if len(x_day) == 0:
                del x_day, y_day
                gc.collect()
                _log_mem("empty day skipped")
                continue

            n = len(x_day)
            x_out[total : total + n] = x_day
            y_out[total : total + n] = y_day
            total += n
            ticker_written += n

            _log_mem(
                f"after append {os.path.basename(pf)}",
                {
                    "last_x_day": x_day,
                    "last_y_day": y_day,
                    "x_out_used": x_out[:total],
                    "y_out_used": y_out[:total],
                },
            )
            LOGGER.info(
                "total rows now=%d / %d | ticker rows=%d / %d",
                total,
                max_total_samples,
                ticker_written,
                ticker_target,
            )

            gc.collect()
            _log_mem("after gc file")

        if total >= max_total_samples:
            LOGGER.info("Reached sample cap. stopping early.")
            break

    if total == 0:
        raise FileNotFoundError(
            "No usable samples found. Ensure data/processed/<TICKER>/*.parquet exists "
            "with 10-level columns ask/bid price/size."
        )

    x = x_out[:total]
    y = y_out[:total]
    LOGGER.info("Filled preallocated buffers from %d files. final_rows=%d", files_seen, total)
    _log_mem("after buffer fill", {"x": x, "y": y})

    del x_out, y_out
    gc.collect()
    _log_mem("after dropping prealloc buffers")

    return x, y


def temporal_split(
    x: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    n = len(x)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return (x[:t1], y[:t1]), (x[t1:t2], y[t1:t2]), (x[t2:], y[t2:])


def evaluate_horizon(y_true: np.ndarray, y_pred: np.ndarray, h: int) -> dict[str, float]:
    return {
        f"h{h}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"h{h}_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"h{h}_f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"h{h}_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        f"h{h}_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _class_count_str(y: np.ndarray, n_classes: int = 3) -> str:
    counts = np.bincount(y.astype(np.int32, copy=False), minlength=n_classes)
    return " ".join([f"c{i}={int(c)}" for i, c in enumerate(counts)])


def _unique_class_count(y: np.ndarray) -> int:
    return int(np.unique(y.astype(np.int32, copy=False)).size)


def resolve_tickers(cfg: dict) -> list[str]:
    if cfg["tickers"]:
        return cfg["tickers"]
    if cfg["ticker"]:
        return [cfg["ticker"]]
    return _discover_tickers(cfg["data_dir"])


def main(config: dict | None = None) -> dict:
    cfg = CONFIG if config is None else config

    os.makedirs(cfg["weights_dir"], exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)

    tickers = resolve_tickers(cfg)
    if not tickers:
        raise FileNotFoundError(f"No tickers with parquet files found in: {cfg['data_dir']}")

    horizons = DEFAULT_HORIZONS

    auto_cap = _estimate_auto_max_samples(
        seq_len=cfg["seq_len"],
        n_horizons=len(horizons),
        ram_fraction=cfg["ram_fraction_for_dataset"],
    )
    sample_cap = auto_cap if cfg["max_total_samples"] is None else int(cfg["max_total_samples"])

    LOGGER.info("=" * 80)
    LOGGER.info("Random Forest (Wallbridge-style inputs, RAM-aware partial loading, RAM diagnostics)")
    LOGGER.info("=" * 80)
    LOGGER.info("Tickers=%s", tickers)
    LOGGER.info("Horizons=%s", horizons)
    LOGGER.info("Seq len=%d", cfg["seq_len"])
    LOGGER.info("Alpha=%.6f", cfg["alpha"])
    LOGGER.info("Max files per ticker=%s", cfg.get("max_files_per_ticker"))
    LOGGER.info("Sample cap=%d (auto=%d)", sample_cap, auto_cap)
    _log_mem("main start")

    t0 = time.time()
    x, y = load_dataset_partial(
        cfg["data_dir"],
        tickers,
        horizons,
        cfg["seq_len"],
        cfg["alpha"],
        sample_cap,
        cfg,
    )
    _log_mem("after dataset load", {"x": x, "y": y})

    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = temporal_split(x, y)
    _log_mem("after split", {"x_tr": x_tr, "x_va": x_va, "x_te": x_te})

    x_tr = x_tr.astype(np.float32, copy=False)
    x_te = x_te.astype(np.float32, copy=False)
    n_val = int(x_va.shape[0])

    del x, y, x_va
    gc.collect()
    _log_mem("after split cleanup", {"x_tr": x_tr, "y_tr": y_tr, "x_te": x_te, "y_te": y_te, "y_va": y_va})

    metrics = {}
    model_paths = {}

    for i, h in enumerate(horizons):
        LOGGER.info("-" * 80)
        LOGGER.info("[h=%d] training classifier", h)

        y_train_h = y_tr[:, i].astype(np.int32, copy=False)
        y_val_h = y_va[:, i].astype(np.int32, copy=False)
        y_test_h = y_te[:, i].astype(np.int32, copy=False)

        LOGGER.info("[h=%d] train label counts: %s", h, _class_count_str(y_train_h))
        LOGGER.info("[h=%d] val   label counts: %s", h, _class_count_str(y_val_h))
        LOGGER.info("[h=%d] test  label counts: %s", h, _class_count_str(y_test_h))

        if cfg.get("fail_if_single_class_split", True):
            train_k = _unique_class_count(y_train_h)
            val_k = _unique_class_count(y_val_h)
            test_k = _unique_class_count(y_test_h)
            if train_k < 2 or val_k < 2 or test_k < 2:
                raise ValueError(
                    f"Degenerate labels at h={h}: unique classes train/val/test="
                    f"{train_k}/{val_k}/{test_k}. "
                    "Aborting because at least one split has a single class. "
                    "Adjust alpha/smoothing/subsampling/sample cap."
                )

        clf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            class_weight="balanced",
            n_jobs=cfg["n_jobs"],
            random_state=cfg["random_state"],
        )

        _log_mem(f"h={h} before fit", {"x_tr": x_tr, "y_train_h": y_train_h})
        fit_t0 = time.time()
        clf.fit(x_tr, y_train_h)
        LOGGER.info("[h=%d] fit time %.2fs", h, time.time() - fit_t0)
        _log_mem(f"h={h} after fit")

        pred_t0 = time.time()
        y_pred_h = clf.predict(x_te)
        LOGGER.info("[h=%d] predict time %.2fs", h, time.time() - pred_t0)
        _log_mem(f"h={h} after predict", {"x_te": x_te, "y_pred_h": y_pred_h})
        LOGGER.info("[h=%d] pred  label counts: %s", h, _class_count_str(y_pred_h))

        pred_k = _unique_class_count(y_pred_h)
        if pred_k < 2:
            LOGGER.warning(
                "[h=%d] Model predicts a single class on test set (unique=%d).",
                h,
                pred_k,
            )

        m = evaluate_horizon(y_test_h, y_pred_h, h)
        metrics.update(m)

        out_path = os.path.join(cfg["weights_dir"], f"rf_wallbridge_classifier_h{h}.joblib")
        save_t0 = time.time()
        joblib.dump(clf, out_path)
        LOGGER.info("[h=%d] model saved in %.2fs at %s", h, time.time() - save_t0, out_path)
        model_paths[f"h{h}"] = out_path

        LOGGER.info(
            "[h=%d] acc=%.4f f1_macro=%.4f f1_weighted=%.4f",
            h, m[f"h{h}_accuracy"], m[f"h{h}_f1_macro"], m[f"h{h}_f1_weighted"]
        )

        del clf, y_pred_h, y_train_h, y_val_h, y_test_h
        gc.collect()
        _log_mem(f"h={h} after cleanup")

    del y_va
    gc.collect()

    total_s = time.time() - t0
    run_meta = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "tickers": tickers,
        "horizons": horizons,
        "seq_len": cfg["seq_len"],
        "alpha": cfg["alpha"],
        "feature_spec": "raw_lob_10_levels_only",
        "features_per_event": 40,
        "flattened_dim": int(x_tr.shape[1]),
        "n_train": int(x_tr.shape[0]),
        "n_val": n_val,
        "n_test": int(x_te.shape[0]),
        "max_total_samples": int(sample_cap),
        "max_files_per_ticker": int(cfg.get("max_files_per_ticker", 0) or 0),
        "ram_fraction_for_dataset": float(cfg["ram_fraction_for_dataset"]),
        "rf_params": {
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "min_samples_leaf": cfg["min_samples_leaf"],
            "class_weight": "balanced",
            "n_jobs": cfg["n_jobs"],
            "random_state": cfg["random_state"],
        },
        "test_metrics": metrics,
        "model_paths": model_paths,
        "runtime_seconds": round(float(total_s), 2),
    }

    results_path = os.path.join(cfg["results_dir"], "rf_wallbridge_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    LOGGER.info("=" * 80)
    LOGGER.info("Done")
    LOGGER.info("Results: %s", results_path)
    LOGGER.info("Models:")
    for h in horizons:
        LOGGER.info("  - %s", model_paths[f"h{h}"])
    _log_mem("final")
    LOGGER.info("=" * 80)

    return run_meta


run_meta = main()
