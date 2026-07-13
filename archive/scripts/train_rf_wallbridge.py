"""
Train a paper-faithful Random Forest baseline using Wallbridge/FI-2010 inputs.

What this script reproduces from the paper setup:
- Raw LOB features only: 10 levels * (ask_price, ask_size, bid_price, bid_size) = 40
- Event-time windows: 100 most recent events per sample
- Fixed-threshold labels: alpha=0.002 over horizons [10, 20, 50, 100]
- Temporal split (no shuffling)

Usage:
    python scripts/train_rf_wallbridge.py
    python scripts/train_rf_wallbridge.py --ticker AAPL
    python scripts/train_rf_wallbridge.py --tickers AAPL MSFT
    python scripts/train_rf_wallbridge.py --n-estimators 300 --max-depth 20
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.labels import (  # noqa: E402
    DEFAULT_HORIZONS,
    make_fixed_threshold_classification_labels,
)


RAW_LOB_10_COLS = [
    f"{side}_{field}_{lvl}"
    for lvl in range(1, 11)
    for side, field in (("ask", "price"), ("ask", "size"), ("bid", "price"), ("bid", "size"))
]


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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert one day of parquet to windowed RF samples.

    X_day: (n_samples, seq_len * 40)
    y_day: (n_samples, len(horizons))
    """
    df = pd.read_parquet(parquet_path, columns=RAW_LOB_10_COLS)

    # Fixed-threshold FI-style labels.
    cls_df = make_fixed_threshold_classification_labels(
        df,
        horizons=horizons,
        alpha=alpha,
        use_smoothing=True,
    )

    raw = df.values.astype(np.float32)
    y_all = cls_df.values.astype(np.float32)

    max_h = max(horizons)
    valid_end = len(df) - max_h
    if valid_end <= seq_len:
        return np.empty((0, seq_len * raw.shape[1]), dtype=np.float32), np.empty((0, len(horizons)), dtype=np.int32)

    X_rows = []
    y_rows = []

    # Target time index t is the end of the lookback window.
    for t in range(seq_len - 1, valid_end):
        window = raw[t - seq_len + 1 : t + 1]
        y_t = y_all[t]
        if np.isnan(y_t).any() or np.isnan(window).any():
            continue
        X_rows.append(window.reshape(-1))
        y_rows.append(y_t.astype(np.int32))

    if not X_rows:
        return np.empty((0, seq_len * raw.shape[1]), dtype=np.float32), np.empty((0, len(horizons)), dtype=np.int32)

    return np.vstack(X_rows).astype(np.float32), np.vstack(y_rows).astype(np.int32)


def load_dataset(
    data_dir: str,
    tickers: list[str],
    horizons: list[int],
    seq_len: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    all_x = []
    all_y = []

    for ticker in tickers:
        files = sorted(glob.glob(os.path.join(data_dir, ticker, "*.parquet")))
        if not files:
            continue
        print(f"[{ticker}] loading {len(files)} parquet files...")
        for pf in files:
            x_day, y_day = _build_day_samples(pf, horizons, seq_len, alpha)
            if len(x_day) == 0:
                continue
            all_x.append(x_day)
            all_y.append(y_day)

    if not all_x:
        raise FileNotFoundError(
            "No usable samples found. Ensure data/processed/<TICKER>/*.parquet exists "
            "with 10-level columns ask/bid price/size."
        )

    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Wallbridge-style RF on raw 10-level LOB windows")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "processed"))
    parser.add_argument("--weights-dir", type=str, default=os.path.join(PROJECT_ROOT, "model_weights"))
    parser.add_argument("--results-dir", type=str, default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker")
    parser.add_argument("--tickers", nargs="+", default=None, help="Multiple tickers")
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.002)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    if args.tickers:
        tickers = args.tickers
    elif args.ticker:
        tickers = [args.ticker]
    else:
        tickers = _discover_tickers(args.data_dir)

    if not tickers:
        raise FileNotFoundError(f"No tickers with parquet files found in: {args.data_dir}")

    horizons = DEFAULT_HORIZONS
    print("=" * 80)
    print("Random Forest (Wallbridge-style inputs)")
    print("=" * 80)
    print(f"Tickers: {tickers}")
    print(f"Horizons: {horizons}")
    print(f"Seq len: {args.seq_len}")
    print(f"Fixed alpha: {args.alpha}")
    print("Features per event: 40 raw LOB")

    t0 = time.time()
    x, y = load_dataset(args.data_dir, tickers, horizons, args.seq_len, args.alpha)
    print(f"Loaded samples: {x.shape[0]:,}  |  Dim: {x.shape[1]:,}")

    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = temporal_split(x, y)

    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_va_s = scaler.transform(x_va)
    x_te_s = scaler.transform(x_te)

    del x, y, x_tr, x_va

    metrics = {}
    model_paths = {}

    for i, h in enumerate(horizons):
        print(f"\n[h={h}] training classifier...")
        y_train_h = y_tr[:, i].astype(np.int32)
        y_test_h = y_te[:, i].astype(np.int32)

        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )
        clf.fit(x_tr_s, y_train_h)

        y_pred_h = clf.predict(x_te_s)
        m = evaluate_horizon(y_test_h, y_pred_h, h)
        metrics.update(m)

        out_path = os.path.join(args.weights_dir, f"rf_wallbridge_classifier_h{h}.joblib")
        joblib.dump(clf, out_path)
        model_paths[f"h{h}"] = out_path

        print(
            f"  acc={m[f'h{h}_accuracy']:.4f} "
            f"f1_macro={m[f'h{h}_f1_macro']:.4f} "
            f"f1_weighted={m[f'h{h}_f1_weighted']:.4f}"
        )

    scaler_path = os.path.join(args.weights_dir, "rf_wallbridge_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    total_s = time.time() - t0
    run_meta = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "tickers": tickers,
        "horizons": horizons,
        "seq_len": args.seq_len,
        "alpha": args.alpha,
        "feature_spec": "raw_lob_10_levels_only",
        "features_per_event": 40,
        "flattened_dim": int(x_tr_s.shape[1]),
        "n_train": int(x_tr_s.shape[0]),
        "n_val": int(x_va_s.shape[0]),
        "n_test": int(x_te_s.shape[0]),
        "rf_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "class_weight": "balanced",
            "n_jobs": args.n_jobs,
            "random_state": args.random_state,
        },
        "test_metrics": metrics,
        "model_paths": model_paths,
        "scaler_path": scaler_path,
        "runtime_seconds": round(float(total_s), 2),
    }

    results_path = os.path.join(args.results_dir, "rf_wallbridge_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print("\n" + "=" * 80)
    print("Done")
    print(f"Results: {results_path}")
    print(f"Scaler : {scaler_path}")
    print("Models :")
    for h in horizons:
        print(f"  - {model_paths[f'h{h}']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
