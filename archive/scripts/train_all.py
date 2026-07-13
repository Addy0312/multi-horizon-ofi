"""
Train ALL models sequentially and collect results.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --epochs 10  # quick test
    python scripts/train_all.py --models lstm transformer temporal_attention
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.train import CONFIGS, train_linear, train_deep
from src.features.labels import DEFAULT_HORIZONS


ALL_MODELS = [
    "ols",
    "ridge",
    "lasso",
    "mlp",
    "lstm",
    "cnn",
    "deeplob",
    "seq2seq",
    "transformer",
    "temporal_attention",
]


def main():
    parser = argparse.ArgumentParser(description="Train all LOB models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        help="Models to train (default: all)",
    )
    parser.add_argument(
        "--ticker", type=str, default="AMZN",
        help="Single ticker to train on (use --tickers for multiple)",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Multiple tickers to train on sequentially (overrides --ticker)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/processed",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override epochs for deep models",
    )
    args = parser.parse_args()

    horizons = DEFAULT_HORIZONS
    all_results = {}

    total_t0 = time.time()

    # Support multi-ticker training
    tickers = args.tickers if args.tickers else [args.ticker]

    # Auto-discover tickers if 'all' is specified
    if tickers == ["all"]:
        ticker_dirs = sorted([
            d.name for d in Path(args.data_dir).iterdir()
            if d.is_dir() and list(d.glob("*.parquet"))
        ])
        if ticker_dirs:
            tickers = ticker_dirs
            print(f"Auto-discovered tickers: {tickers}")

    for ticker in tickers:
        print(f"\n{'#'*80}")
        print(f"# TICKER: {ticker}")
        print(f"{'#'*80}")

        ticker_results = {}

        for model_name in args.models:
            if model_name not in CONFIGS:
                print(f"Unknown model: {model_name}, skipping.")
                continue

            cfg = CONFIGS[model_name].copy()
            if args.epochs is not None and cfg["type"] != "linear":
                cfg["epochs"] = args.epochs

            try:
                if cfg["type"] == "linear":
                    metrics = train_linear(
                        model_name, cfg, args.data_dir, ticker,
                        horizons, args.results_dir,
                    )
                else:
                    metrics = train_deep(
                        model_name, cfg, args.data_dir, ticker,
                        horizons, args.results_dir, args.device,
                    )
                ticker_results[model_name] = metrics
            except Exception as e:
                print(f"\nERROR training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                ticker_results[model_name] = {"error": str(e)}

        all_results[ticker] = ticker_results

    total_elapsed = time.time() - total_t0

    # ── Summary table ────────────────────────────────────────────────
    for ticker in tickers:
        if ticker not in all_results:
            continue

        print(f"\n{'='*80}")
        print(f"SUMMARY — {ticker} — All Models ({total_elapsed:.0f}s total)")
        print(f"{'='*80}")

        header = f"{'Model':<25}"
        for h in horizons:
            header += f"  h{h}_acc  h{h}_f1"
        print(header)
        print("-" * len(header))

        ticker_results = all_results[ticker]
        for model_name in args.models:
            if model_name not in ticker_results:
                continue
            metrics = ticker_results[model_name]
            if "error" in metrics:
                print(f"{model_name:<25}  ERROR: {metrics['error']}")
                continue

            # Handle nested dict from deep models
            if "test_metrics" in metrics:
                metrics = metrics["test_metrics"]

            row = f"{model_name:<25}"
            for h in horizons:
                acc = metrics.get(f"h{h}_accuracy", 0)
                f1 = metrics.get(f"h{h}_f1_macro", 0)
                row += f"  {acc:.3f}  {f1:.3f}"
            print(row)

    # Save summary
    os.makedirs(args.results_dir, exist_ok=True)
    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull summary saved to {summary_path}")


if __name__ == "__main__":
    main()
