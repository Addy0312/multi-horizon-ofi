"""
End-to-end pipeline: Download Databento data → Convert → Validate → Ready for training.

This is the master orchestration script that:
    1. Downloads MBP-10 data from Databento (day-by-day, resumable)
    2. Converts DBN files to LOBSTER-compatible Parquet
    3. Validates all Parquet files for pipeline compatibility
    4. Prints a summary with data statistics

Usage:
    # Quick test (1 day, 1 stock — costs ~$0.50)
    python scripts/download_and_prepare.py --config quick -y

    # 1 week, 2 stocks (~$10-20)
    python scripts/download_and_prepare.py --config one_week -y

    # Full dataset (4 weeks, 5 stocks — ~$50-100)
    python scripts/download_and_prepare.py --config full -y

    # Custom
    python scripts/download_and_prepare.py --tickers AAPL NVDA \\
        --start 2024-03-01 --end 2024-03-16 -y

    # Convert only (already downloaded)
    python scripts/download_and_prepare.py --convert-only

    # Estimate cost only (no download)
    python scripts/download_and_prepare.py --estimate-only

Environment:
    export DATABENTO_API_KEY="db-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.etl.databento_config import (
    DOWNLOAD_CONFIG,
    QUICK_TEST_CONFIG,
    ONE_WEEK_CONFIG,
    DownloadConfig,
)
from src.etl.databento_download import download_all
from src.etl.databento_to_lobster import convert_all, validate_all


# ──────────────────────────────────────────────────────────────────────────
# Data Summary & Statistics
# ──────────────────────────────────────────────────────────────────────────

def print_data_summary(processed_dir: str):
    """
    Print a comprehensive summary of all downloaded & converted data.
    Shows statistics useful for the research paper's methodology section.
    """
    root = Path(processed_dir)

    print(f"\n{'=' * 70}")
    print(f"  DATA SUMMARY — Ready for Training")
    print(f"{'=' * 70}")

    total_rows = 0
    total_days = 0
    total_size_mb = 0

    ticker_stats = {}

    for ticker_dir in sorted(root.iterdir()):
        if not ticker_dir.is_dir():
            continue

        pq_files = sorted(ticker_dir.glob("*.parquet"))
        if not pq_files:
            continue

        ticker = ticker_dir.name
        n_days = len(pq_files)
        dates = []
        rows = 0
        size_mb = 0
        spreads = []
        daily_rows = []

        for pq in pq_files:
            df = pd.read_parquet(pq)
            n = len(df)
            rows += n
            daily_rows.append(n)
            size_mb += pq.stat().st_size / (1024 ** 2)

            date_str = pq.stem
            dates.append(date_str)

            # Spread statistics
            if "ask_price_1" in df.columns and "bid_price_1" in df.columns:
                spread = (df["ask_price_1"] - df["bid_price_1"]).median()
                spreads.append(spread)

        ticker_stats[ticker] = {
            "n_days": n_days,
            "total_rows": rows,
            "avg_rows_per_day": rows / n_days if n_days > 0 else 0,
            "size_mb": size_mb,
            "date_range": f"{dates[0]} → {dates[-1]}" if dates else "N/A",
            "median_spread": np.median(spreads) if spreads else 0,
        }

        total_rows += rows
        total_days += n_days
        total_size_mb += size_mb

    # Print per-ticker stats
    print(f"\n  {'Ticker':<8} {'Days':>5} {'Events':>12} {'Avg/Day':>10} "
          f"{'Size(MB)':>9} {'Spread':>8}  Date Range")
    print(f"  {'─'*8} {'─'*5} {'─'*12} {'─'*10} {'─'*9} {'─'*8}  {'─'*23}")

    for ticker, stats in ticker_stats.items():
        print(f"  {ticker:<8} {stats['n_days']:>5} {stats['total_rows']:>12,} "
              f"{stats['avg_rows_per_day']:>10,.0f} "
              f"{stats['size_mb']:>9.1f} "
              f"{stats['median_spread']:>8.4f}  "
              f"{stats['date_range']}")

    print(f"  {'─'*8} {'─'*5} {'─'*12} {'─'*10} {'─'*9}")
    print(f"  {'TOTAL':<8} {total_days:>5} {total_rows:>12,} "
          f"{total_rows/total_days if total_days > 0 else 0:>10,.0f} "
          f"{total_size_mb:>9.1f}")

    # Paper-ready methodology stats
    print(f"\n  {'─' * 50}")
    print(f"  PAPER METHODOLOGY STATISTICS:")
    print(f"  {'─' * 50}")
    print(f"  Tickers:           {len(ticker_stats)}")
    print(f"  Trading days:      {total_days}")
    print(f"  Stock-days:        {total_days}")
    print(f"  Total LOB events:  {total_rows:,}")
    print(f"  Data source:       Databento (XNAS.ITCH / Nasdaq TotalView)")
    print(f"  Schema:            MBP-10 (Market By Price, 10 levels)")
    print(f"  Granularity:       Nanosecond-resolution event data")
    print(f"  Market hours:      09:30 - 16:00 ET")
    if ticker_stats:
        first = list(ticker_stats.values())[0]
        print(f"  Period:            {first['date_range']}")
    print(f"{'=' * 70}")


# ──────────────────────────────────────────────────────────────────────────
# Pipeline Compatibility Check
# ──────────────────────────────────────────────────────────────────────────

def test_pipeline_compatibility(processed_dir: str, ticker: str):
    """
    Run a smoke test to verify the converted data works with the
    existing OFI pipeline (dataset.py → ofi.py → labels.py).
    """
    print(f"\n  PIPELINE COMPATIBILITY TEST ({ticker})")
    print(f"  {'─' * 50}")

    ticker_dir = Path(processed_dir) / ticker
    pq_files = sorted(ticker_dir.glob("*.parquet"))

    if not pq_files:
        print(f"  ✗ No Parquet files found for {ticker}")
        return False

    # Test with the first available file
    test_file = str(pq_files[0])
    print(f"  Testing with: {pq_files[0].name}")

    try:
        # 1. Load parquet
        df = pd.read_parquet(test_file)
        print(f"    ✓ Loaded: {len(df):,} rows, {len(df.columns)} columns")

        # 2. Test OFI computation
        from src.features.ofi import compute_multi_level_ofi
        ofi_df = compute_multi_level_ofi(df, max_level=5)
        print(f"    ✓ OFI computed: {ofi_df.shape[1]} features, "
              f"OFI range [{ofi_df['ofi_1'].min():.0f}, {ofi_df['ofi_1'].max():.0f}]")

        # 3. Test microstructure features
        from src.features.microstructure import compute_all_features
        micro_df = compute_all_features(df, max_level=5)
        print(f"    ✓ Microstructure: {micro_df.shape[1]} features")

        # 4. Test label generation
        from src.features.labels import make_regression_labels, make_classification_labels
        reg_labels = make_regression_labels(df)
        cls_labels = make_classification_labels(df)
        print(f"    ✓ Regression labels: {reg_labels.shape}")
        print(f"    ✓ Classification labels: {cls_labels.shape}")

        # 5. Test full build_features_and_labels
        from src.data.dataset import build_features_and_labels
        X, y_reg, y_cls, fnames = build_features_and_labels(test_file)
        print(f"    ✓ Full feature build: X={X.shape}, y_reg={y_reg.shape}, "
              f"y_cls={y_cls.shape}")
        print(f"      Features: {len(fnames)} ({fnames[:5]}...)")

        print(f"  ✓ ALL PIPELINE TESTS PASSED for {ticker}")
        return True

    except Exception as e:
        print(f"  ✗ PIPELINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────

def run_pipeline(
    cfg: DownloadConfig,
    estimate_only: bool = False,
    convert_only: bool = False,
    skip_confirmation: bool = False,
    validate: bool = True,
    test_pipeline: bool = True,
):
    """
    Run the full download → convert → validate → test pipeline.
    """
    t0 = time.time()

    print("\n" + "█" * 70)
    print("█  MULTI-HORIZON OFI — DATA PIPELINE")
    print("█" + "─" * 69)
    print(f"█  Tickers:  {cfg.tickers}")
    print(f"█  Range:    {cfg.start_date} → {cfg.end_date}")
    print(f"█  Dataset:  {cfg.dataset} / {cfg.schema}")
    print(f"█  Budget:   ${cfg.max_total_cost:.2f}")
    print("█" * 70)

    # ── Step 1: Download ──────────────────────────────────────────────
    if not convert_only:
        print("\n\n" + "=" * 70)
        print("  STEP 1/4: DOWNLOAD FROM DATABENTO")
        print("=" * 70)

        download_results = download_all(
            cfg,
            estimate_only=estimate_only,
            skip_confirmation=skip_confirmation,
        )

        if estimate_only:
            return

    # ── Step 2: Convert DBN → Parquet ─────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  STEP 2/4: CONVERT DBN → LOBSTER PARQUET")
    print("=" * 70)

    convert_results = convert_all(
        dbn_dir=cfg.dbn_cache_dir,
        output_dir=cfg.processed_dir,
        n_levels=cfg.n_levels,
        market_open_sec=cfg.market_open_sec,
        market_close_sec=cfg.market_close_sec,
    )

    # ── Step 3: Validate ──────────────────────────────────────────────
    if validate:
        print("\n\n" + "=" * 70)
        print("  STEP 3/4: VALIDATE CONVERTED DATA")
        print("=" * 70)

        all_valid = validate_all(cfg.processed_dir, cfg.n_levels)
        if not all_valid:
            print("\n  ⚠ Some files failed validation. Check output above.")

    # ── Step 4: Pipeline smoke test ───────────────────────────────────
    if test_pipeline and convert_results:
        print("\n\n" + "=" * 70)
        print("  STEP 4/4: PIPELINE COMPATIBILITY TEST")
        print("=" * 70)

        # Test with the first available ticker
        for ticker in cfg.tickers:
            ticker_dir = Path(cfg.processed_dir) / ticker
            if ticker_dir.exists() and list(ticker_dir.glob("*.parquet")):
                test_pipeline_compatibility(cfg.processed_dir, ticker)
                break

    # ── Summary ───────────────────────────────────────────────────────
    print_data_summary(cfg.processed_dir)

    elapsed = time.time() - t0
    print(f"\n  Total pipeline time: {elapsed/60:.1f} minutes")
    print(f"  Data ready in: {cfg.processed_dir}/")
    print(f"\n  Next step:")
    print(f"    python scripts/train_all.py --ticker AAPL --data-dir {cfg.processed_dir}")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download & prepare Databento data for OFI training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  COST GUIDE (approximate, depends on market activity):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  --config quick       1 day  × 1 stock    ~$0.50
  --config one_week    5 days × 2 stocks   ~$10-20
  --config full        20 days × 5 stocks  ~$50-100

  Run --estimate-only first to see exact costs!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Examples:
  # See how much it will cost (no download)
  python scripts/download_and_prepare.py --estimate-only

  # Quick test
  python scripts/download_and_prepare.py --config quick -y

  # Full research dataset
  python scripts/download_and_prepare.py --config full -y

  # Custom setup
  python scripts/download_and_prepare.py --tickers AAPL NVDA TSLA \\
      --start 2024-06-01 --end 2024-07-01 --max-cost 150 -y

  # Already downloaded? Just convert
  python scripts/download_and_prepare.py --convert-only
        """,
    )

    parser.add_argument(
        "--config",
        choices=["full", "quick", "one_week"],
        default="full",
        help="Predefined config profile (default: full)",
    )
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Override tickers")
    parser.add_argument("--start", type=str, default=None,
                        help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="Override end date (YYYY-MM-DD, exclusive)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Databento API key")
    parser.add_argument("--max-cost", type=float, default=None,
                        help="Override max total budget (USD)")
    parser.add_argument("--estimate-only", action="store_true",
                        help="Only estimate costs, don't download")
    parser.add_argument("--convert-only", action="store_true",
                        help="Skip download, only convert existing DBN files")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompt")
    parser.add_argument("--no-test", action="store_true",
                        help="Skip pipeline compatibility test")

    args = parser.parse_args()

    # Select config
    configs = {
        "full": DOWNLOAD_CONFIG,
        "quick": QUICK_TEST_CONFIG,
        "one_week": ONE_WEEK_CONFIG,
    }
    cfg = configs[args.config]

    # Apply CLI overrides
    if args.tickers:
        cfg.tickers = args.tickers
    if args.start:
        cfg.start_date = args.start
    if args.end:
        cfg.end_date = args.end
    if args.api_key:
        cfg.api_key = args.api_key
    if args.max_cost:
        cfg.max_total_cost = args.max_cost

    run_pipeline(
        cfg,
        estimate_only=args.estimate_only,
        convert_only=args.convert_only,
        skip_confirmation=args.yes,
        test_pipeline=not args.no_test,
    )


if __name__ == "__main__":
    main()
