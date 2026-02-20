"""
Convert Databento MBP-10 (DBN) files to LOBSTER-compatible Parquet format.

This module bridges Databento's data format with the existing OFI pipeline.
It reads MBP-10 (Market By Price, 10 levels) DBN files and produces the
exact same Parquet schema that src/etl/preprocess.py generates from LOBSTER
CSV files.

Target Parquet Schema (matches preprocess.py output):
    time           : float64   — seconds since midnight
    event_type     : int       — LOBSTER event code (1-7)
    order_id       : int       — (set to 0; Databento MBP doesn't expose this)
    size           : int       — event size
    price          : float64   — event price (÷ 10,000 for LOBSTER scale)
    direction      : int       — 1=buy, -1=sell
    ask_price_1..N : float64   — ask prices at each level (÷ 10,000)
    ask_size_1..N  : int       — ask sizes at each level
    bid_price_1..N : float64   — bid prices at each level (÷ 10,000)
    bid_size_1..N  : int       — bid sizes at each level
    datetime       : datetime  — full datetime for pandas

The key insight: The OFI computation only uses:
    bid_price_{1..5}, bid_size_{1..5}, ask_price_{1..5}, ask_size_{1..5}
These come from the MBP-10 snapshots directly.

Usage:
    python -m src.etl.databento_to_lobster
    python -m src.etl.databento_to_lobster --dbn-dir data/dbn_cache --out-dir data/processed
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

try:
    import databento as db
except ImportError:
    print("ERROR: databento package not installed. pip install databento")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm package not installed. pip install tqdm")
    sys.exit(1)

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.etl.databento_config import DOWNLOAD_CONFIG, DownloadConfig


# ──────────────────────────────────────────────────────────────────────────
# MBP-10 Column Mapping
# ──────────────────────────────────────────────────────────────────────────

def _build_mbp10_level_columns(n_levels: int = 10) -> dict:
    """
    Map Databento MBP-10 DataFrame columns to LOBSTER-style column names.

    Databento MBP-10 to_df() produces columns like:
        bid_px_00, bid_sz_00, ask_px_00, ask_sz_00,  (Level 0 = best)
        bid_px_01, bid_sz_01, ask_px_01, ask_sz_01,  (Level 1)
        ...
        bid_px_09, bid_sz_09, ask_px_09, ask_sz_09   (Level 9)

    LOBSTER convention:
        ask_price_1, ask_size_1, bid_price_1, bid_size_1,  (Level 1 = best)
        ask_price_2, ask_size_2, bid_price_2, bid_size_2,
        ...

    Note: Databento levels are 0-indexed, LOBSTER is 1-indexed.
    """
    mapping = {}
    for i in range(n_levels):
        db_idx = f"{i:02d}"
        lob_idx = i + 1  # LOBSTER is 1-indexed

        mapping[f"ask_px_{db_idx}"] = f"ask_price_{lob_idx}"
        mapping[f"ask_sz_{db_idx}"] = f"ask_size_{lob_idx}"
        mapping[f"bid_px_{db_idx}"] = f"bid_price_{lob_idx}"
        mapping[f"bid_sz_{db_idx}"] = f"bid_size_{lob_idx}"

    return mapping


# ──────────────────────────────────────────────────────────────────────────
# Core Conversion
# ──────────────────────────────────────────────────────────────────────────

def convert_dbn_to_lobster_parquet(
    dbn_path: str,
    output_path: str,
    n_levels: int = 10,
    market_open_sec: int = 34200,
    market_close_sec: int = 57600,
) -> Optional[str]:
    """
    Convert a single Databento MBP-10 DBN file to LOBSTER-compatible Parquet.

    Parameters
    ----------
    dbn_path : str
        Path to the input .dbn.zst file.
    output_path : str
        Path for the output .parquet file.
    n_levels : int
        Number of LOB levels to include (default: 10).
    market_open_sec : int
        Market open in seconds from midnight (09:30 = 34200).
    market_close_sec : int
        Market close in seconds from midnight (16:00 = 57600).

    Returns
    -------
    str or None
        Path to the saved Parquet file, or None if no data.
    """
    # ── Load DBN file ─────────────────────────────────────────────────
    try:
        store = db.DBNStore.from_file(dbn_path)
    except Exception as e:
        print(f"    ERROR reading {dbn_path}: {e}")
        return None

    # Convert to DataFrame with human-readable prices & timestamps
    try:
        df = store.to_df(
            price_type="float",  # Convert prices to float (from fixed-point)
            pretty_ts=True,      # Convert timestamps to pandas Timestamp
            map_symbols=True,    # Include symbol column
        )
    except Exception as e:
        print(f"    ERROR converting to DataFrame: {e}")
        return None

    if df.empty:
        print(f"    WARNING: No data in {dbn_path}")
        return None

    # ── Extract date from the data ────────────────────────────────────
    # The index is ts_recv; use ts_event for event timing
    if "ts_event" in df.columns:
        event_times = pd.to_datetime(df["ts_event"], utc=True)
    else:
        # Fall back to index (ts_recv)
        event_times = df.index.to_series()
        if event_times.dt.tz is None:
            event_times = event_times.dt.tz_localize("UTC")

    # Convert to US/Eastern for market hours
    event_times_et = event_times.dt.tz_convert("US/Eastern")
    trade_date = event_times_et.iloc[0].date()

    # ── Compute 'time' column (seconds since midnight ET) ─────────────
    midnight_et = pd.Timestamp(
        year=trade_date.year,
        month=trade_date.month,
        day=trade_date.day,
        tz="US/Eastern",
    )
    time_sec = (event_times_et - midnight_et).dt.total_seconds().values

    # ── Filter to regular trading hours ───────────────────────────────
    mask = (time_sec >= market_open_sec) & (time_sec <= market_close_sec)
    df = df.loc[mask].copy()
    time_sec = time_sec[mask]
    event_times_et = event_times_et.loc[mask]

    if df.empty:
        print(f"    WARNING: No data within market hours for {trade_date}")
        return None

    # ── Rename LOB columns: Databento → LOBSTER ───────────────────────
    col_mapping = _build_mbp10_level_columns(n_levels)

    # Check which columns exist (Databento format may vary)
    available_mappings = {k: v for k, v in col_mapping.items() if k in df.columns}

    if len(available_mappings) < 4:
        print(f"    ERROR: Missing LOB columns. Available: {list(df.columns)}")
        return None

    # ── Build output DataFrame ────────────────────────────────────────
    out = pd.DataFrame()

    # Time column (seconds since midnight, matching LOBSTER format)
    out["time"] = time_sec

    # Event metadata (approximated — MBP-10 doesn't have LOBSTER event codes)
    # We set event_type=5 (execution visible in book change)
    # order_id=0 (not available), direction based on side
    out["event_type"] = 5  # Book update indicator
    out["order_id"] = 0

    # Size: use the best bid or ask size change as proxy
    if "size" in df.columns:
        out["size"] = df["size"].values
    else:
        # If no explicit size, use best level size
        best_bid_sz_col = [c for c in df.columns if "bid_sz_00" in c]
        if best_bid_sz_col:
            out["size"] = df[best_bid_sz_col[0]].values
        else:
            out["size"] = 0

    # Price: use the event price or mid-price
    if "price" in df.columns:
        # Databento prices are already in float after pretty_px=True
        out["price"] = df["price"].values
    else:
        # Compute mid-price from best bid/ask
        if "bid_px_00" in df.columns and "ask_px_00" in df.columns:
            out["price"] = (df["bid_px_00"].values + df["ask_px_00"].values) / 2.0
        else:
            out["price"] = 0.0

    # Direction: side from Databento
    if "side" in df.columns:
        # Databento side: 'B' = buy, 'A' = sell/ask
        out["direction"] = df["side"].map({"B": 1, "A": -1}).fillna(0).astype(int).values
    else:
        out["direction"] = 0

    # ── LOB Level Columns ─────────────────────────────────────────────
    for db_col, lob_col in available_mappings.items():
        out[lob_col] = df[db_col].values

    # ── Ensure all expected LOB columns exist ─────────────────────────
    for lvl in range(1, n_levels + 1):
        for col in [f"ask_price_{lvl}", f"ask_size_{lvl}",
                    f"bid_price_{lvl}", f"bid_size_{lvl}"]:
            if col not in out.columns:
                out[col] = 0

    # ── Drop crossed markets (safety) ─────────────────────────────────
    valid = out["bid_price_1"] < out["ask_price_1"]
    n_crossed = (~valid).sum()
    if n_crossed > 0:
        print(f"    Dropped {n_crossed} crossed-market rows "
              f"({n_crossed/len(out)*100:.2f}%)")
        out = out[valid].copy()

    # ── Add datetime column (matching preprocess.py output) ───────────
    out["datetime"] = event_times_et.values[:len(out)]

    # ── Drop NaN/inf ──────────────────────────────────────────────────
    price_cols = [c for c in out.columns if "price" in c]
    for pc in price_cols:
        out[pc] = out[pc].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["bid_price_1", "ask_price_1"])

    if out.empty:
        print(f"    WARNING: No valid data after filtering for {trade_date}")
        return None

    # ── Save to Parquet ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_parquet(output_path, compression="snappy", index=False)

    return output_path


# ──────────────────────────────────────────────────────────────────────────
# Batch Conversion
# ──────────────────────────────────────────────────────────────────────────

def convert_all(
    dbn_dir: str = "data/dbn_cache",
    output_dir: str = "data/processed",
    n_levels: int = 10,
    market_open_sec: int = 34200,
    market_close_sec: int = 57600,
    max_workers: int = 4,
) -> dict:
    """
    Convert all DBN files in the cache directory to LOBSTER-compatible Parquet
    using parallel processing with progress tracking.

    Expects structure:
        dbn_dir/
            AAPL/
                AAPL_2024-01-08.mbp-10.dbn.zst
                AAPL_2024-01-09.mbp-10.dbn.zst
            MSFT/
                ...

    Outputs:
        output_dir/
            AAPL/
                2024-01-08.parquet
                2024-01-09.parquet
            MSFT/
                ...

    Returns
    -------
    dict
        {ticker: [list of parquet paths]}
    """
    results = {}

    # Discover ticker directories
    dbn_root = Path(dbn_dir)
    if not dbn_root.exists():
        print(f"ERROR: DBN cache directory not found: {dbn_dir}")
        return results

    ticker_dirs = sorted([
        d for d in dbn_root.iterdir()
        if d.is_dir() and d.name != "__pycache__" and d.name != "download_log.json"
    ])

    if not ticker_dirs:
        # Maybe files are flat (no subdirectories)
        dbn_files = sorted(dbn_root.glob("*.dbn.zst"))
        if dbn_files:
            ticker_dirs = [dbn_root]

    # Collect all conversion tasks
    tasks = []
    for ticker_dir in ticker_dirs:
        ticker = ticker_dir.name
        if ticker in ["download_log.json", "__pycache__"]:
            continue

        dbn_files = sorted(ticker_dir.glob("*.dbn.zst"))
        if not dbn_files:
            continue

        for dbn_file in dbn_files:
            # Extract date from filename
            fname = dbn_file.stem.replace(".dbn", "").replace(".mbp-10", "")
            parts = fname.split("_")
            date_str = parts[-1] if len(parts) >= 2 else fname

            # Output path
            out_path = os.path.join(output_dir, ticker, f"{date_str}.parquet")

            # Skip if already converted
            if os.path.exists(out_path):
                if ticker not in results:
                    results[ticker] = []
                results[ticker].append(out_path)
                continue

            tasks.append((str(dbn_file), out_path, ticker, date_str, n_levels, 
                         market_open_sec, market_close_sec))

    total_files = len(tasks) + sum(len(v) for v in results.values())
    
    if not tasks:
        print(f"\n  All {total_files} files already converted!")
        return results

    print(f"\n{'━' * 60}")
    print(f"  Converting {len(tasks)} files ({total_files - len(tasks)} cached)")
    print(f"{'━' * 60}\n")

    # Sequential conversion with progress bar
    with tqdm(
        total=total_files,
        desc="Converting",
        unit="file",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        initial=total_files - len(tasks),
    ) as pbar:
        for dbn_path, out_path, ticker, date_str, levels, open_sec, close_sec in tasks:
            try:
                result = convert_dbn_to_lobster_parquet(
                    dbn_path, out_path, levels, open_sec, close_sec
                )
                if result:
                    pq_size = os.path.getsize(result) / (1024 ** 2)
                    pq_df = pd.read_parquet(result)
                    if ticker not in results:
                        results[ticker] = []
                    results[ticker].append(result)
                    pbar.set_postfix_str(f"{ticker} {date_str} ✓ {len(pq_df):,} rows {pq_size:.1f}MB")
                else:
                    pbar.set_postfix_str(f"{ticker} {date_str} ⊘")
            except Exception as e:
                pbar.set_postfix_str(f"{ticker} {date_str} ✗ {str(e)[:30]}")
            
            pbar.update(1)

    print(f"\n{'=' * 60}")
    print(f"  CONVERSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total files:     {total_files}")
    print(f"  Newly converted: {len(tasks)}")
    print(f"  Output dir:      {output_dir}")
    print(f"{'=' * 60}")

    return results


# ──────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────

def validate_parquet(parquet_path: str, n_levels: int = 10) -> bool:
    """
    Validate that a converted Parquet file is compatible with the OFI pipeline.

    Checks:
        1. Required columns exist
        2. No NaN in critical columns
        3. Bid < Ask (no crossed markets)
        4. Positive sizes
        5. Time is within market hours
    """
    df = pd.read_parquet(parquet_path)

    errors = []

    # 1. Required columns
    required = ["time", "bid_price_1", "ask_price_1", "bid_size_1", "ask_size_1"]
    for lvl in range(1, min(6, n_levels + 1)):  # At least 5 levels for OFI
        required.extend([
            f"bid_price_{lvl}", f"bid_size_{lvl}",
            f"ask_price_{lvl}", f"ask_size_{lvl}",
        ])
    required = list(set(required))

    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    # 2. NaN check
    for col in ["bid_price_1", "ask_price_1"]:
        if col in df.columns:
            nans = df[col].isna().sum()
            if nans > 0:
                errors.append(f"{col} has {nans} NaN values")

    # 3. Crossed markets
    if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
        crossed = (df["bid_price_1"] >= df["ask_price_1"]).sum()
        if crossed > 0:
            errors.append(f"{crossed} crossed-market rows")

    # 4. Positive sizes
    for col in ["bid_size_1", "ask_size_1"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                errors.append(f"{col} has {neg} negative values")

    # 5. Time range
    if "time" in df.columns:
        t_min, t_max = df["time"].min(), df["time"].max()
        if t_min < 34000 or t_max > 58000:
            errors.append(f"Time range suspect: {t_min:.0f} to {t_max:.0f}")

    if errors:
        print(f"    VALIDATION FAILED for {parquet_path}:")
        for e in errors:
            print(f"      ✗ {e}")
        return False

    print(f"    ✓ Valid: {len(df):,} rows, "
          f"time {df['time'].min():.0f}-{df['time'].max():.0f}s, "
          f"spread {(df['ask_price_1'] - df['bid_price_1']).median():.4f}")
    return True


def validate_all(processed_dir: str = "data/processed", n_levels: int = 10):
    """Validate all converted Parquet files."""
    print("\n  VALIDATION")
    print("  " + "─" * 40)

    root = Path(processed_dir)
    all_valid = True

    for ticker_dir in sorted(root.iterdir()):
        if not ticker_dir.is_dir():
            continue

        pq_files = sorted(ticker_dir.glob("*.parquet"))
        if not pq_files:
            continue

        print(f"\n  {ticker_dir.name}: {len(pq_files)} files")
        for pq in pq_files:
            valid = validate_parquet(str(pq), n_levels)
            if not valid:
                all_valid = False

    return all_valid


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert Databento MBP-10 DBN files to LOBSTER-compatible Parquet",
    )
    parser.add_argument(
        "--dbn-dir",
        type=str,
        default="data/dbn_cache",
        help="Directory with cached DBN files (default: data/dbn_cache)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed",
        help="Output directory for Parquet files (default: data/processed)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=10,
        help="Number of LOB levels to include (default: 10)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing Parquet files, don't convert",
    )

    args = parser.parse_args()

    if args.validate_only:
        validate_all(args.out_dir, args.levels)
    else:
        convert_all(
            dbn_dir=args.dbn_dir,
            output_dir=args.out_dir,
            n_levels=args.levels,
        )
        validate_all(args.out_dir, args.levels)


if __name__ == "__main__":
    main()
