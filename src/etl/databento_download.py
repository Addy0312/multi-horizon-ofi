"""
Databento Historical Data Downloader for Multi-Horizon OFI Project.

Downloads MBP-10 (Market By Price, 10 levels) order book snapshots from
Nasdaq TotalView (XNAS.ITCH) via Databento's Historical API.

Features:
    - Cost estimation before any download
    - Day-by-day chunked downloads (resumable)
    - DBN file caching (never re-downloads the same day)
    - Multi-ticker support with progress tracking
    - Safety guards for budget 

Usage:
    # From project root
    python -m src.etl.databento_download                    # Full download
    python -m src.etl.databento_download --config quick     # Quick test (1 day, 1 stock)
    python -m src.etl.databento_download --config one_week  # 1 week, 2 stocks
    python -m src.etl.databento_download --estimate-only    # Just show cost estimate

Requires:
    pip install databento
    export DATABENTO_API_KEY="db-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

try:
    import databento as db
except ImportError:
    print("ERROR: databento package not installed.")
    print("  Install it with: pip install databento")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm package not installed.")
    print("  Install it with: pip install tqdm")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.etl.databento_config import (
    DOWNLOAD_CONFIG,
    QUICK_TEST_CONFIG,
    ONE_WEEK_CONFIG,
    DownloadConfig,
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def get_trading_days(start: str, end: str) -> List[date]:
    """
    Generate list of weekdays (Mon-Fri) between start and end (exclusive).
    Does NOT account for market holidays — Databento will simply return
    no data for holidays, so this is safe.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()

    days = []
    current = start_dt
    while current < end_dt:
        if current.weekday() < 5:  # Mon=0 .. Fri=4
            days.append(current)
        current += timedelta(days=1)
    return days


def dbn_cache_path(cache_dir: str, ticker: str, day: date) -> Path:
    """Return the expected cache path for a day's DBN file."""
    p = Path(cache_dir) / ticker
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{ticker}_{day.isoformat()}.mbp-10.dbn.zst"


def download_log_path(cache_dir: str) -> Path:
    """Path to the JSON log tracking completed downloads."""
    return Path(cache_dir) / "download_log.json"


def load_download_log(cache_dir: str) -> dict:
    """Load or create the download progress log."""
    log_path = download_log_path(cache_dir)
    if log_path.exists():
        with open(log_path, "r") as f:
            return json.load(f)
    return {"completed": [], "costs": {}, "total_cost": 0.0}


def save_download_log(cache_dir: str, log: dict):
    """Persist the download progress log."""
    log_path = download_log_path(cache_dir)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────
# Cost Estimation
# ──────────────────────────────────────────────────────────────────────────

def estimate_costs(
    client: db.Historical,
    cfg: DownloadConfig,
) -> Tuple[dict, float]:
    """
    Estimate download costs for all tickers and dates.

    Returns
    -------
    breakdown : dict
        Per-ticker cost breakdown.
    total : float
        Total estimated cost in USD.
    """
    print("\n" + "=" * 70)
    print("  COST ESTIMATION")
    print("=" * 70)

    breakdown = {}
    total = 0.0

    for ticker in cfg.tickers:
        try:
            cost = client.metadata.get_cost(
                dataset=cfg.dataset,
                symbols=[ticker],
                schema=cfg.schema,
                stype_in=cfg.stype_in,
                start=cfg.start_date,
                end=cfg.end_date,
            )
            breakdown[ticker] = cost
            total += cost
            print(f"  {ticker:6s}  ${cost:>10.4f}")
        except Exception as e:
            print(f"  {ticker:6s}  ERROR: {e}")
            breakdown[ticker] = -1

        # Rate limit: 20 metadata requests/sec
        time.sleep(0.1)

    print(f"  {'─' * 20}")
    print(f"  {'TOTAL':6s}  ${total:>10.4f}")
    print(f"  Date range: {cfg.start_date} → {cfg.end_date}")
    print(f"  Dataset: {cfg.dataset}, Schema: {cfg.schema}")
    print("=" * 70)

    return breakdown, total


def estimate_data_size(
    client: db.Historical,
    cfg: DownloadConfig,
) -> Tuple[dict, int]:
    """
    Estimate download sizes for all tickers.

    Returns
    -------
    breakdown : dict
        Per-ticker size in bytes.
    total : int
        Total size in bytes.
    """
    print("\n  DATA SIZE ESTIMATION")
    print("  " + "─" * 40)

    breakdown = {}
    total = 0

    for ticker in cfg.tickers:
        try:
            size = client.metadata.get_billable_size(
                dataset=cfg.dataset,
                symbols=[ticker],
                schema=cfg.schema,
                stype_in=cfg.stype_in,
                start=cfg.start_date,
                end=cfg.end_date,
            )
            breakdown[ticker] = size
            total += size
            size_gb = size / (1024 ** 3)
            print(f"  {ticker:6s}  {size_gb:>8.2f} GB")
        except Exception as e:
            print(f"  {ticker:6s}  ERROR: {e}")
            breakdown[ticker] = -1

        time.sleep(0.1)

    total_gb = total / (1024 ** 3)
    print(f"  {'─' * 20}")
    print(f"  {'TOTAL':6s}  {total_gb:>8.2f} GB")

    return breakdown, total


# ──────────────────────────────────────────────────────────────────────────
# Data Availability Check
# ──────────────────────────────────────────────────────────────────────────

def check_data_availability(
    client: db.Historical,
    cfg: DownloadConfig,
) -> bool:
    """
    Verify that the dataset has data available for the requested range.
    """
    print("\n  DATA AVAILABILITY CHECK")
    print("  " + "─" * 40)

    try:
        dataset_range = client.metadata.get_dataset_range(dataset=cfg.dataset)
        avail_start = dataset_range["start"][:10]
        avail_end = dataset_range["end"][:10]
        print(f"  Dataset: {cfg.dataset}")
        print(f"  Available: {avail_start} → {avail_end}")
        print(f"  Requested: {cfg.start_date} → {cfg.end_date}")

        # Check schemas
        schemas = client.metadata.list_schemas(dataset=cfg.dataset)
        if cfg.schema in schemas:
            print(f"  Schema '{cfg.schema}': ✓ Available")
        else:
            print(f"  Schema '{cfg.schema}': ✗ NOT AVAILABLE")
            print(f"  Available schemas: {schemas}")
            return False

        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────
# Main Download Logic
# ──────────────────────────────────────────────────────────────────────────

def download_one_day(
    client: db.Historical,
    cfg: DownloadConfig,
    ticker: str,
    day: date,
    pbar: Optional[tqdm] = None,
) -> Optional[Path]:
    """
    Download a single day's MBP-10 data for one ticker with progress tracking.

    Saves as compressed DBN (.dbn.zst) to the cache directory.
    Returns the path to the saved file, or None on failure.
    """
    cache_path = dbn_cache_path(cfg.dbn_cache_dir, ticker, day)

    # Skip if already cached
    if cache_path.exists() and cache_path.stat().st_size > 0:
        if pbar:
            pbar.set_postfix_str(f"{ticker} {day} ✓ cached")
            pbar.update(1)
        return cache_path

    # Build time range for the trading day
    day_start = f"{day.isoformat()}T00:00:00"
    day_end_dt = day + timedelta(days=1)
    day_end = f"{day_end_dt.isoformat()}T00:00:00"

    try:
        if pbar:
            pbar.set_postfix_str(f"{ticker} {day} ↓ downloading...")
        
        data = client.timeseries.get_range(
            dataset=cfg.dataset,
            symbols=[ticker],
            schema=cfg.schema,
            stype_in=cfg.stype_in,
            start=day_start,
            end=day_end,
        )

        # Save to disk
        data.to_file(str(cache_path))
        
        size_mb = cache_path.stat().st_size / (1024 ** 2)
        if pbar:
            pbar.set_postfix_str(f"{ticker} {day} ✓ {size_mb:.1f}MB")
            pbar.update(1)
        
        return cache_path

    except db.BentoClientError as e:
        if "404" in str(e) or "Not Found" in str(e):
            if pbar:
                pbar.set_postfix_str(f"{ticker} {day} ⊘ no data (holiday)")
                pbar.update(1)
            return None
        raise
    except Exception as e:
        if pbar:
            pbar.set_postfix_str(f"{ticker} {day} ✗ error")
            pbar.update(1)
        return None


def download_all(
    cfg: DownloadConfig,
    estimate_only: bool = False,
    skip_confirmation: bool = False,
) -> dict:
    """
    Main entry point: download all tickers for all days in the config.

    Parameters
    ----------
    cfg : DownloadConfig
        Download configuration.
    estimate_only : bool
        If True, only show cost/size estimates without downloading.
    skip_confirmation : bool
        If True, skip the user confirmation prompt.

    Returns
    -------
    dict
        Summary of downloads: {ticker: [list of dbn file paths]}
    """
    # ── Create client ─────────────────────────────────────────────────
    try:
        client = db.Historical(cfg.api_key)
    except Exception as e:
        print(f"ERROR: Failed to create Databento client: {e}")
        print("  Make sure DATABENTO_API_KEY is set or pass --api-key")
        sys.exit(1)

    # ── Pre-flight checks ─────────────────────────────────────────────
    available = check_data_availability(client, cfg)
    if not available:
        print("\nABORTING: Data not available for the requested parameters.")
        sys.exit(1)

    cost_breakdown, total_cost = estimate_costs(client, cfg)
    size_breakdown, total_size = estimate_data_size(client, cfg)

    if estimate_only:
        print("\n  [--estimate-only] No data will be downloaded.")
        return {}

    # ── Budget guard ──────────────────────────────────────────────────
    if total_cost > cfg.max_total_cost:
        print(f"\n  ⚠ BUDGET EXCEEDED: ${total_cost:.2f} > ${cfg.max_total_cost:.2f}")
        print("  Increase max_total_cost in config or reduce scope.")
        sys.exit(1)

    # ── Confirmation ──────────────────────────────────────────────────
    if not skip_confirmation:
        print(f"\n  Ready to download ~{total_size / (1024**3):.1f} GB "
              f"for ~${total_cost:.2f}")
        response = input("  Proceed? [y/N] ").strip().lower()
        if response != "y":
            print("  Aborted by user.")
            sys.exit(0)

    # ── Generate trading days ─────────────────────────────────────────
    trading_days = get_trading_days(cfg.start_date, cfg.end_date)
    print(f"\n  Trading days to download: {len(trading_days)}")
    print(f"  Tickers: {cfg.tickers}")
    print(f"  Total requests: {len(trading_days) * len(cfg.tickers)}")

    # ── Create directories ────────────────────────────────────────────
    os.makedirs(cfg.dbn_cache_dir, exist_ok=True)
    os.makedirs(cfg.raw_dir, exist_ok=True)

    # ── Load progress log ─────────────────────────────────────────────
    log = load_download_log(cfg.dbn_cache_dir)
    log_lock = Lock()  # Thread-safe log updates
    results = {}
    session_cost = 0.0

    # ── Build task list ───────────────────────────────────────────────
    tasks = []
    for ticker in cfg.tickers:
        for day in trading_days:
            task_key = f"{ticker}_{day.isoformat()}"
            
            # Skip if already completed
            if task_key in log["completed"]:
                cached = dbn_cache_path(cfg.dbn_cache_dir, ticker, day)
                if cached.exists():
                    if ticker not in results:
                        results[ticker] = []
                    results[ticker].append(cached)
                    continue
            
            tasks.append((ticker, day, task_key))

    total_tasks = len(tasks) + len([k for k in log["completed"] 
                                     if any(t in k for t in cfg.tickers)])
    
    print(f"\n  {'━' * 60}")
    print(f"  Starting parallel download: {len(tasks)} new files")
    print(f"  (Already cached: {total_tasks - len(tasks)})")
    print(f"  {'━' * 60}\n")

    # ── Parallel download with progress bar ──────────────────────────
    def download_task(ticker: str, day: date, task_key: str):
        """Worker function for parallel download."""
        # Each thread gets its own client to avoid race conditions
        thread_client = db.Historical(cfg.api_key)
        
        # Estimate cost
        try:
            day_cost = thread_client.metadata.get_cost(
                dataset=cfg.dataset,
                symbols=[ticker],
                schema=cfg.schema,
                stype_in=cfg.stype_in,
                start=day.isoformat(),
                end=(day + timedelta(days=1)).isoformat(),
            )
        except Exception:
            day_cost = 0.0
        
        # Budget check
        if day_cost > cfg.max_cost_per_request:
            return None, None, 0.0
        
        # Download
        path = download_one_day(thread_client, cfg, ticker, day, pbar=None)
        
        return ticker, path, day_cost

    # Create progress bar
    with tqdm(
        total=total_tasks,
        desc="Downloading",
        unit="file",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        initial=total_tasks - len(tasks),  # Account for cached files
    ) as pbar:
        
        # Use ThreadPoolExecutor for parallel downloads (max 4 concurrent)
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_task = {
                executor.submit(download_task, ticker, day, task_key): (ticker, task_key, day)
                for ticker, day, task_key in tasks
            }
            
            for future in as_completed(future_to_task):
                ticker, task_key, day = future_to_task[future]
                
                try:
                    result_ticker, path, day_cost = future.result()
                    
                    if path is not None:
                        # Thread-safe log update
                        with log_lock:
                            if result_ticker not in results:
                                results[result_ticker] = []
                            results[result_ticker].append(path)
                            
                            log["completed"].append(task_key)
                            log["costs"][task_key] = day_cost
                            log["total_cost"] = log.get("total_cost", 0) + day_cost
                            session_cost += day_cost
                            save_download_log(cfg.dbn_cache_dir, log)
                        
                        size_mb = path.stat().st_size / (1024 ** 2)
                        pbar.set_postfix_str(f"{result_ticker} {day} ✓ {size_mb:.1f}MB")
                    else:
                        pbar.set_postfix_str(f"{result_ticker} {day} ⊘")
                    
                    pbar.update(1)
                    
                except Exception as e:
                    pbar.set_postfix_str(f"{ticker} {day} ✗ {str(e)[:30]}")
                    pbar.update(1)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"{'=' * 60}")
    total_files = sum(len(v) for v in results.values())
    print(f"  Files downloaded: {total_files}")
    print(f"  Session cost:     ${session_cost:.4f}")
    print(f"  Cumulative cost:  ${log.get('total_cost', 0):.4f}")
    print(f"  Cache directory:  {cfg.dbn_cache_dir}")
    print(f"{'=' * 60}")

    return results


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Databento MBP-10 data for OFI research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate costs only (no download)
  python -m src.etl.databento_download --estimate-only

  # Quick test: 1 day, 1 stock
  python -m src.etl.databento_download --config quick

  # 1 week, 2 stocks
  python -m src.etl.databento_download --config one_week

  # Full download (4 weeks, 5 stocks)
  python -m src.etl.databento_download --config full -y

  # Custom tickers and dates
  python -m src.etl.databento_download --tickers AAPL MSFT \\
      --start 2024-01-08 --end 2024-01-13
        """,
    )

    parser.add_argument(
        "--config",
        choices=["full", "quick", "one_week"],
        default="full",
        help="Predefined config (default: full)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Override tickers (e.g., --tickers AAPL MSFT)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Override start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Override end date (YYYY-MM-DD, exclusive)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Databento API key (or set DATABENTO_API_KEY env var)",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate costs, don't download",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Override max total budget (USD)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

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

    print("\n" + "=" * 60)
    print("  DATABENTO MBP-10 DOWNLOADER")
    print("  Multi-Horizon OFI Research Project")
    print("=" * 60)
    print(f"  Config:   {args.config}")
    print(f"  Tickers:  {cfg.tickers}")
    print(f"  Range:    {cfg.start_date} → {cfg.end_date}")
    print(f"  Dataset:  {cfg.dataset}")
    print(f"  Schema:   {cfg.schema}")
    print(f"  Budget:   ${cfg.max_total_cost:.2f}")

    download_all(
        cfg,
        estimate_only=args.estimate_only,
        skip_confirmation=args.yes,
    )


if __name__ == "__main__":
    main()
