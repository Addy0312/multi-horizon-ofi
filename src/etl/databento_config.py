"""
Configuration for Databento data downloads.

Centralises all parameters for the multi-stock, multi-day download pipeline.
Edit this file to change tickers, date ranges, dataset, etc.

Usage:
    from src.etl.databento_config import DOWNLOAD_CONFIG
"""

from dataclasses import dataclass, field
from typing import List
from datetime import date


@dataclass
class DownloadConfig:
    """Configuration for a Databento download session."""

    # ── Authentication ────────────────────────────────────────────────
    # Set DATABENTO_API_KEY env var, or pass directly here.
    api_key: str | None = None  # None → reads DATABENTO_API_KEY env var

    # ── Dataset & Schema ──────────────────────────────────────────────
    dataset: str = "XNAS.ITCH"       # Nasdaq TotalView (same source as LOBSTER)
    schema: str = "mbp-10"           # Market By Price, 10 levels of depth
    stype_in: str = "raw_symbol"     # Symbol type for queries

    # ── Tickers ───────────────────────────────────────────────────────
    # Highly liquid US large-cap stocks.
    # These are ideal for OFI research: tight spreads, deep books, high activity.
    tickers: List[str] = field(default_factory=lambda: [
        "AAPL",    # Apple — most liquid US equity
        "MSFT",    # Microsoft — consistently deep book
        "AMZN",    # Amazon — wide spread, interesting OFI dynamics
        "GOOG",    # Alphabet — good mid-cap comparison
        "TSLA",    # Tesla — high volatility, tests model robustness
    ])

    # ── Date Range ────────────────────────────────────────────────────
    # 4 weeks of data: covers multiple volatility regimes, FOMC weeks,
    # OpEx, month-end rebalancing — all essential for a strong paper.
    #
    # Strategy:
    #   Week 1 (Jan 8-12, 2024):  Normal week
    #   Week 2 (Jan 16-19, 2024): Post-MLK, potential vol pickup
    #   Week 3 (Jan 22-26, 2024): Earnings season ramp-up
    #   Week 4 (Jan 29-Feb 2, 2024): FOMC week + mega-cap earnings
    #
    # This gives ~20 trading days × 5 stocks = 100 stock-days.
    start_date: str = "2024-01-08"
    end_date: str = "2024-02-03"     # exclusive end date

    # ── Trading Hours Filter (Eastern Time) ───────────────────────────
    # Regular trading hours: 09:30 - 16:00 ET
    # In seconds from midnight (matches LOBSTER convention)
    market_open_sec: int = 34200     # 09:30:00
    market_close_sec: int = 57600    # 16:00:00

    # ── File Paths ────────────────────────────────────────────────────
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    dbn_cache_dir: str = "data/dbn_cache"   # Cache raw DBN files to avoid re-download

    # ── Download Options ──────────────────────────────────────────────
    # Download one day at a time to keep memory manageable
    # and allow resumption if interrupted.
    chunk_by_day: bool = True

    # Maximum cost (USD) before aborting — safety guard
    max_cost_per_request: float = 25.0
    max_total_cost: float = 200.0

    # ── LOBSTER Compatibility ─────────────────────────────────────────
    # Number of LOB levels to keep in converted output
    n_levels: int = 10

    # LOBSTER price scale: prices × 10,000 (4 decimal places)
    # Databento prices are in fixed-point with 1e-9 scale
    lobster_price_scale: int = 10000


# ── Singleton config instance ─────────────────────────────────────────
DOWNLOAD_CONFIG = DownloadConfig()


# ── Alternate configs for quick testing ───────────────────────────────
QUICK_TEST_CONFIG = DownloadConfig(
    tickers=["AAPL"],
    start_date="2024-01-08",
    end_date="2024-01-09",   # Just 1 day
    max_cost_per_request=5.0,
    max_total_cost=10.0,
)

ONE_WEEK_CONFIG = DownloadConfig(
    tickers=["AAPL", "MSFT"],
    start_date="2024-01-08",
    end_date="2024-01-13",   # Mon-Fri (1 week)
    max_cost_per_request=15.0,
    max_total_cost=50.0,
)
