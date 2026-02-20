"""ETL pipeline: download, convert, and preprocess market data."""

from src.etl.preprocess import process_day

__all__ = ["process_day"]
