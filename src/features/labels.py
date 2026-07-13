from typing import List, Optional
import numpy as np
import pandas as pd

DEFAULT_HORIZONS: List[int] = [10, 20, 50, 100]

def _mid_price_array(df: pd.DataFrame, dtype=np.float64) -> np.ndarray:
    bp = df["bid_price_1"].to_numpy(dtype=dtype, copy=False)
    ap = df["ask_price_1"].to_numpy(dtype=dtype, copy=False)
    return (ap + bp) * 0.5

def _smoothed_mid(m: np.ndarray, k: int) -> np.ndarray:
    n = len(m)
    out = np.full(n, np.nan, dtype=m.dtype)
    if k >= n:
        return out
    cumsum = np.cumsum(m, dtype=m.dtype)
    out[: n - k] = (cumsum[k:] - cumsum[:-k]) / k
    return out

def make_fixed_threshold_classification_labels(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    alpha: float = 0.002,
    use_smoothing: bool = True,
    dtype=np.float64,
) -> pd.DataFrame:
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    m = _mid_price_array(df, dtype)
    n = len(m)
    out = np.empty((n, len(horizons)), dtype=dtype)
    for i, k in enumerate(horizons):
        future_m = _smoothed_mid(m, k) if use_smoothing else np.full(n, np.nan, dtype=dtype)
        if not use_smoothing and k < n:
            future_m[: n - k] = m[k:]
        pct_change = (future_m - m) / m
        labels = np.full(n, np.nan, dtype=dtype)
        labels[pct_change > alpha] = 2
        labels[pct_change < -alpha] = 0
        mid_mask = (~np.isnan(pct_change)) & (np.isnan(labels))
        labels[mid_mask] = 1
        out[:, i] = labels
    cols = [f"label_{k}" for k in horizons]
    return pd.DataFrame(out, index=df.index, columns=cols)
