"""
Multi-horizon label generation for LOB mid-price prediction.

Supports both:
    - Regression targets (continuous price changes / returns)
    - Classification targets (UP / STATIONARY / DOWN)

Label conventions (classification):
    0 = DOWN
    1 = STATIONARY
    2 = UP

References:
    - Zhang, Zohren & Roberts (2019): smoothed labelling with
      k-event moving average and threshold α = mean ± 0.5σ
    - Zhang & Zohren (2021): multi-horizon extension
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


# ── Default horizons (events ahead) used throughout the project ──────────
DEFAULT_HORIZONS: List[int] = [10, 20, 50, 100]


def mid_price(df: pd.DataFrame) -> np.ndarray:
    """Return mid-price as a numpy array."""
    return ((df["ask_price_1"].values + df["bid_price_1"].values) / 2.0)


# ──────────────────────────────────────────────────────────────────────────
# Regression targets
# ──────────────────────────────────────────────────────────────────────────

def make_regression_labels(
    df: pd.DataFrame,
    horizons: List[int] | None = None,
) -> pd.DataFrame:
    """
    Create regression targets: future mid-price change (raw difference).

        y_t^{(k)} = m_{t+k} − m_t

    where m_t is the mid-price at event t.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed LOB dataframe.
    horizons : list of int
        Forecast horizons in number of events.

    Returns
    -------
    pd.DataFrame
        Columns: delta_mid_10, delta_mid_20, ...
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    m = mid_price(df)
    result = {}
    for k in horizons:
        shifted = np.empty_like(m)
        shifted[:] = np.nan
        if k < len(m):
            shifted[: len(m) - k] = m[k:]
        result[f"delta_mid_{k}"] = shifted - m
    return pd.DataFrame(result, index=df.index)


def make_return_labels(
    df: pd.DataFrame,
    horizons: List[int] | None = None,
) -> pd.DataFrame:
    """
    Create log-return regression targets:
        y_t^{(k)} = log(m_{t+k}) − log(m_t)
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    m = mid_price(df)
    log_m = np.log(m)
    result = {}
    for k in horizons:
        shifted = np.empty_like(log_m)
        shifted[:] = np.nan
        if k < len(log_m):
            shifted[: len(log_m) - k] = log_m[k:]
        result[f"log_return_{k}"] = shifted - log_m
    return pd.DataFrame(result, index=df.index)


# ──────────────────────────────────────────────────────────────────────────
# Classification targets
# ──────────────────────────────────────────────────────────────────────────

def _smoothed_mid(m: np.ndarray, k: int) -> np.ndarray:
    """
    Smoothed future mid-price: average of m_{t+1} .. m_{t+k}.

    Used by Zhang et al. (2019) and Zhang & Zohren (2021) to
    reduce noise in the label.
    """
    cumsum = np.cumsum(m)
    # avg of m[t+1:t+k+1]
    smoothed = np.full_like(m, np.nan)
    if k < len(m):
        # sum from t+1 to t+k
        end_sum = cumsum[k:]  # cumsum at t+k
        start_sum = cumsum[:-k]  # cumsum at t
        smoothed[: len(m) - k] = (end_sum - start_sum) / k
    return smoothed


def make_classification_labels(
    df: pd.DataFrame,
    horizons: List[int] | None = None,
    alpha: float = 0.5,
    use_smoothing: bool = True,
) -> pd.DataFrame:
    """
    Create 3-class classification labels (DOWN=0, STAT=1, UP=2).

    Method (Zhang et al. 2019):
        1. Compute smoothed future mid-price: m̄_{t,k} = mean(m_{t+1}..m_{t+k})
        2. Compute percentage change: l_t = (m̄_{t,k} − m_t) / m_t
        3. Threshold: σ = std(l), label = UP if l > α·σ, DOWN if l < −α·σ

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed LOB dataframe.
    horizons : list of int
        Forecast horizons.
    alpha : float
        Threshold multiplier for stationary band (default 0.5).
    use_smoothing : bool
        If True, use smoothed future mid (Zhang 2019). If False, use
        raw future mid-price at event t+k.

    Returns
    -------
    pd.DataFrame
        Columns: label_10, label_20, ... with values in {0, 1, 2}.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    m = mid_price(df)
    result = {}

    for k in horizons:
        if use_smoothing:
            future_m = _smoothed_mid(m, k)
        else:
            future_m = np.empty_like(m)
            future_m[:] = np.nan
            if k < len(m):
                future_m[: len(m) - k] = m[k:]

        pct_change = (future_m - m) / m

        # Compute threshold on non-NaN values
        valid = pct_change[~np.isnan(pct_change)]
        sigma = np.std(valid)
        threshold = alpha * sigma

        labels = np.full(len(m), np.nan)
        labels[pct_change > threshold] = 2    # UP
        labels[pct_change < -threshold] = 0   # DOWN
        mask = (~np.isnan(pct_change)) & np.isnan(labels)
        labels[mask] = 1                      # STATIONARY

        result[f"label_{k}"] = labels

    return pd.DataFrame(result, index=df.index)


# ──────────────────────────────────────────────────────────────────────────
# Convenience: build everything at once
# ──────────────────────────────────────────────────────────────────────────

def make_all_labels(
    df: pd.DataFrame,
    horizons: List[int] | None = None,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """
    Generate both regression and classification labels for all horizons.
    """
    reg = make_regression_labels(df, horizons)
    cls = make_classification_labels(df, horizons, alpha=alpha)
    return pd.concat([reg, cls], axis=1)


def get_class_weights(
    labels: np.ndarray,
) -> np.ndarray:
    """
    Compute inverse-frequency class weights for imbalanced classes.
    Useful for CrossEntropyLoss(weight=...).

    Returns
    -------
    np.ndarray of shape (3,)
        Weight for each class [DOWN, STAT, UP].
    """
    valid = labels[~np.isnan(labels)].astype(int)
    counts = np.bincount(valid, minlength=3).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = len(valid) / (3.0 * counts)
    return weights
