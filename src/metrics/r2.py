"""
R² (coefficient of determination) metrics for multi-horizon LOB prediction.

Provides:
    - Standard R²
    - Adjusted R² (penalises for number of features)
    - Out-of-sample R² (R²_oos) — appropriate for time-series evaluation
      where the naive benchmark is the historical mean (or a random walk)
    - Multi-horizon helpers and a formatted summary table

Notes
-----
Standard in-sample R² can be misleadingly high in time-series settings
because the model can "remember" trends.  Out-of-sample R² (R²_oos) is a
much stricter metric: a positive R²_oos means the model beats the
historical-mean benchmark out-of-sample, which is the relevant bar for
LOB mid-price forecasting.

References
----------
- Campbell & Thompson (2008): Predicting excess stock returns out of sample:
  Can anything beat the historical average?
- Kling & Mose (2009): Out-of-sample performance of discrete-time spot
  interest rate models.
- López de Prado (2018): Advances in Financial Machine Learning, ch. 5
  (fractional differencing & information content).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────
# Core scalar metrics
# ──────────────────────────────────────────────────────────────────────────


def compute_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Standard coefficient of determination R².

    R² = 1 − SS_res / SS_tot

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
    y_pred : np.ndarray, shape (N,)

    Returns
    -------
    float
        R² score.  1.0 = perfect prediction; 0.0 = same as mean baseline;
        negative = worse than mean baseline.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot < 1e-12:
        # Degenerate case: all targets are identical
        return 1.0 if ss_res < 1e-12 else 0.0

    return float(1.0 - ss_res / ss_tot)


def compute_adjusted_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
) -> float:
    """
    Adjusted R² — penalises R² for the number of predictors.

    R²_adj = 1 − (1 − R²) * (N − 1) / (N − p − 1)

    where N is the number of observations and p is the number of features.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
    y_pred : np.ndarray, shape (N,)
    n_features : int
        Number of predictor features used by the model.

    Returns
    -------
    float
        Adjusted R².  Can be lower than R² when adding weak features.
    """
    n = len(y_true)
    if n <= n_features + 1:
        # Cannot compute — not enough degrees of freedom
        return float("nan")

    r2 = compute_r2(y_true, y_pred)
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1)
    return float(adj)


def compute_oos_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_benchmark: Optional[np.ndarray] = None,
) -> float:
    """
    Out-of-sample R² (R²_oos).

    R²_oos = 1 − MSPE_model / MSPE_benchmark

    The default benchmark is the historical mean of y_true (i.e., always
    predicting the mean — the toughest naive baseline for zero-mean returns).

    A positive R²_oos means the model beats the benchmark out-of-sample.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Actual values on the *test* set.
    y_pred : np.ndarray, shape (N,)
        Model predictions on the *test* set.
    y_benchmark : np.ndarray or None, shape (N,)
        Benchmark predictions.  If None, uses np.mean(y_true) as the
        prevailing-mean benchmark (Campbell & Thompson 2008).

    Returns
    -------
    float
        R²_oos.  Positive → model beats benchmark; 0 → ties benchmark;
        negative → model worse than benchmark.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_benchmark is None:
        y_benchmark = np.full_like(y_true, np.mean(y_true))
    else:
        y_benchmark = np.asarray(y_benchmark, dtype=np.float64)

    mspe_model = np.mean((y_true - y_pred) ** 2)
    mspe_bench = np.mean((y_true - y_benchmark) ** 2)

    if mspe_bench < 1e-16:
        return 1.0 if mspe_model < 1e-16 else float("-inf")

    return float(1.0 - mspe_model / mspe_bench)


# ──────────────────────────────────────────────────────────────────────────
# Multi-horizon helpers
# ──────────────────────────────────────────────────────────────────────────


def compute_horizon_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_name: str = "",
    n_features: Optional[int] = None,
    compute_oos: bool = True,
    y_benchmark: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute R², adjusted R², and R²_oos for a single horizon.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
    y_pred : np.ndarray, shape (N,)
    horizon_name : str
        Prefix for metric keys (e.g., "h10").
    n_features : int or None
        Number of features used.  Required for adjusted R².
    compute_oos : bool
        Whether to compute out-of-sample R².
    y_benchmark : np.ndarray or None
        Custom benchmark for R²_oos; defaults to mean(y_true).

    Returns
    -------
    dict
        Keys: ``{prefix}_r2``, ``{prefix}_r2_adj`` (if n_features given),
        ``{prefix}_r2_oos`` (if compute_oos).
    """
    prefix = f"{horizon_name}_" if horizon_name else ""
    metrics: Dict[str, float] = {}

    metrics[f"{prefix}r2"] = compute_r2(y_true, y_pred)

    if n_features is not None:
        metrics[f"{prefix}r2_adj"] = compute_adjusted_r2(y_true, y_pred, n_features)

    if compute_oos:
        metrics[f"{prefix}r2_oos"] = compute_oos_r2(y_true, y_pred, y_benchmark)

    return metrics


def compute_all_horizon_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: List[int],
    n_features: Optional[int] = None,
    compute_oos: bool = True,
) -> Dict[str, float]:
    """
    Compute R² metrics for all horizons.

    Parameters
    ----------
    y_true : np.ndarray, shape (N, H)
        Ground truth regression targets.
    y_pred : np.ndarray, shape (N, H)
        Model predictions.
    horizons : list of int
        Horizon labels (used for key prefixes, e.g., [10, 20, 50, 100]).
    n_features : int or None
        Number of predictor features (for adjusted R²).
    compute_oos : bool
        Whether to compute out-of-sample R².

    Returns
    -------
    dict
        Flat dict with prefixed metric names, e.g. ``h10_r2``, ``h10_r2_oos``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    all_metrics: Dict[str, float] = {}
    for i, h in enumerate(horizons):
        m = compute_horizon_r2(
            y_true[:, i],
            y_pred[:, i],
            horizon_name=f"h{h}",
            n_features=n_features,
            compute_oos=compute_oos,
        )
        all_metrics.update(m)

    return all_metrics


# ──────────────────────────────────────────────────────────────────────────
# Formatted summary
# ──────────────────────────────────────────────────────────────────────────


def r2_summary_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: List[int],
    n_features: Optional[int] = None,
    label: str = "R² Summary",
) -> pd.DataFrame:
    """
    Produce a pretty DataFrame summary of R² metrics across horizons.

    Parameters
    ----------
    y_true : np.ndarray, shape (N, H)
    y_pred : np.ndarray, shape (N, H)
    horizons : list of int
    n_features : int or None
    label : str
        Title printed above the table.

    Returns
    -------
    pd.DataFrame
        Columns: horizon, r2, r2_adj (if n_features), r2_oos.
    """
    metrics = compute_all_horizon_r2(
        y_true, y_pred, horizons, n_features=n_features, compute_oos=True
    )

    rows = []
    for h in horizons:
        row: Dict[str, object] = {"horizon": h}
        row["R²"] = metrics.get(f"h{h}_r2", float("nan"))
        if n_features is not None:
            row["R²_adj"] = metrics.get(f"h{h}_r2_adj", float("nan"))
        row["R²_oos"] = metrics.get(f"h{h}_r2_oos", float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("horizon")
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"{'='*50}\n")
    return df


def print_r2_table(
    metrics: Dict[str, float],
    horizons: List[int],
) -> None:
    """
    Pretty-print R² metrics from a flat metrics dict.

    Parameters
    ----------
    metrics : dict
        Output from :func:`compute_all_horizon_r2`.
    horizons : list of int
    """
    header = f"\n{'='*55}\n{'Horizon':>10}  {'R²':>10}  {'R²_adj':>10}  {'R²_oos':>10}\n{'='*55}"
    print(header)
    for h in horizons:
        r2 = metrics.get(f"h{h}_r2", float("nan"))
        r2_adj = metrics.get(f"h{h}_r2_adj", float("nan"))
        r2_oos = metrics.get(f"h{h}_r2_oos", float("nan"))
        r2_adj_str = f"{r2_adj:.4f}" if not np.isnan(r2_adj) else "  n/a  "
        print(f"  k={h:>5}    {r2:>8.4f}    {r2_adj_str:>8}    {r2_oos:>8.4f}")
    print("=" * 55)


# ──────────────────────────────────────────────────────────────────────────
# Convenience: compute from flat arrays (e.g., linear model output)
# ──────────────────────────────────────────────────────────────────────────


def evaluate_r2_flat(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: List[int],
    n_features: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate R² for multi-horizon predictions and optionally print a table.

    Parameters
    ----------
    y_true : np.ndarray, shape (N, H) or (N,)
    y_pred : np.ndarray, shape (N, H) or (N,)
    horizons : list of int
    n_features : int or None
    verbose : bool
        If True, print the summary table.

    Returns
    -------
    dict
        All R² metrics keyed by horizon prefix.
    """
    metrics = compute_all_horizon_r2(
        y_true, y_pred, horizons, n_features=n_features, compute_oos=True
    )
    if verbose:
        print_r2_table(metrics, horizons)
    return metrics
