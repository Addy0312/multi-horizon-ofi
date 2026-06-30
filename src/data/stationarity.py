"""
Stationarity testing and transformation utilities for LOB feature engineering.

LOB mid-price is a near-random-walk (non-stationary), and raw price features
cause spurious correlations in machine learning models.  This module provides:

Testing
-------
    - Augmented Dickey-Fuller (ADF) test — H₀: unit root (non-stationary)
    - KPSS test — H₀: stationary (complementary to ADF)
    - Batch testing across all features with a summary DataFrame

Transformations
---------------
    - First differencing  (destroys all autocorrelation memory)
    - Log differencing    (for strictly positive price series)
    - Fractional differencing (FFD) — key innovation from López de Prado (2018):
      finds the minimal d ∈ (0,1) that achieves stationarity while preserving
      the maximum amount of memory / information in the series.

Why fractional differencing?
-----------------------------
Full (integer) differencing of price series destroys most of the predictive
memory in the data.  Fractional differencing with d < 1 achieves stationarity
while retaining long-range dependence, which is precisely what deep models
need to learn meaningful patterns from LOB data.

The ``find_min_d()`` function binary-searches for the minimum d that passes
an ADF stationarity test, giving models the best of both worlds: stationary
inputs + preserved memory.

References
----------
- Engle & Granger (1987): Co-integration and error correction.
- Kwiatkowski et al. (1992): Testing the null hypothesis of stationarity (KPSS).
- López de Prado (2018): Advances in Financial Machine Learning, ch. 5.
  "Stationary Features" — fractional differencing for financial time series.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────
# ADF test
# ──────────────────────────────────────────────────────────────────────────


def adf_test(
    series: Union[np.ndarray, pd.Series],
    maxlag: Optional[int] = None,
    regression: str = "c",
    autolag: str = "AIC",
    p_threshold: float = 0.05,
) -> Dict[str, object]:
    """
    Augmented Dickey-Fuller unit-root test.

    H₀: the series has a unit root (is non-stationary).
    Reject H₀ (p < threshold) → series is stationary.

    Parameters
    ----------
    series : array-like, shape (N,)
        The time series to test.  NaNs are dropped before testing.
    maxlag : int or None
        Maximum number of lags to include.  None = auto (default).
    regression : str
        ``'c'``   — constant only (default, suitable for most LOB features)
        ``'ct'``  — constant + trend
        ``'ctt'`` — constant + linear + quadratic trend
        ``'n'``   — no constant
    autolag : str
        Lag selection criterion: ``'AIC'`` (default), ``'BIC'``, ``'t-stat'``.
    p_threshold : float
        Significance level for the stationarity decision (default 0.05).

    Returns
    -------
    dict with keys:
        ``statistic``    : float — ADF test statistic
        ``p_value``      : float — MacKinnon p-value
        ``n_lags``       : int   — number of lags used
        ``n_obs``        : int   — number of observations used
        ``critical``     : dict  — critical values at 1%, 5%, 10%
        ``is_stationary``: bool  — True if p_value < p_threshold
        ``method``       : str   — 'ADF'
    """
    from statsmodels.tsa.stattools import adfuller

    x = np.asarray(series, dtype=np.float64)
    x = x[~np.isnan(x)]

    if len(x) < 20:
        warnings.warn(
            f"ADF test: only {len(x)} non-NaN observations — result may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    result = adfuller(x, maxlag=maxlag, regression=regression, autolag=autolag)
    stat, pval, n_lags, n_obs, critical_vals = result[0], result[1], result[2], result[3], result[4]

    return {
        "statistic": float(stat),
        "p_value": float(pval),
        "n_lags": int(n_lags),
        "n_obs": int(n_obs),
        "critical": {k: float(v) for k, v in critical_vals.items()},
        "is_stationary": bool(pval < p_threshold),
        "method": "ADF",
        "p_threshold": p_threshold,
    }


# ──────────────────────────────────────────────────────────────────────────
# KPSS test
# ──────────────────────────────────────────────────────────────────────────


def kpss_test(
    series: Union[np.ndarray, pd.Series],
    regression: str = "c",
    nlags: str = "auto",
    p_threshold: float = 0.05,
) -> Dict[str, object]:
    """
    KPSS (Kwiatkowski–Phillips–Schmidt–Shin) stationarity test.

    H₀: the series IS stationary (opposite of ADF).
    Reject H₀ (p < threshold) → series is non-stationary.

    Used together with ADF to distinguish:
        - ADF non-reject + KPSS non-reject → likely stationary
        - ADF reject + KPSS reject         → likely non-stationary
        - ADF reject + KPSS non-reject     → stationary confirmed
        - ADF non-reject + KPSS reject     → ambiguous (possibly fractionally integrated)

    Parameters
    ----------
    series : array-like, shape (N,)
    regression : str
        ``'c'``  — level stationarity (default)
        ``'ct'`` — trend stationarity
    nlags : str or int
        Lag truncation for the long-run variance estimate.
        ``'auto'`` uses Newey-West automatic bandwidth.
    p_threshold : float
        Significance level (default 0.05).

    Returns
    -------
    dict with keys:
        ``statistic``    : float
        ``p_value``      : float  (note: statsmodels interpolates; may be 0.01 or 0.10)
        ``n_lags``       : int
        ``critical``     : dict
        ``is_stationary``: bool  — True if we FAIL to reject H₀ (p > threshold)
        ``method``       : str   — 'KPSS'
    """
    from statsmodels.tsa.stattools import kpss

    x = np.asarray(series, dtype=np.float64)
    x = x[~np.isnan(x)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS emits interpolation warnings
        result = kpss(x, regression=regression, nlags=nlags)

    stat, pval, n_lags, critical_vals = result[0], result[1], result[2], result[3]

    return {
        "statistic": float(stat),
        "p_value": float(pval),
        "n_lags": int(n_lags),
        "critical": {k: float(v) for k, v in critical_vals.items()},
        "is_stationary": bool(pval > p_threshold),  # KPSS: fail-to-reject H₀ → stationary
        "method": "KPSS",
        "p_threshold": p_threshold,
    }


# ──────────────────────────────────────────────────────────────────────────
# Batch testing
# ──────────────────────────────────────────────────────────────────────────


def test_feature_stationarity(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    run_kpss: bool = True,
    adf_p_threshold: float = 0.05,
    kpss_p_threshold: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run ADF (and optionally KPSS) on every feature column of X.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
    feature_names : list of str or None
        Column names for the summary.  If None, uses ``f0, f1, ...``.
    run_kpss : bool
        Whether to also run the KPSS test (slower, more informative).
    adf_p_threshold : float
    kpss_p_threshold : float
    verbose : bool
        If True, prints a summary after testing.

    Returns
    -------
    pd.DataFrame
        One row per feature with columns:
        ``feature``, ``adf_stat``, ``adf_pval``, ``adf_stationary``,
        [``kpss_stat``, ``kpss_pval``, ``kpss_stationary``],
        ``verdict``  (one of: ``'stationary'``, ``'non-stationary'``, ``'ambiguous'``).
    """
    X = np.asarray(X, dtype=np.float64)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    rows = []
    for j, name in enumerate(feature_names):
        col = X[:, j]
        row: Dict[str, object] = {"feature": name}

        adf = adf_test(col, p_threshold=adf_p_threshold)
        row["adf_stat"] = adf["statistic"]
        row["adf_pval"] = adf["p_value"]
        row["adf_stationary"] = adf["is_stationary"]

        if run_kpss:
            try:
                ks = kpss_test(col, p_threshold=kpss_p_threshold)
                row["kpss_stat"] = ks["statistic"]
                row["kpss_pval"] = ks["p_value"]
                row["kpss_stationary"] = ks["is_stationary"]

                # Combined verdict
                if adf["is_stationary"] and ks["is_stationary"]:
                    row["verdict"] = "stationary"
                elif not adf["is_stationary"] and not ks["is_stationary"]:
                    row["verdict"] = "non-stationary"
                else:
                    row["verdict"] = "ambiguous"
            except Exception as e:
                row["kpss_stat"] = float("nan")
                row["kpss_pval"] = float("nan")
                row["kpss_stationary"] = None
                row["verdict"] = "adf-only"
                warnings.warn(f"KPSS failed for '{name}': {e}", UserWarning, stacklevel=2)
        else:
            row["verdict"] = "stationary" if adf["is_stationary"] else "non-stationary"

        rows.append(row)

    df = pd.DataFrame(rows)

    if verbose:
        n_stat = (df["verdict"] == "stationary").sum()
        n_nonstat = (df["verdict"] == "non-stationary").sum()
        n_ambig = df["verdict"].isin(["ambiguous", "adf-only"]).sum()
        print(f"\n{'='*60}")
        print(f"  Stationarity Test Summary  ({n_features} features)")
        print(f"{'='*60}")
        print(f"  Stationary    : {n_stat}")
        print(f"  Non-stationary: {n_nonstat}")
        print(f"  Ambiguous     : {n_ambig}")
        print(f"{'='*60}")
        non_stat = df[df["verdict"] == "non-stationary"]["feature"].tolist()
        if non_stat:
            print(f"\n  Non-stationary features ({len(non_stat)}):")
            for f in non_stat[:20]:
                print(f"    - {f}")
            if len(non_stat) > 20:
                print(f"    ... and {len(non_stat) - 20} more")
        print()

    return df


# ──────────────────────────────────────────────────────────────────────────
# Transformations
# ──────────────────────────────────────────────────────────────────────────


def difference_series(
    x: np.ndarray,
    d: int = 1,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Apply integer differencing of order d.

    x_diff[t] = x[t] − x[t−1]  (for d=1)

    Parameters
    ----------
    x : np.ndarray, shape (N,) or (N, F)
    d : int
        Order of differencing (default 1).
    fill_value : float
        Value used for the first d rows (which have no prior observation).

    Returns
    -------
    np.ndarray, same shape as x.
    """
    x = np.asarray(x, dtype=np.float64)
    result = x.copy()
    for _ in range(d):
        result = np.diff(result, n=1, axis=0, prepend=result[:1] * np.nan)
    result[:d] = fill_value
    return result.astype(np.float32)


def log_difference_series(
    x: np.ndarray,
    fill_value: float = 0.0,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Apply log-differencing (log returns).

    x_logdiff[t] = log(x[t] + eps) − log(x[t-1] + eps)

    Appropriate for strictly positive price series.  Produces log returns
    which are approximately stationary for most financial instruments.

    Parameters
    ----------
    x : np.ndarray, shape (N,) or (N, F)
    fill_value : float
        Value for the first row (no prior).
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    np.ndarray, same shape as x.
    """
    x = np.asarray(x, dtype=np.float64)
    log_x = np.log(np.abs(x) + eps)
    result = np.diff(log_x, n=1, axis=0, prepend=log_x[:1] * np.nan)
    result[0] = fill_value
    return result.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────
# Fractional differencing (FFD — Fixed-Width Window)
# ──────────────────────────────────────────────────────────────────────────


def _ffd_weights(d: float, size: int, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute FFD (Fixed-width fractional differencing) weights.

    w_k = Π_{j=1}^{k} (j − 1 − d) / j,  w_0 = 1

    Parameters
    ----------
    d : float
        Fractional differencing order (0 < d < 1 for partial memory retention).
    size : int
        Maximum window size.
    threshold : float
        Truncate weights below this absolute value to save computation.

    Returns
    -------
    np.ndarray, shape (K,) — weights in convolution order (most recent first).
    """
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    # Reverse so index 0 = most recent
    return np.array(w[::-1], dtype=np.float64)


def fractional_difference(
    x: np.ndarray,
    d: float,
    threshold: float = 1e-3,
    fill_nan: bool = True,
) -> np.ndarray:
    """
    Apply fractional differencing of order d using the FFD method.

    FFD applies a fixed-width window equal to the number of weights above
    ``threshold``, so the window size is consistent across the series
    (unlike the expanding-window variant).

    For d = 0   → original series (no transformation)
    For d = 0.5 → intermediate: some memory preserved, partially stationary
    For d = 1   → equivalent to first differencing (all memory lost)

    Parameters
    ----------
    x : np.ndarray, shape (N,) or (N, F)
        Input series.  For 2-D input each column is differenced independently.
    d : float
        Fractional differencing order, typically in [0, 1].
    threshold : float
        Weight truncation threshold (controls the effective window size).
    fill_nan : bool
        If True, the first (window−1) rows are filled with NaN (they are
        undefined due to insufficient history).  If False, filled with 0.

    Returns
    -------
    np.ndarray, same shape as x, dtype float32.

    References
    ----------
    López de Prado (2018): Advances in Financial Machine Learning, ch. 5.
    """
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]

    n, n_features = x.shape
    w = _ffd_weights(d, n, threshold)
    window = len(w)

    out = np.full((n, n_features), np.nan, dtype=np.float64)

    for j in range(n_features):
        col = x[:, j]
        for t in range(window - 1, n):
            segment = col[t - window + 1 : t + 1]
            if np.any(np.isnan(segment)):
                continue
            out[t, j] = float(np.dot(w, segment))

    if not fill_nan:
        out = np.where(np.isnan(out), 0.0, out)

    if squeeze:
        out = out[:, 0]

    return out.astype(np.float32)


def find_min_d(
    x: np.ndarray,
    max_d: float = 1.0,
    step: float = 0.1,
    refine_step: float = 0.01,
    target_pvalue: float = 0.05,
    threshold: float = 1e-3,
    verbose: bool = True,
) -> float:
    """
    Binary-search for the minimum fractional differencing order d that
    achieves stationarity (ADF p-value < target_pvalue).

    The key insight from López de Prado (2018): we want the *minimum* d
    that makes the series stationary, because higher d destroys more
    of the autocorrelation structure (memory) that ML models rely on.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        A single time series (1-D).
    max_d : float
        Maximum d to consider (default 1.0).
    step : float
        Coarse search step size.
    refine_step : float
        Fine-grained search step after coarse search.
    target_pvalue : float
        ADF p-value threshold for declaring stationarity.
    threshold : float
        FFD weight truncation threshold.
    verbose : bool

    Returns
    -------
    float
        Minimum d that achieves stationarity.  Returns max_d if no
        stationary d is found within the search range.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[~np.isnan(x)]

    if verbose:
        print(f"  Searching for min d in [0.0, {max_d}] (coarse step={step}, fine step={refine_step}) ...")

    # --- Coarse pass ---
    coarse_stationary_d = None
    d = step
    while d <= max_d + 1e-9:
        x_d = fractional_difference(x, d, threshold=threshold, fill_nan=True)
        x_d_clean = x_d[~np.isnan(x_d)]
        if len(x_d_clean) < 20:
            d += step
            continue
        try:
            result = adf_test(x_d_clean, p_threshold=target_pvalue)
            if result["is_stationary"]:
                coarse_stationary_d = d
                break
        except Exception:
            pass
        d += step
        d = round(d, 6)

    if coarse_stationary_d is None:
        if verbose:
            print(f"  No stationary d found up to {max_d}. Returning {max_d}.")
        return max_d

    # --- Fine pass (refine around coarse_stationary_d) ---
    start_fine = max(0.0, coarse_stationary_d - step + refine_step)
    min_d_found = coarse_stationary_d

    d = start_fine
    while d < coarse_stationary_d + 1e-9:
        x_d = fractional_difference(x, d, threshold=threshold, fill_nan=True)
        x_d_clean = x_d[~np.isnan(x_d)]
        if len(x_d_clean) < 20:
            d += refine_step
            continue
        try:
            result = adf_test(x_d_clean, p_threshold=target_pvalue)
            if result["is_stationary"]:
                min_d_found = d
                break
        except Exception:
            pass
        d += refine_step
        d = round(d, 6)

    if verbose:
        print(f"  Minimum d for stationarity: {min_d_found:.3f}")

    return float(min_d_found)


# ──────────────────────────────────────────────────────────────────────────
# High-level: make features stationary
# ──────────────────────────────────────────────────────────────────────────


def make_stationary_features(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    method: str = "auto",
    d_fixed: float = 0.5,
    adf_p_threshold: float = 0.05,
    ffd_threshold: float = 1e-3,
    log_diff_features: Optional[List[int]] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Transform features to achieve stationarity.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
        Feature matrix (train split only for fitting; then apply the same
        transformation to val/test).
    feature_names : list of str or None
    method : str
        Transformation method:
        ``'diff'``       — First differencing (all features).
        ``'log_diff'``   — Log-differencing (all features; requires X > 0).
        ``'frac_diff'``  — Fractional differencing with fixed d=d_fixed.
        ``'frac_min_d'`` — Fractional differencing, find minimum d per feature
                           via ADF (slower but preserves maximum memory).
        ``'auto'``       — Test stationarity first; only transform non-stationary
                           features using fractional differencing with min-d search.
    d_fixed : float
        d value used when method='frac_diff' (default 0.5).
    adf_p_threshold : float
        ADF significance level.
    ffd_threshold : float
        FFD weight truncation threshold.
    log_diff_features : list of int or None
        Column indices for which to apply log-differencing instead of the
        default method.  Useful for price columns.
    verbose : bool

    Returns
    -------
    X_stationary : np.ndarray, shape (N, F), dtype float32
        Transformed feature matrix.
    metadata : dict
        ``'method'``: str
        ``'d_per_feature'``: list of float (d used per column; 0 = untransformed)
        ``'transformed_indices'``: list of int
        ``'stationarity_df'``: pd.DataFrame (if method='auto', else None)
    """
    X = np.asarray(X, dtype=np.float64)
    n, n_features = X.shape

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    log_diff_set = set(log_diff_features or [])
    X_out = X.copy()
    d_per_feature = [0.0] * n_features
    transformed_indices: List[int] = []
    stationarity_df = None

    if method == "diff":
        if verbose:
            print("Applying first differencing to all features ...")
        for j in range(n_features):
            if j in log_diff_set:
                X_out[:, j] = log_difference_series(X[:, j])
            else:
                X_out[:, j] = difference_series(X[:, j], d=1)
            d_per_feature[j] = 1.0
            transformed_indices.append(j)

    elif method == "log_diff":
        if verbose:
            print("Applying log-differencing to all features ...")
        for j in range(n_features):
            X_out[:, j] = log_difference_series(X[:, j])
            d_per_feature[j] = 1.0
            transformed_indices.append(j)

    elif method == "frac_diff":
        if verbose:
            print(f"Applying fractional differencing (d={d_fixed}) to all features ...")
        for j in range(n_features):
            if j in log_diff_set:
                X_out[:, j] = log_difference_series(X[:, j])
                d_per_feature[j] = 1.0
            else:
                X_out[:, j] = fractional_difference(X[:, j], d=d_fixed,
                                                     threshold=ffd_threshold)
                d_per_feature[j] = d_fixed
            transformed_indices.append(j)

    elif method == "frac_min_d":
        if verbose:
            print("Finding minimum d per feature (fractional differencing) ...")
        for j, name in enumerate(feature_names):
            if j in log_diff_set:
                X_out[:, j] = log_difference_series(X[:, j])
                d_per_feature[j] = 1.0
                transformed_indices.append(j)
                continue
            if verbose:
                print(f"  [{j+1}/{n_features}] {name}")
            d_min = find_min_d(
                X[:, j],
                target_pvalue=adf_p_threshold,
                threshold=ffd_threshold,
                verbose=verbose,
            )
            if d_min > 0:
                X_out[:, j] = fractional_difference(X[:, j], d=d_min,
                                                     threshold=ffd_threshold)
                d_per_feature[j] = d_min
                transformed_indices.append(j)

    elif method == "auto":
        if verbose:
            print("Running stationarity tests to identify non-stationary features ...")
        stationarity_df = test_feature_stationarity(
            X, feature_names=feature_names,
            run_kpss=True,
            adf_p_threshold=adf_p_threshold,
            verbose=verbose,
        )
        non_stat_mask = stationarity_df["verdict"] == "non-stationary"
        non_stat_indices = list(stationarity_df[non_stat_mask].index)

        if verbose:
            print(f"\nApplying fractional differencing (min-d search) to "
                  f"{len(non_stat_indices)} non-stationary features ...")

        for j in non_stat_indices:
            name = feature_names[j]
            if j in log_diff_set:
                X_out[:, j] = log_difference_series(X[:, j])
                d_per_feature[j] = 1.0
                transformed_indices.append(j)
                continue
            if verbose:
                print(f"  [{j+1}/{n_features}] {name}")
            d_min = find_min_d(
                X[:, j],
                target_pvalue=adf_p_threshold,
                threshold=ffd_threshold,
                verbose=verbose,
            )
            if d_min > 0:
                X_out[:, j] = fractional_difference(X[:, j], d=d_min,
                                                     threshold=ffd_threshold)
                d_per_feature[j] = d_min
                transformed_indices.append(j)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: "
            "'diff', 'log_diff', 'frac_diff', 'frac_min_d', 'auto'"
        )

    # Replace NaN from differencing with 0 in final output
    X_out = np.where(np.isnan(X_out), 0.0, X_out)

    if verbose:
        n_transformed = len(set(transformed_indices))
        print(f"\nStationarity transformation complete: "
              f"{n_transformed}/{n_features} features transformed.")

    metadata = {
        "method": method,
        "d_per_feature": d_per_feature,
        "transformed_indices": list(set(transformed_indices)),
        "stationarity_df": stationarity_df,
    }
    return X_out.astype(np.float32), metadata


def apply_stationarity_transform(
    X: np.ndarray,
    metadata: Dict[str, object],
    ffd_threshold: float = 1e-3,
) -> np.ndarray:
    """
    Apply a previously fitted stationarity transformation to new data
    (val or test split) using the d values discovered on the training set.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
    metadata : dict
        Output ``metadata`` from :func:`make_stationary_features`.
    ffd_threshold : float

    Returns
    -------
    np.ndarray, shape (N, F), dtype float32
    """
    X = np.asarray(X, dtype=np.float64)
    d_per_feature: List[float] = metadata["d_per_feature"]
    X_out = X.copy()

    for j, d in enumerate(d_per_feature):
        if d == 0.0:
            continue
        elif d == 1.0:
            # First differencing or log-differencing — use simple diff
            X_out[:, j] = difference_series(X[:, j], d=1)
        else:
            X_out[:, j] = fractional_difference(X[:, j], d=d,
                                                 threshold=ffd_threshold)

    X_out = np.where(np.isnan(X_out), 0.0, X_out)
    return X_out.astype(np.float32)
