"""
Regression metrics for multi-horizon LOB prediction.
"""

import numpy as np
from typing import Dict, List


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_name: str = "",
) -> Dict[str, float]:
    """
    Compute regression metrics for a single horizon.

    Returns MSE, RMSE, MAE, R².
    """
    prefix = f"{horizon_name}_" if horizon_name else ""
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {
        f"{prefix}mse": float(mse),
        f"{prefix}rmse": float(rmse),
        f"{prefix}mae": float(mae),
        f"{prefix}r2": float(r2),
    }


def compute_all_horizon_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: List[int],
) -> Dict[str, float]:
    """
    Compute regression metrics for all horizons.

    Parameters
    ----------
    y_true : np.ndarray, shape (N, H)
    y_pred : np.ndarray, shape (N, H)

    Returns
    -------
    Flat dict with prefixed metric names.
    """
    metrics = {}
    for i, h in enumerate(horizons):
        m = compute_regression_metrics(
            y_true[:, i], y_pred[:, i], horizon_name=f"h{h}"
        )
        metrics.update(m)
    return metrics


def information_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Information Coefficient (IC) = Pearson correlation between
    predicted and actual returns.

    Higher is better. Typical values in finance: 0.02–0.10.
    """
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rank_information_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Rank IC = Spearman correlation between predicted and actual returns.
    More robust to outliers than Pearson IC.
    """
    from scipy.stats import spearmanr
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr) if not np.isnan(corr) else 0.0
