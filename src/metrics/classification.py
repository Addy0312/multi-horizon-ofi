"""
Classification metrics for multi-horizon LOB prediction.
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_name: str = "",
) -> Dict[str, float]:
    """
    Compute classification metrics for a single horizon.

    Parameters
    ----------
    y_true : np.ndarray of int, shape (N,)
    y_pred : np.ndarray of int, shape (N,)
    horizon_name : str
        Label prefix for the metrics dict keys.

    Returns
    -------
    dict with accuracy, f1_macro, f1_weighted, precision, recall.
    """
    prefix = f"{horizon_name}_" if horizon_name else ""
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def compute_all_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: List[int],
) -> Dict[str, float]:
    """
    Compute classification metrics for all horizons.

    Parameters
    ----------
    y_true : np.ndarray, shape (N, H) — ground truth per horizon
    y_pred : np.ndarray, shape (N, H) — predictions per horizon

    Returns
    -------
    Flat dict with prefixed metric names.
    """
    metrics = {}
    for i, h in enumerate(horizons):
        m = compute_classification_metrics(
            y_true[:, i], y_pred[:, i], horizon_name=f"h{h}"
        )
        metrics.update(m)
    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    target_names: List[str] | None = None,
) -> str:
    """Pretty-print sklearn classification report."""
    if target_names is None:
        target_names = ["DOWN", "STATIONARY", "UP"]
    header = f"\n{'='*50}\nHorizon k={horizon}\n{'='*50}"
    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    return header + "\n" + report


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return 3×3 confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
