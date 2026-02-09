"""
Feature normalization / scaling utilities.

Supports:
    - Z-score (mean=0, std=1) — default
    - Min-max [0, 1]
    - Robust (median / IQR) — less sensitive to outliers

All scalers are fitted on train data only to avoid look-ahead bias.
"""

import numpy as np
from typing import Tuple


class ZScoreScaler:
    """Standard (Z-score) normalisation fitted on training data."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ZScoreScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # prevent div-by-zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """Min-max scaling to [0, 1]."""

    def __init__(self):
        self.min_: np.ndarray | None = None
        self.range_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.min_ = X.min(axis=0)
        self.range_ = X.max(axis=0) - self.min_
        self.range_[self.range_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.range_ + self.min_


class RobustScaler:
    """Robust scaling using median and IQR (less sensitive to outliers)."""

    def __init__(self):
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "RobustScaler":
        self.median_ = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        self.iqr_ = q75 - q25
        self.iqr_[self.iqr_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.median_) / self.iqr_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.iqr_ + self.median_


def normalize_splits(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    method: str = "zscore",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Normalise train/val/test using only train statistics.

    Parameters
    ----------
    method : str
        One of 'zscore', 'minmax', 'robust'.

    Returns
    -------
    X_train_n, X_val_n, X_test_n, scaler
    """
    scalers = {
        "zscore": ZScoreScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    if method not in scalers:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(scalers.keys())}")

    scaler = scalers[method]()
    X_train_n = scaler.fit_transform(X_train)
    X_val_n = scaler.transform(X_val)
    X_test_n = scaler.transform(X_test)

    return X_train_n, X_val_n, X_test_n, scaler
