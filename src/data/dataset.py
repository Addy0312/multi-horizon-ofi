"""
PyTorch Dataset for LOB / OFI data with sliding-window sequences.

Handles:
    - Loading preprocessed parquet → feature engineering → labels
    - Sliding window creation for sequential models
    - Train / val / test temporal split (no future leakage)
    - Both classification and regression targets
"""

import os
import glob
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.features.ofi import compute_multi_level_ofi
from src.features.microstructure import compute_all_features, compute_mid_price
from src.features.labels import (
    make_regression_labels,
    make_classification_labels,
    DEFAULT_HORIZONS,
)


# ──────────────────────────────────────────────────────────────────────────
# Feature builder: parquet → numpy arrays
# ──────────────────────────────────────────────────────────────────────────

def build_features_and_labels(
    parquet_path: str,
    ofi_levels: int = 5,
    horizons: List[int] | None = None,
    alpha: float = 0.5,
    include_raw_lob: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load one day's parquet and return feature matrix + label arrays.

    Parameters
    ----------
    parquet_path : str
        Path to preprocessed parquet file.
    ofi_levels : int
        Number of LOB levels for OFI computation.
    horizons : list of int
        Prediction horizons (events ahead).
    alpha : float
        Classification threshold multiplier.
    include_raw_lob : bool
        Whether to include raw LOB prices/sizes as features.

    Returns
    -------
    X : np.ndarray, shape (N, F)
        Feature matrix.
    y_reg : np.ndarray, shape (N, H)
        Regression targets (delta mid-price) for each horizon.
    y_cls : np.ndarray, shape (N, H)
        Classification targets (0/1/2) for each horizon.
    feature_names : list of str
        Column names for X.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    df = pd.read_parquet(parquet_path)

    # --- Features ---
    ofi_df = compute_multi_level_ofi(df, max_level=ofi_levels)
    micro_df = compute_all_features(df, max_level=ofi_levels)

    feature_parts = [ofi_df, micro_df]

    if include_raw_lob:
        lob_cols = []
        for lvl in range(1, ofi_levels + 1):
            lob_cols.extend([
                f"ask_price_{lvl}", f"ask_size_{lvl}",
                f"bid_price_{lvl}", f"bid_size_{lvl}",
            ])
        feature_parts.append(df[lob_cols])

    features = pd.concat(feature_parts, axis=1)
    feature_names = list(features.columns)

    # --- Labels ---
    reg_df = make_regression_labels(df, horizons)
    cls_df = make_classification_labels(df, horizons, alpha=alpha)

    # --- Drop rows with NaN labels (tail of the day) ---
    max_horizon = max(horizons)
    valid_end = len(df) - max_horizon
    if valid_end <= 0:
        raise ValueError(
            f"Day has {len(df)} events but max horizon is {max_horizon}"
        )

    X = features.values[:valid_end].astype(np.float32)
    y_reg = reg_df.values[:valid_end].astype(np.float32)
    y_cls = cls_df.values[:valid_end].astype(np.float32)

    return X, y_reg, y_cls, feature_names


# ──────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────

class LOBDataset(Dataset):
    """
    Sliding-window dataset over LOB features.

    Each sample is:
        X[i] : (seq_len, n_features) — lookback window
        y_reg[i] : (n_horizons,)     — regression targets at last event
        y_cls[i] : (n_horizons,)     — classification targets at last event
    """

    def __init__(
        self,
        X: np.ndarray,
        y_reg: np.ndarray,
        y_cls: np.ndarray,
        seq_len: int = 100,
    ):
        """
        Parameters
        ----------
        X : np.ndarray, shape (N, F)
        y_reg : np.ndarray, shape (N, H)
        y_cls : np.ndarray, shape (N, H)
        seq_len : int
            Lookback window size.
        """
        self.X = X
        self.y_reg = y_reg
        self.y_cls = y_cls
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.X[idx : idx + self.seq_len]
        # Labels correspond to the *last* event in the window
        target_idx = idx + self.seq_len - 1
        yr = self.y_reg[target_idx]
        yc = self.y_cls[target_idx]

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y_reg": torch.tensor(yr, dtype=torch.float32),
            "y_cls": torch.tensor(yc, dtype=torch.long),
        }


class FlatDataset(Dataset):
    """
    Non-sequential dataset for linear / MLP baselines.
    Each sample is a single row of features (no lookback).
    """

    def __init__(self, X: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray):
        self.X = X
        self.y_reg = y_reg
        self.y_cls = y_cls

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.tensor(self.X[idx], dtype=torch.float32),
            "y_reg": torch.tensor(self.y_reg[idx], dtype=torch.float32),
            "y_cls": torch.tensor(self.y_cls[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────
# Temporal train / val / test split
# ──────────────────────────────────────────────────────────────────────────

def temporal_split(
    X: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Temporal split: first train_frac → train, next val_frac → val, rest → test.
    No shuffling — preserves time order to prevent look-ahead bias.
    """
    n = len(X)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))

    train = (X[:t1], y_reg[:t1], y_cls[:t1])
    val = (X[t1:t2], y_reg[t1:t2], y_cls[t1:t2])
    test = (X[t2:], y_reg[t2:], y_cls[t2:])
    return train, val, test


# ──────────────────────────────────────────────────────────────────────────
# High-level loader factory
# ──────────────────────────────────────────────────────────────────────────

def load_all_days(
    processed_dir: str,
    ticker: str,
    ofi_levels: int = 5,
    horizons: List[int] | None = None,
    alpha: float = 0.5,
    include_raw_lob: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load and concatenate all days for a ticker.

    Returns concatenated (X, y_reg, y_cls, feature_names).
    """
    ticker_dir = os.path.join(processed_dir, ticker)
    parquet_files = sorted(glob.glob(os.path.join(ticker_dir, "*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {ticker_dir}")

    all_X, all_yr, all_yc = [], [], []
    feature_names = None

    for pf in parquet_files:
        X, yr, yc, fnames = build_features_and_labels(
            pf,
            ofi_levels=ofi_levels,
            horizons=horizons,
            alpha=alpha,
            include_raw_lob=include_raw_lob,
        )
        all_X.append(X)
        all_yr.append(yr)
        all_yc.append(yc)
        if feature_names is None:
            feature_names = fnames

    return (
        np.concatenate(all_X, axis=0),
        np.concatenate(all_yr, axis=0),
        np.concatenate(all_yc, axis=0),
        feature_names,
    )


def create_dataloaders(
    processed_dir: str,
    ticker: str,
    seq_len: int = 100,
    batch_size: int = 256,
    horizons: List[int] | None = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    flat: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], int]:
    """
    End-to-end: load data → features → labels → split → DataLoaders.

    Parameters
    ----------
    flat : bool
        If True, return FlatDataset (for linear / MLP models).
        If False, return LOBDataset (sliding window for sequential models).

    Returns
    -------
    train_loader, val_loader, test_loader, feature_names, n_features
    """
    X, y_reg, y_cls, feature_names = load_all_days(
        processed_dir, ticker, horizons=horizons
    )

    (X_tr, yr_tr, yc_tr), (X_va, yr_va, yc_va), (X_te, yr_te, yc_te) = \
        temporal_split(X, y_reg, y_cls, train_frac, val_frac)

    DsCls = FlatDataset if flat else LOBDataset

    if flat:
        ds_tr = DsCls(X_tr, yr_tr, yc_tr)
        ds_va = DsCls(X_va, yr_va, yc_va)
        ds_te = DsCls(X_te, yr_te, yc_te)
    else:
        ds_tr = DsCls(X_tr, yr_tr, yc_tr, seq_len=seq_len)
        ds_va = DsCls(X_va, yr_va, yc_va, seq_len=seq_len)
        ds_te = DsCls(X_te, yr_te, yc_te, seq_len=seq_len)

    train_loader = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, feature_names, X.shape[1]
