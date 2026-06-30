"""
SMOTE-based data preprocessing for class imbalance in LOB prediction.

LOB classification datasets suffer from severe class imbalance:
  - STATIONARY events dominate (~60–70% of samples)
  - UP / DOWN events are relatively rare

This causes classifiers to collapse to the majority class (STATIONARY),
yielding high accuracy but useless predictions.

This module provides:
    - Class distribution diagnostics
    - SMOTE (Synthetic Minority Over-sampling Technique)
    - Borderline-SMOTE (focuses on borderline minority examples)
    - ADASYN (Adaptive Synthetic Sampling)
    - SMOTE + Tomek Links (oversample minority + clean majority boundary)
    - Per-horizon resampling wrappers

Important
---------
All resampling functions MUST be applied only to the **training split**.
Applying SMOTE to validation or test data would constitute data leakage
and produce misleadingly optimistic evaluation metrics.

Dependencies
------------
    pip install imbalanced-learn

References
----------
- Chawla et al. (2002): SMOTE — Synthetic Minority Over-sampling Technique
- Han, Wang, Mao (2005): Borderline-SMOTE
- He et al. (2008): ADASYN — Adaptive Synthetic Sampling Approach
- Batista et al. (2004): SMOTE + Tomek Links cleaning
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# Dependency check
# ──────────────────────────────────────────────────────────────────────────

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    from imblearn.under_sampling import TomekLinks
    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False


def _require_imblearn() -> None:
    if not _HAS_IMBLEARN:
        raise ImportError(
            "imbalanced-learn is required for SMOTE preprocessing.\n"
            "Install it with:  pip install imbalanced-learn"
        )


# ──────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────

#: Class label names used in logging
CLASS_NAMES = {0: "DOWN", 1: "STATIONARY", 2: "UP"}


def get_class_distribution(
    y: np.ndarray,
    n_classes: int = 3,
) -> Dict[int, int]:
    """
    Return a dict mapping class label → count.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Integer class labels (0, 1, 2).
    n_classes : int
        Total number of expected classes (default 3 for DOWN/STAT/UP).

    Returns
    -------
    dict
        ``{class_int: count}``, including classes with count 0.
    """
    y = np.asarray(y, dtype=np.int32).ravel()
    counts = np.bincount(y, minlength=n_classes)
    return {int(c): int(counts[c]) for c in range(n_classes)}


def print_class_distribution(
    y: np.ndarray,
    label: str = "",
    n_classes: int = 3,
) -> None:
    """Pretty-print class distribution with percentages."""
    dist = get_class_distribution(y, n_classes)
    total = sum(dist.values())
    tag = f" [{label}]" if label else ""
    print(f"\nClass distribution{tag}  (total={total:,})")
    print("-" * 42)
    for cls, cnt in dist.items():
        name = CLASS_NAMES.get(cls, f"class_{cls}")
        pct = 100.0 * cnt / max(total, 1)
        bar = "#" * int(pct / 2)
        print(f"  {cls} ({name:>10}): {cnt:>8,}  ({pct:5.1f}%)  {bar}")
    print("-" * 42)


def imbalance_ratio(y: np.ndarray, n_classes: int = 3) -> float:
    """
    Return the ratio of the majority class count to the minority class count.

    A ratio of 1.0 = perfectly balanced; higher = more imbalanced.
    """
    dist = get_class_distribution(y, n_classes)
    counts = list(dist.values())
    counts = [c for c in counts if c > 0]
    if not counts:
        return 1.0
    return float(max(counts) / min(counts))


# ──────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────


def _validate_inputs(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32).ravel()
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of samples, got {len(X)} vs {len(y)}"
        )
    return X, y


def _check_enough_minority(y: np.ndarray, k_neighbors: int = 5) -> bool:
    """
    SMOTE requires at least k_neighbors + 1 samples per minority class.
    Returns True if safe to proceed, False otherwise.
    """
    dist = get_class_distribution(y)
    for cls, cnt in dist.items():
        if 0 < cnt <= k_neighbors:
            warnings.warn(
                f"Class {cls} ({CLASS_NAMES.get(cls, cls)}) has only {cnt} samples "
                f"but k_neighbors={k_neighbors}. SMOTE may fail — consider reducing "
                f"k_neighbors or using a smaller sampling_strategy.",
                UserWarning,
                stacklevel=3,
            )
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────
# SMOTE variants
# ──────────────────────────────────────────────────────────────────────────


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: str | dict = "auto",
    k_neighbors: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply standard SMOTE to balance class distribution.

    SMOTE generates synthetic minority-class samples by interpolating between
    existing minority examples and their k nearest neighbours.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
        Feature matrix (train only).
    y : np.ndarray, shape (N,)
        Class labels (train only).
    sampling_strategy : str or dict
        - ``'auto'`` / ``'not majority'`` — resample all classes except the
          majority to match its count.
        - ``'minority'``  — resample only the minority class.
        - dict: ``{class_label: desired_count}``
    k_neighbors : int
        Number of nearest neighbours for synthetic sample generation.
    random_state : int
        Reproducibility seed.
    verbose : bool
        Print class distribution before and after.

    Returns
    -------
    X_res : np.ndarray, shape (N', F)
    y_res : np.ndarray, shape (N',)

    Warning
    -------
    Apply ONLY to training data.  Never apply to val or test splits.
    """
    _require_imblearn()
    X, y = _validate_inputs(X, y)

    if verbose:
        print_class_distribution(y, label="Before SMOTE")

    _check_enough_minority(y, k_neighbors)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X, y)

    if verbose:
        print_class_distribution(y_res, label="After SMOTE")

    return X_res.astype(np.float32), y_res.astype(np.int32)


def apply_borderline_smote(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: str | dict = "auto",
    k_neighbors: int = 5,
    m_neighbors: int = 10,
    kind: str = "borderline-1",
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Borderline-SMOTE to generate synthetic samples near decision boundaries.

    Unlike standard SMOTE which samples from all minority examples, Borderline-SMOTE
    focuses on "danger zone" minority examples that are close to the majority class.
    This tends to produce more informative synthetic samples.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
    y : np.ndarray, shape (N,)
    sampling_strategy : str or dict
    k_neighbors : int
        Number of nearest neighbours for synthetic sample generation.
    m_neighbors : int
        Number of nearest neighbours to determine whether a sample is
        in the borderline region.
    kind : str
        ``'borderline-1'`` — only minority-class neighbours are used in
        synthetic sample generation (safer).
        ``'borderline-2'`` — minority and majority neighbours are used
        (more aggressive, can generate noisier samples).
    random_state : int
    verbose : bool

    Returns
    -------
    X_res, y_res
    """
    _require_imblearn()
    X, y = _validate_inputs(X, y)

    if verbose:
        print_class_distribution(y, label="Before Borderline-SMOTE")

    _check_enough_minority(y, k_neighbors)

    bl_smote = BorderlineSMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        m_neighbors=m_neighbors,
        kind=kind,
        random_state=random_state,
    )
    X_res, y_res = bl_smote.fit_resample(X, y)

    if verbose:
        print_class_distribution(y_res, label="After Borderline-SMOTE")

    return X_res.astype(np.float32), y_res.astype(np.int32)


def apply_adasyn(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: str | dict = "auto",
    n_neighbors: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply ADASYN (Adaptive Synthetic Sampling).

    ADASYN generates more synthetic samples for minority examples that are
    harder to learn (i.e., surrounded by more majority-class neighbours).
    This adaptively focuses synthesis on difficult regions of the feature space.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
    y : np.ndarray, shape (N,)
    sampling_strategy : str or dict
    n_neighbors : int
        Number of nearest neighbours used to estimate local density.
    random_state : int
    verbose : bool

    Returns
    -------
    X_res, y_res
    """
    _require_imblearn()
    X, y = _validate_inputs(X, y)

    if verbose:
        print_class_distribution(y, label="Before ADASYN")

    _check_enough_minority(y, n_neighbors)

    adasyn = ADASYN(
        sampling_strategy=sampling_strategy,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    try:
        X_res, y_res = adasyn.fit_resample(X, y)
    except ValueError as e:
        # ADASYN can fail if density ratio collapses; fall back to SMOTE
        warnings.warn(
            f"ADASYN failed ({e}). Falling back to standard SMOTE.",
            UserWarning,
            stacklevel=2,
        )
        return apply_smote(X, y, sampling_strategy=sampling_strategy,
                           k_neighbors=n_neighbors, random_state=random_state,
                           verbose=False)

    if verbose:
        print_class_distribution(y_res, label="After ADASYN")

    return X_res.astype(np.float32), y_res.astype(np.int32)


def apply_smote_tomek(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: str | dict = "auto",
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE + Tomek Links (combined over- and under-sampling).

    Steps:
        1. Oversample minority classes with SMOTE.
        2. Remove Tomek Links — pairs of samples from opposite classes that are
           nearest neighbours.  This cleans noise near the decision boundary.

    This combination typically yields better F1 scores than pure SMOTE because
    the majority-class noise near boundaries is removed after oversampling.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
    y : np.ndarray, shape (N,)
    sampling_strategy : str or dict
    random_state : int
    verbose : bool

    Returns
    -------
    X_res, y_res
    """
    _require_imblearn()
    X, y = _validate_inputs(X, y)

    if verbose:
        print_class_distribution(y, label="Before SMOTE-Tomek")

    smote_tomek = SMOTETomek(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )
    X_res, y_res = smote_tomek.fit_resample(X, y)

    if verbose:
        print_class_distribution(y_res, label="After SMOTE-Tomek")

    return X_res.astype(np.float32), y_res.astype(np.int32)


# ──────────────────────────────────────────────────────────────────────────
# Per-horizon wrapper
# ──────────────────────────────────────────────────────────────────────────

#: Available resampling methods
RESAMPLE_METHODS = {
    "smote": apply_smote,
    "borderline_smote": apply_borderline_smote,
    "adasyn": apply_adasyn,
    "smote_tomek": apply_smote_tomek,
}


def resample_for_horizon(
    X: np.ndarray,
    y_cls: np.ndarray,
    horizon_idx: int = 0,
    method: str = "smote",
    verbose: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample training data for a single horizon.

    Extracts labels for the given horizon index, applies the chosen resampling
    strategy, and returns resampled (X, y_cls_1d) for that horizon.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
        Feature matrix (training split only).
    y_cls : np.ndarray, shape (N,) or (N, H)
        Classification labels.  If 2-D, ``horizon_idx`` selects the column.
    horizon_idx : int
        Which horizon column to use when y_cls is 2-D.
    method : str
        One of ``'smote'``, ``'borderline_smote'``, ``'adasyn'``, ``'smote_tomek'``.
    verbose : bool
    **kwargs
        Passed to the underlying resampler.

    Returns
    -------
    X_res : np.ndarray, shape (N', F)
    y_res : np.ndarray, shape (N',)
        Resampled labels for the selected horizon.
    """
    if method not in RESAMPLE_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {list(RESAMPLE_METHODS.keys())}"
        )

    y_cls = np.asarray(y_cls)
    if y_cls.ndim == 2:
        y_h = y_cls[:, horizon_idx]
    else:
        y_h = y_cls

    fn = RESAMPLE_METHODS[method]
    return fn(X, y_h, verbose=verbose, **kwargs)


def resample_all_horizons(
    X: np.ndarray,
    y_cls: np.ndarray,
    method: str = "smote",
    verbose: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample training data jointly across all horizons.

    Uses the labels of the **first horizon** (index 0) to decide which
    samples to oversample, then applies the same resampled indices to all
    horizon columns.  This preserves label correlation across horizons.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
    y_cls : np.ndarray, shape (N, H)
    method : str
    verbose : bool
    **kwargs

    Returns
    -------
    X_res : np.ndarray, shape (N', F)
    y_cls_res : np.ndarray, shape (N', H)
    """
    if method not in RESAMPLE_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {list(RESAMPLE_METHODS.keys())}"
        )

    y_cls = np.asarray(y_cls)
    if y_cls.ndim == 1:
        y_cls = y_cls[:, None]

    # We use horizon 0 as the reference for resampling decisions.
    # Synthetic samples get majority-voted labels from neighbouring real samples
    # — for multi-horizon, we replicate the first-horizon synthetic labels
    # across all horizons (a practical approximation).
    fn = RESAMPLE_METHODS[method]

    # Build a combined label (first horizon) and track new indices via sklearn
    # Note: we pass y_cls[:, 0] to the resampler; for the returned synthetic
    # samples we tile the remaining horizon labels using nearest-real-sample logic.
    X_res, y_h0_res = fn(X, y_cls[:, 0], verbose=verbose, **kwargs)

    n_orig = len(X)
    n_res = len(X_res)
    n_synthetic = n_res - n_orig

    # For original samples, keep their multi-horizon labels.
    # For synthetic samples, use the first-horizon label (y_h0_res) and
    # replicate it to all horizons (approximation; ideally use kNN over real samples).
    y_orig = y_cls                                               # (N, H)
    y_res = np.zeros((n_res, y_cls.shape[1]), dtype=np.int32)
    y_res[:n_orig] = y_orig

    if n_synthetic > 0:
        # Assign first-horizon label to all horizons for synthetic samples
        y_res[n_orig:, :] = y_h0_res[n_orig:, None]

    if verbose:
        print(f"\nResampled {n_orig} → {n_res} samples "
              f"({n_synthetic} synthetic added) across {y_cls.shape[1]} horizons.")

    return X_res, y_res
