import numpy as np
import warnings
from .helpers import _deep_seed_from_path
from typing import Tuple, List, Dict, Optional

_CLASS_NAMES = {0: 'DOWN', 1: 'STATIONARY', 2: 'UP'}

def get_class_distribution(y, n_classes=3):
    y = np.asarray(y, dtype=np.int32).ravel()
    c = np.bincount(y, minlength=n_classes)
    return {int(i): int(c[i]) for i in range(n_classes)}

def print_class_distribution(y, label='', n_classes=3):
    dist = get_class_distribution(y, n_classes)
    total = sum(dist.values())
    tag = f' [{label}]' if label else ''
    print(f'\nClass distribution{tag}  (total={total:,})')
    print('-' * 44)
    for cls, cnt in dist.items():
        name = _CLASS_NAMES.get(cls, f'cls{cls}')
        pct = 100.0 * cnt / max(total, 1)
        bar = '█' * int(pct / 2)
        print(f'  {cls} ({name:>10}): {cnt:>8,}  ({pct:5.1f}%)  {bar}')
    print('-' * 44)

def imbalance_ratio(y, n_classes=3):
    d = get_class_distribution(y, n_classes)
    c = [v for v in d.values() if v > 0]
    return float(max(c) / min(c)) if c else 1.0

def _smote_available():
    try:
        from imblearn.over_sampling import SMOTE
        return True
    except ImportError:
        return False

def apply_smote(X, y, sampling_strategy='auto', k_neighbors=5, random_state=42, verbose=True):
    if not _smote_available():
        raise ImportError('pip install imbalanced-learn')
    from imblearn.over_sampling import SMOTE
    if verbose:
        print_class_distribution(y, 'Before SMOTE')
    sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    Xr, yr = sm.fit_resample(X, y)
    if verbose:
        print_class_distribution(yr, 'After SMOTE')
    return (Xr.astype(np.float32), yr.astype(np.int32))

def apply_borderline_smote(X, y, sampling_strategy='auto', k_neighbors=5, random_state=42, verbose=True):
    if not _smote_available():
        raise ImportError('pip install imbalanced-learn')
    from imblearn.over_sampling import BorderlineSMOTE
    if verbose:
        print_class_distribution(y, 'Before Borderline-SMOTE')
    sm = BorderlineSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    Xr, yr = sm.fit_resample(X, y)
    if verbose:
        print_class_distribution(yr, 'After Borderline-SMOTE')
    return (Xr.astype(np.float32), yr.astype(np.int32))

def apply_adasyn(X, y, sampling_strategy='auto', k_neighbors=5, random_state=42, verbose=True):
    if not _smote_available():
        raise ImportError('pip install imbalanced-learn')
    from imblearn.over_sampling import ADASYN
    if verbose:
        print_class_distribution(y, 'Before ADASYN')
    sm = ADASYN(sampling_strategy=sampling_strategy, n_neighbors=k_neighbors, random_state=random_state)
    try:
        Xr, yr = sm.fit_resample(X, y)
    except ValueError:
        warnings.warn('ADASYN failed, falling back to SMOTE')
        return apply_smote(X, y, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state, verbose=False)
    if verbose:
        print_class_distribution(yr, 'After ADASYN')
    return (Xr.astype(np.float32), yr.astype(np.int32))

def apply_smote_tomek(X, y, sampling_strategy='auto', random_state=42, verbose=True):
    if not _smote_available():
        raise ImportError('pip install imbalanced-learn')
    from imblearn.combine import SMOTETomek
    if verbose:
        print_class_distribution(y, 'Before SMOTE-Tomek')
    sm = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
    Xr, yr = sm.fit_resample(X, y)
    if verbose:
        print_class_distribution(yr, 'After SMOTE-Tomek')
    return (Xr.astype(np.float32), yr.astype(np.int32))

def _apply_smote_to_day(raw: np.ndarray, starts: np.ndarray, y_day: np.ndarray, cfg: dict, parquet_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply SMOTE to the flattened window features for a single day.
    Returns (raw_aug, starts_aug, y_day_aug) — synthetic samples appended.

    SMOTE is applied to the FLATTENED window vectors (seq_len × n_features)
    so the resampler sees the full temporal context per sample.
    NOTE: This is memory-intensive. Keep seq_len × batch small in Colab.
    """
    seq_len = int(cfg['seq_len'])
    smote_meth = str(cfg.get('smote_method', 'smote'))
    k_neighbors = int(cfg.get('smote_k', 5))
    min_per_cls = int(cfg.get('smote_min_per_class', 10))
    seed = _deep_seed_from_path(parquet_path, int(cfg.get('seed', 42)))
    y_h = y_day[:, 0].astype(np.int32)
    counts = np.bincount(y_h, minlength=3)
    if int(counts.min()) < min_per_cls:
        return (raw, starts, y_day)
    n_samples = len(starts)
    n_features = raw.shape[1]
    X_flat = np.zeros((n_samples, seq_len * n_features), dtype=np.float32)
    for i, s in enumerate(starts):
        X_flat[i] = raw[s:s + seq_len].ravel()
    try:
        fn = RESAMPLE_METHODS.get(smote_meth, apply_smote)
        X_res, y_res = fn(X_flat, y_h, k_neighbors=k_neighbors, random_state=seed, verbose=False)
    except Exception as _e:
        warnings.warn(f'SMOTE failed ({_e}), using original data')
        return (raw, starts, y_day)
    n_synth = len(X_res) - n_samples
    if n_synth <= 0:
        return (raw, starts, y_day)
    X_synth = X_res[n_samples:].reshape(n_synth, seq_len, n_features)
    n_raw = len(raw)
    raw_ext = np.vstack([raw, X_synth.reshape(-1, n_features)])
    synth_starts = np.array([n_raw + i * seq_len for i in range(n_synth)], dtype=np.int64)
    starts_ext = np.concatenate([starts, synth_starts])
    y_synth = np.tile(y_res[n_samples:, None], (1, y_day.shape[1])).astype(np.int64)
    y_ext = np.vstack([y_day, y_synth])
    return (raw_ext, starts_ext, y_ext)


RESAMPLE_METHODS = {
    'smote': apply_smote,
    'borderline_smote': apply_borderline_smote,
    'adasyn': apply_adasyn,
    'smote_tomek': apply_smote_tomek,
}
