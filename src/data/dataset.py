import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .preprocessor import _apply_day_normalization
from .smote_preprocessing import _apply_smote_to_day

@dataclass
class DayStats:
    ticker: str
    file_name: str
    rows_kept: int
    step: int
    max_rows: int

class StableDayWindowDataset(Dataset):
    """
    Per-day normalized sliding-window dataset.

    Supports multiple normalization methods via DEEP_CONFIG['normalization_method'].
    The normalization is fitted and applied per day to prevent cross-day leakage.
    """

    def __init__(self, raw: np.ndarray, starts: np.ndarray, labels: np.ndarray, seq_len: int, norm_method: str='robust'):
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        raw_normed = _apply_day_normalization(raw, norm_method)
        self.raw = raw_normed
        self.starts = starts.astype(np.int64, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        x = self.raw[s:s + self.seq_len].copy()
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)
        y = self.labels[idx]
        return (torch.from_numpy(x.astype(np.float32, copy=False)), torch.from_numpy(y))

def _deep_build_day_dataset(parquet_path: str, cfg: dict, is_train: bool) -> Tuple['StableDayWindowDataset | None', 'DayStats | None']:
    horizons = list(cfg['horizons'])
    seq_len = int(cfg['seq_len'])
    alpha = float(cfg['alpha'])
    step = _deep_choose_subsample(cfg)
    max_rows = _deep_choose_max_rows(cfg, is_train=is_train)
    norm_method = str(cfg.get('normalization_method', 'robust'))
    try:
        df = pd.read_parquet(parquet_path, columns=DEEP_RAW_LOB_10_COLS)
    except Exception as e:
        print(f'  [dataset] Failed to read {parquet_path}: {e}')
        return (None, None)
    raw = np.ascontiguousarray(df.to_numpy(dtype=np.float32, copy=False))
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_full = make_fixed_threshold_classification_labels(df, horizons=horizons, alpha=alpha, use_smoothing=True).to_numpy(dtype=np.float32, copy=False)
    if cfg.get('enable_stationarity') and '_STATIONARITY_META' in globals() and (_STATIONARITY_META is not None):
        try:
            raw = apply_stationarity_transform(raw, _STATIONARITY_META)
        except Exception as _se:
            warnings.warn(f'Stationarity transform failed: {_se}')
    max_h = int(max(horizons))
    valid_end = len(df) - max_h
    del df
    if valid_end <= seq_len:
        gc.collect()
        return (None, None)
    labels = y_full[seq_len - 1:valid_end]
    valid_mask = ~np.isnan(labels).any(axis=1)
    starts = np.flatnonzero(valid_mask).astype(np.int64, copy=False)
    if step > 1:
        starts = starts[::step]
    if max_rows > 0:
        starts = starts[:max_rows]
    if starts.size == 0:
        gc.collect()
        return (None, None)
    y_day = labels[starts].astype(np.int64, copy=False)
    class_mask = ((y_day >= 0) & (y_day <= 2)).all(axis=1)
    starts = starts[class_mask]
    y_day = y_day[class_mask]
    if starts.size == 0:
        gc.collect()
        return (None, None)
    if is_train and cfg.get('enable_smote', False):
        try:
            raw, starts, y_day = _apply_smote_to_day(raw, starts, y_day, cfg, parquet_path)
        except Exception as _se:
            warnings.warn(f'SMOTE failed for {parquet_path}: {_se}')
    ds = StableDayWindowDataset(raw, starts, y_day, seq_len, norm_method=norm_method)
    stats = DayStats(ticker=os.path.basename(os.path.dirname(parquet_path)), file_name=os.path.basename(parquet_path), rows_kept=int(starts.size), step=step, max_rows=max_rows)
    del raw, y_full, labels, valid_mask, y_day, class_mask
    gc.collect()
    return (ds, stats)

def _deep_make_loader(dataset: Dataset, cfg: dict, is_train: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=int(cfg.get('batch_size', 256)), shuffle=is_train, num_workers=int(cfg.get('num_workers', 0)), pin_memory=DEEP_DEVICE.type == 'cuda', drop_last=is_train)

