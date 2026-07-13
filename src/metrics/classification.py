import numpy as np
from typing import Dict, List

def _deep_update_confusion(confusion: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    mask = (y_true >= 0) & (y_true < 3) & (y_pred >= 0) & (y_pred < 3)
    yt = y_true[mask].astype(np.int64)
    yp = y_pred[mask].astype(np.int64)
    np.add.at(confusion, (yt, yp), 1)

def _deep_metrics_from_confusion(confusion: np.ndarray, h: int) -> dict:
    metrics: Dict[str, float] = {}
    n_cls = confusion.shape[0]
    total = float(confusion.sum())
    metrics[f'h{h}_accuracy'] = float(np.trace(confusion)) / max(total, 1.0)
    f1s = []
    for c in range(n_cls):
        tp = float(confusion[c, c])
        fp = float(confusion[:, c].sum()) - tp
        fn = float(confusion[c, :].sum()) - tp
        prec = tp / max(tp + fp, 1e-09)
        rec = tp / max(tp + fn, 1e-09)
        f1 = 2.0 * prec * rec / max(prec + rec, 1e-09)
        f1s.append(f1)
        metrics[f'h{h}_f1_c{c}'] = f1
    metrics[f'h{h}_f1_macro'] = float(np.mean(f1s))
    metrics[f'h{h}_f1_weighted'] = float(sum((f1s[c] * float(confusion[c, :].sum()) for c in range(n_cls))) / max(total, 1.0))
    return metrics

def _mean_macro_f1(metrics: dict, horizons: list) -> float:
    vals = [float(metrics.get(f'h{h}_f1_macro', 0.0)) for h in horizons]
    return float(np.mean(vals)) if vals else 0.0

