import numpy as np

def compute_r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)

def compute_adjusted_r2(y_true, y_pred, n_features: int):
    r2 = compute_r2(y_true, y_pred)
    n = len(y_true)
    if n <= n_features + 1:
        return float('nan')
    return 1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1)

def compute_oos_r2(y_true, y_pred):
    """Out-of-sample R² vs. historical mean benchmark."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_bm = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / max(ss_bm, 1e-12)

def compute_all_horizon_r2(Y_true, Y_pred, horizons, n_features=40):
    results = {}
    for i, h in enumerate(horizons):
        yt = Y_true[:, i].astype(np.float64)
        yp = Y_pred[:, i].astype(np.float64)
        results[f'h{h}'] = {'r2': compute_r2(yt, yp), 'r2_adj': compute_adjusted_r2(yt, yp, n_features), 'r2_oos': compute_oos_r2(yt, yp)}
    return results

def print_r2_table(results, horizons):
    print(f"\n{'=' * 55}")
    print(f"   {'Horizon':>8}   {'R²':>8}   {'R²_adj':>8}   {'R²_oos':>8}")
    print(f"{'=' * 55}")
    for h in horizons:
        d = results.get(f'h{h}', {})
        print(f"   {'k=' + str(h):>8}   {d.get('r2', float('nan')):>8.4f}   {d.get('r2_adj', float('nan')):>8.4f}   {d.get('r2_oos', float('nan')):>8.4f}")
    print(f"{'=' * 55}")

