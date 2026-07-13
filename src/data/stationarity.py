import numpy as np
import pandas as pd
import warnings

def adf_test(series, p_threshold=0.05):
    from statsmodels.tsa.stattools import adfuller
    x = np.asarray(series, dtype=np.float64)
    x = x[~np.isnan(x)]
    res = adfuller(x, autolag='AIC')
    return {'statistic': float(res[0]), 'p_value': float(res[1]), 'is_stationary': bool(res[1] < p_threshold), 'method': 'ADF'}

def kpss_test(series, p_threshold=0.05):
    from statsmodels.tsa.stattools import kpss
    x = np.asarray(series, dtype=np.float64)
    x = x[~np.isnan(x)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = kpss(x, regression='c', nlags='auto')
    return {'statistic': float(res[0]), 'p_value': float(res[1]), 'is_stationary': bool(res[1] > p_threshold), 'method': 'KPSS'}

def _ffd_weights(d, size, threshold=0.001):
    w = [1.0]
    for k in range(1, size):
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < threshold:
            break
        w.append(wk)
    return np.array(w[::-1], dtype=np.float64)

def fractional_difference(x, d, threshold=0.001, fill_nan=True):
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    n, nf = x.shape
    w = _ffd_weights(d, n, threshold)
    win = len(w)
    out = np.full((n, nf), np.nan)
    for j in range(nf):
        col = x[:, j]
        for t in range(win - 1, n):
            seg = col[t - win + 1:t + 1]
            if not np.any(np.isnan(seg)):
                out[t, j] = np.dot(w, seg)
    if not fill_nan:
        out = np.where(np.isnan(out), 0.0, out)
    return (out[:, 0] if squeeze else out).astype(np.float32)

def test_feature_stationarity(X, feature_names=None, run_kpss=True, adf_p_threshold=0.05, verbose=True):
    import pandas as _pd
    X = np.asarray(X, dtype=np.float64)
    nf = X.shape[1]
    if feature_names is None:
        feature_names = [f'f{i}' for i in range(nf)]
    rows = []
    for j, name in enumerate(feature_names):
        r = {'feature': name}
        a = adf_test(X[:, j], adf_p_threshold)
        r['adf_stat'] = a['statistic']
        r['adf_pval'] = a['p_value']
        r['adf_stationary'] = a['is_stationary']
        if run_kpss:
            try:
                k = kpss_test(X[:, j])
                r['kpss_stationary'] = k['is_stationary']
                r['verdict'] = 'stationary' if a['is_stationary'] and k['is_stationary'] else 'non-stationary' if not a['is_stationary'] and (not k['is_stationary']) else 'ambiguous'
            except Exception:
                r['verdict'] = 'stationary' if a['is_stationary'] else 'non-stationary'
        else:
            r['verdict'] = 'stationary' if a['is_stationary'] else 'non-stationary'
        rows.append(r)
    df = _pd.DataFrame(rows)
    if verbose:
        print(f"  Non-stationary: {(df['verdict'] == 'non-stationary').sum()}/{nf}  Stationary: {(df['verdict'] == 'stationary').sum()}/{nf}  Ambiguous: {(df['verdict'] == 'ambiguous').sum()}/{nf}")
    return df

def make_stationary_features(X, feature_names=None, method='diff', d_fixed=0.4, adf_p_threshold=0.05, ffd_threshold=0.001, verbose=True):
    X = np.asarray(X, dtype=np.float64)
    n, nf = X.shape
    X_out = X.copy()
    d_per = [0.0] * nf
    idx_t = []
    if method == 'diff':
        for j in range(nf):
            d = np.diff(X[:, j], prepend=X[0, j])
            d[0] = 0.0
            X_out[:, j] = d
            d_per[j] = 1.0
            idx_t.append(j)
    elif method == 'frac_diff':
        for j in range(nf):
            fd = fractional_difference(X[:, j], d_fixed, ffd_threshold)
            X_out[:, j] = np.where(np.isnan(fd), 0.0, fd)
            d_per[j] = d_fixed
            idx_t.append(j)
    elif method == 'auto':
        stat_df = test_feature_stationarity(X, feature_names, adf_p_threshold=adf_p_threshold, verbose=verbose)
        for j, row in stat_df.iterrows():
            if row['verdict'] == 'non-stationary':
                fd = fractional_difference(X[:, j], d_fixed, ffd_threshold)
                X_out[:, j] = np.where(np.isnan(fd), 0.0, fd)
                d_per[j] = d_fixed
                idx_t.append(j)
    X_out = np.where(np.isnan(X_out), 0.0, X_out)
    if verbose:
        print(f'  Stationarity transform: {len(set(idx_t))}/{nf} features transformed (method={method})')
    return (X_out.astype(np.float32), {'method': method, 'd_per_feature': d_per, 'transformed_indices': idx_t})

def apply_stationarity_transform(X, metadata, ffd_threshold=0.001):
    X = np.asarray(X, dtype=np.float64)
    d_per = metadata['d_per_feature']
    X_out = X.copy()
    for j, d in enumerate(d_per):
        if d == 0.0:
            continue
        elif d == 1.0:
            di = np.diff(X[:, j], prepend=X[0, j])
            di[0] = 0.0
            X_out[:, j] = di
        else:
            fd = fractional_difference(X[:, j], d, ffd_threshold)
            X_out[:, j] = np.where(np.isnan(fd), 0.0, fd)
    return np.where(np.isnan(X_out), 0.0, X_out).astype(np.float32)

