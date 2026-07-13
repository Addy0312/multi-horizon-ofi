import numpy as np

class ZScoreScaler:

    def fit(self, X):
        self.mean_ = X.mean(0)
        self.std_ = np.where(X.std(0) < 1e-08, 1.0, X.std(0))
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class MinMaxScaler:

    def fit(self, X):
        self.min_ = X.min(0)
        self.max_ = X.max(0)
        self.range_ = np.where(self.max_ - self.min_ < 1e-08, 1.0, self.max_ - self.min_)
        return self

    def transform(self, X):
        return (X - self.min_) / self.range_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class RobustScaler:

    def fit(self, X):
        self.median_ = np.median(X, 0)
        q75, q25 = np.percentile(X, [75, 25], 0)
        self.iqr_ = np.where(q75 - q25 < 1e-08, 1.0, q75 - q25)
        return self

    def transform(self, X):
        return (X - self.median_) / self.iqr_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class MaxAbsScaler:

    def fit(self, X):
        self.max_abs_ = np.where(np.abs(X).max(0) < 1e-08, 1.0, np.abs(X).max(0))
        return self

    def transform(self, X):
        return X / self.max_abs_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class DecimalScaler:

    def fit(self, X):
        max_abs = np.abs(X).max(0)
        self.scale_ = np.power(10, np.ceil(np.log10(np.where(max_abs < 1e-08, 1.0, max_abs))))
        return self

    def transform(self, X):
        return X / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class QuantileScaler:

    def __init__(self, output_distribution='uniform', n_quantiles=1000):
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles

    def fit(self, X):
        try:
            from sklearn.preprocessing import QuantileTransformer
            self._sk = QuantileTransformer(output_distribution=self.output_distribution, n_quantiles=min(self.n_quantiles, len(X)), random_state=42)
            self._sk.fit(X)
        except ImportError:
            self._sk = None
        return self

    def transform(self, X):
        if self._sk is None:
            return X
        return self._sk.transform(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class PowerTransformer:

    def fit(self, X):
        try:
            from sklearn.preprocessing import PowerTransformer as _PT
            self._sk = _PT(method='yeo-johnson', standardize=True)
            self._sk.fit(X)
        except ImportError:
            self._sk = None
        return self

    def transform(self, X):
        if self._sk is None:
            return X
        return self._sk.transform(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

def normalize_splits(X_train, X_val, X_test, method='zscore'):
    factory = _SCALER_MAP.get(method, ZScoreScaler)
    scaler = factory() if callable(factory) else factory
    X_tr = scaler.fit_transform(X_train.astype(np.float64)).astype(np.float32)
    X_va = scaler.transform(X_val.astype(np.float64)).astype(np.float32)
    X_te = scaler.transform(X_test.astype(np.float64)).astype(np.float32)
    return (X_tr, X_va, X_te, scaler)

def _apply_day_normalization(raw: np.ndarray, method: str) -> np.ndarray:
    """
    Normalise a single day's raw feature matrix.

    Per-day normalization is important: different days can have very different
    price scales, so a globally-fitted scaler may not generalise well.
    We fit on the current day and apply within that day only.
    """
    raw = raw.astype(np.float64)
    try:
        if method == 'zscore':
            mean = raw.mean(0)
            std = np.where(raw.std(0) < 1e-08, 1.0, raw.std(0))
            return ((raw - mean) / std).astype(np.float32)
        elif method == 'minmax':
            mn = raw.min(0)
            mx = raw.max(0)
            r = np.where(mx - mn < 1e-08, 1.0, mx - mn)
            return ((raw - mn) / r).astype(np.float32)
        elif method == 'robust':
            med = np.median(raw, 0)
            q75, q25 = np.percentile(raw, [75, 25], 0)
            iqr = np.where(q75 - q25 < 1e-08, 1.0, q75 - q25)
            return ((raw - med) / iqr).astype(np.float32)
        elif method == 'maxabs':
            ma = np.where(np.abs(raw).max(0) < 1e-08, 1.0, np.abs(raw).max(0))
            return (raw / ma).astype(np.float32)
        elif method == 'decimal':
            ma = np.abs(raw).max(0)
            scale = np.power(10, np.ceil(np.log10(np.where(ma < 1e-08, 1.0, ma))))
            return (raw / scale).astype(np.float32)
        elif method in ('quantile', 'quantile_normal'):
            try:
                from sklearn.preprocessing import QuantileTransformer
                dist = 'normal' if method == 'quantile_normal' else 'uniform'
                qt = QuantileTransformer(output_distribution=dist, n_quantiles=min(1000, len(raw)), random_state=42)
                return qt.fit_transform(raw).astype(np.float32)
            except ImportError:
                pass
        elif method == 'power':
            try:
                from sklearn.preprocessing import PowerTransformer
                pt = PowerTransformer(method='yeo-johnson', standardize=True)
                return pt.fit_transform(raw).astype(np.float32)
            except ImportError:
                pass
    except Exception:
        pass
    mean = raw.mean(0)
    std = np.where(raw.std(0) < 1e-08, 1.0, raw.std(0))
    return ((raw - mean) / std).astype(np.float32)


_SCALER_MAP = {
    "zscore": ZScoreScaler, "minmax": MinMaxScaler, "robust": RobustScaler,
    "maxabs": MaxAbsScaler, "decimal": DecimalScaler,
    "quantile": lambda: QuantileScaler(output_distribution='uniform'),
    "quantile_normal": lambda: QuantileScaler(output_distribution='normal'),
    "power": PowerTransformer,
}
