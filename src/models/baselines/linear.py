"""
Linear baseline models for LOB mid-price prediction.

Implements:
    1. OLS Regression — Cont, Kukanov & Stoikov (2014)
       - Uses OFI as the primary feature
       - Contemporaneous + forecasting variants
    2. Ridge / Lasso Regression — Xu, Gould & Howison (2019)
       - Multi-level OFI features with regularisation
       - Evaluates feature importance across LOB levels

These are non-sequential models: each sample is a single feature vector
(no lookback window), fitted per horizon.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from src.features.labels import DEFAULT_HORIZONS
from src.metrics.regression import compute_regression_metrics
from src.metrics.classification import compute_classification_metrics


# ──────────────────────────────────────────────────────────────────────────
# OLS Baseline  (Cont, Kukanov & Stoikov 2014)
# ──────────────────────────────────────────────────────────────────────────

class OLSBaseline:
    """
    Ordinary Least Squares regression: ΔP_t = β · OFI_t + ε

    Following Cont et al. (2014), this is primarily an *explanatory*
    model (contemporaneous R²), but we also evaluate it as a
    1-step-ahead forecaster using lagged OFI.

    For multi-horizon: one independent OLS per horizon.
    """

    def __init__(self, horizons: List[int] | None = None):
        self.horizons = horizons or DEFAULT_HORIZONS
        self.models: Dict[int, sm.OLS] = {}
        self.results: Dict[int, sm.regression.linear_model.RegressionResultsWrapper] = {}
        self.scalers: Dict[int, StandardScaler] = {}

    def fit(
        self,
        X: np.ndarray,
        y_reg: np.ndarray,
    ) -> "OLSBaseline":
        """
        Fit one OLS model per horizon.

        Parameters
        ----------
        X : np.ndarray, shape (N, F)
            Feature matrix (OFI columns).
        y_reg : np.ndarray, shape (N, H)
            Regression targets for each horizon.
        """
        for i, h in enumerate(self.horizons):
            y = y_reg[:, i]
            mask = ~np.isnan(y)
            X_h = X[mask]
            y_h = y[mask]

            # Standardise
            scaler = StandardScaler()
            X_h = scaler.fit_transform(X_h)
            self.scalers[h] = scaler

            # Add constant for intercept
            X_const = sm.add_constant(X_h)
            model = sm.OLS(y_h, X_const)
            self.results[h] = model.fit()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict delta mid-price for all horizons.

        Returns
        -------
        np.ndarray, shape (N, H)
        """
        preds = np.zeros((len(X), len(self.horizons)))
        for i, h in enumerate(self.horizons):
            X_s = self.scalers[h].transform(X)
            X_const = sm.add_constant(X_s)
            preds[:, i] = self.results[h].predict(X_const)
        return preds

    def predict_classes(
        self,
        X: np.ndarray,
        thresholds: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Convert regression predictions to 3-class labels
        using per-horizon thresholds.

        Parameters
        ----------
        thresholds : np.ndarray, shape (H,)
            If None, defaults to 0 (sign-based labelling).
        """
        preds = self.predict(X)
        if thresholds is None:
            thresholds = np.zeros(len(self.horizons))

        classes = np.ones_like(preds, dtype=int)  # default STATIONARY
        for i in range(len(self.horizons)):
            classes[:, i][preds[:, i] > thresholds[i]] = 2   # UP
            classes[:, i][preds[:, i] < -thresholds[i]] = 0  # DOWN
        return classes

    def summary(self, horizon: int | None = None) -> str:
        """Print statsmodels summary for a given horizon."""
        if horizon is None:
            horizon = self.horizons[0]
        return str(self.results[horizon].summary())

    def get_r_squared(self) -> Dict[int, float]:
        """R² for each horizon."""
        return {h: self.results[h].rsquared for h in self.horizons}


# ──────────────────────────────────────────────────────────────────────────
# Ridge / Lasso  (Xu, Gould & Howison 2019)
# ──────────────────────────────────────────────────────────────────────────

class RegularizedLinearModel:
    """
    Ridge or Lasso regression for multi-level OFI prediction.

    Following Xu et al. (2019):
        - Uses OFI from levels 1..K as features
        - Regularisation controls overfitting and reveals which
          levels carry the most predictive information
        - Evaluated on RMSE improvement over single-level baseline

    For multi-horizon: one model per horizon.
    """

    def __init__(
        self,
        method: str = "ridge",
        alpha: float = 1.0,
        horizons: List[int] | None = None,
    ):
        """
        Parameters
        ----------
        method : str
            'ridge' or 'lasso'.
        alpha : float
            Regularisation strength.
        """
        self.method = method
        self.alpha = alpha
        self.horizons = horizons or DEFAULT_HORIZONS
        self.models: Dict[int, object] = {}
        self.scalers: Dict[int, StandardScaler] = {}

    def _make_model(self):
        if self.method == "ridge":
            return Ridge(alpha=self.alpha)
        elif self.method == "lasso":
            return Lasso(alpha=self.alpha, max_iter=5000)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(
        self,
        X: np.ndarray,
        y_reg: np.ndarray,
    ) -> "RegularizedLinearModel":
        """Fit one model per horizon."""
        for i, h in enumerate(self.horizons):
            y = y_reg[:, i]
            mask = ~np.isnan(y)
            X_h = X[mask]
            y_h = y[mask]

            scaler = StandardScaler()
            X_h = scaler.fit_transform(X_h)
            self.scalers[h] = scaler

            model = self._make_model()
            model.fit(X_h, y_h)
            self.models[h] = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for all horizons. Returns (N, H)."""
        preds = np.zeros((len(X), len(self.horizons)))
        for i, h in enumerate(self.horizons):
            X_s = self.scalers[h].transform(X)
            preds[:, i] = self.models[h].predict(X_s)
        return preds

    def predict_classes(
        self,
        X: np.ndarray,
        thresholds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert regression predictions to 3-class labels."""
        preds = self.predict(X)
        if thresholds is None:
            thresholds = np.zeros(len(self.horizons))

        classes = np.ones_like(preds, dtype=int)
        for i in range(len(self.horizons)):
            classes[:, i][preds[:, i] > thresholds[i]] = 2
            classes[:, i][preds[:, i] < -thresholds[i]] = 0
        return classes

    def get_coefficients(self) -> Dict[int, np.ndarray]:
        """Return fitted coefficients per horizon."""
        return {h: self.models[h].coef_ for h in self.horizons}

    def get_feature_importance(
        self, feature_names: List[str]
    ) -> Dict[int, Dict[str, float]]:
        """Return absolute coefficient values as importance."""
        result = {}
        for h in self.horizons:
            coefs = self.models[h].coef_
            importance = {
                name: abs(float(c))
                for name, c in zip(feature_names, coefs)
            }
            result[h] = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)
            )
        return result


# ──────────────────────────────────────────────────────────────────────────
# Convenience: evaluate any linear model
# ──────────────────────────────────────────────────────────────────────────

def evaluate_linear_model(
    model,
    X_test: np.ndarray,
    y_reg_test: np.ndarray,
    y_cls_test: np.ndarray,
    horizons: List[int] | None = None,
    thresholds: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    Evaluate a linear model on both regression and classification metrics.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    # Regression
    preds_reg = model.predict(X_test)
    reg_metrics = {}
    for i, h in enumerate(horizons):
        mask = ~np.isnan(y_reg_test[:, i])
        m = compute_regression_metrics(
            y_reg_test[mask, i], preds_reg[mask, i], horizon_name=f"h{h}"
        )
        reg_metrics.update(m)

    # Classification
    preds_cls = model.predict_classes(X_test, thresholds)
    cls_metrics = {}
    for i, h in enumerate(horizons):
        mask = ~np.isnan(y_cls_test[:, i])
        m = compute_classification_metrics(
            y_cls_test[mask, i].astype(int),
            preds_cls[mask, i],
            horizon_name=f"h{h}",
        )
        cls_metrics.update(m)

    return {**reg_metrics, **cls_metrics}
