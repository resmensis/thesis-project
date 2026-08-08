from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

logger = logging.getLogger("eap_ml.models_linear")


def predictive_r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    num = np.sum((y_true - y_pred) ** 2)
    return 1.0 - num / denom if denom > 0 else np.nan


def fit_huber_regression(X_train, y_train, epsilon=1.35, alpha=0.0001):
    """
    Fit Huber regression - this is what Gu et al. (2020) call "OLS".
    
    Gu et al. (2020) use robust regression with Huber loss as their baseline OLS model.
    """
    logger.debug(f"Fitting Huber regression: epsilon={epsilon}, alpha={alpha}, shape={X_train.shape}")
    model = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=500)
    model.fit(X_train, y_train)
    logger.debug("Huber regression fitting completed")
    return model


def tune_huber_regression(X_train, y_train, X_val, y_val, eps_grid=(1.1,1.35,1.5,1.75,2.0), alpha_grid=(1e-5,1e-4,1e-3,1e-2)):
    """
    Tune Huber regression hyperparameters.
    """
    logger.info(f"Tuning Huber regression on {len(X_train)} samples")
    best = {"model": None, "val_r2": -np.inf, "params": None}
    for eps in eps_grid:
        for alpha in alpha_grid:
            model = fit_huber_regression(X_train, y_train, epsilon=eps, alpha=alpha)
            score = predictive_r2(y_val, model.predict(X_val))
            if score > best["val_r2"]:
                best = {"model": model, "val_r2": score, "params": {"epsilon": eps, "alpha": alpha}}
    logger.info(f"Best Huber val R2: {best['val_r2']:.4f}, params: {best['params']}")
    return best


def fit_pooled_huber_3(X_train, y_train, epsilon=1.35, alpha=0.0001):
    """
    Fit pooled Huber regression with 3 factors (size, value, momentum).
    This is Gu et al. (2020) OLS-3 benchmark.
    """
    logger.debug(f"Fitting pooled Huber 3-factor model: shape={X_train.shape}")
    return fit_huber_regression(X_train, y_train, epsilon=epsilon, alpha=alpha)


def fit_pooled_huber_full(X_train, y_train, epsilon=1.35, alpha=0.0001):
    """
    Fit pooled Huber regression with all features.
    This is Gu et al. (2020) OLS-full model.
    """
    logger.debug(f"Fitting pooled Huber full model: shape={X_train.shape}")
    return fit_huber_regression(X_train, y_train, epsilon=epsilon, alpha=alpha)


class PCRModel:
    """
    Principal Component Regression.
    
    Gu et al. (2020) use PCR with varying number of principal components.
    """
    def __init__(self, n_components, robust=True, huber_epsilon=1.35, huber_alpha=0.0001):
        self.n_components = n_components
        self.robust = robust  # Always use Huber (Gu et al. 2020)
        self.huber_epsilon = huber_epsilon
        self.huber_alpha = huber_alpha
        self.pca = PCA(n_components=n_components)
        self.regressor = None

    def fit(self, X, y):
        logger.debug(f"Fitting PCR with {self.n_components} components, robust={self.robust}")
        Z = self.pca.fit_transform(X)
        # Gu et al. (2020) always use Huber regression
        self.regressor = HuberRegressor(epsilon=self.huber_epsilon, alpha=self.huber_alpha, max_iter=500)
        self.regressor.fit(Z, y)
        return self

    def predict(self, X):
        Z = self.pca.transform(X)
        return self.regressor.predict(Z)


def tune_pcr(X_train, y_train, X_val, y_val, k_grid=(3,5,10,20,30,50), robust=True):
    """
    Tune PCR number of components.
    """
    logger.info(f"Tuning PCR: k_grid={k_grid}, robust={robust}")
    best = {"model": None, "val_r2": -np.inf, "params": None}
    max_k = min(X_train.shape[0], X_train.shape[1])
    for k in k_grid:
        if k > max_k:
            continue
        model = PCRModel(n_components=k, robust=robust).fit(X_train, y_train)
        score = predictive_r2(y_val, model.predict(X_val))
        if score > best["val_r2"]:
            best = {"model": model, "val_r2": score, "params": {"n_components": k, "robust": robust}}
    logger.info(f"Best PCR val R2: {best['val_r2']:.4f}, params: {best['params']}")
    return best


def fit_pls(X_train, y_train, n_components):
    """
    Partial Least Squares regression.
    """
    logger.debug(f"Fitting PLS with {n_components} components")
    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(X_train, y_train)
    logger.debug("PLS fitting completed")
    return model


def tune_pls(X_train, y_train, X_val, y_val, k_grid=(2,3,5,10,15,20)):
    """
    Tune PLS number of components.
    """
    logger.info(f"Tuning PLS: k_grid={k_grid}")
    best = {"model": None, "val_r2": -np.inf, "params": None}
    max_k = min(X_train.shape[1], max(1, X_train.shape[0] - 1))
    for k in k_grid:
        if k > max_k:
            continue
        model = fit_pls(X_train, y_train, n_components=k)
        score = predictive_r2(y_val, model.predict(X_val).ravel())
        if score > best["val_r2"]:
            best = {"model": model, "val_r2": score, "params": {"n_components": k}}
    logger.info(f"Best PLS val R2: {best['val_r2']:.4f}, params: {best['params']}")
    return best