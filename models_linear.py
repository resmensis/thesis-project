from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer

from evaluation import gu_pred_r2
from models_huber import HuberLinearRegressor


logger = logging.getLogger("eap_ml.models_linear")


gu_pred_r2_scorer = make_scorer(gu_pred_r2)


def fit_custom_huber(X_train, y_train, xi_quantile=0.999):
    """
    Fit Huber regression - this is what Gu et al. (2020) call "OLS".
    
    Gu et al. (2020) use robust regression with Huber loss as their baseline OLS model.
    """
    logger.debug(f"Fitting custom Huber regression: xi at {xi_quantile} quantile, shape={X_train.shape}")
    model = HuberLinearRegressor(xi_quantile=xi_quantile)
    model.fit(X_train, y_train)
    logger.debug("Huber regression fitting completed")
    return model



def tune_huber_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    param_grid: dict | None = None,
):
    """
    Tune Huber regression with GridSearchCV.

    Uses a fixed train/validation split, so validation observations are
    never used for fitting. Selects hyperparameters by Gu et al. (2020)
    predicted R2 on the validation set.
    """
    if param_grid is None:
        param_grid = {
            "epsilon": [2.576, 3.291],
            "alpha": [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
            "max_iter": [500],
        }

    logger.info(
        f"Tuning Huber regression on {len(X_train)} train and "
        f"{len(X_val)} validation samples"
    )

    X = np.vstack([X_train, X_val])
    y = np.concatenate([y_train, y_val])

    test_fold = np.concatenate([
        np.full(len(X_train), -1, dtype=int),
        np.zeros(len(X_val), dtype=int),
    ])
    splitter = PredefinedSplit(test_fold=test_fold)

    grid = GridSearchCV(
        estimator=HuberRegressor(),
        param_grid=param_grid,
        scoring=gu_pred_r2_scorer,
        cv=splitter,
        refit=False,
        n_jobs=-1,
        return_train_score=False,
        verbose=0,
    )

    grid.fit(X, y)

    best_params = grid.best_params_

    # Refit only on the training set, exactly as in the previous implementation.
    best_model = HuberRegressor(**best_params)
    best_model.fit(X_train, y_train)

    best_score = gu_pred_r2(y_val, best_model.predict(X_val))

    logger.info(
        f"Best Huber val R2: {best_score:.4f}, params: {best_params}"
    )

    return {
        "model": best_model,
        "val_r2": best_score,
        "params": best_params,
        "cv_results": grid.cv_results_,
    }


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
        score = gu_pred_r2(y_val, model.predict(X_val))
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
        score = gu_pred_r2(y_val, model.predict(X_val).ravel())
        if score > best["val_r2"]:
            best = {"model": model, "val_r2": score, "params": {"n_components": k}}
    logger.info(f"Best PLS val R2: {best['val_r2']:.4f}, params: {best['params']}")
    return best