from __future__ import annotations

import logging
from itertools import product
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer

from evaluation import gu_pred_r2


logger = logging.getLogger("eap_ml.models_alternative")




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


def tune_huber_regression(X_train, y_train, X_val, y_val, eps_grid=(2.576,3.291), alpha_grid=(1e-3,1e-2,1e-1,1e0,1e1,1e2)):
    """
    Tune Huber regression hyperparameters.
    """
    logger.info(f"Tuning Huber regression on {len(X_train)} samples")
    best = {"model": None, "val_r2": -np.inf, "params": None}
    for eps in eps_grid:
        for alpha in alpha_grid:
            model = fit_huber_regression(X_train, y_train, epsilon=eps, alpha=alpha)
            score = gu_pred_r2(y_val, model.predict(X_val))
            if score > best["val_r2"]:
                best = {"model": model, "val_r2": score, "params": {"epsilon": eps, "alpha": alpha}}
    logger.info(f"Best Huber val R2: {best['val_r2']:.4f}, params: {best['params']}")
    return best


def fit_gbrt(X_train, y_train, n_estimators=200, learning_rate=0.05, max_depth=2, min_samples_leaf=100, random_state=42):
    logger.debug(f"Fitting GBRT: n_estimators={n_estimators}, lr={learning_rate}, depth={max_depth}")
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state, loss="squared_error")
    model.fit(X_train, y_train)
    logger.debug("GBRT fitting completed")
    return model


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


def tune_gbrt(X_train, y_train, X_val, y_val, n_estimators_grid=(100,200,500), learning_rate_grid=(0.01,0.05,0.1), depth_grid=(1,2,3), leaf_grid=(25,50,100), random_state=42):
    logger.info("Tuning GBRT")
    best = {"model": None, "val_r2": -np.inf, "params": None}
    for n_estimators in n_estimators_grid:
        for lr in learning_rate_grid:
            for depth in depth_grid:
                for leaf in leaf_grid:
                    model = fit_gbrt(X_train, y_train, n_estimators=n_estimators, learning_rate=lr, max_depth=depth, min_samples_leaf=leaf, random_state=random_state)
                    score = gu_pred_r2(y_val, model.predict(X_val))
                    if score > best["val_r2"]:
                        best = {"model": model, "val_r2": score, "params": {"n_estimators": n_estimators, "learning_rate": lr, "max_depth": depth, "min_samples_leaf": leaf}}
    logger.info(f"Best GBRT val R2: {best['val_r2']:.4f}, params: {best['params']}")
    return best


def fit_random_forest(X_train, y_train, n_estimators=500, max_depth=6, max_features=3, min_samples_leaf=100, n_jobs=-1, random_state=42):
    logger.debug(f"Fitting RF: n_estimators={n_estimators}, depth={max_depth}, max_features={max_features}")
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, n_jobs=n_jobs, random_state=random_state)
    model.fit(X_train, y_train)
    logger.debug("RF fitting completed")
    return model


def tune_random_forest(X_train, y_train, X_val, y_val, n_estimators_grid=(200,500), depth_grid=(4,6,8,None), max_features_grid=("sqrt","log2",0.3,0.5), leaf_grid=(25,50,100), random_state=42):
    logger.info("Tuning Random Forest")
    best = {"model": None, "val_r2": -np.inf, "params": None}
    for n_estimators in n_estimators_grid:
        for depth in depth_grid:
            for max_features in max_features_grid:
                for leaf in leaf_grid:
                    model = fit_random_forest(X_train, y_train, n_estimators=n_estimators, max_depth=depth, max_features=max_features, min_samples_leaf=leaf, random_state=random_state)
                    score = gu_pred_r2(y_val, model.predict(X_val))
                    if score > best["val_r2"]:
                        best = {"model": model, "val_r2": score, "params": {"n_estimators": n_estimators, "max_depth": depth, "max_features": max_features, "min_samples_leaf": leaf}}
    logger.info(f"Best RF val R2: {best['val_r2']:.4f}, params: {best['params']}")
    return best


def fit_mlp_model(
    X_train,
    y_train,
    X_val,
    y_val,
    arch_params: dict,
    random_state: int = 42,
) -> dict:
    """Fit one MLP configuration and score it on the external validation set."""
    logger.debug("Fitting MLP: %s", arch_params)

    model = MLPRegressor(
        hidden_layer_sizes=tuple(arch_params["hidden_layer_sizes"]),
        activation=arch_params["activation"],
        alpha=arch_params["alpha"],
        learning_rate_init=arch_params["learning_rate_init"],
        max_iter=arch_params["max_iter"],
        solver="adam",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=random_state,
    )

    model.fit(X_train, np.asarray(y_train).ravel())

    val_r2 = float(
        gu_pred_r2(
            np.asarray(y_val).ravel(),
            model.predict(X_val).ravel(),
        )
    )

    return {
        "model": model,
        "val_r2": val_r2,
        "params": dict(arch_params),
    }


def tune_mlp_models(
    X_train,
    y_train,
    X_val,
    y_val,
    architectures: dict,
    param_grid: dict,
    random_state: int = 42,
) -> dict:
    """
    Tune each named architecture independently.

    Returns:
        {
            "NN1": {"model": ..., "val_r2": ..., "params": ...},
            "NN2": {"model": ..., "val_r2": ..., "params": ...},
            ...
        }
    """
    if not architectures:
        raise ValueError("At least one MLP architecture must be supplied.")

    required = {
        "activation",
        "alpha_grid",
        "learning_rate_grid",
        "max_iter",
    }

    missing = required.difference(param_grid)
    if missing:
        raise KeyError(f"Missing MLP grid settings: {sorted(missing)}")

    alpha_grid = tuple(param_grid["alpha_grid"])
    learning_rate_grid = tuple(param_grid["learning_rate_grid"])

    if not alpha_grid or not learning_rate_grid:
        raise ValueError("MLP tuning grids must not be empty.")

    results = {}

    for model_name, hidden_layers in architectures.items():
        hidden_layers = tuple(hidden_layers)

        if not hidden_layers or any(
            not isinstance(width, (int, np.integer)) or width < 1
            for width in hidden_layers
        ):
            raise ValueError(
                f"Invalid hidden-layer specification for "
                f"{model_name}: {hidden_layers}"
            )

        logger.info(
            "Tuning %s with hidden layers %s",
            model_name,
            hidden_layers,
        )

        best = {
            "model": None,
            "val_r2": -np.inf,
            "params": None,
        }

        for alpha, learning_rate in product(
            alpha_grid,
            learning_rate_grid,
        ):
            arch_params = {
                "hidden_layer_sizes": hidden_layers,
                "activation": param_grid["activation"],
                "alpha": alpha,
                "learning_rate_init": learning_rate,
                "max_iter": param_grid["max_iter"],
            }

            result = fit_mlp_model(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                arch_params=arch_params,
                random_state=random_state,
            )

            if (
                np.isfinite(result["val_r2"])
                and result["val_r2"] > best["val_r2"]
            ):
                best = result

        if best["model"] is None:
            raise RuntimeError(
                f"No valid configuration found for {model_name}. "
                "Check validation targets and validation scores."
            )

        results[model_name] = best

        logger.info(
            "%s: best validation R2=%.6f; params=%s",
            model_name,
            best["val_r2"],
            best["params"],
        )

    return results