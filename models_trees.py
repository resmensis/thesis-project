from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

logger = logging.getLogger("eap_ml.models_trees")


def predictive_r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    num = np.sum((y_true - y_pred) ** 2)
    return 1.0 - num / denom if denom > 0 else np.nan


def fit_gbrt(X_train, y_train, n_estimators=200, learning_rate=0.05, max_depth=2, min_samples_leaf=100, random_state=42):
    logger.debug(f"Fitting GBRT: n_estimators={n_estimators}, lr={learning_rate}, depth={max_depth}")
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state, loss="squared_error")
    model.fit(X_train, y_train)
    logger.debug("GBRT fitting completed")
    return model


def tune_gbrt(X_train, y_train, X_val, y_val, n_estimators_grid=(100,200,500), learning_rate_grid=(0.01,0.05,0.1), depth_grid=(1,2,3), leaf_grid=(25,50,100), random_state=42):
    logger.info("Tuning GBRT")
    best = {"model": None, "val_r2": -np.inf, "params": None}
    for n_estimators in n_estimators_grid:
        for lr in learning_rate_grid:
            for depth in depth_grid:
                for leaf in leaf_grid:
                    model = fit_gbrt(X_train, y_train, n_estimators=n_estimators, learning_rate=lr, max_depth=depth, min_samples_leaf=leaf, random_state=random_state)
                    score = predictive_r2(y_val, model.predict(X_val))
                    if score > best["val_r2"]:
                        best = {"model": model, "val_r2": score, "params": {"n_estimators": n_estimators, "learning_rate": lr, "max_depth": depth, "min_samples_leaf": leaf}}
    logger.info(f"Best GBRT val R2: {best['val_r2']:.4f}, params: {best['params']}")
    return best


def fit_random_forest(X_train, y_train, n_estimators=500, max_depth=6, max_features="sqrt", min_samples_leaf=100, n_jobs=-1, random_state=42):
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
                    score = predictive_r2(y_val, model.predict(X_val))
                    if score > best["val_r2"]:
                        best = {"model": model, "val_r2": score, "params": {"n_estimators": n_estimators, "max_depth": depth, "max_features": max_features, "min_samples_leaf": leaf}}
    logger.info(f"Best RF val R2: {best['val_r2']:.4f}, params: {best['params']}")
    return best


def feature_importance_table(model, feature_names, top_n=25):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    return importances.head(top_n).reset_index().rename(columns={"index": "feature", 0: "importance"})