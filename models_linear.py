from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
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


def tune_pcr_huber(
    X_train,
    y_train,
    X_val,
    y_val,
    random_state,
    param_grid: dict,
):
    """
    Tune PCR with Huber with GridSearchCV.

    Uses a fixed train/validation split, so validation observations are
    never used for fitting. Selects hyperparameters by Gu et al. (2020)
    predicted R2 on the validation set.
    """


    logger.info(
        f"Tuning PCR with Huber on {len(X_train)} train and "
        f"{len(X_val)} validation samples"
    )

    X = np.vstack([X_train, X_val])
    y = np.concatenate([y_train, y_val])

    test_fold = np.concatenate([
        np.full(len(X_train), -1, dtype=int),
        np.zeros(len(X_val), dtype=int),
    ])
    splitter = PredefinedSplit(test_fold=test_fold)

    pipeline = Pipeline([
        ("pca", PCA(random_state=random_state)),
        ("huber", HuberLinearRegressor()),
    ])

    grid = GridSearchCV(
        estimator=pipeline,
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
    best_model = pipeline.set_params(**best_params)
    best_model.fit(X_train, y_train)

    best_score = gu_pred_r2(y_val, best_model.predict(X_val))

    logger.info(
        f"Best PCR with Huber val R2: {best_score:.4f}, params: {best_params}"
    )

    return {
        "model": best_model,
        "val_r2": best_score,
        "params": best_params,
        "cv_results": grid.cv_results_,
    }


def tune_pls(
    X_train,
    y_train,
    X_val,
    y_val,
    param_grid: dict,
):
    """
    Tune PLS with GridSearchCV.

    Uses a fixed train/validation split, so validation observations are
    never used for fitting. Selects hyperparameters by Gu et al. (2020)
    predicted R2 on the validation set.
    """


    logger.info(
        f"Tuning PLS on {len(X_train)} train and "
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
        estimator=PLSRegression(),
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
    best_model = PLSRegression(**best_params)
    best_model.fit(X_train, y_train)

    best_score = gu_pred_r2(y_val, best_model.predict(X_val))

    logger.info(
        f"Best PLS val R2: {best_score:.4f}, params: {best_params}"
    )

    return {
        "model": best_model,
        "val_r2": best_score,
        "params": best_params,
        "cv_results": grid.cv_results_,
    }


def tune_pcr_linear(
    X_train,
    y_train,
    X_val,
    y_val,
    random_state,
    param_grid: dict,
):
    """
    Tune PCR with Linear with GridSearchCV.

    Uses a fixed train/validation split, so validation observations are
    never used for fitting. Selects hyperparameters by Gu et al. (2020)
    predicted R2 on the validation set.
    """


    logger.info(
        f"Tuning PCR with Linear on {len(X_train)} train and "
        f"{len(X_val)} validation samples"
    )

    X = np.vstack([X_train, X_val])
    y = np.concatenate([y_train, y_val])

    test_fold = np.concatenate([
        np.full(len(X_train), -1, dtype=int),
        np.zeros(len(X_val), dtype=int),
    ])
    splitter = PredefinedSplit(test_fold=test_fold)

    pipeline = Pipeline([
        ("pca", PCA(random_state=random_state)),
        ("linear", LinearRegression()),
    ])

    grid = GridSearchCV(
        estimator=pipeline,
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
    best_model = pipeline.set_params(**best_params)
    best_model.fit(X_train, y_train)

    best_score = gu_pred_r2(y_val, best_model.predict(X_val))

    logger.info(
        f"Best PCR with Linear val R2: {best_score:.4f}, params: {best_params}"
    )

    return {
        "model": best_model,
        "val_r2": best_score,
        "params": best_params,
        "cv_results": grid.cv_results_,
    }