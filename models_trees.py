from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer

from evaluation import gu_pred_r2

logger = logging.getLogger("eap_ml.models_trees")


gu_pred_r2_scorer = make_scorer(gu_pred_r2)


def tune_random_forest(
    X_train,
    y_train,
    X_val,
    y_val,
    random_state,
    param_grid: dict,
):
    """
    Tune Random Forest with GridSearchCV.

    Uses a fixed train/validation split, so validation observations are
    never used for fitting. Selects hyperparameters by Gu et al. (2020)
    predicted R2 on the validation set.
    """


    logger.info(
        f"Tuning Random Forest on {len(X_train)} train and "
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
        estimator=RandomForestRegressor(
            n_jobs=1,
            random_state=random_state,
        ),
        param_grid=param_grid,
        scoring=gu_pred_r2_scorer,
        cv=splitter,
        refit=False,
        n_jobs=-1,
    )

    grid.fit(X, y)

    best_params = grid.best_params_

    # Refit only on the training set, exactly as in the previous implementation.
    best_model = RandomForestRegressor(random_state=random_state,**best_params)
    best_model.fit(X_train, y_train)

    feature_importance = pd.Series(
    best_model.feature_importances_,
    index=X_train.columns,
    name="importance",
    )

    best_score = gu_pred_r2(y_val, best_model.predict(X_val))

    logger.info(
        f"Best Random Forest val R2: {best_score:.4f}, params: {best_params}"
    )

    return {
        "model": best_model,
        "val_r2": best_score,
        "params": best_params,
        "cv_results": grid.cv_results_,
        "feature_importance": feature_importance,
    }


def tune_gbrt(
    X_train,
    y_train,
    X_val,
    y_val,
    random_state,
    param_grid: dict,
    
):
    """
    Tune GBRT with GridSearchCV.

    Uses a fixed train/validation split, so validation observations are
    never used for fitting. Selects hyperparameters by Gu et al. (2020)
    predicted R2 on the validation set.
    """


    logger.info(
        f"Tuning GBRT on {len(X_train)} train and "
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
        estimator=GradientBoostingRegressor(random_state=random_state),
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
    best_model = GradientBoostingRegressor(random_state=random_state,**best_params)
    best_model.fit(X_train, y_train)

    feature_importance = pd.Series(
        best_model.feature_importances_,
        index=X_train.columns,
        name="importance",
    )

    best_score = gu_pred_r2(y_val, best_model.predict(X_val))

    logger.info(
        f"Best GBRT val R2: {best_score:.4f}, params: {best_params}"
    )

    return {
        "model": best_model,
        "val_r2": best_score,
        "params": best_params,
        "cv_results": grid.cv_results_,
        "feature_importance": feature_importance,
    }