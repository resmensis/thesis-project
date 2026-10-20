from __future__ import annotations

import logging
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.base import clone
from sklearn.metrics import make_scorer

from evaluation import gu_pred_r2


logger = logging.getLogger("eap_ml.models_mlp")


gu_pred_r2_scorer = make_scorer(gu_pred_r2)





def tune_mlp_models(
    X_train,
    y_train,
    X_val,
    y_val,
    architectures: dict,
    param_grid: dict,
    random_state: int = 42,
):
    """
    Tune each MLP architecture independently using GridSearchCV.

    Uses a fixed train/validation split, so validation observations are
    never used for fitting. Selects hyperparameters by Gu et al. (2020)
    predicted R2 on the validation set.

    Returns:
        {
            "NN1": {
                "model": ...,
                "val_r2": ...,
                "params": ...,
                "cv_results": ...,
            },
            "NN2": {...},
        }
    """
    logger.info(
        f"Tuning MLP on {len(X_train)} train and "
        f"{len(X_val)} validation samples"
    )

    if not architectures:
        raise ValueError("At least one MLP architecture must be supplied.")

    X = np.vstack([X_train, X_val])
    y = np.concatenate([
        np.asarray(y_train).ravel(),
        np.asarray(y_val).ravel(),
    ])

    test_fold = np.concatenate([
        np.full(len(X_train), -1, dtype=int),
        np.zeros(len(X_val), dtype=int),
    ])
    splitter = PredefinedSplit(test_fold=test_fold)

    results = {}

    for model_name, hidden_layers in architectures.items():
        hidden_layers = tuple(hidden_layers)

        estimator = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            solver="adam",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=random_state,
        )

        grid = GridSearchCV(
            estimator=estimator,
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

        # Create a fresh estimator with the selected hyperparameters.
        best_model = clone(estimator).set_params(**best_params)
        best_model.fit(X_train, np.asarray(y_train).ravel())

        best_score = float(
            gu_pred_r2(
                np.asarray(y_val).ravel(),
                best_model.predict(X_val).ravel(),
            )
        )

        logger.info(
            "%s: best MLP validation R2=%.6f; params=%s",
            model_name,
            best_score,
            best_params,
        )

        results[model_name] = {
            "model": best_model,
            "val_r2": best_score,
            "params": best_params,
            "cv_results": grid.cv_results_,
        }

    return results