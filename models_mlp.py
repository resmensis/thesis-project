from __future__ import annotations

import logging
from itertools import product

import numpy as np
from sklearn.neural_network import MLPRegressor

from evaluation import gu_pred_r2

logger = logging.getLogger("eap_ml.models_mlp")


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