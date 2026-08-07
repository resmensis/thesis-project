from __future__ import annotations
import numpy as np
from sklearn.neural_network import MLPRegressor


def predictive_r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    num = np.sum((y_true - y_pred) ** 2)
    return 1.0 - num / denom if denom > 0 else np.nan


def fit_mlp_model(X_train, y_train, X_val, y_val, arch_params, random_state=42):
    model = MLPRegressor(
        hidden_layer_sizes=arch_params.get("hidden_layer_sizes", (64,64,64)),
        activation=arch_params.get("activation", "relu"),
        alpha=arch_params.get("alpha", 1e-4),
        learning_rate_init=arch_params.get("learning_rate_init", 1e-3),
        max_iter=arch_params.get("max_iter", 300),
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    val_r2 = predictive_r2(y_val, model.predict(X_val))
    return {"model": model, "val_r2": val_r2, "params": arch_params}


def tune_mlp_models(X_train, y_train, X_val, y_val, depths=(1,2,3,4,5), widths=(32,64,128), activations=("relu","tanh"), alpha_grid=(1e-5,1e-4,1e-3), learning_rate_grid=(1e-3,5e-4), max_iter=300, random_state=42):
    best = {"model": None, "val_r2": -np.inf, "params": None}
    for depth in depths:
        for width in widths:
            for activation in activations:
                for alpha in alpha_grid:
                    for lr in learning_rate_grid:
                        arch = {
                            "hidden_layer_sizes": tuple([width] * depth),
                            "activation": activation,
                            "alpha": alpha,
                            "learning_rate_init": lr,
                            "max_iter": max_iter,
                        }
                        res = fit_mlp_model(X_train, y_train, X_val, y_val, arch, random_state=random_state)
                        if res["val_r2"] > best["val_r2"]:
                            best = res
    return best
