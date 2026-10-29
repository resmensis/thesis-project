from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

import keras
from keras import layers, models

from scikeras.wrappers import KerasRegressor

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer

from evaluation import gu_pred_r2

logger = logging.getLogger("eap_ml.models_lstm")

gu_pred_r2_scorer = make_scorer(gu_pred_r2)


def build_panel_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a stock-month panel into fixed-length sequences.

    For stock i at month t, the target is excess_ret_total at t and the
    input contains features from months t - seq_len + 1 through t.
    """
    sequences = []
    targets = []

    for _, stock_df in df.sort_values(
        ["permno", "date"]
    ).groupby("permno", sort=False):

        stock_df = stock_df.reset_index(drop=True)

        if len(stock_df) < seq_len:
            continue

        X = stock_df[feature_cols].to_numpy(dtype=np.float32)
        y = stock_df[target_col].to_numpy(dtype=np.float32)

        for end_index in range(seq_len, len(stock_df) + 1):
            sequences.append(X[end_index - seq_len:end_index])
            targets.append(y[end_index - 1])

    if not sequences:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def build_lstm_model(
    input_dim: int,
    seq_len: int,
    hidden_dim: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build a slim Keras LSTM regression model.
    """
    inputs = keras.Input(shape=(seq_len, input_dim))

    x = inputs

    for layer_index in range(num_layers):
        x = layers.LSTM(
            units=hidden_dim,
            return_sequences=layer_index < num_layers - 1,
            dropout=dropout,
        )(x)

    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="mse",
    )

    return model


def tune_lstm_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    param_grid: dict,
    seq_len: int = 12,
    epochs: int = 20,
    batch_size: int = 1024,
    random_state: int = 42,
) -> dict:
    """
    Tune a Keras LSTM using GridSearchCV and the project's fixed
    train/validation split.

    seq_len is fixed because it determines the LSTM input shape.
    """
    logger.info(
        f"Tuning Keras LSTM: {len(train_df)} train and "
        f"{len(val_df)} validation observations; seq_len={seq_len}"
    )

    X_train, y_train = build_panel_sequences(
        df=train_df,
        feature_cols=feature_cols,
        target_col="excess_ret_total",
        seq_len=seq_len,
    )

    X_val, y_val = build_panel_sequences(
        df=val_df,
        feature_cols=feature_cols,
        target_col="excess_ret_total",
        seq_len=seq_len,
    )

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(
            "Insufficient observations to construct LSTM sequences."
        )

    X = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)

    test_fold = np.concatenate([
        np.full(len(X_train), -1, dtype=int),
        np.zeros(len(X_val), dtype=int),
    ])

    splitter = PredefinedSplit(test_fold=test_fold)

    def model_builder(
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
    ) -> keras.Model:
        return build_lstm_model(
            input_dim=len(feature_cols),
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
        )

    regressor = KerasRegressor(
        model=model_builder,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        random_state=random_state,
        input_dim=len(feature_cols),
        seq_len=seq_len,
    )

    sklearn_param_grid = {
        "model__hidden_dim": param_grid["hidden_dim"],
        "model__num_layers": param_grid["num_layers"],
        "model__dropout": param_grid["dropout"],
        "model__learning_rate": param_grid["learning_rate"],
    }

    grid = GridSearchCV(
        estimator=regressor,
        param_grid=sklearn_param_grid,
        scoring=gu_pred_r2_scorer,
        cv=splitter,
        refit=False,
        n_jobs=1,
        return_train_score=False,
        verbose=1,
    )

    grid.fit(X, y)

    best_params = {
        key.removeprefix("model__"): value
        for key, value in grid.best_params_.items()
    }

    best_model = model_builder(**best_params)

    best_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    best_score = float(
        gu_pred_r2(
            y_val,
            best_model.predict(X_val, verbose=0).ravel(),
        )
    )

    logger.info(
        "Best LSTM validation R2=%.6f; params=%s",
        best_score,
        {"seq_len": seq_len, **best_params},
    )

    return {
        "model": best_model,
        "val_r2": best_score,
        "params": {"seq_len": seq_len, **best_params},
        "cv_results": grid.cv_results_,
    }