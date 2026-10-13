from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from itertools import combinations
from statsmodels.tsa.stattools import diebold_mariano_test

from pathlib import Path
from io_utils import ensure_dir, save_parquet, save_csv

logger = logging.getLogger("eap_ml.evaluation")


def predictive_r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    num = np.sum((y_true - y_pred) ** 2)
    return 1.0 - num / denom if denom > 0 else np.nan


def compute_sharpe_ratio(returns_series):
    r = pd.Series(returns_series).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return np.sqrt(12.0) * r.mean() / r.std(ddof=1)


def aggregate_portfolio_predictions(meta, y_true, y_pred, weight_col="me"):
    df = meta.copy()
    df["y_true"] = np.asarray(y_true)
    df["y_pred"] = np.asarray(y_pred).ravel()
    if weight_col not in df.columns:
        df[weight_col] = 1.0
    df["w"] = df[weight_col].fillna(0.0).clip(lower=0.0)
    def _agg(g):
        if g["w"].sum() <= 0:
            return pd.Series({"port_true": np.nan, "port_pred": np.nan})
        return pd.Series({"port_true": np.average(g["y_true"], weights=g["w"]), "port_pred": np.average(g["y_pred"], weights=g["w"])})
    return df.groupby("date").apply(_agg).reset_index()


def build_market_timing_strategy(meta, y_true, y_pred, threshold=0.0):
    port = aggregate_portfolio_predictions(meta, y_true, y_pred)
    port["signal"] = (port["port_pred"] > threshold).astype(float)
    port["strategy_ret"] = port["signal"] * port["port_true"]
    return port


def build_decile_portfolios(predictions, returns, weights, dates, n_bins=10):
    df = pd.DataFrame({"date": pd.to_datetime(dates), "pred": np.asarray(predictions).ravel(), "ret": np.asarray(returns), "w": np.asarray(weights) if weights is not None else 1.0}).dropna(subset=["pred", "ret"])
    ew_rows, vw_rows = [], []
    for dt, g in df.groupby("date"):
        if len(g) < n_bins:
            continue
        g = g.copy()
        g["bin"] = pd.qcut(g["pred"], q=n_bins, labels=False, duplicates="drop")
        if g["bin"].nunique() < n_bins:
            continue
        top = g[g["bin"] == g["bin"].max()]
        bottom = g[g["bin"] == g["bin"].min()]
        ew_ret = top["ret"].mean() - bottom["ret"].mean()
        top_w = pd.Series(top["w"]).clip(lower=0.0)
        bot_w = pd.Series(bottom["w"]).clip(lower=0.0)
        vw_ret = np.average(top["ret"], weights=top_w) - np.average(bottom["ret"], weights=bot_w) if top_w.sum() > 0 and bot_w.sum() > 0 else np.nan
        ew_rows.append({"date": dt, "ls_ret": ew_ret})
        vw_rows.append({"date": dt, "ls_ret": vw_ret})
    return {"equal_weight": pd.DataFrame(ew_rows), "value_weight": pd.DataFrame(vw_rows)}


def evaluate_all_models(models_dict, data_dict):
    logger.info(f"Evaluating {len(models_dict)} models")
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    meta_test = data_dict["meta_test"]
    stock_rows, portfolio_rows, sharpe_rows = [], [], []
    for name, model in models_dict.items():
        pred = np.asarray(model.predict(X_test)).ravel()
        stock_r2 = predictive_r2(y_test, pred)
        port_df = aggregate_portfolio_predictions(meta_test, y_test, pred)
        port_r2 = predictive_r2(port_df["port_true"], port_df["port_pred"])
        timing = build_market_timing_strategy(meta_test, y_test, pred)
        timing_sharpe = compute_sharpe_ratio(timing["strategy_ret"])
        deciles = build_decile_portfolios(predictions=pred, returns=y_test, weights=meta_test["me"] if "me" in meta_test.columns else None, dates=meta_test["date"])
        ew_sharpe = compute_sharpe_ratio(deciles["equal_weight"]["ls_ret"]) if not deciles["equal_weight"].empty else np.nan
        vw_sharpe = compute_sharpe_ratio(deciles["value_weight"]["ls_ret"]) if not deciles["value_weight"].empty else np.nan
        stock_rows.append({"model": name, "stock_r2": stock_r2})
        portfolio_rows.append({"model": name, "portfolio_r2": port_r2})
        sharpe_rows.append({"model": name, "timing_sharpe": timing_sharpe, "decile_ls_ew_sharpe": ew_sharpe, "decile_ls_vw_sharpe": vw_sharpe})
    logger.info("Model evaluation completed")
    return {
        "stock_level": pd.DataFrame(stock_rows).sort_values("stock_r2", ascending=False),
        "portfolio_level": pd.DataFrame(portfolio_rows).sort_values("portfolio_r2", ascending=False),
        "sharpe": pd.DataFrame(sharpe_rows).sort_values("decile_ls_vw_sharpe", ascending=False),
    }


from scipy import stats


def _get_subsample_size(regime_config) -> int:
    """Return the number of stocks in the top/bottom mv subsamples."""
    if regime_config.mode == "coding":
        return regime_config.top_bottom_coding
    return regime_config.top_bottom


def diebold_mariano_test(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    h: int = 1,
) -> dict:
    """
    Diebold-Mariano test comparing two models' forecast losses.

    A negative DM statistic means model A has a lower average loss
    than model B.
    """
    loss_diff = np.asarray(losses_a) - np.asarray(losses_b)
    loss_diff = loss_diff[~np.isnan(loss_diff)]

    n = len(loss_diff)
    if n < 2:
        return {"dm": np.nan, "p_value": np.nan}

    mean_diff = loss_diff.mean()

    # Newey-West-style variance estimate for autocorrelated forecast errors.
    gamma_0 = np.mean((loss_diff - mean_diff) ** 2)

    max_lag = h - 1
    gamma_sum = 0.0
    for lag in range(1, max_lag + 1):
        gamma_lag = np.mean(
            (loss_diff[lag:] - mean_diff)
            * (loss_diff[:-lag] - mean_diff)
        )
        gamma_sum += 2.0 * gamma_lag

    variance = (gamma_0 + gamma_sum) / n

    if variance <= 0:
        return {"dm": np.nan, "p_value": np.nan}

    dm = mean_diff / np.sqrt(variance)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))

    return {"dm": dm, "p_value": p_value}


def compute_model_r2_metrics(
    predictions: pd.DataFrame,
    regime_config,
) -> pd.DataFrame:
    """
    Compute overall, top-mv, and bottom-mv out-of-sample R2 for every model.

    For each month:
      - top: the N stocks with the highest market value (mv)
      - bottom: the N stocks with the lowest market value (mv)

    N is determined by the data regime:
      - coding regime: regime_config.top_bottom_coding
      - full regime: regime_config.top_bottom
    """
    model_cols = [
        column for column in predictions.columns
        if column.startswith("pred_")
    ]

    subsample_size = _get_subsample_size(regime_config)

    rows = []

    for model_col in model_cols:
        model_name = model_col.removeprefix("pred_")

        df = predictions[
            ["permno", "date", "excess_ret_total", "mv", model_col]
        ].dropna(
            subset=["excess_ret_total", "mv", model_col]
        )

        overall_r2 = predictive_r2(
            df["excess_ret_total"],
            df[model_col],
        )

        top_frames = []
        bottom_frames = []

        for _, month_df in df.groupby("date"):
            month_df = month_df.sort_values("mv", ascending=False)

            top_frames.append(month_df.head(subsample_size))
            bottom_frames.append(month_df.tail(subsample_size))

        top_df = pd.concat(top_frames, ignore_index=True)
        bottom_df = pd.concat(bottom_frames, ignore_index=True)

        top_r2 = predictive_r2(
            top_df["excess_ret_total"],
            top_df[model_col],
        )

        bottom_r2 = predictive_r2(
            bottom_df["excess_ret_total"],
            bottom_df[model_col],
        )

        rows.append({
            "model": model_name,
            "r2_overall": overall_r2,
            "r2_top": top_r2,
            "r2_bottom": bottom_r2,
            "subsample_size": subsample_size,
            "n_obs_overall": len(df),
            "n_obs_top": len(top_df),
            "n_obs_bottom": len(bottom_df),
        })

    return pd.DataFrame(rows).sort_values(
        "r2_overall",
        ascending=False,
    )


def compute_diebold_mariano_tests(
    predictions: pd.DataFrame,
    loss: str = "squared_error",
    horizon: int = 1,
    harvey_adjustment: bool = False,
) -> pd.DataFrame:
    """
    Compute Diebold-Mariano tests for every unique pair of models.

    The null hypothesis is that both models have equal predictive accuracy.

    Parameters
    ----------
    predictions : DataFrame
        Wide prediction store with columns:
        permno, date, excess_ret_total, mv, pred_<MODEL>.

    loss : str
        'squared_error' or 'absolute_error'.

    horizon : int
        Forecast horizon. Use 1 for monthly one-step-ahead forecasts.

    harvey_adjustment : bool
        Whether to apply the Harvey et al. small-sample correction.

    Returns
    -------
    DataFrame
        One row per model pair, with DM statistic and p-value.
    """
    model_cols = [
        column for column in predictions.columns
        if column.startswith("pred_")
    ]

    if len(model_cols) < 2:
        raise ValueError(
            "At least two prediction columns are required for "
            "Diebold-Mariano tests."
        )

    if loss not in {"squared_error", "absolute_error"}:
        raise ValueError(
            "loss must be either 'squared_error' or 'absolute_error'."
        )

    rows = []

    for model_a_col, model_b_col in combinations(model_cols, 2):
        comparison = predictions[
            ["permno", "date", model_a_col, model_b_col]
        ].dropna()

        if comparison.empty:
            continue

        y_true = comparison["excess_ret_total"]
        pred_a = comparison[model_a_col]
        pred_b = comparison[model_b_col]

        result = diebold_mariano_test(
            y_true,
            pred_a,
            pred_b,
            criterion="mse" if loss == "squared_error" else "mae",
            power=2 if loss == "squared_error" else 1,
            harvey_adj=harvey_adjustment,
            horizon=horizon,
        )

        rows.append({
            "model_a": model_a_col.removeprefix("pred_"),
            "model_b": model_b_col.removeprefix("pred_"),
            "loss": loss,
            "dm_statistic": result.statistic,
            "p_value": result.pvalue,
            "n_obs": len(comparison),
        })

    return pd.DataFrame(rows)


def build_ml_portfolios(
    predictions: pd.DataFrame,
    n_quantiles: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build monthly quantile portfolios for every prediction model.

    For each month and model:
      1. Sort stocks by the model prediction.
      2. Assign stocks to n_quantile portfolios.
      3. Compute value-weighted portfolio returns using mv.
      4. Construct the high-minus-low portfolio:
         highest-prediction portfolio minus lowest-prediction portfolio.

    Returns
    -------
    portfolio_returns : DataFrame
        One row per month and model, with returns for each quantile
        portfolio and the high-minus-low portfolio.

    portfolio_r2 : DataFrame
        One row per model, with the portfolio-level predictive R2.
    """
    model_cols = [
        c for c in predictions.columns
        if c.startswith("pred_")
    ]

    portfolio_rows = []
    portfolio_r2_rows = []

    for model_col in model_cols:
        model_name = model_col.removeprefix("pred_")

        df = predictions[
            ["permno", "date", "excess_ret_total", "mv", model_col]
        ].dropna(
            subset=["excess_ret_total", "mv", model_col]
        ).copy()

        df = df[df["mv"] > 0]

        monthly_portfolios = []

        for date, month_df in df.groupby("date"):
            if len(month_df) < n_quantiles:
                continue

            month_df = month_df.copy()

            try:
                month_df["portfolio"] = pd.qcut(
                    month_df[model_col],
                    q=n_quantiles,
                    labels=False,
                    duplicates="drop",
                )
            except ValueError:
                continue

            n_portfolios = month_df["portfolio"].nunique()

            if n_portfolios < n_quantiles:
                continue

            month_summary = {
                "date": date,
                "model": model_name,
            }

            for portfolio in range(n_quantiles):
                portfolio_df = month_df[
                    month_df["portfolio"] == portfolio
                ]

                weights = portfolio_df["mv"]
                portfolio_return = np.average(
                    portfolio_df["excess_ret_total"],
                    weights=weights,
                )

                month_summary[f"portfolio_{portfolio + 1}"] = (
                    portfolio_return
                )

            low = month_summary["portfolio_1"]
            high = month_summary[f"portfolio_{n_quantiles}"]

            month_summary["high_minus_low"] = high - low

            monthly_portfolios.append(month_summary)

        model_portfolios = pd.DataFrame(monthly_portfolios)

        if model_portfolios.empty:
            continue

        portfolio_rows.append(model_portfolios)

        # Portfolio-level predictive R2:
        # compare the average realized return of the ten portfolios with
        # the average predicted return of the ten portfolios.
        portfolio_prediction = []

        for date, month_df in df.groupby("date"):
            if len(month_df) < n_quantiles:
                continue

            month_df = month_df.copy()

            try:
                month_df["portfolio"] = pd.qcut(
                    month_df[model_col],
                    q=n_quantiles,
                    labels=False,
                    duplicates="drop",
                )
            except ValueError:
                continue

            if month_df["portfolio"].nunique() < n_quantiles:
                continue

            realized = []
            predicted = []

            for portfolio in range(n_quantiles):
                portfolio_df = month_df[
                    month_df["portfolio"] == portfolio
                ]

                realized.append(
                    np.average(
                        portfolio_df["excess_ret_total"],
                        weights=portfolio_df["mv"],
                    )
                )
                predicted.append(
                    np.average(
                        portfolio_df[model_col],
                        weights=portfolio_df["mv"],
                    )
                )

            portfolio_prediction.append({
                "date": date,
                "port_true": np.mean(realized),
                "port_pred": np.mean(predicted),
            })

        portfolio_prediction_df = pd.DataFrame(portfolio_prediction)

        portfolio_r2_rows.append({
            "model": model_name,
            "portfolio_r2": predictive_r2(
                portfolio_prediction_df["port_true"],
                portfolio_prediction_df["port_pred"],
            ),
            "hl_sharpe": compute_sharpe_ratio(
                model_portfolios["high_minus_low"]
            ),
            "n_months": len(model_portfolios),
        })

    portfolio_returns = (
        pd.concat(portfolio_rows, ignore_index=True)
        if portfolio_rows
        else pd.DataFrame()
    )

    portfolio_metrics = (
        pd.DataFrame(portfolio_r2_rows)
        if portfolio_r2_rows
        else pd.DataFrame()
    )

    return portfolio_returns, portfolio_metrics


def evaluate_predictions(
    predictions: pd.DataFrame,
    regime_config,
    output_dir: str | Path = "output",
    n_quantiles: int = 10,
    dm_loss: str = "squared_error",
    horizon: int = 1,
    harvey_adjustment: bool = False,
    save_outputs: bool = True,
) -> dict:
    """
    Evaluate all model prediction columns in the wide prediction store.

    Parameters
    ----------
    predictions : DataFrame
        Output of run_expanding_window(). Must contain:
        permno, date, excess_ret_total, mv, and pred_<MODEL> columns.

    regime_config : DataRegimeConfig
        Determines the number of stocks in top/bottom mv subsamples.

    benchmark_model : str, optional
        If provided, Diebold-Mariano tests compare every other model
        with this benchmark.

    n_quantiles : int
        Number of monthly prediction-sorted portfolios.

    dm_loss : str
        Loss function for Diebold-Mariano tests:
        'squared_error' or 'absolute_error'.

    Returns
    -------
    dict
        Dictionary with evaluation tables.
    """
    logger.info("Starting prediction evaluation.")

    required = [
        "permno",
        "date",
        "excess_ret_total",
        "mv",
    ]

    missing = [
        column for column in required
        if column not in predictions.columns
    ]

    if missing:
        raise KeyError(
            f"predictions is missing required columns: {missing}"
        )

    model_cols = [
        column for column in predictions.columns
        if column.startswith("pred_")
    ]

    if not model_cols:
        raise ValueError(
            "No prediction columns found. Expected columns beginning "
            "with 'pred_'."
        )

    r2_metrics = compute_model_r2_metrics(
        predictions=predictions,
        regime_config=regime_config,
    )
    out_path_r2_metrics = f"{output_dir}/{regime_config.mode}/r2_metrics.csv"
    save_csv(r2_metrics,out_path_r2_metrics,enabled=save_outputs)

    dm_tests = compute_diebold_mariano_tests(
        predictions=predictions,
        loss=dm_loss,
        horizon=horizon,
        harvey_adjustment=harvey_adjustment,
    )
    out_path_dm_tests = f"{output_dir}/{regime_config.mode}/dm_tests.csv"
    save_csv(dm_tests,out_path_dm_tests,enabled=save_outputs)

    portfolio_returns, portfolio_metrics = build_ml_portfolios(
        predictions=predictions,
        n_quantiles=n_quantiles,
    )
    out_path_portfolio_returns = f"{output_dir}/{regime_config.mode}/portfolio_returns.parquet"
    save_parquet(portfolio_returns,out_path_portfolio_returns,enabled=save_outputs)
    out_path_portfolio_metrics = f"{output_dir}/{regime_config.mode}/portfolio_metrics.csv"
    save_csv(portfolio_metrics,out_path_portfolio_metrics,enabled=save_outputs)

    logger.info(
        f"Evaluation completed for {len(model_cols)} models."
        f"Metrics save to {output_dir}/{regime_config.mode}"
    )

    return {
        "r2_metrics": r2_metrics,
        "dm_tests": dm_tests,
        "portfolio_returns": portfolio_returns,
        "portfolio_metrics": portfolio_metrics,
    }