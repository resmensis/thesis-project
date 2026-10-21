from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from itertools import combinations
from scipy import stats


from pathlib import Path
from io_utils import ensure_dir, save_parquet, save_csv

logger = logging.getLogger("eap_ml.evaluation")


def predictive_r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    num = np.sum((y_true - y_pred) ** 2)
    return 1.0 - num / denom if denom > 0 else np.nan

def gu_pred_r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum((y_true) ** 2)
    num = np.sum((y_true - y_pred) ** 2)
    return 1.0 - num / denom if denom > 0 else np.nan


def compute_sharpe_ratio(returns_series):
    r = pd.Series(returns_series).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return np.sqrt(12.0) * r.mean() / r.std(ddof=1)


def _get_subsample_size(regime_config) -> int:
    """Return the number of stocks in the top/bottom mv subsamples."""
    if regime_config.mode == "coding":
        return regime_config.top_bottom_coding
    return regime_config.top_bottom


def filter_predictions_until(
    predictions: pd.DataFrame,
    end_year: int,
) -> pd.DataFrame:
    """
    Return only prediction observations dated up to December 31 of end_year.
    """
    cutoff = pd.Timestamp(f"{end_year}-12-31")

    dates = pd.to_datetime(predictions["date"])

    return predictions.loc[dates <= cutoff].copy()


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

        overall_r2 = gu_pred_r2(
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

        top_r2 = gu_pred_r2(
            top_df["excess_ret_total"],
            top_df[model_col],
        )

        bottom_r2 = gu_pred_r2(
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
            ["permno", "date", "excess_ret_total", model_a_col, model_b_col]
        ].dropna()

        if comparison.empty:
            continue

        y_true = comparison["excess_ret_total"]
        pred_a = comparison[model_a_col]
        pred_b = comparison[model_b_col]

        if loss == "squared_error":
            losses_a = (y_true - pred_a) ** 2
            losses_b = (y_true - pred_b) ** 2
        else:
            losses_a = (y_true - pred_a).abs()
            losses_b = (y_true - pred_b).abs()

        dm_result = diebold_mariano_test(
            losses_a.to_numpy(),
            losses_b.to_numpy(),
            h=horizon,
        )

        rows.append({
            "model_a": model_a_col.removeprefix("pred_"),
            "model_b": model_b_col.removeprefix("pred_"),
            "loss": loss,
            "dm_statistic": dm_result["dm"],
            "p_value": dm_result["p_value"],
            "n_obs": len(comparison),
        })

    return pd.DataFrame(rows)


def compute_portfolio_metrics(
    portfolio_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute summary statistics for monthly portfolio returns.

    For every model, weighting scheme, and portfolio, this function
    computes:
      - average realized monthly return
      - monthly standard deviation of realized returns
      - annualized Sharpe ratio of realized returns
      - average predicted portfolio return
      - number of monthly observations
    """
    if portfolio_returns.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "weighting",
                "portfolio",
                "mean_realized_return",
                "std_realized_return",
                "sharpe_ratio",
                "mean_predicted_return",
                "n_months",
            ]
        )

    rows = []

    for (
        model_name,
        weighting,
        portfolio,
    ), group in portfolio_returns.groupby(
        ["model", "weighting", "portfolio"],
        sort=False,
    ):
        realized = group["realized_return"].dropna()
        predicted = group["predicted_return"].dropna()

        mean_realized = realized.mean() if len(realized) else np.nan
        std_realized = (
            realized.std(ddof=1) if len(realized) > 1 else np.nan
        )

        if len(realized) > 1 and std_realized != 0:
            sharpe_ratio = (
                np.sqrt(12.0) * mean_realized / std_realized
            )
        else:
            sharpe_ratio = np.nan

        rows.append({
            "model": model_name,
            "weighting": weighting,
            "portfolio": portfolio,
            "mean_predicted_return": (
                predicted.mean() if len(predicted) else np.nan
            ),
            "mean_realized_return": mean_realized,
            "std_realized_return": std_realized,
            "sharpe_ratio": sharpe_ratio,
            "n_months": len(realized),
        })

    return pd.DataFrame(rows)


def build_ml_portfolios(
    predictions: pd.DataFrame,
    n_quantiles: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build monthly prediction-sorted decile portfolios for every model.

    For each month and model:
      1. Sort stocks by model prediction.
      2. Assign stocks to n_quantiles portfolios.
      3. Compute value-weighted and equal-weighted realized returns.
      4. Compute the corresponding predicted portfolio returns.
      5. Construct high-minus-low portfolios:
         highest-prediction decile minus lowest-prediction decile.

    Portfolio 1 contains the lowest predictions.
    Portfolio n_quantiles contains the highest predictions.

    Returns
    -------
    portfolio_returns : DataFrame
        Long-format monthly realized and predicted portfolio returns.

    portfolio_metrics : DataFrame
        Summary statistics for realized portfolio returns.
    """
    required_cols = [
        "permno",
        "date",
        "excess_ret_total",
        "mv",
    ]

    missing = [
        column for column in required_cols
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

    portfolio_returns = []

    for model_col in model_cols:
        model_name = model_col.removeprefix("pred_")

        df = predictions[
            required_cols + [model_col]
        ].dropna(
            subset=["excess_ret_total", "mv", model_col]
        ).copy()

        df = df[df["mv"] > 0]

        for date, month_df in df.groupby("date"):
            if len(month_df) < n_quantiles:
                continue

            month_df = month_df.copy()

            # Rank before cutting so tied predictions do not reduce
            # the number of portfolios.
            month_df["prediction_rank"] = month_df[model_col].rank(
                method="first",
                ascending=True,
            )

            try:
                month_df["portfolio"] = pd.qcut(
                    month_df["prediction_rank"],
                    q=n_quantiles,
                    labels=False,
                ) + 1
            except ValueError:
                continue

            for portfolio in range(1, n_quantiles + 1):
                portfolio_df = month_df[
                    month_df["portfolio"] == portfolio
                ]

                realized_vw = np.average(
                    portfolio_df["excess_ret_total"],
                    weights=portfolio_df["mv"],
                )
                predicted_vw = np.average(
                    portfolio_df[model_col],
                    weights=portfolio_df["mv"],
                )

                realized_ew = portfolio_df["excess_ret_total"].mean()
                predicted_ew = portfolio_df[model_col].mean()

                portfolio_returns.append({
                    "date": date,
                    "model": model_name,
                    "weighting": "value_weighted",
                    "portfolio": portfolio,
                    "realized_return": realized_vw,
                    "predicted_return": predicted_vw,
                    "n_stocks": len(portfolio_df),
                })

                portfolio_returns.append({
                    "date": date,
                    "model": model_name,
                    "weighting": "equal_weighted",
                    "portfolio": portfolio,
                    "realized_return": realized_ew,
                    "predicted_return": predicted_ew,
                    "n_stocks": len(portfolio_df),
                })

            low_df = month_df[month_df["portfolio"] == 1]
            high_df = month_df[
                month_df["portfolio"] == n_quantiles
            ]

            realized_vw_low = np.average(
                low_df["excess_ret_total"],
                weights=low_df["mv"],
            )
            realized_vw_high = np.average(
                high_df["excess_ret_total"],
                weights=high_df["mv"],
            )
            predicted_vw_low = np.average(
                low_df[model_col],
                weights=low_df["mv"],
            )
            predicted_vw_high = np.average(
                high_df[model_col],
                weights=high_df["mv"],
            )

            realized_ew_low = low_df["excess_ret_total"].mean()
            realized_ew_high = high_df["excess_ret_total"].mean()
            predicted_ew_low = low_df[model_col].mean()
            predicted_ew_high = high_df[model_col].mean()

            portfolio_returns.append({
                "date": date,
                "model": model_name,
                "weighting": "value_weighted",
                "portfolio": "high_minus_low",
                "realized_return": realized_vw_high - realized_vw_low,
                "predicted_return": predicted_vw_high - predicted_vw_low,
                "n_stocks": len(high_df) + len(low_df),
            })

            portfolio_returns.append({
                "date": date,
                "model": model_name,
                "weighting": "equal_weighted",
                "portfolio": "high_minus_low",
                "realized_return": realized_ew_high - realized_ew_low,
                "predicted_return": predicted_ew_high - predicted_ew_low,
                "n_stocks": len(high_df) + len(low_df),
            })

    portfolio_returns = (
        pd.DataFrame(portfolio_returns)
        if portfolio_returns
        else pd.DataFrame(
            columns=[
                "date",
                "model",
                "weighting",
                "portfolio",
                "realized_return",
                "predicted_return",
                "n_stocks",
            ]
        )
    )

    portfolio_metrics = compute_portfolio_metrics(
        portfolio_returns=portfolio_returns,
    )

    return portfolio_returns, portfolio_metrics

def compute_average_feature_importance(
    feature_importance_dir: str | Path,
    models: list[str] = ["RandomForest", "GBRT"],
    output_dir: str | Path = "output",
    regime_mode: str = "coding",
    save_outputs: bool = True,
) -> dict:
    """
    Average feature importance across all yearly refits.

    Parameters
    ----------
    feature_importance_dir : directory containing yearly importance CSVs
    models : model names to aggregate
    """
    logger.info("Computing average feature importance.")

    feature_importance_dir = Path(feature_importance_dir)
    output_dir = Path(output_dir) / regime_mode

    results = {}

    for model_name in models:
        yearly_frames = []

        for path in sorted(feature_importance_dir.glob(
            f"{model_name.lower()}_importance_year_*.csv"
        )):
            yearly = pd.read_csv(path, index_col=0)

            # Expected CSV format:
            # index: feature name
            # column: importance
            if "importance" not in yearly.columns:
                raise ValueError(
                    f"Feature-importance file {path} must contain "
                    "an 'importance' column."
                )

            yearly = yearly.rename(columns={"importance": path.stem})
            yearly_frames.append(yearly)

        if not yearly_frames:
            logger.warning(
                "No feature-importance files found for %s in %s.",
                model_name,
                feature_importance_dir,
            )
            continue

        importance_panel = pd.concat(yearly_frames, axis=1)
        average_importance = importance_panel.mean(axis=1)
        std_importance = importance_panel.std(axis=1, ddof=1)
        n_years = importance_panel.notna().sum(axis=1)

        summary = pd.DataFrame({
            "feature": average_importance.index,
            "mean_importance": average_importance.values,
            "std_importance": std_importance.values,
            "n_years": n_years.values,
        }).sort_values("mean_importance", ascending=False)

        results[model_name] = summary

        if save_outputs:
            output_path = output_dir / (
                f"feature_importance_average_{model_name.lower()}.csv"
            )
            save_csv(summary, output_path, enabled=save_outputs)

    logger.info(
        "Computed average feature importance for: %s",
        list(results.keys()),
    )

    return results


def evaluate_predictions(
    predictions: pd.DataFrame,
    regime_config,
    output_dir: str | Path = "output",
    n_quantiles: int = 10,
    dm_loss: str = "squared_error",
    horizon: int = 1,
    save_outputs: bool = True,
    feature_importance_dir: str | Path | None = None,
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

    predictions_original = filter_predictions_until(
        predictions=predictions,
        end_year=regime_config.test_end_year_original,
    )

    r2_metrics_original = compute_model_r2_metrics(
        predictions=predictions_original,
        regime_config=regime_config,
    )
    out_path_r2_metrics_original = f"{output_dir}/{regime_config.mode}/r2_metrics_original.csv"
    save_csv(r2_metrics_original,out_path_r2_metrics_original,enabled=save_outputs)

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
    )
    out_path_dm_tests = f"{output_dir}/{regime_config.mode}/dm_tests.csv"
    save_csv(dm_tests,out_path_dm_tests,enabled=save_outputs)

    portfolio_returns, portfolio_metrics = build_ml_portfolios(
        predictions=predictions,
        n_quantiles=n_quantiles,
    )
    """
    out_path_portfolio_returns = f"{output_dir}/{regime_config.mode}/portfolio_returns.parquet"
    save_parquet(portfolio_returns,out_path_portfolio_returns,enabled=save_outputs)    
    """

    out_path_portfolio_metrics = (
        f"{output_dir}/{regime_config.mode}/portfolio_metrics.csv"
    )
    save_csv(
        portfolio_metrics,
        out_path_portfolio_metrics,
        enabled=save_outputs,
)

    logger.info(
        f"Evaluation completed for {len(model_cols)} models."
        f"Metrics save to {output_dir}/{regime_config.mode}"
    )

    feature_importance_results = None

    if feature_importance_dir is not None:
        feature_importance_results = compute_average_feature_importance(
            feature_importance_dir=feature_importance_dir,
            models=["RandomForest", "GBRT"],
            output_dir=output_dir,
            regime_mode=regime_config.mode,
            save_outputs=save_outputs,
        )

    return {
        "r2_metrics_original": r2_metrics_original,
        "r2_metrics": r2_metrics,
        "dm_tests": dm_tests,
        "portfolio_returns": portfolio_returns,
        "portfolio_metrics": portfolio_metrics,
        "feature_importance": feature_importance_results,
    }