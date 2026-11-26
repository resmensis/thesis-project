from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Literal

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
            ["permno", "date", "excess_ret_total", "mvel1", model_col]
        ].dropna(
            subset=["excess_ret_total", "mvel1", model_col]
        )

        overall_r2 = gu_pred_r2(
            df["excess_ret_total"],
            df[model_col],
        )

        top_frames = []
        bottom_frames = []

        for _, month_df in df.groupby("date"):
            month_df = month_df.sort_values("mvel1", ascending=False)

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
    model_order: list[str] | None = None,
    nw_lags: int | None = None,
) -> pd.DataFrame:
    """
    Pairwise DM tests on monthly cross-sectional mean squared-loss differences.

    Uses Bartlett-weighted Newey-West standard errors without a finite-sample
    correction. Negative DM favors model_a.

    nw_lags=None uses floor(4 * (T / 100)**(2 / 9)), capped at T - 1.
    This bandwidth rule is an implementation choice, not specified in the
    supplied Gu et al. excerpt.

    Bonferroni adjustment covers all planned unique model pairs.
    """
    required = ["year", "excess_ret_total"]
    missing = set(required).difference(predictions.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    if not predictions.columns.is_unique:
        raise ValueError("Prediction column names must be unique.")
    if nw_lags is not None and (
        isinstance(nw_lags, (bool, np.bool_))
        or not isinstance(nw_lags, (int, np.integer))
        or nw_lags < 0
    ):
        raise ValueError("nw_lags must be a nonnegative integer or None.")

    model_cols = [
        col for col in predictions.columns
        if isinstance(col, str) and col.startswith("pred_")
    ]
    if len(model_cols) < 2:
        raise ValueError("At least two prediction models are required.")

    preferred_order = model_order or [
        "OLS_full", "OLS_3", "PLS", "PCR_Linear",
        "RandomForest", "GBRT",
        "NN1", "NN2", "NN3", "NN4", "NN5",
        "PCR_Huber", "LinearSVR", "OLS_3_old",
    ]
    model_cols = list(dict.fromkeys(
        [f"pred_{name}" for name in preferred_order
         if f"pred_{name}" in model_cols]
        + model_cols
    ))

    data = predictions[required + model_cols].copy()

    years = pd.to_numeric(data["year"], errors="raise")
    if (
        years.isna().any()
        or not np.isfinite(years.to_numpy(dtype=float)).all()
        or (years % 1 != 0).any()
    ):
        raise ValueError("year must contain finite, nonmissing integer years.")

    data["year"] = years.astype(int)

    numeric_cols = ["excess_ret_total"] + model_cols
    data[numeric_cols] = data[numeric_cols].apply(
        pd.to_numeric, errors="raise"
    )

    n_tests = len(model_cols) * (len(model_cols) - 1) // 2
    rows = []

    for model_a, model_b in combinations(model_cols, 2):
        pair = data[["year", "excess_ret_total", model_a, model_b]]
        values = pair[["excess_ret_total", model_a, model_b]].to_numpy(
            dtype=float, na_value=np.nan
        )
        valid = np.isfinite(values).all(axis=1)
        y, pred_a, pred_b = values[valid].T

        with np.errstate(over="ignore", invalid="ignore"):
            loss_diff = (y - pred_a) ** 2 - (y - pred_b) ** 2

        if not np.isfinite(loss_diff).all():
            raise ValueError(
                f"Nonfinite squared-loss differences: {model_a}, {model_b}."
            )

        yearly = (
            pd.Series(loss_diff, index=pair.loc[valid, "year"])
            .groupby(level=0, sort=True)
            .mean()
        )

        t = len(yearly)
        dm = p_value = np.nan

        if t >= 2:
            expected = np.arange(yearly.index[0], yearly.index[-1] + 1)
            if not np.array_equal(yearly.index.to_numpy(), expected):
                raise ValueError(
                    f"Internal missing years: {model_a}, {model_b}."
                )

            if nw_lags is not None and nw_lags >= t:
                raise ValueError(
                    f"nw_lags must be smaller than {t} for "
                    f"{model_a}, {model_b}."
                )

            lags = (
                min(int(np.floor(4 * (t / 100) ** (2 / 9))), t - 1)
                if nw_lags is None else int(nw_lags)
            )
            d = yearly.to_numpy(dtype=float)
            u = d - d.mean()

            with np.errstate(over="ignore", invalid="ignore"):
                long_run_variance = u @ u / t
                for lag in range(1, lags + 1):
                    weight = 1 - lag / (lags + 1)
                    long_run_variance += (
                        2 * weight * (u[lag:] @ u[:-lag]) / t
                    )

                se = (
                    np.sqrt(long_run_variance / t)
                    if long_run_variance > 0 else np.nan
                )

            if np.isfinite(se) and se > 0:
                dm = float(d.mean() / se)
                p_value = float(2 * stats.norm.sf(abs(dm)))

        rows.append({
            "model_a": model_a.removeprefix("pred_"),
            "model_b": model_b.removeprefix("pred_"),
            "dm_statistic": dm,
            "p_value": p_value,
            "p_value_bonferroni": (
                np.minimum(n_tests * p_value, 1.0)
            ),
            "n_periods": t,
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
        "mvel1",
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
            subset=["excess_ret_total", "mvel1", model_col]
        ).copy()

        df = df[df["mvel1"] > 0]

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
                    weights=portfolio_df["mvel1"],
                )
                predicted_vw = np.average(
                    portfolio_df[model_col],
                    weights=portfolio_df["mvel1"],
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
                weights=low_df["mvel1"],
            )
            realized_vw_high = np.average(
                high_df["excess_ret_total"],
                weights=high_df["mvel1"],
            )
            predicted_vw_low = np.average(
                low_df[model_col],
                weights=low_df["mvel1"],
            )
            predicted_vw_high = np.average(
                high_df[model_col],
                weights=high_df["mvel1"],
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

    portfolio_returns["portfolio"] = (
        portfolio_returns["portfolio"].astype("string")
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


def compute_fama_french_portfolio_metrics(
    portfolio_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute portfolio-level out-of-sample R2 using gu_pred_r2().

    Returns one row per panel, family, portfolio, model, and
    weighting scheme.

    R2 is stored as a decimal, consistently with the project's
    existing model R2 metrics.

    Only months with both finite realized and predicted returns
    enter the calculation.
    """
    group_cols = [
        "panel",
        "family",
        "portfolio",
        "model",
        "weighting",
    ]
    return_cols = [
        "realized_return",
        "predicted_return",
    ]
    output_cols = [*group_cols, "r2"]

    required = [*group_cols, "date", *return_cols]
    missing = sorted(set(required).difference(portfolio_returns.columns))

    if missing:
        raise KeyError(
            f"portfolio_returns is missing required columns: {missing}"
        )

    if portfolio_returns.empty:
        return pd.DataFrame(columns=output_cols)

    if portfolio_returns[group_cols + ["date"]].isna().any().any():
        raise ValueError(
            "Portfolio identifiers and dates must not be missing."
        )

    if portfolio_returns.duplicated(
        [*group_cols, "date"]
    ).any():
        raise ValueError(
            "Expected one observation per portfolio, model, "
            "weighting scheme, and date."
        )

    rows = []

    for keys, group in portfolio_returns.groupby(
        group_cols,
        sort=False,
        observed=True,
    ):
        values = (
            group[return_cols]
            .apply(pd.to_numeric, errors="raise")
            .to_numpy(dtype=float, na_value=np.nan)
        )

        valid = np.isfinite(values).all(axis=1)
        paired_values = values[valid]

        if len(paired_values):
            r2 = gu_pred_r2(
                paired_values[:, 0],
                paired_values[:, 1],
            )
        else:
            r2 = np.nan

        rows.append({
            **dict(zip(group_cols, keys)),
            "r2": r2,
        })

    return pd.DataFrame(rows, columns=output_cols)


def build_fama_french_portfolios(
    predictions: pd.DataFrame,
    *,
    smb_definition: str = "ff5",
    output_dir: str | Path = "output",
    save_outputs: bool = True,
) -> pd.DataFrame:
    """
    Build monthly characteristic-sorted portfolio proxies and forecasts.

    Produces:
        - 24 size x characteristic component portfolios
        - SMB, HML, RMW, CMA, UMD

    Important
    ---------
    This is a monthly-sort approximation using the available evaluation
    universe, not an exact recreation of official French portfolios.

    Inputs must already be aligned to the return month:
        excess_ret_total and pred_* refer to the same realized return.
        mvel1 and sorting characteristics must be known ex ante.

    mvel1 must be positive, unscaled lagged market equity.

    Missing returns or predictions do not change portfolio membership.
    Instead, the affected aggregate return is NaN.

    Parameters
    ----------
    smb_definition:
        "ff3": SMB from the six size/value portfolios.
        "ff5": Average of size/value, size/profitability,
               and size/investment SMB.

    Returns
    -------
    DataFrame
        Monthly portfolio returns in long format.

        n_positions counts component positions for factors; a stock
        appearing in multiple component portfolios can be counted
        more than once.
    """
    if smb_definition not in {"ff3", "ff5"}:
        raise ValueError("smb_definition must be 'ff3' or 'ff5'.")

    if not predictions.columns.is_unique:
        raise ValueError("Input column names must be unique.")

    characteristics = ["bm", "operprof", "agr", "mom12m"]
    required = [
        "permno", "date", "excess_ret_total", "mvel1",
        *characteristics,
    ]
    missing = sorted(set(required).difference(predictions.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    model_cols = [
        col for col in predictions.columns
        if isinstance(col, str) and col.startswith("pred_")
    ]
    if not model_cols:
        raise ValueError("No pred_* columns found.")

    data = predictions[required + model_cols].copy()

    data["date"] = (
        pd.to_datetime(data["date"], errors="raise")
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    if data[["permno", "date"]].isna().any().any():
        raise ValueError("permno and date must not be missing.")

    if data.duplicated(["permno", "date"]).any():
        raise ValueError(
            "Input must have one observation per permno and month."
        )

    numeric_cols = [
        "excess_ret_total", "mvel1",
        *characteristics, *model_cols,
    ]
    data[numeric_cols] = data[numeric_cols].apply(
        pd.to_numeric, errors="raise"
    )
    data[numeric_cols] = data[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    data = data.loc[data["mvel1"].gt(0)].copy()
    if data.empty:
        raise ValueError("No observations with positive finite mvel1.")

    # Low, middle, high characteristic labels.
    specifications = {
        "value": ("bm", ("growth", "neutral", "value"), "HML"),
        "profitability": (
            "operprof", ("weak", "neutral", "robust"), "RMW"
        ),
        "investment": (
            "agr", ("conservative", "neutral", "aggressive"), "CMA"
        ),
        "momentum": ("mom12m", ("down", "medium", "up"), "UMD"),
    }

    rows = []

    def aggregate(values, weights):
        finite = np.isfinite(values)
        coverage = float(weights[finite].sum())
        result = (
            float(weights @ values)
            if finite.all()
            else np.nan
        )
        return result, coverage

    for date, month in data.groupby("date", sort=True):
        size_breakpoint = month["mvel1"].median()

        for family, (characteristic, labels, factor) in (
            specifications.items()
        ):
            eligible = month.loc[
                month[characteristic].notna()
            ].copy()

            if eligible.empty:
                continue

            low_cut, high_cut = eligible[characteristic].quantile(
                [0.30, 0.70]
            )

            # Do not split tied values arbitrarily to force six cells.
            if low_cut > high_cut:
                raise ValueError(
                    f"Degenerate {family} breakpoints in {date:%Y-%m}."
                )

            eligible["_size"] = np.where(
                eligible["mvel1"] <= size_breakpoint,
                "Small",
                "Big",
            )
            eligible["_bucket"] = np.select(
                [
                    eligible[characteristic] <= low_cut,
                    eligible[characteristic] <= high_cut,
                ],
                [0, 1],
                default=2,
            )

            cells = {
                (size, bucket): eligible.loc[
                    eligible["_size"].eq(size)
                    & eligible["_bucket"].eq(bucket)
                ]
                for size in ("Small", "Big")
                for bucket in range(3)
            }

            for model_col in model_cols:
                model = model_col.removeprefix("pred_")
                component_results = {}

                for (size, bucket), cell in cells.items():
                    n_stocks = len(cell)

                    if n_stocks:
                        weights = cell["mvel1"].to_numpy(dtype=float)
                        weights = weights / weights.sum()

                        realized, realized_coverage = aggregate(
                            cell["excess_ret_total"].to_numpy(
                                dtype=float
                            ),
                            weights,
                        )
                        predicted, prediction_coverage = aggregate(
                            cell[model_col].to_numpy(dtype=float),
                            weights,
                        )
                    else:
                        realized = predicted = np.nan
                        realized_coverage = prediction_coverage = 0.0

                    component_results[(size, bucket)] = (
                        realized, predicted, n_stocks
                    )

                    rows.append({
                        "date": date,
                        "model": model,
                        "panel": "B",
                        "family": family,
                        "portfolio": f"{size} {labels[bucket]}",
                        "weighting": "value_weighted",
                        "realized_return": realized,
                        "predicted_return": predicted,
                        "n_positions": n_stocks,
                        "realized_weight_coverage": realized_coverage,
                        "prediction_weight_coverage": prediction_coverage,
                    })

                # CMA is low investment minus high investment.
                long_bucket, short_bucket = (
                    (0, 2) if factor == "CMA" else (2, 0)
                )

                factor_keys = [
                    ("Small", long_bucket),
                    ("Big", long_bucket),
                    ("Small", short_bucket),
                    ("Big", short_bucket),
                ]
                coefficients = np.array([0.5, 0.5, -0.5, -0.5])

                values = np.array([
                    component_results[key][:2]
                    for key in factor_keys
                ])
                factor_values = coefficients @ values

                rows.append({
                    "date": date,
                    "model": model,
                    "panel": "A",
                    "family": family,
                    "portfolio": factor,
                    "weighting": "value_weighted",
                    "realized_return": factor_values[0],
                    "predicted_return": factor_values[1],
                    "n_positions": sum(
                        component_results[key][2]
                        for key in factor_keys
                    ),
                    "realized_weight_coverage": np.nan,
                    "prediction_weight_coverage": np.nan,
                })

                # Retain family-specific SMBs internally.
                if family != "momentum":
                    smb_keys = [
                        (size, bucket)
                        for size in ("Small", "Big")
                        for bucket in range(3)
                    ]
                    smb_coefficients = np.array(
                        [1 / 3] * 3 + [-1 / 3] * 3
                    )
                    smb_values = smb_coefficients @ np.array([
                        component_results[key][:2]
                        for key in smb_keys
                    ])

                    rows.append({
                        "date": date,
                        "model": model,
                        "panel": "internal",
                        "family": family,
                        "portfolio": f"SMB_{family}",
                        "weighting": "value_weighted",
                        "realized_return": smb_values[0],
                        "predicted_return": smb_values[1],
                        "n_positions": sum(
                            component_results[key][2]
                            for key in smb_keys
                        ),
                        "realized_weight_coverage": np.nan,
                        "prediction_weight_coverage": np.nan,
                    })

    if not rows:
        raise ValueError("No portfolios could be constructed.")

    result = pd.DataFrame(rows)
    smb_rows = []

    for (date, model), group in result.loc[
        result["panel"].eq("internal")
    ].groupby(["date", "model"], sort=False):
        families = (
            ["value"]
            if smb_definition == "ff3"
            else ["value", "profitability", "investment"]
        )
        selected = group.set_index("family").reindex(families)

        # np.mean deliberately propagates missing component returns.
        smb_rows.append({
            "date": date,
            "model": model,
            "panel": "A",
            "family": "size",
            "portfolio": "SMB",
            "weighting": "value_weighted",
            "realized_return": np.mean(
                selected["realized_return"].to_numpy(dtype=float)
            ),
            "predicted_return": np.mean(
                selected["predicted_return"].to_numpy(dtype=float)
            ),
            "n_positions": selected["n_positions"].sum(min_count=1),
            "realized_weight_coverage": np.nan,
            "prediction_weight_coverage": np.nan,
        })

    result = pd.concat(
        [
            result.loc[~result["panel"].eq("internal")],
            pd.DataFrame(smb_rows),
        ],
        ignore_index=True,
    )

    result["construction"] = "monthly_available_universe_2x3"
    result["smb_definition"] = smb_definition

    result = result.sort_values(
        ["date", "model", "panel", "family", "portfolio"]
    ).reset_index(drop=True)

    portfolio_metrics = compute_fama_french_portfolio_metrics(
        portfolio_returns=result,
    )

    output_dir = Path(output_dir)

    save_csv(
        portfolio_metrics,
        output_dir / "fama_french_portfolio_metrics.csv",
        enabled=save_outputs,
    )

    logger.info(
        "Fama-French portfolio R2 evaluation completed: %s rows.",
        len(portfolio_metrics),
    )

    return portfolio_metrics


def build_sp500_portfolio(
    predictions: pd.DataFrame,
    *,
    n_constituents: int = 500,
    membership_update: str = "monthly",
    mvel1_is_log: bool = False,
    output_dir: str | Path = "output",
    save_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rebuild an S&P 500 proxy from the largest stocks by mvel1.

    Timing
    ------
    mvel1 on row date t must be market equity known before month t, so
    excess_ret_total and pred_* in row t are the month-t outcomes and
    forecasts. Membership and weights use only mvel1 at the start of
    month t.

    membership_update
    -----------------
    "monthly":   top n_constituents are re-selected every month.
    "quarterly": selected in March/June/September/December rows (and the
                 first sample month), then held until the next selection.
                 Stocks missing from a later month drop out and weights
                 are renormalized among the remaining members.

    Weights are mvel1 / sum(mvel1) across members. Missing realized or
    predicted returns keep the stock as a member; the affected portfolio
    return is NaN and the coverage columns record the missing weight.

    Months with fewer than n_constituents eligible stocks are skipped.

    Returns
    -------
    (portfolio_returns, portfolio_metrics)
        portfolio_metrics uses compute_fama_french_portfolio_metrics().
    """
    if membership_update not in {"monthly", "quarterly"}:
        raise ValueError("membership_update must be 'monthly' or 'quarterly'.")
    if (
        isinstance(n_constituents, (bool, np.bool_))
        or not isinstance(n_constituents, (int, np.integer))
        or n_constituents < 1
    ):
        raise ValueError("n_constituents must be a positive integer.")

    if not predictions.columns.is_unique:
        raise ValueError("Input column names must be unique.")

    required = ["permno", "date", "excess_ret_total", "mvel1"]
    missing = sorted(set(required).difference(predictions.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    model_cols = [
        col for col in predictions.columns
        if isinstance(col, str) and col.startswith("pred_")
    ]
    if not model_cols:
        raise ValueError("No pred_* columns found.")

    data = predictions[required + model_cols].copy()

    data["date"] = (
        pd.to_datetime(data["date"], errors="raise")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    if data[["permno", "date"]].isna().any().any():
        raise ValueError("permno and date must not be missing.")
    if data.duplicated(["permno", "date"]).any():
        raise ValueError("Input must have one observation per permno and month.")

    numeric_cols = ["excess_ret_total", "mvel1", *model_cols]
    data[numeric_cols] = (
        data[numeric_cols]
        .apply(pd.to_numeric, errors="raise")
        .replace([np.inf, -np.inf], np.nan)
    )

    if mvel1_is_log:
        data["mvel1"] = np.exp(data["mvel1"])

    if data["mvel1"].max() <= 1.0:
        raise ValueError(
            "mvel1 looks scaled to [-1, 1]. Use the unscaled market equity "
            "saved before the quantile transformation."
        )

    data = data.loc[data["mvel1"].gt(0)].copy()

    counts = data.groupby("date").size()
    short_dates = counts.index[counts < n_constituents]
    if len(short_dates):
        logger.warning(
            "Skipping %s months with fewer than %s eligible stocks.",
            len(short_dates), n_constituents,
        )
        data = data.loc[~data["date"].isin(short_dates)].copy()
    if data.empty:
        raise ValueError("No months with enough eligible stocks.")

    # Deterministic ranking: largest mvel1 first, permno breaks ties.
    data = data.sort_values(
        ["date", "mvel1", "permno"],
        ascending=[True, False, True],
    )
    data["_rank"] = data.groupby("date").cumcount() + 1

    if membership_update == "monthly":
        members = data.loc[data["_rank"] <= n_constituents].copy()
    else:
        dates = pd.DatetimeIndex(
            data["date"].drop_duplicates().sort_values()
        )
        rebalance_dates = dates[dates.month.isin([3, 6, 9, 12])].union(
            dates[:1]
        )

        date_map = pd.merge_asof(
            pd.DataFrame({"date": dates}),
            pd.DataFrame({"_rebalance_date": rebalance_dates}),
            left_on="date",
            right_on="_rebalance_date",
        )
        constituents = (
            data.loc[
                data["date"].isin(rebalance_dates)
                & (data["_rank"] <= n_constituents),
                ["permno", "date"],
            ]
            .rename(columns={"date": "_rebalance_date"})
        )
        members = (
            data.merge(date_map, on="date", how="inner")
            .merge(constituents, on=["permno", "_rebalance_date"], how="inner")
        )

    members["_w"] = (
        members["mvel1"] / members.groupby("date")["mvel1"].transform("sum")
    )
    by_date = members["date"]

    def weighted_aggregate(column: str):
        values = members[column]
        finite = values.notna()
        total = (
            (members["_w"] * values).where(finite, 0.0)
            .groupby(by_date).sum()
        )
        n_missing = (~finite).groupby(by_date).sum()
        coverage = members["_w"].where(finite, 0.0).groupby(by_date).sum()
        return total.where(n_missing.eq(0)), coverage

    n_positions = members.groupby("date").size()
    market_cap_share = (
        members.groupby("date")["mvel1"].sum()
        / data.groupby("date")["mvel1"].sum()
    ).reindex(n_positions.index)

    realized, realized_coverage = weighted_aggregate("excess_ret_total")

    frames = []
    for model_col in model_cols:
        predicted, prediction_coverage = weighted_aggregate(model_col)

        frames.append(pd.DataFrame({
            "date": realized.index,
            "model": model_col.removeprefix("pred_"),
            "panel": "A",
            "family": "market",
            "portfolio": "S&P 500",
            "weighting": "value_weighted",
            "realized_return": realized.to_numpy(),
            "predicted_return": predicted.to_numpy(),
            "n_positions": n_positions.to_numpy(),
            "realized_weight_coverage": realized_coverage.to_numpy(),
            "prediction_weight_coverage": prediction_coverage.to_numpy(),
            "market_cap_share": market_cap_share.to_numpy(),
            "construction": (
                f"top_{n_constituents}_mvel1_{membership_update}"
            ),
        }))

    portfolio_returns = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "model"])
        .reset_index(drop=True)
    )

    portfolio_metrics = compute_fama_french_portfolio_metrics(
        portfolio_returns=portfolio_returns,
    )

    output_dir = Path(output_dir)

    save_parquet(
        portfolio_returns,
        output_dir / "sp500_portfolio_returns.parquet",
        enabled=save_outputs,
    )
    save_csv(
        portfolio_returns,
        output_dir / "sp500_portfolio_returns.csv",
        enabled=save_outputs,
    )
    save_csv(
        portfolio_metrics,
        output_dir / "sp500_portfolio_metrics.csv",
        enabled=save_outputs,
    )

    logger.info(
        "S&P 500 proxy built: %s months, %s models.",
        realized.shape[0], len(model_cols),
    )

    return portfolio_returns, portfolio_metrics


def create_evaulation_dataset(
        predictions: pd.DataFrame,
        fama_french_charas_df: pd.DataFrame,
        output_dir: str | Path = "output",
        save_outputs: bool = True
) -> pd.DataFrame:

    cols_to_drop = ["mv"]
    cols_to_drop = [col for col in cols_to_drop if col in predictions.columns]

    if cols_to_drop:
        predictions = predictions.drop(columns=cols_to_drop)

    df = predictions.merge(
        fama_french_charas_df,
        on=["permno", "date"],
        how="inner",
        # validate="one_to_one",
    )
    out_path = f"{output_dir}/all_predictions_fama_french.parquet"
    save_parquet(df,out_path,enabled=save_outputs)

    return df


def evaluate_predictions(
    predictions: pd.DataFrame,
    regime_config,
    eval_config,
    output_dir: str | Path = "output",
    n_quantiles: int = 10,
    save_outputs: bool = True,
    feature_importance_dir: str | Path | None = None,
) -> dict:
    """
    Evaluate all model prediction columns in the wide prediction store.

    Parameters
    ----------
    predictions : DataFrame
        Output of run_expanding_window(). Must contain:
        permno, date, excess_ret_total, mvel1, and pred_<MODEL> columns.

    regime_config : DataRegimeConfig
        Determines the number of stocks in top/bottom mvel1 subsamples.

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
        "mvel1",
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
    out_path_r2_metrics_original = f"{output_dir}/r2_metrics_original.csv"
    save_csv(r2_metrics_original,out_path_r2_metrics_original,enabled=save_outputs)
    logger.info(
        f"Evaluation part finished: "
        f"r2_metrics_original.csv saved to: {output_dir}"
    )

    r2_metrics = compute_model_r2_metrics(
        predictions=predictions,
        regime_config=regime_config,
    )
    out_path_r2_metrics = f"{output_dir}/r2_metrics.csv"
    save_csv(r2_metrics,out_path_r2_metrics,enabled=save_outputs)
    logger.info(
        f"Evaluation part finished: "
        f"r2_metrics.csv saved to: {output_dir}"
    )

    dm_tests = compute_diebold_mariano_tests(
        predictions=predictions,
        model_order=eval_config.preferred_order,
    )
    out_path_dm_tests = f"{output_dir}/dm_tests.csv"
    save_csv(dm_tests,out_path_dm_tests,enabled=save_outputs)
    logger.info(
        f"Evaluation part finished: "
        f"dm_tests.csv saved to: {output_dir}"
    )

    portfolio_returns, portfolio_metrics = build_ml_portfolios(
        predictions=predictions,
        n_quantiles=n_quantiles,
    )

    out_path_portfolio_returns = f"{output_dir}/portfolio_returns.parquet"
    save_parquet(portfolio_returns,out_path_portfolio_returns,enabled=save_outputs)
    logger.info(
        f"Evaluation part finished: "
        f"portfolio_returns.parquet saved to: {output_dir}"
    )   

    out_path_portfolio_metrics = (f"{output_dir}/portfolio_metrics.csv")
    save_csv(portfolio_metrics,out_path_portfolio_metrics,enabled=save_outputs,)
    logger.info(
        f"Evaluation part finished: "
        f"portfolio_metrics.csv saved to: {output_dir}"
    )

    
    fama_french_portfolio_metrics = build_fama_french_portfolios(
        predictions=predictions,
        smb_definition="ff5",
        output_dir=output_dir,
        save_outputs=save_outputs,
    )

    logger.info(
        "Fama-French-style portfolio construction completed: "
        "%s monthly portfolio/model observations.",
        len(fama_french_portfolio_metrics),
    )    
    

    sp500_portfolio_returns, sp500_portfolio_metrics = build_sp500_portfolio(
        predictions=predictions,
        n_constituents=500,
        membership_update="quarterly",
        output_dir=output_dir,
        save_outputs=save_outputs,
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

    logger.info(
        f"Evaluation completed for {len(model_cols)} models."
        f"Metrics save to {output_dir}"
    )

    return {
        "r2_metrics_original": r2_metrics_original,
        "r2_metrics": r2_metrics,
        "dm_tests": dm_tests,
        "portfolio_returns": portfolio_returns,
        "portfolio_metrics": portfolio_metrics,
        "fama_french_portfolio_metrics": fama_french_portfolio_metrics,
        "sp500_portfolio_returns": sp500_portfolio_returns,
        "sp500_portfolio_metrics": sp500_portfolio_metrics,
        "feature_importance": feature_importance_results,
    }

    """
    return {
        "r2_metrics_original": r2_metrics_original,
        "r2_metrics": r2_metrics,
        "dm_tests": dm_tests,
        "portfolio_returns": portfolio_returns,
        "portfolio_metrics": portfolio_metrics,
        "fama_french_portfolio_returns": fama_french_portfolio_returns,
        "feature_importance": feature_importance_results,
    }

    """
