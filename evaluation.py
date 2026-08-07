from __future__ import annotations
import logging
import numpy as np
import pandas as pd

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