from __future__ import annotations
import logging
import numpy as np
import pandas as pd


import re
from pathlib import Path
from io_utils import ensure_dir, save_parquet, save_csv

logger = logging.getLogger("eap_ml.visualization")


def r2_metrics(
    r2_metrics: pd.DataFrame,
    eval_config,
    output_path: str | Path,
    file_name: str,
    sample_years: str,
    label: str = "tab:stock_prediction_performance",
) -> Path:
    """
    Write a Gu et al. Table 1-style LaTeX table with a grouped bar chart.

    Required columns:
        model, r2_overall, r2_top, r2_bottom, subsample_size

    R2 values must be fractions, not percentages.
    Model labels supplied by the caller are treated as LaTeX.
    """
    required = {
        "model",
        "r2_overall",
        "r2_top",
        "r2_bottom",
        "subsample_size",
    }
    missing = required.difference(r2_metrics.columns)
    if missing:
        raise KeyError(f"Missing r2_metrics columns: {sorted(missing)}")

    if r2_metrics.empty:
        raise ValueError("r2_metrics is empty.")

    table = r2_metrics.copy()

    if table["model"].isna().any():
        raise ValueError("Model names must not be missing.")

    table["model"] = table["model"].astype(str)

    if table["model"].duplicated().any():
        raise ValueError("Expected exactly one metrics row per model.")

    sizes = pd.to_numeric(
        table["subsample_size"], errors="raise"
    ).dropna().unique()

    if len(sizes) != 1 or sizes[0] <= 0 or sizes[0] != int(sizes[0]):
        raise ValueError("Expected one positive integer subsample_size.")

    subsample_size = int(sizes[0])

    value_cols = ["r2_overall", "r2_top", "r2_bottom"]
    table[value_cols] = table[value_cols].apply(
        pd.to_numeric, errors="raise"
    )
    table[value_cols] = table[value_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    preferred_order = eval_config.preferred_order

    available = table["model"].tolist()
    ordered = list(dict.fromkeys(
        [name for name in preferred_order if name in available]
        + available
    ))

    table = table.set_index("model").loc[ordered]
    percent = table[value_cols] * 100.0

    def escape_latex(text):
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(c, c) for c in str(text))

    default_labels = eval_config.default_labels
    labels = {
        name: default_labels.get(name, escape_latex(name))
        for name in ordered
    }

    default_sublabels = eval_config.default_sublabels
    sublabels = {
        name: default_sublabels.get(name, escape_latex(name))
        for name in ordered
    }

    if not re.fullmatch(r"[A-Za-z0-9:._-]+", label):
        raise ValueError("Use a simple LaTeX label without special characters.")

    def format_value(value):
        return "--" if pd.isna(value) else f"{value:.2f}"

    def table_row(title, column):
        values = " & ".join(
            format_value(percent.loc[name, column])
            for name in ordered
        )
        return f"{title} & {values}" + r" \\"

    plot_models = [
        name for name in ordered
        if name not in eval_config.exclude_from_plot
    ]

    excluded = [name for name in ordered if name in eval_config.exclude_from_plot]
    incomplete = [
        name for name in ordered
        if name not in eval_config.exclude_from_plot and name not in plot_models
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{threeparttable}",
        (
            r"\caption{Monthly out-of-sample stock-level prediction "
            r"performance (percentage $R^2_{\mathrm{OOS}}$)}"
            f": {escape_latex(sample_years)}"
        ),
        rf"\label{{{label}}}",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        (
            r"\begin{tabular*}{\textwidth}"
            r"{@{\extracolsep{\fill}}l"
            + "c" * len(ordered)
            + r"@{}}"
        ),
        r"\toprule",
        " & " + " & ".join(labels[name] for name in ordered) + r" \\",
        " & " + " & ".join(sublabels[name] for name in ordered) + r" \\",
        r"\midrule",
        table_row("All", "r2_overall"),
        table_row(f"Top {subsample_size:,}", "r2_top"),
        table_row(f"Bottom {subsample_size:,}", "r2_bottom"),
        r"\bottomrule",
        r"\end{tabular*}",
        r"\endgroup",
    ]

    if plot_models:
        ticks = ",".join(str(i) for i in range(len(plot_models)))
        tick_labels = ",".join(
            "{" + labels[name] + " " + sublabels[name] + "}" for name in plot_models
        )

        lines.extend([
            r"\par\medskip",
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"width=\linewidth,",
            r"height=6.5cm,",
            r"ybar=0pt,",
            r"bar width=5pt,",
            r"enlarge x limits=0.07,",
            rf"xtick={{{ticks}}},",
            rf"xticklabels={{{tick_labels}}},",
            r"xticklabel style={rotate=-45,anchor=west,font=\scriptsize},",
            r"yticklabel style={font=\scriptsize},",
            r"ylabel={$R^2_{\mathrm{OOS}}$ (\%)},",
            r"axis x line*=bottom,",
            r"axis y line*=left,",
            r"ymajorgrids=true,",
            r"grid style={gray!25},",
            r"legend style={",
            r"at={(0.01,0.99)},anchor=north west,",
            r"draw=none,fill=white,font=\scriptsize",
            r"},",
            r"legend cell align=left,",
            r"]",
        ])

        series = [
            ("r2_overall", "blue!55!black", "All"),
            ("r2_top", "teal!85!green", "Top"),
            ("r2_bottom", "yellow!85!green", "Bottom"),
        ]

        for column, color, legend in series:
            coordinates = " ".join(
                f"({i},{percent.loc[name, column]:.8f})"
                for i, name in enumerate(plot_models)
            )
            lines.extend([
                rf"\addplot[fill={color},draw=none] "
                rf"coordinates {{{coordinates}}};",
                rf"\addlegendentry{{{legend}}}",
            ])

        lines.extend([
            r"\end{axis}",
            r"\end{tikzpicture}",
        ])

    notes = (
        r"\item This table reports monthly out-of-sample stock-level "
        r"$R^2_{\mathrm{OOS}}$ in percent for each model. "
        r"All denotes the full evaluation sample. "
        f"Top {subsample_size:,} and Bottom {subsample_size:,} denote "
        "the largest and smallest market-value stocks selected "
        "within each month, respectively. "
        f"The out-of-sample period is {escape_latex(sample_years)}. "
        "The lower panel compares the statistics reported above."
    )

    if excluded:
        notes += (
            " Models excluded from the lower panel: "
            + ", ".join(labels[name] for name in excluded)
            + "."
        )

    if incomplete:
        notes += (
            " Models with incomplete metrics are omitted from the chart: "
            + ", ".join(labels[name] for name in incomplete)
            + "."
        )

    lines.extend([
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        notes,
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ])

    output_path = Path(f"{output_path}/{file_name}.tex")
    if output_path.suffix.lower() != ".tex":
        raise ValueError("output_path must have a .tex extension.")

    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(
        f"Visualisation part finished: "
        f"{file_name}.tex saved to: {output_path}"
    )

    return output_path


def dm_tests(
    dm_tests: pd.DataFrame,
    eval_config,
    output_path: str | Path,
    file_name: str,
    sample_years: str,
    label: str = "tab:diebold_mariano_tests",
) -> Path:
    """
    Export a Gu et al. Table 3-style LaTeX table with decimal alignment.

    Positive entries mean the column model has lower loss than the row model.
    Bold: individual p-value <= 0.05.
    Asterisk: supplied Bonferroni-adjusted p-value <= 0.05.

    Required columns:
        model_a, model_b, dm_statistic, p_value, p_value_bonferroni

    Uses eval_config.preferred_order, default_labels, and default_sublabels.
    Configured labels and sublabels are treated as LaTeX.
    """
    required = {
        "model_a",
        "model_b",
        "dm_statistic",
        "p_value",
        "p_value_bonferroni",
    }
    missing = required.difference(dm_tests.columns)
    if missing:
        raise KeyError(f"Missing DM columns: {sorted(missing)}")
    if dm_tests.empty:
        raise ValueError("dm_tests is empty.")
    if dm_tests[["model_a", "model_b"]].isna().any().any():
        raise ValueError("Model names must not be missing.")
    if not re.fullmatch(r"[A-Za-z0-9:._-]+", label):
        raise ValueError("Use a simple LaTeX label.")

    output_path = Path(f"{output_path}/{file_name}.tex")
    if output_path.suffix.lower() != ".tex":
        raise ValueError("output_path must have a .tex extension.")

    table = dm_tests.copy()
    table[["model_a", "model_b"]] = table[
        ["model_a", "model_b"]
    ].astype(str)

    numeric_cols = [
        "dm_statistic",
        "p_value",
        "p_value_bonferroni",
    ]
    table[numeric_cols] = table[numeric_cols].apply(
        pd.to_numeric,
        errors="raise",
    )

    def escape_latex(text):
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(
            replacements.get(character, character)
            for character in str(text)
        )

    available = list(dict.fromkeys(
        table[["model_a", "model_b"]].to_numpy().ravel()
    ))
    ordered = list(dict.fromkeys(
        [
            name
            for name in eval_config.preferred_order
            if name in available
        ]
        + available
    ))

    if len(ordered) < 2:
        raise ValueError("At least two models are required.")

    labels = {
        name: eval_config.default_labels.get(
            name,
            escape_latex(name),
        )
        for name in ordered
    }
    sublabels = {
        name: eval_config.default_sublabels.get(name, "")
        for name in ordered
    }

    pairs = {}
    for row in table.itertuples(index=False):
        key = frozenset((row.model_a, row.model_b))
        if row.model_a == row.model_b or key in pairs:
            raise ValueError(
                "Expected one row per unique, distinct model pair."
            )
        pairs[key] = row

    def format_cell(row_model, column_model):
        pair = pairs.get(frozenset((row_model, column_model)))

        if pair is None or not np.isfinite(pair.dm_statistic):
            return r"\multicolumn{1}{c}{--}"

        value = (
            pair.dm_statistic
            if pair.model_a == row_model
            else -pair.dm_statistic
        )

        text = f"{value:.2f}"
        if text == "-0.00":
            text = "0.00"

        if pd.notna(pair.p_value) and pair.p_value <= 0.05:
            text = r"\bfseries " + text

        if (
            pd.notna(pair.p_value_bonferroni)
            and pair.p_value_bonferroni <= 0.05
        ):
            text += r"{\textsuperscript{*}}"

        return text

    column_models = ordered[1:]

    # Allow enough integer digits for the rounded statistics.
    finite_values = table.loc[
        np.isfinite(table["dm_statistic"]),
        "dm_statistic",
    ]
    integer_digits = max(
        (
            len(f"{abs(value):.2f}".split(".")[0])
            for value in finite_values
        ),
        default=1,
    )

    numeric_column = (
        r"S["
        + f"table-format=-{integer_digits}.2,"
        + r"mode=text,"
        r"detect-weight=true,"
        r"table-space-text-post={\textsuperscript{*}},"
        r"table-align-text-post=false"
        r"]"
    )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{threeparttable}",
        (
            r"\caption{Comparison of out-of-sample prediction using "
            r"Diebold--Mariano tests: "
            + escape_latex(sample_years)
            + "}"
        ),
        rf"\label{{{label}}}",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        (
            r"\begin{tabular*}{\textwidth}"
            r"{@{\extracolsep{\fill}}l"
            + numeric_column * len(column_models)
            + r"@{}}"
        ),
        r"\toprule",
        " & " + " & ".join(
            rf"\multicolumn{{1}}{{c}}{{{labels[name]}}}"
            for name in column_models
        ) + r" \\",
        " & " + " & ".join(
            rf"\multicolumn{{1}}{{c}}{{{sublabels[name]}}}"
            for name in column_models
        ) + r" \\",
        r"\midrule",
    ]

    for i, row_model in enumerate(ordered[:-1]):
        suffix = sublabels[row_model]
        separator = (
            r"\ "
            if suffix and not suffix.startswith("+")
            else ""
        )
        row_label = labels[row_model] + separator + suffix

        cells = [
            ""
            if j <= i
            else format_cell(row_model, column_model)
            for j, column_model in enumerate(
                column_models,
                start=1,
            )
        ]

        lines.append(
            row_label + " & " + " & ".join(cells) + r" \\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\endgroup",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        (
            r"\item This table reports pairwise Diebold--Mariano test "
            r"statistics comparing out-of-sample prediction performance. "
            r"Positive numbers indicate that the column model outperforms "
            r"the row model; negative numbers indicate the reverse. "
            r"Bold font indicates significance at the 5\% level using "
            r"individual two-sided p-values. An asterisk indicates "
            r"significance at the 5\% level using the supplied "
            r"Bonferroni-adjusted p-values, whose testing family includes "
            r"all planned unique model pairs in the corresponding DM run. "
            r"The tests use yearly averages of squared-loss differences "
            r"and Newey--West standard errors. "
            r"The out-of-sample period is "
            + escape_latex(sample_years)
            + r". Undefined or unavailable comparisons are shown as --."
        ),
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ])

    ensure_dir(output_path.parent)
    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    logger.info(
        f"Visualisation part finished: "
        f"{file_name}.tex saved to: {output_path}"
    )

    return output_path


def r2_comparison(
    r2_metrics_original: pd.DataFrame,
    r2_metrics: pd.DataFrame,
    eval_config,
    output_path: str | Path,
    file_name: str,
    original_years: str,
    extended_years: str,
    label: str = "tab:r2_comparison",
) -> Path:
    """
    Compare supplied Gu et al. benchmarks with two replication periods.

    Replication inputs require model and r2_overall columns.
    Replication R2 values are fractions; benchmark values below are percent.

    Fixed model IDs determine column placement, never DataFrame row positions.
    Configured model labels and sublabels are treated as LaTeX.
    """
    # model ID, fallback heading, fallback suffix, Gu benchmark in percent
    columns = [
        ("OLS_full", "OLS", "+H", -3.46),
        ("OLS_3", "OLS-3", "+H", 0.16),
        ("PLS", "PLS", "", 0.27),
        ("PCR_Linear", "PCR", "", 0.26),
        ("ENet", "ENet", "+H", 0.11),
        ("GLM", "GLM", "+H", 0.19),
        ("RandomForest", "RF", "", 0.33),
        ("GBRT", "GBRT", "+H", 0.34),
        ("NN1", "NN1", "", 0.33),
        ("NN2", "NN2", "", 0.39),
        ("NN3", "NN3", "", 0.40),
        ("NN4", "NN4", "", 0.39),
        ("NN5", "NN5", "", 0.36),
        ("PCR_Huber", "PCR", "+H", np.nan),
        ("LinearSVR", "LinSVR", "", np.nan),
        ("OLS_3_old", "Huber-3", "sklearn", np.nan),
    ]
    model_ids = [model for model, _, _, _ in columns]

    def escape_latex(text):
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(c, c) for c in str(text))

    def extract_percent(frame, frame_name):
        missing = {"model", "r2_overall"}.difference(frame.columns)
        if missing:
            raise KeyError(f"{frame_name}: missing columns {sorted(missing)}")

        data = frame[["model", "r2_overall"]].copy()
        if data["model"].isna().any():
            raise ValueError(f"{frame_name}: missing model names.")

        data["model"] = data["model"].astype(str)
        if data["model"].duplicated().any():
            raise ValueError(f"{frame_name}: expected one row per model.")

        unknown = sorted(set(data["model"]).difference(model_ids))
        if unknown:
            raise ValueError(
                f"{frame_name}: add these models to the column mapping: "
                f"{unknown}"
            )

        values = pd.to_numeric(
            data.set_index("model")["r2_overall"],
            errors="raise",
        )
        values = (values * 100.0).replace([np.inf, -np.inf], np.nan)
        return values.reindex(model_ids)

    original = extract_percent(r2_metrics_original, "r2_metrics_original")
    extended = extract_percent(r2_metrics, "r2_metrics")
    benchmark = pd.Series(
        [value for _, _, _, value in columns],
        index=model_ids,
        dtype=float,
    )

    labels = {
        model: eval_config.default_labels.get(model, heading)
        for model, heading, _, _ in columns
    }
    sublabels = {
        model: eval_config.default_sublabels.get(model, suffix)
        for model, _, suffix, _ in columns
    }

    if not re.fullmatch(r"[A-Za-z0-9:._-]+", label):
        raise ValueError("Use a simple LaTeX label.")

    output_path = Path(f"{output_path}/{file_name}.tex")
    if output_path.suffix.lower() != ".tex":
        raise ValueError("output_path must have a .tex extension.")

    all_values = pd.concat([benchmark, original, extended]).dropna()
    integer_digits = max(
        (len(f"{abs(value):.2f}".split(".")[0]) for value in all_values),
        default=1,
    )
    numeric_column = (
        f"S[table-format=-{integer_digits}.2,"
        "output-decimal-marker={.}]"
    )

    def format_value(value):
        if pd.isna(value):
            return r"\multicolumn{1}{c}{--}"
        text = f"{value:.2f}"
        return "0.00" if text == "-0.00" else text

    def table_row(title, values):
        return (
            title + " & "
            + " & ".join(format_value(values.loc[m]) for m in model_ids)
            + r" \\"
        )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{threeparttable}",
        (
            r"\caption{Comparing predictive performance: "
            r"\citet{gu_empirical_2020} vs.\ replication}"
        ),
        rf"\label{{{label}}}",
        r"\begingroup",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{2pt}",
        (
            r"\begin{tabular*}{\textwidth}"
            r"{@{\extracolsep{\fill}}l"
            + numeric_column * len(model_ids)
            + r"@{}}"
        ),
        r"\toprule",
        " & " + " & ".join(
            rf"\multicolumn{{1}}{{c}}{{{labels[m]}}}"
            for m in model_ids
        ) + r" \\",
        " & " + " & ".join(
            rf"\multicolumn{{1}}{{c}}{{{sublabels[m]}}}"
            for m in model_ids
        ) + r" \\",
        r"\midrule",
        table_row("Gu et al.", benchmark),
        table_row(r"Replication$_{2016}$", original),
        table_row(r"Replication$_{2021}$", extended),
        r"\bottomrule",
        r"\end{tabular*}",
        r"\endgroup",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        (
            r"\item Notes: Entries report monthly out-of-sample "
            r"$R^2_{\mathrm{OOS}}$ in percent for the full stock panel. "
            r"The Gu et al.\ row contains the supplied published benchmark "
            r"values. Replication$_{2016}$ covers "
            + escape_latex(original_years)
            + r"; Replication$_{2021}$ covers "
            + escape_latex(extended_years)
            + r". Models are aligned by the explicit column mapping; "
            r"shared headings do not imply identical specifications, "
            r"estimation procedures, or data. "
            r"ENet and GLM benchmark columns are retained even when those "
            r"models are absent from the replication. PCR+H, LinSVR, "
            r"and Huber-3 sklearn have no supplied Gu et al.\ benchmark. "
            r"A dash denotes an unavailable or nonfinite result."
        ),
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]

    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(
        f"Visualisation part finished: "
        f"{file_name}.tex saved to: {output_path}"
    )
    return output_path


def portfolio_metrics(
    portfolio_metrics: pd.DataFrame,
    eval_config,
    output_path: str | Path,
    file_name: str,
    sample_years: str,
    label: str = "tab:portfolio_performance",
) -> dict[str, Path]:
    """
    Export separate value- and equal-weighted Gu Table 7-style LaTeX tables.

    Three models per panel, with Pred, Avg, SD, and SR for each model.
    Return means and standard deviations are converted from fractions to
    percent. The supplied annualized Sharpe ratios are left unchanged.

    Returns a dictionary mapping weighting identifiers to output files.
    """
    metrics = [
        "mean_predicted_return",
        "mean_realized_return",
        "std_realized_return",
        "sharpe_ratio",
    ]
    required = {"model", "weighting", "portfolio", *metrics}
    missing = required.difference(portfolio_metrics.columns)
    if missing:
        raise KeyError(f"Missing portfolio columns: {sorted(missing)}")
    if portfolio_metrics.empty:
        raise ValueError("portfolio_metrics is empty.")
    if not re.fullmatch(r"[A-Za-z0-9:._-]+", label):
        raise ValueError("Use a simple LaTeX label.")

    def escape_latex(text):
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(c, c) for c in str(text))

    data = portfolio_metrics.copy()
    if data[["model", "weighting", "portfolio"]].isna().any().any():
        raise ValueError("Model, weighting, and portfolio must not be missing.")

    data["model"] = data["model"].astype(str)
    data = data[
        ~data["model"].isin(eval_config.exclude_from_table)
    ].copy()

    if data.empty:
        raise ValueError("No models remain after exclude_from_table.")

    # Normalize integer deciles and the long-short identifier.
    raw_portfolios = data["portfolio"].astype(str)
    is_hl = raw_portfolios.eq("high_minus_low")
    deciles = pd.to_numeric(raw_portfolios.where(~is_hl), errors="coerce")
    invalid = ~is_hl & (
        deciles.isna()
        | ~deciles.between(1, 10)
        | deciles.mod(1).ne(0)
    )
    if invalid.any():
        raise ValueError(
            "Expected deciles 1–10 or 'high_minus_low'. "
            f"Unexpected portfolios: {raw_portfolios[invalid].unique().tolist()}"
        )

    data["portfolio"] = [
        "high_minus_low" if hl else str(int(decile))
        for hl, decile in zip(is_hl, deciles)
    ]

    if data.duplicated(["model", "weighting", "portfolio"]).any():
        raise ValueError(
            "Expected one row per model, weighting, and portfolio."
        )

    data[metrics] = (
        data[metrics]
        .apply(pd.to_numeric, errors="raise")
        .replace([np.inf, -np.inf], np.nan)
    )
    data[metrics[:3]] *= 100.0

    weightings = {
        "value_weighted": "Value-weighted",
        "equal_weighted": "Equal-weighted",
    }
    unexpected = set(data["weighting"]).difference(weightings)
    if unexpected:
        raise ValueError(f"Unknown weighting identifiers: {sorted(unexpected)}")

    for weighting in weightings:
        if not data["weighting"].eq(weighting).any():
            raise ValueError(f"No data available for {weighting}.")

    def model_heading(model):
        heading = eval_config.default_labels.get(
            model, escape_latex(model)
        )
        suffix = eval_config.default_sublabels.get(model, "")
        separator = r"\ " if suffix and not suffix.startswith("+") else ""
        return heading + separator + suffix

    def format_value(value):
        if pd.isna(value):
            return r"\multicolumn{1}{c}{--}"
        text = f"{value:.2f}"
        return "0.00" if text == "-0.00" else text

    portfolio_rows = [
        ("1", "Low(L)"),
        *[(str(i), str(i)) for i in range(2, 10)],
        ("10", "High(H)"),
        ("high_minus_low", "H--L"),
    ]

    output_dir = Path(output_path)
    stem = Path(file_name).stem
    outputs = {}

    for weighting, weighting_title in weightings.items():
        subset = data[data["weighting"].eq(weighting)]
        available = subset["model"].drop_duplicates().tolist()
        ordered = list(dict.fromkeys(
            [m for m in eval_config.preferred_order if m in available]
            + available
        ))
        lookup = subset.set_index(["model", "portfolio"])[metrics]

        # A common width per metric keeps panels consistently aligned.
        numeric_columns = []
        for metric in metrics:
            values = subset[metric].dropna()
            digits = max(
                (
                    len(f"{abs(value):.2f}".split(".")[0])
                    for value in values
                ),
                default=1,
            )
            numeric_columns.append(
                f"S[table-format=-{digits}.2,"
                "output-decimal-marker={.}]"
            )

        # Keep three four-column model slots, even in the last panel.
        column_spec = (
            r"@{\extracolsep{\fill}}l"
            + "".join(numeric_columns) * 3
            + r"@{}"
        )

        lines = [
            r"\begin{table}[p]",
            r"\centering",
            r"\begin{threeparttable}",
            (
                r"\caption{Performance of the machine learning portfolios: "
                + weighting_title
                + " ("
                + escape_latex(sample_years)
                + ")}"
            ),
            rf"\label{{{label}:{weighting}}}",
            r"\begingroup",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2pt}",
            r"\renewcommand{\arraystretch}{1.05}",
            r"\begin{tabular*}{\textwidth}{" + column_spec + "}",
            r"\toprule",
        ]

        for start in range(0, len(ordered), 3):
            panel_models = ordered[start:start + 3]
            slots = panel_models + [None] * (3 - len(panel_models))

            if start:
                lines.append(r"\midrule")

            lines.append(
                " & "
                + " & ".join(
                    rf"\multicolumn{{4}}{{c}}{{{model_heading(model)}}}"
                    if model is not None
                    else r"\multicolumn{4}{c}{}"
                    for model in slots
                )
                + r" \\"
            )

            lines.append(" ".join(
                rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}"
                for i in range(len(panel_models))
            ))

            lines.append(
                " & "
                + " & ".join(
                    rf"\multicolumn{{1}}{{c}}{{{heading}}}"
                    if model is not None else r"\multicolumn{1}{c}{}"
                    for model in slots
                    for heading in ("Pred", "Avg", "SD", "SR")
                )
                + r" \\"
            )
            lines.append(r"\midrule")

            for portfolio, row_label in portfolio_rows:
                if portfolio == "high_minus_low":
                    lines.append(r"\addlinespace[3pt]")

                cells = []
                for model in slots:
                    if model is None:
                        cells.extend([""] * 4)
                    elif (model, portfolio) not in lookup.index:
                        cells.extend([r"\multicolumn{1}{c}{--}"] * 4)
                    else:
                        values = lookup.loc[(model, portfolio)]
                        cells.extend(
                            format_value(values[metric])
                            for metric in metrics
                        )

                lines.append(
                    row_label + " & " + " & ".join(cells) + r" \\"
                )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular*}",
            r"\endgroup",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            (
                r"\item Notes: Stocks are sorted into deciles each month "
                r"by their predicted returns. Low(L) and High(H) denote "
                r"the lowest- and highest-prediction deciles. "
                r"Pred is the average predicted monthly return; Avg is "
                r"the average realized monthly excess return; SD is the "
                r"standard deviation of realized monthly excess returns. "
                r"Pred, Avg, and SD are expressed in percent. "
                r"SR is the annualized Sharpe ratio supplied by the "
                r"evaluation function. H--L denotes the high-minus-low "
                r"portfolio; its statistics are taken directly from the "
                r"portfolio metrics, not calculated by subtracting "
                r"decile standard deviations or Sharpe ratios. "
                r"Portfolios are "
                + weighting_title.lower()
                + r". The evaluation period is "
                + escape_latex(sample_years)
                + r". Models in exclude\_from\_table are omitted. "
                r"Unavailable or nonfinite results are shown as --."
            ),
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ])

        destination = output_dir / f"{stem}_{weighting}.tex"
        ensure_dir(destination.parent)
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        outputs[weighting] = destination

        logger.info(
            f"Visualisation part finished: "
            f"{stem}_{weighting}.tex saved to: {output_dir}"
        )

    return outputs
