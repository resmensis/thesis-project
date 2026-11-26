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
            r"\caption{Monthly out-of-sample stock-level prediction " #
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


def cumulative_returns(
    portfolio_returns: pd.DataFrame,
    eval_config,
    output_path: str | Path,
    file_name: str,
    sample_years: str,
    sp500_returns: pd.DataFrame | None = None,
    models: list[str] | None = None,
    label: str = "fig:cumulative_returns",
    invert_short: bool = False,
) -> Path:
    """
    Export a Gu et al. Figure 9-style LaTeX (pgfplots) figure.

    Plots cumulative log returns of value-weighted decile portfolios sorted
    on out-of-sample forecasts. Solid = long (portfolio "10"), dashed =
    short (portfolio "1"), black = cumulative S&P 500 proxy excess return.

    portfolio_returns required columns:
        date, model, weighting, portfolio, realized_return

    sp500_returns: output of build_sp500_portfolio() (the content of
        sp500_portfolio_returns.parquet), long format with columns
        date, model, realized_return and optionally portfolio / weighting.
        The S&P 500 realized return does not depend on the model, so the
        rows are collapsed to one series per month. The function raises if
        models disagree for a month.

    realized_return must be a decimal monthly excess return (not percent).
    portfolio must be a string ("1".."10" or "high_minus_low").
    invert_short: if True, plot the negative of the bottom-decile
        cumulative log return (economic short position).

    Uses eval_config.preferred_order, default_labels, default_sublabels,
    exclude_from_plot, cumulative_return_models and nber_recessions.
    Configured labels are treated as LaTeX.
    """
    required = {"date", "model", "weighting", "portfolio", "realized_return"}
    missing = required.difference(portfolio_returns.columns)
    if missing:
        raise KeyError(f"Missing portfolio_returns columns: {sorted(missing)}")
    if portfolio_returns.empty:
        raise ValueError("portfolio_returns is empty.")
    if not re.fullmatch(r"[A-Za-z0-9:._-]+", label):
        raise ValueError("Use a simple LaTeX label without special characters.")

    output_path = Path(f"{output_path}/{file_name}.tex")
    if output_path.suffix.lower() != ".tex":
        raise ValueError("output_path must have a .tex extension.")

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

    def to_month_start(values):
        dates = pd.to_datetime(values)
        if not pd.DatetimeIndex(dates).is_month_start.all():
            raise ValueError("Dates must be month-start timestamps.")
        return dates
    
    def clean_returns(values, name):
        out = pd.to_numeric(values, errors="raise").replace(
            [np.inf, -np.inf], np.nan
        )
        if (out <= -1.0).any():
            raise ValueError(f"{name}: returns <= -100% make log returns undefined.")
        return out

    def x_start(ts):
        return ts.year + (ts.month - 1) / 12.0

    def x_end(ts):
        return ts.year + ts.month / 12.0

    data = portfolio_returns.loc[
        portfolio_returns["weighting"].astype(str).eq("value_weighted")
        & portfolio_returns["portfolio"].astype(str).isin(["1", "10"]),
        ["date", "model", "portfolio", "realized_return"],
    ].copy()
    if data.empty:
        raise ValueError("No value-weighted rows for portfolios '1' and '10'.")

    data["date"] = to_month_start(data["date"])
    data["model"] = data["model"].astype(str)
    data["portfolio"] = data["portfolio"].astype(str)
    if data[["date", "model", "portfolio"]].isna().any().any():
        raise ValueError("date, model, and portfolio must not be missing.")
    if data.duplicated(["model", "date", "portfolio"]).any():
        raise ValueError("Expected one row per model, date, and portfolio.")
    data["realized_return"] = clean_returns(data["realized_return"], "portfolio_returns")

    available = data["model"].drop_duplicates().tolist()
    if models is None:
        requested = [
            m for m in eval_config.cumulative_return_models if m in available
        ]
        absent = [
            m for m in eval_config.cumulative_return_models
            if m not in available
        ]
        if absent:
            logger.info(
                f"cumulative_returns: configured models not in data: {absent}"
            )
    else:
        unknown = [m for m in models if m not in available]
        if unknown:
            raise ValueError(f"Requested models not in data: {unknown}")
        requested = list(dict.fromkeys(models))

    ordered = list(dict.fromkeys(
        [m for m in eval_config.preferred_order if m in requested]
        + requested
    ))
    plot_models = [m for m in ordered if m not in eval_config.exclude_from_plot]
    if not plot_models:
        raise ValueError("No models remain after exclude_from_plot.")

    long_wide = data[data["portfolio"].eq("10")].pivot(
        index="date", columns="model", values="realized_return"
    )
    short_wide = data[data["portfolio"].eq("1")].pivot(
        index="date", columns="model", values="realized_return"
    )
    for name in plot_models:
        if name not in long_wide.columns or name not in short_wide.columns:
            raise ValueError(f"Model {name} needs both portfolios '1' and '10'.")

    all_dates = long_wide.index.union(short_wide.index).sort_values()
    long_wide = long_wide.reindex(all_dates)[plot_models]
    short_wide = short_wide.reindex(all_dates)[plot_models]

    valid = long_wide.notna().all(axis=1) & short_wide.notna().all(axis=1)
    for name in plot_models:
        dropped = int(
            (long_wide[name].isna() | short_wide[name].isna()).sum()
        )
        logger.info(f"cumulative_returns: {name}: {dropped} months with missing returns.")

    market = None
    if sp500_returns is not None:
        sp_required = {"date", "model", "realized_return"}
        sp_missing = sp_required.difference(sp500_returns.columns)
        if sp_missing:
            raise KeyError(f"Missing sp500_returns columns: {sorted(sp_missing)}")
        if sp500_returns.empty:
            raise ValueError("sp500_returns is empty.")

        sp = sp500_returns.copy()
        if "portfolio" in sp.columns:
            sp = sp[sp["portfolio"].astype(str).eq("S&P 500")]
        if "weighting" in sp.columns:
            sp = sp[sp["weighting"].astype(str).eq("value_weighted")]
        if sp.empty:
            raise ValueError("No value-weighted 'S&P 500' rows in sp500_returns.")

        sp["date"] = to_month_start(sp["date"])
        sp["model"] = sp["model"].astype(str)
        if sp[["date", "model"]].isna().any().any():
            raise ValueError("sp500_returns: date and model must not be missing.")
        if sp.duplicated(["model", "date"]).any():
            raise ValueError("sp500_returns: expected one row per model and date.")
        sp["realized_return"] = clean_returns(sp["realized_return"], "sp500_returns")

        # The S&P 500 return is model-independent: collapse and verify.
        grouped = sp.groupby("date")["realized_return"]
        spread = grouped.max() - grouped.min()
        n_nan = grouped.apply(lambda s: int(s.isna().sum()))
        n_rows = grouped.size()
        mixed_nan = (n_nan > 0) & (n_nan < n_rows)
        if (spread > 1e-12).any() or mixed_nan.any():
            raise ValueError(
                "sp500_returns: realized_return differs across models "
                "within a month; expected a model-independent series."
            )
        market = grouped.max().reindex(all_dates)
        logger.info(
            f"cumulative_returns: S&P 500: {int(market.isna().sum())} "
            "months missing after alignment."
        )
        valid &= market.notna()

    common = all_dates[valid.to_numpy()]
    if len(common) < 2:
        raise ValueError("Fewer than two common months across models and S&P 500.")
    if len(common) != len(all_dates):
        logger.info(
            f"cumulative_returns: using {len(common)} of {len(all_dates)} months "
            "(common finite months only)."
        )

    def cumulate(series):
        return np.log1p(series.loc[common].to_numpy(dtype=float)).cumsum()

    long_cum = {m: cumulate(long_wide[m]) for m in plot_models}
    short_cum = {m: cumulate(short_wide[m]) for m in plot_models}
    if invert_short:
        short_cum = {m: -v for m, v in short_cum.items()}
    market_cum = cumulate(market) if market is not None else None

    x_origin = x_start(common[0])
    xs = [x_end(d) for d in common]
    x_min, x_max = x_origin, xs[-1]

    all_values = np.concatenate(
        [np.zeros(1)]
        + list(long_cum.values())
        + list(short_cum.values())
        + ([market_cum] if market_cum is not None else [])
    )
    y_lo, y_hi = float(all_values.min()), float(all_values.max())
    margin = max(0.05 * (y_hi - y_lo), 0.1)
    y_min, y_max = y_lo - margin, y_hi + margin

    palette = [
        "0072B2", "D55E00", "009E73", "CC79A7",
        "E69F00", "56B4E9", "882255", "117733",
    ]
    if len(plot_models) > len(palette):
        raise ValueError(f"At most {len(palette)} models are supported.")
    colors = {m: f"crmodel{i}" for i, m in enumerate(plot_models)}

    def model_label(model):
        heading = eval_config.default_labels.get(model, escape_latex(model))
        suffix = eval_config.default_sublabels.get(model, "")
        separator = r"\ " if suffix and not suffix.startswith("+") else ""
        return heading + separator + suffix

    def coords(values):
        points = [f"({x_origin:.6f},0.000000)"] + [
            f"({x:.6f},{v:.6f})" for x, v in zip(xs, values)
        ]
        return " ".join(points)

    first_tick = int(np.ceil(x_min))
    last_tick = int(np.floor(x_max))
    ticks = ",".join(str(t) for t in range(first_tick, last_tick + 1, 3))

    recessions = []
    for start, end in eval_config.nber_recessions:
        x0 = max(x_start(pd.Timestamp(start)), x_min)
        x1 = min(x_end(pd.Timestamp(end)), x_max)
        if x0 < x1:
            recessions.append((x0, x1))

    lines = [
        r"\begin{figure}[htbp]",
        r"\centering",
        (
            r"\caption{Cumulative return of machine learning portfolios: "
            + escape_latex(sample_years) + "}"
        ),
        rf"\label{{{label}}}",
    ]
    for i, color in enumerate(palette[:len(plot_models)]):
        lines.append(rf"\definecolor{{crmodel{i}}}{{HTML}}{{{color}}}")

    lines.extend([
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"width=\linewidth,",
        r"height=7cm,",
        rf"xmin={x_min:.6f}, xmax={x_max:.6f},",
        rf"ymin={y_min:.6f}, ymax={y_max:.6f},",
        rf"xtick={{{ticks}}},",
        r"xticklabel={\pgfmathprintnumber[fixed,precision=0,"
        r"set thousands separator={}]{\tick}},",
        r"xticklabel style={font=\scriptsize},",
        r"yticklabel style={font=\scriptsize},",
        r"ylabel={Cumulative log return},",
        r"ylabel style={font=\scriptsize},",
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

    for x0, x1 in recessions:
        lines.append(
            rf"\fill[gray,opacity=0.15] (axis cs:{x0:.6f},{y_min:.6f}) "
            rf"rectangle (axis cs:{x1:.6f},{y_max:.6f});"
        )

    for name in plot_models:
        lines.extend([
            rf"\addplot[{colors[name]},solid,no marks,thick] "
            rf"coordinates {{{coords(long_cum[name])}}};",
            rf"\addlegendentry{{{model_label(name)}}}",
        ])
    for name in plot_models:
        lines.append(
            rf"\addplot[{colors[name]},dashed,no marks,thick,forget plot] "
            rf"coordinates {{{coords(short_cum[name])}}};"
        )

    if market_cum is not None:
        lines.extend([
            rf"\addplot[black,solid,no marks,thick] "
            rf"coordinates {{{coords(market_cum)}}};",
            r"\addlegendentry{S\&P 500}",
        ])

    lines.extend([
        r"\addlegendimage{black,solid,thick}",
        r"\addlegendentry{Long}",
        r"\addlegendimage{black,dashed,thick}",
        r"\addlegendentry{Short}",
        r"\end{axis}",
        r"\end{tikzpicture}",
    ])

    short_text = (
        "the negative of the bottom-decile cumulative log return"
        if invert_short
        else "the cumulative log return of the bottom decile itself"
    )
    notes = (
        r"\par\medskip\begingroup\footnotesize\emph{Notes:} "
        r"The figure shows cumulative log returns of value-weighted decile "
        r"portfolios sorted on out-of-sample return forecasts. Solid lines "
        r"denote the long position (top decile); dashed lines denote the "
        f"short position ({short_text}). "
        r"Shaded areas indicate NBER recessions. "
    )
    if market_cum is not None:
        notes += (
            r"The S\&P 500 line is the cumulative excess return of a "
            r"value-weighted proxy built from the largest 500 stocks by "
            r"lagged market equity. "
        )
    else:
        notes += r"No market benchmark was supplied. "
    notes += (
        f"The out-of-sample period is {escape_latex(sample_years)}; "
        f"the plotted sample spans {common[0]:%Y-%m} to {common[-1]:%Y-%m} "
        f"({len(common)} common months)."
        r"\endgroup"
    )
    lines.extend([notes, r"\end{figure}"])

    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(
        f"Visualisation part finished: "
        f"{file_name}.tex saved to: {output_path}"
    )

    return output_path


def portfolio_r2(
    fama_french_portfolio_metrics: pd.DataFrame,
    sp500_portfolio_metrics: pd.DataFrame,
    eval_config,
    output_path: str | Path,
    file_name: str,
    sample_years: str,
    label: str = "tab:portfolio_r2",
) -> Path:
    """
    Recreate a Gu et al. Table 5-style LaTeX table of portfolio-level R2.

    Required columns in both inputs:
        panel, family, portfolio, model, r2
    An optional weighting column must contain only 'value_weighted'.

    R2 values must be fractions; they are displayed in percent.
    Model columns follow eval_config.preferred_order, then any further
    models in order of appearance, as in r2_metrics().
    Rows are looked up by (panel, family, portfolio, model), so the
    'Big neutral' and 'Small neutral' rows of different families
    never collide.

    Missing row/model combinations are printed as '--'.
    Configured model labels and sublabels are treated as LaTeX.
    """
    # panel, family, portfolio identifier, displayed label
    panel_a_rows = [
        ("A", "market", "S&P 500", "S&P 500"),
        ("A", "size", "SMB", "SMB"),
        ("A", "value", "HML", "HML"),
        ("A", "profitability", "RMW", "RMW"),
        ("A", "investment", "CMA", "CMA"),
        ("A", "momentum", "UMD", "UMD"),
    ]
    panel_b_blocks = [
        ("value", ["value", "growth", "neutral"]),
        ("investment", ["conservative", "aggressive", "neutral"]),
        ("profitability", ["robust", "weak", "neutral"]),
        ("momentum", ["up", "down", "medium"]),
    ]
    panel_b_rows = [
        [
            ("B", family, f"{size} {name}", f"{size} {name}")
            for size in ("Big", "Small")
            for name in names
        ]
        for family, names in panel_b_blocks
    ]
    expected_rows = {
        (panel, family, portfolio)
        for panel, family, portfolio, _ in (
            panel_a_rows + [row for block in panel_b_rows for row in block]
        )
    }

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

    key_cols = ["panel", "family", "portfolio", "model"]
    required = {*key_cols, "r2"}

    def prepare(frame, frame_name):
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{frame_name}: missing columns {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{frame_name} is empty.")

        if "weighting" in frame.columns:
            weightings = frame["weighting"].dropna().astype(str)
            if not weightings.eq("value_weighted").all():
                raise ValueError(
                    f"{frame_name}: expected only 'value_weighted' "
                    "weighting."
                )

        data = frame[[*key_cols, "r2"]].copy()
        if data[key_cols].isna().any().any():
            raise ValueError(
                f"{frame_name}: panel, family, portfolio, and model "
                "must not be missing."
            )
        data[key_cols] = data[key_cols].astype(str)
        data["r2"] = pd.to_numeric(data["r2"], errors="raise")
        return data

    data = pd.concat(
        [
            prepare(fama_french_portfolio_metrics, "fama_french_portfolio_metrics"),
            prepare(sp500_portfolio_metrics, "sp500_portfolio_metrics"),
        ],
        ignore_index=True,
    )

    excluded = sorted(
        set(data["model"]).intersection(eval_config.exclude_from_table)
    )
    if excluded:
        logger.info(
            f"portfolio_r2: excluding models via exclude_from_table: "
            f"{excluded}"
        )
    data = data[
        ~data["model"].isin(eval_config.exclude_from_table)
    ].copy()

    if data.empty:
        raise ValueError("No models remain after exclude_from_table.")

    if data.duplicated(key_cols).any():
        raise ValueError(
            "Expected one row per panel, family, portfolio, and model."
        )

    observed_rows = set(
        data[["panel", "family", "portfolio"]].itertuples(
            index=False, name=None
        )
    )
    unexpected = sorted(observed_rows.difference(expected_rows))
    if unexpected:
        raise ValueError(
            "Unexpected (panel, family, portfolio) combinations: "
            f"{unexpected}"
        )

    if not re.fullmatch(r"[A-Za-z0-9:._-]+", label):
        raise ValueError("Use a simple LaTeX label.")

    output_path = Path(f"{output_path}/{file_name}.tex")
    if output_path.suffix.lower() != ".tex":
        raise ValueError("output_path must have a .tex extension.")

    available = data["model"].drop_duplicates().tolist()
    ordered = list(dict.fromkeys(
        [name for name in eval_config.preferred_order if name in available]
        + available
    ))

    labels = {
        name: eval_config.default_labels.get(name, escape_latex(name))
        for name in ordered
    }
    sublabels = {
        name: eval_config.default_sublabels.get(name, "")
        for name in ordered
    }

    data["r2"] = (
        data["r2"].replace([np.inf, -np.inf], np.nan) * 100.0
    )
    lookup = data.set_index(key_cols)["r2"]

    def get_value(panel, family, portfolio, model):
        return lookup.get((panel, family, portfolio, model), np.nan)

    def format_value(value):
        if pd.isna(value):
            return r"\multicolumn{1}{c}{--}"
        text = f"{value:.2f}"
        return "0.00" if text == "-0.00" else text

    finite_values = data["r2"].dropna()
    integer_digits = max(
        (len(f"{abs(value):.2f}".split(".")[0]) for value in finite_values),
        default=1,
    )
    numeric_column = (
        f"S[table-format=-{integer_digits}.2,"
        "output-decimal-marker={.}]"
    )

    n_columns = len(ordered) + 1

    def table_row(row):
        panel, family, portfolio, display = row
        values = " & ".join(
            format_value(get_value(panel, family, portfolio, model))
            for model in ordered
        )
        return f"{escape_latex(display)} & {values}" + r" \\"

    def panel_title(title):
        return (
            rf"\multicolumn{{{n_columns}}}{{l}}"
            rf"{{\textit{{{escape_latex(title)}}}}} \\"
        )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{threeparttable}",
        (
            r"\caption{Monthly portfolio-level out-of-sample predictive "
            r"$R^2$ (percent): "
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
            + numeric_column * len(ordered)
            + r"@{}}"
        ),
        r"\toprule",
        " & " + " & ".join(
            rf"\multicolumn{{1}}{{c}}{{{labels[m]}}}" for m in ordered
        ) + r" \\",
        " & " + " & ".join(
            rf"\multicolumn{{1}}{{c}}{{{sublabels[m]}}}" for m in ordered
        ) + r" \\",
        r"\midrule",
        panel_title("A. Common factor portfolios"),
        *[table_row(row) for row in panel_a_rows],
        r"\addlinespace",
        panel_title("B. Subcomponents of factor portfolios"),
    ]

    for i, block in enumerate(panel_b_rows):
        if i > 0:
            lines.append(r"\addlinespace")
        lines.extend(table_row(row) for row in block)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\endgroup",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        (
            r"\item Notes: Entries report monthly out-of-sample "
            r"$R^2_{\mathrm{OOS}}$ in percent, computed on portfolio-level "
            r"returns and forecasts. The 29 portfolios (an S\&P 500 proxy, "
            r"five factor portfolios, and 24 size double-sorted portfolios) "
            r"are constructed from stock-level predictions using value "
            r"weights. The S\&P 500 is a proxy formed from the 500 stocks "
            r"with the largest market equity, not the index itself. "
            r"Factor portfolios are monthly sorts on the evaluation "
            r"universe, not the official French portfolios. "
            r"$R^2_{\mathrm{OOS}}$ uses the zero-benchmark denominator "
            r"$\sum r^2$. Model columns follow the order and labels of "
            r"the stock-level performance table. "
            r"The out-of-sample period is "
            + escape_latex(sample_years)
            + r". A dash denotes an unavailable or nonfinite result."
        ),
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ])

    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(
        f"Visualisation part finished: "
        f"{file_name}.tex saved to: {output_path}"
    )

    return output_path


def _boundaries(w, test_year: int, kind: str):
    """Cumulative-year boundaries (inclusive end years) for one refit."""
    train_len = w.initial_train_end_year - w.train_start_year + 1
    val_len = w.val_end_year - w.val_start_year + 1
    val_end = test_year - 1
    train_end = val_end - val_len
    start = (w.train_start_year - 1) if kind == "expanding" else (train_end - train_len)
    return start, train_end, val_end, test_year


def _cum_by_year(obs: pd.DataFrame, years) -> dict[int, float]:
    yearly = obs.groupby(obs["month"].dt.year)["n_obs"].sum().sort_index()
    cum = yearly.cumsum()
    first = int(yearly.index.min())

    def f(y):
        if y < first:
            return 0.0
        return float(cum.loc[:y].iloc[-1])

    return {y: f(y) for y in years}


def _panel_coords(w, cum_fn, test_years, kind, scale=1.0):
    rows = []
    for t in test_years:
        rows.append([cum_fn(y) * scale for y in _boundaries(w, t, kind)])
    return np.array(rows)  # shape (n_years, 4): start, train_end, val_end, test_end


def _coords_str(xs, ys):
    return " ".join(f"({x},{y:.6g})" for x, y in zip(xs, ys))

def write_split_tex(
    obs_csv_path: str,
    window_cfg,
    out_path: str,
    test_end_year: int | None = None,
):
    obs = pd.read_csv(obs_csv_path, parse_dates=["month"])
    test_end_year = test_end_year or window_cfg.test_end_year_extended
    test_years = list(range(window_cfg.test_start_year, test_end_year + 1))

    cum_act = _cum_by_year(obs, range(window_cfg.train_start_year - 1, test_end_year + 1))
    act_fn = lambda y: cum_act[max(y, window_cfg.train_start_year - 1)]
    n_years = test_end_year - window_cfg.train_start_year + 1
    th_fn = lambda y: max(0, y - window_cfg.train_start_year + 1) / n_years

    MILL = 1e-6
    panels = [("expanding", th_fn, 1.0), ("rolling", th_fn, 1.0),
              ("expanding", act_fn, MILL), ("rolling", act_fn, MILL)]
    coords = [_panel_coords(window_cfg, fn, test_years, kind, scale=s) for kind, fn, s in panels]
    y_top_act = float(np.ceil(max(c.max() for c in coords[2:]) * 2) / 2)  # shared, rounded to 0.5

    x0, x1 = test_years[0], test_years[-1]
    xticks = ",".join(str(y) for y in range(int(np.ceil(x0 / 5) * 5), x1 + 1, 5))

    L = [
        r"% Requires: \usepackage{pgfplots}, \pgfplotsset{compat=1.18},",
        r"% \usepgfplotslibrary{groupplots,fillbetween}",
        r"\begin{tikzpicture}",
        r"\definecolor{splitTrain}{HTML}{0072B2}",
        r"\definecolor{splitVal}{HTML}{E69F00}",
        r"\definecolor{splitTest}{HTML}{D55E00}",
        r"\begin{groupplot}[",
        r"  group style={group size=2 by 2, horizontal sep=0.5cm, vertical sep=1.5cm,",
        r"    xticklabels at=edge bottom, xlabels at=edge bottom,",
        r"    yticklabels at=edge left, ylabels at=edge left},",
        rf"  width=0.50\textwidth, height=5cm, xmin={x0}, xmax={x1}, ymin=0,",
        rf"  xtick={{{xticks}}}, xticklabel style={{/pgf/number format/1000 sep={{}}}},",
        r"  xlabel={Testing year}, tick align=outside, axis on top,",
        r"  ymajorgrids, grid style={gray!25, very thin}, tick style={gray},",
        r"  every axis title/.style={font=\small\bfseries}, label style={font=\small},",
        r"  tick label style={font=\footnotesize},",
        r"]",
    ]

    cols = ["splitTrain", "splitVal", "splitTest"]
    names = ["Training", "Validation", "Testing"]
    titles = ["Expanding window", "Rolling window", "", ""]

    for i, ((kind, _, _), c) in enumerate(zip(panels, coords)):
        theory = i < 2
        opts = []
        if titles[i]:
            opts.append(f"title={{{titles[i]}}}")
        if i % 2 == 0:
            opts.append("ylabel={Share of timeline}" if theory else "ylabel={Observations (millions)}")
        opts.append("ymax=1, ytick={0,0.25,0.5,0.75,1}" if theory else f"ymax={y_top_act:g}")
        if i == 0:
            opts += [
                "legend columns=3", "legend to name=splitleg",
                "legend style={draw=none, font=\\small, /tikz/every even column/.append style={column sep=0.4cm}}",
            ]
        L.append(r"\nextgroupplot[" + ", ".join(opts) + "]")

        for j in range(4):
            L.append(rf"\addplot[name path=b{i}{j}, gray!70!black, very thin, forget plot] "
                     rf"coordinates {{{_coords_str(test_years, c[:, j])}}};")
        for j in range(3):
            L.append(rf"\addplot[{cols[j]}, fill opacity=0.65, forget plot] "
                     rf"fill between[of=b{i}{j} and b{i}{j+1}];")
        if i == 0:
            for j in range(3):
                L.append(rf"\addlegendimage{{area legend, fill={cols[j]}, fill opacity=0.65, draw=gray}}")
                L.append(rf"\addlegendentry{{{names[j]}}}")

    L += [
        r"\end{groupplot}",
        r"\path (group c1r2.south west) -- (group c2r2.south east) coordinate[midway] (splitmid);",
        r"\node[anchor=north, yshift=-1.1cm] at (splitmid) {\ref{splitleg}};",
        r"\end{tikzpicture}",
    ]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    return out

