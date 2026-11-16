from __future__ import annotations
import logging
import numpy as np
import pandas as pd


import re
from pathlib import Path
from io_utils import ensure_dir, save_parquet, save_csv

logger = logging.getLogger("eap_ml.visualization")


def r2_metrics_to_gu_table(
    r2_metrics: pd.DataFrame,
    eval_config,
    output_path: str | Path,
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

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".tex":
        raise ValueError("output_path must have a .tex extension.")

    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return output_path