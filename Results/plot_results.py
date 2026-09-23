"""
Visualise federated imputation benchmark results.

Usage:
    python Results/plot_results.py
    python Results/plot_results.py --input Results/imputation_results.csv

Outputs (Results/figures/):
    --scope grid
        table_results.csv / .tex      — NRMSE per dataset x method, averaged over the grid
        figure_comparison.png         — one row per dataset, columns MR / MFR / N_clients
    --scope hetero
        figure_hetero.png             — 5 methods, homogeneous + 4 heterogeneity scenarios
        table_hp_selection.csv        — leave-one-dataset-out choice of FedProx mu / FedOpt lr
        table_hp_candidates.csv       — every candidate tag's per-scenario NRMSE (tidy)
        table_strategy.csv / .tex     — the same strategy numbers as a table
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

METHOD_ORDER = ["Mean", "MIWAE", "CAFE", "FedHF", "ReMasker"]
METHOD_LABELS = {
    "Mean": "Mean",
    "MIWAE": "Fed-MIWAE",
    "ReMasker": "Fed-ReMasker",
    "CAFE": "Cafe",
    "FedHF": "FedHF-Impute",
}
METHOD_COLORS = {
    "Mean": "#9e9e9e",
    "MIWAE": "#4c8cbf",
    "ReMasker": "#e08c3a",
    "CAFE": "#d94f3d",
    "FedHF": "#5ab46e",
}
METHOD_MARKERS = {
    "Mean": "o", "MIWAE": "s", "ReMasker": "^",
    "CAFE": "D", "FedHF": "P",
}

DATASET_ORDER = [
    "linear_d20", "linear_d50", "linear_d100",
    "nn_d20", "nn_d50", "nn_d100",
    "codon", "physionet", "nhanes",
]

# Displayed name for the homogeneous reference. Defined once: it appears as a
# tick label in both bar figures and as a row label in the strategy table, and
# those must agree. Duplicating the string is what let the seed caption drift
# out of sync with the table earlier.
HOMOGENEOUS_LABEL = "Homogeneous"

VARIANT_ORDER = [
    "iid",
    "clients_imbalanced_gini0.2",
    "clients_imbalanced_gini0.35",
    "features_imbalanced",
    "non_iid",
]
VARIANT_LABELS = {
    "iid": HOMOGENEOUS_LABEL,
    "clients_imbalanced_gini0.2": "Quantity \nskew \n(Gini 0.2)",
    "clients_imbalanced_gini0.35": "Quantity \nskew \n(Gini 0.35)",
    "features_imbalanced": "Missingness\nskew",
    # "Feature-distribution" unabbreviated is 1.43in against ~1.35in of slot, so
    # it overruns its neighbours. The table and body text carry the full name.
    "non_iid": "Feature-\ndistribution \nskew",
}

# Default fixed values when not the varying axis
FIX_MFR = 0.2
FIX_MR = 0.3
FIX_N = 5

# --- Fed-ReMasker ablation study: server-side aggregation strategy ---
# Canonical strategy names. The concrete run tags (ReMasker_prox_mu0.01,
# ReMasker_fedopt_lr0.1, ...) are collapsed onto these three by the LODO
# selection below, so each panel can use a different mu/lr while every panel
# still shares one legend.
STRATEGY_ORDER = ["FedAvg", "FedProx", "FedOpt"]

# Displayed as FedAdam, not FedOpt. FedOpt (Reddi et al., 2020) is the general
# server-optimizer framework; FedAdagrad, FedAdam and FedYogi are its named
# instantiations. Every run here left server_opt_variant at its "adam" default,
# so what the figures show is specifically FedAdam. The dict key stays "FedOpt"
# because it mirrors --hp strategy=fedopt and the ReMasker_fedopt_lr* tag names.
#
# NB: the tag name does not record the variant, so if server_opt_variant is ever
# swept, encode it in the --tag (e.g. fedyogi_lr0.01) and give it its own label
# here -- otherwise a FedYogi run would silently be plotted as FedAdam.
STRATEGY_LABELS = {"FedAvg": "FedAvg", "FedProx": "FedProx", "FedOpt": "FedAdam"}

REAL_DATASETS = ["codon", "physionet", "nhanes"]

# --- Figure styling -------------------------------------------------------
# Sizes in points, dimensions in inches. Large enough to read on screen and in
# drafts; LaTeX scales them down for the paper.
CMP_TITLE, CMP_LABEL, CMP_TICK, CMP_LEGEND = 18, 17, 15, 16
CMP_WIDTH, CMP_ROW_H = 13.0, 3.3

HET_TITLE, HET_TICK, HET_LABEL, HET_LEGEND, HET_XTICK = 20, 15, 19, 17, 13
HET_WIDTH_PER_DS, HET_H = 7.5, 5.2

LINEWIDTH, MARKERSIZE, TITLE_PAD = 1.8, 5, 12

# Axis label for the error metric. A bare "NRMSE" is a symbol-only label, which
# the conference guidance rules out ("write Magnetization, or Magnetization, M,
# not just M").
NRMSE_LABEL = "Normalized RMSE"

SAVE_DPI = 300

sns.set_theme(style="whitegrid", font_scale=1.7)
# Serif, to match the IEEE template's body text. seaborn's theme leaves this at
# DejaVu Sans, so every figure was set in a different face from the paper around
# it. Face only -- sizes and dimensions are unchanged.
plt.rcParams.update({"lines.linewidth": LINEWIDTH, "lines.markersize": MARKERSIZE,
                     "font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif",
                                    "Nimbus Roman", "DejaVu Serif"]})


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------

def _load(path):
    df = pd.read_csv(path)
    df["group"] = df["dataset_name"].str.extract(r"^(linear|nn)")[0].fillna("real")
    return df


def _to_per_experiment(df):
    """Weighted RMSE across clients → one row per (method, dataset, MFR, MR, N_clients)."""
    def _wrmse(g, col, n_col):
        valid = g[n_col] > 0
        if not valid.any():
            return np.nan
        return float(np.sqrt(
            (g.loc[valid, n_col] * g.loc[valid, col] ** 2).sum()
            / g.loc[valid, n_col].sum()
        ))

    return (
        df.groupby(["method", "dataset_name", "group", "MFR", "MR", "N_clients"])
        .apply(lambda g: pd.Series({
            "rmse": _wrmse(g, "rmse", "n_missing_numeric"),
            "rmse_mr": _wrmse(g, "rmse_mr", "n_missing_numeric_mr"),
            "rmse_mfr": _wrmse(g, "rmse_mfr", "n_missing_numeric_mfr"),
        }), include_groups=False)
        .reset_index()
    )


def _to_per_dataset(per_exp):
    """Mean across all (MFR, MR, N) → one row per (method, dataset)."""
    return (
        per_exp
        .groupby(["method", "dataset_name", "group"])
        [["rmse_mr", "rmse_mfr"]]
        .mean()
        .reset_index()
    )


_SEED_SUFFIX_RE = re.compile(r"_seed\d+$")


def _variant_base(variant):
    """Strip a trailing '_seedN' tag (create_datasets.py --hetero_seeds) so
    repeated draws of the same scenario group together."""
    return _SEED_SUFFIX_RE.sub("", variant)


def _to_hetero_per_experiment(df):
    """One row per (method, dataset, variant) for the Hetero scenarios.

    Weighted RMSE across clients is computed per seed draw first, then — when
    a variant was run at multiple seeds (create_datasets.py --hetero_seeds,
    tagged '..._seedN') — averaged again across seeds. Returns, for each of
    the combined metric and its value-level / feature-level split ('rmse',
    'rmse_mr', 'rmse_mfr'), the mean across seeds and a '..._std' companion
    (0 when only one seed exists), plus 'n_seeds'.

    Uses only the Hetero client-heterogeneity rows (is_hetero=True). Returns
    an empty frame if the input predates the is_hetero/variant columns, or
    simply has no Hetero results yet.
    """
    empty_cols = ["method", "dataset_name", "group", "variant",
                  "rmse", "rmse_std", "rmse_mr", "rmse_mr_std",
                  "rmse_mfr", "rmse_mfr_std", "n_seeds"]
    if "is_hetero" not in df.columns:
        return pd.DataFrame(columns=empty_cols)

    hetero = df[df["is_hetero"] == True]  # noqa: E712 — explicit bool compare reads clearer here
    if hetero.empty:
        return pd.DataFrame(columns=empty_cols)

    def _wrmse(g, col, n_col):
        valid = g[n_col] > 0
        if not valid.any():
            return np.nan
        return float(np.sqrt(
            (g.loc[valid, n_col] * g.loc[valid, col] ** 2).sum()
            / g.loc[valid, n_col].sum()
        ))

    hetero = hetero.copy()
    hetero["variant_base"] = hetero["variant"].map(_variant_base)

    per_seed = (
        hetero.groupby(["method", "dataset_name", "group", "variant_base", "variant"])
        .apply(lambda g: pd.Series({
            "rmse": _wrmse(g, "rmse", "n_missing_numeric"),
            "rmse_mr": _wrmse(g, "rmse_mr", "n_missing_numeric_mr"),
            "rmse_mfr": _wrmse(g, "rmse_mfr", "n_missing_numeric_mfr"),
        }), include_groups=False)
        .reset_index()
    )

    return (
        per_seed
        .groupby(["method", "dataset_name", "group", "variant_base"])
        .agg(rmse=("rmse", "mean"), rmse_std=("rmse", "std"),
             rmse_mr=("rmse_mr", "mean"), rmse_mr_std=("rmse_mr", "std"),
             rmse_mfr=("rmse_mfr", "mean"), rmse_mfr_std=("rmse_mfr", "std"),
             n_seeds=("rmse", "count"))
        .reset_index()
        .rename(columns={"variant_base": "variant"})
        .fillna({"rmse_std": 0.0, "rmse_mr_std": 0.0, "rmse_mfr_std": 0.0})
    )


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def make_table(per_dataset, out_dir):
    """
    Rows = datasets, columns = methods × {MR-NRMSE, MFR-NRMSE}.
    Values are the mean over all (MFR, MR, N) settings, expressed as NRMSE (%).
    Best per (dataset, sub) marked bold in LaTeX / * in CSV; second underlined / +.
    """
    methods = [m for m in METHOD_ORDER if m in per_dataset["method"].unique()]
    datasets = [d for d in DATASET_ORDER if d in per_dataset["dataset_name"].unique()]

    DS_FULL = {
        "linear_d20": "Linear ($d$=20)", "linear_d50": "Linear ($d$=50)", "linear_d100": "Linear ($d$=100)",
        "nn_d20": "NN ($d$=20)", "nn_d50": "NN ($d$=50)", "nn_d100": "NN ($d$=100)",
        "codon": "Codon", "physionet": "PhysioNet", "nhanes": "NHANES",
    }

    # Build flat lookup
    lookup = {}
    for _, row in per_dataset.iterrows():
        lookup[(row["method"], row["dataset_name"], "MR")] = row["rmse_mr"]
        lookup[(row["method"], row["dataset_name"], "MFR")] = row["rmse_mfr"]

    # Table: rows = datasets, cols = (method, sub)
    col_tuples = [(m, sub) for m in methods for sub in ["MR", "MFR"]]
    table = pd.DataFrame(
        index=pd.Index(datasets, name="Dataset"),
        columns=pd.MultiIndex.from_tuples(col_tuples),
        dtype=float,
    )
    for d in datasets:
        for m in methods:
            table.loc[d, (m, "MR")] = lookup.get((m, d, "MR"), np.nan)
            table.loc[d, (m, "MFR")] = lookup.get((m, d, "MFR"), np.nan)
    table = table.round(3)

    # --- best / second per (dataset, sub) — best method in each row-pair ---
    best = {}
    second = {}
    for d in datasets:
        for sub in ["MR", "MFR"]:
            vals = {m: table.loc[d, (m, sub)] for m in methods}
            vals = {m: v for m, v in vals.items() if not pd.isna(v)}
            if not vals:
                continue
            best_m = min(vals, key=vals.get)
            best[(d, sub)] = best_m
            rest = {m: v for m, v in vals.items() if m != best_m}
            if rest:
                second[(d, sub)] = min(rest, key=rest.get)

    # --- CSV ---
    csv_table = table.copy().astype(object)
    for d in datasets:
        for m in methods:
            for sub in ["MR", "MFR"]:
                v = table.loc[d, (m, sub)]
                if pd.isna(v):
                    csv_table.loc[d, (m, sub)] = "—"
                    continue
                s = f"{v:.3f}"
                if best.get((d, sub)) == m:
                    s += "*"
                elif second.get((d, sub)) == m:
                    s += "+"
                csv_table.loc[d, (m, sub)] = s

    avg_row = {}
    for m in methods:
        for sub in ["MR", "MFR"]:
            v = table[(m, sub)].mean()
            avg_row[(m, sub)] = f"{v:.3f}" if not pd.isna(v) else "—"
    csv_table.loc["Average"] = avg_row

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "table_results.csv"
    csv_table.to_csv(csv_path)
    print(f"  Saved -> {csv_path}")

    # --- LaTeX ---
    n_m = len(methods)
    col_spec = "l" + "".join(["cc"] * n_m)
    lines = [
        r"\begin{table*}[tp]",
        r"\centering",
        r"\caption{Normalized Root Mean Squared Error (NRMSE) results for each method and "
        r"dataset. The Val and Feat columns report the NRMSE on value-level "
        r"($\text{NRMSE}_\text{val}$) and feature-level ($\text{NRMSE}_\text{feat}$) missing "
        r"positions, respectively. "
        r"Values are unweighted arithmetic means of the scenario-level NRMSE values "
        r"over the 36 $(\text{MFR}, \text{MR}, K)$ settings per dataset at a single "
        r"partition seed (27 for $\text{NRMSE}_\text{feat}$, which is undefined at "
        r"$\text{MFR}=0$) and the Average row is an unweighted mean across datasets. "
        r"\textbf{Bold}: best per dataset; \underline{underline}: "
        r"second best.}",
        r"\label{tab1}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Row 1: method names spanning 2 columns each
    m_header = " & ".join(
        r"\multicolumn{2}{c}{" + METHOD_LABELS.get(m, m) + "}" for m in methods
    )
    lines.append(r"\textbf{Dataset} & " + m_header + r" \\")

    # Cmidrule per method pair
    cmidrules = " ".join(
        r"\cmidrule(lr){" + f"{2 + i*2}-{3 + i*2}" + "}"
        for i in range(n_m)
    )
    lines.append(cmidrules)

    # Row 2: Val / Feat sub-headers (= MR / MFR internally — meaning explained
    # in table caption)
    sub_header = " & ".join(["Val & Feat"] * n_m)
    lines.append(r" & " + sub_header + r" \\")
    lines.append(r"\midrule")

    for d in datasets:
        label = DS_FULL.get(d, d.replace("_", r"\_"))
        cells = []
        for m in methods:
            for sub in ["MR", "MFR"]:
                v = table.loc[d, (m, sub)]
                if pd.isna(v):
                    cells.append("—")
                else:
                    s = f"{v:.3f}"
                    if best.get((d, sub)) == m:
                        s = r"\textbf{" + s + "}"
                    elif second.get((d, sub)) == m:
                        s = r"\underline{" + s + "}"
                    cells.append(s)
        lines.append(label + " & " + " & ".join(cells) + r" \\")

    # --- Average row ---
    avg = table.mean()
    avg_best = {}
    avg_second = {}
    for sub in ["MR", "MFR"]:
        vals = {m: avg[(m, sub)] for m in methods if not pd.isna(avg[(m, sub)])}
        if vals:
            best_m = min(vals, key=vals.get)
            avg_best[sub] = best_m
            rest = {m: v for m, v in vals.items() if m != best_m}
            if rest:
                avg_second[sub] = min(rest, key=rest.get)

    lines.append(r"\midrule")
    avg_cells = []
    for m in methods:
        for sub in ["MR", "MFR"]:
            v = avg[(m, sub)]
            if pd.isna(v):
                avg_cells.append("—")
            else:
                s = f"{v:.3f}"
                if avg_best.get(sub) == m:
                    s = r"\textbf{" + s + "}"
                elif avg_second.get(sub) == m:
                    s = r"\underline{" + s + "}"
                avg_cells.append(s)
    lines.append(r"\textbf{Average} & " + " & ".join(avg_cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    tex_path = out_dir / "table_results.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved -> {tex_path}")


# ---------------------------------------------------------------------------
# 3x3 Comparison figure
# ---------------------------------------------------------------------------

def fig_comparison(per_exp, out_dir, fmt):
    """
    Rows    = one per dataset, in DATASET_ORDER (9 when the full sweep has run)
    Columns = varying factor (MR / MFR / N_clients)
    Each subplot has one line per method for that single dataset.
    """
    _ds_labels = {
        "linear_d20": "Linear (d=20)", "linear_d50": "Linear (d=50)",
        "linear_d100": "Linear (d=100)",
        "nn_d20": "Nonlinear (d=20)", "nn_d50": "Nonlinear (d=50)",
        "nn_d100": "Nonlinear (d=100)",
        "codon": "Codon", "physionet": "PhysioNet", "nhanes": "NHANES",
    }

    # One row per dataset. Pooling the three linear_d* into a single "Linear"
    # row (and likewise nn_d*) averaged away the dimensionality effect, which
    # is one of the things the synthetic sweep exists to show.
    # Each entry: (filter_column, filter_value, y_label)
    row_specs = [("dataset_name", d, _ds_labels.get(d, d))
                 for d in DATASET_ORDER if d in per_exp["dataset_name"].unique()]

    methods = [m for m in METHOD_ORDER if m in per_exp["method"].unique()]
    n_rows = len(row_specs)

    factors = [
        ("MR", "Missing Value Ratio (MVR)", FIX_MFR, None, FIX_N),
        ("MFR", "Missing Feature Ratio (MFR)", None, FIX_MR, FIX_N),
        # K, not N: N denotes dataset size in the paper. Only the displayed
        # label changes -- the column name and the N<k> directory names on disk
        # stay as they are.
        ("N_clients", "Number of clients (K)", FIX_MFR, FIX_MR, None),
    ]

    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(CMP_WIDTH, CMP_ROW_H * n_rows),
                             sharey=False, squeeze=False)

    for col_idx, (x_col, x_label, fix_mfr, fix_mr, fix_n) in enumerate(factors):
        sub = per_exp.copy()
        if fix_mfr is not None:
            sub = sub[sub["MFR"] == fix_mfr]
        if fix_mr is not None:
            sub = sub[sub["MR"] == fix_mr]
        if fix_n is not None:
            sub = sub[sub["N_clients"] == fix_n]

        for row_idx, (filter_col, filter_val, row_label) in enumerate(row_specs):
            ax = axes[row_idx, col_idx]
            row_data = sub[sub[filter_col] == filter_val]

            agg = (
                row_data.groupby(["method", x_col])["rmse"]
                .mean()
                .reset_index()
            )

            for method in methods:
                mdata = agg[agg["method"] == method].sort_values(x_col)
                if mdata.empty:
                    continue
                ax.plot(
                    mdata[x_col], mdata["rmse"],
                    label=METHOD_LABELS.get(method, method),
                    color=METHOD_COLORS.get(method),
                    marker=METHOD_MARKERS.get(method, "o"),
                    linewidth=LINEWIDTH, markersize=MARKERSIZE,
                )

            # x-axis formatting — show at most 4 ticks to avoid crowding
            if x_col in ("MR", "MFR"):
                # Decimals, not percentages: MR and MFR are stated as decimals
                # throughout the text and in the formula for the number of
                # features removed, so the axes must match rather than make the
                # reader convert. Percent signs stay reserved for quantities
                # that really are percentages -- win rates, relative changes,
                # confidence levels -- which keeps NRMSE unambiguous as a decimal.
                all_ticks = sorted(per_exp[x_col].unique())
                step = max(1, len(all_ticks) // 4)
                ax.set_xticks(all_ticks[::step])
            else:
                all_ticks = sorted(per_exp["N_clients"].unique())
                step = max(1, len(all_ticks) // 4)
                ax.set_xticks(all_ticks[::step])

            if row_idx == 0:
                ax.set_title(x_label, fontsize=CMP_TITLE, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\n{NRMSE_LABEL}",
                              fontsize=CMP_LABEL, fontweight="bold")
            ax.tick_params(labelsize=CMP_TICK)

    # Single shared legend below
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        loc="lower center", ncol=len(methods),
        bbox_to_anchor=(0.5, 0.002), fontsize=CMP_LEGEND,
        frameon=True,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.tight_layout(rect=[0, 0.035, 1, 1])
    _save(fig, "figure_comparison", out_dir, fmt)


# ---------------------------------------------------------------------------
# Heterogeneity figure
# ---------------------------------------------------------------------------

def fig_hetero(hetero_per_exp, out_dir, fmt):
    """
    Grouped bar chart: one panel per dataset, x = the homogeneous reference
    followed by the heterogeneity scenarios, bars = methods (same colours and
    order as the comparison figure). Error bars are +/- 1 std across the
    --hetero_seeds draws, the homogeneous 'iid' variant included, since it is
    drawn the same way as the skewed scenarios.
    """
    if hetero_per_exp.empty:
        print("  No results found for figure_hetero — skipping.")
        return

    datasets = [d for d in DATASET_ORDER if d in hetero_per_exp["dataset_name"].unique()]
    methods = [m for m in METHOD_ORDER if m in hetero_per_exp["method"].unique()]
    if not methods:
        print("  No matching methods found for figure_hetero — skipping.")
        return
    variants = [v for v in VARIANT_ORDER if v in hetero_per_exp["variant"].unique()]
    variants += [v for v in sorted(hetero_per_exp["variant"].unique()) if v not in variants]

    x_labels = [VARIANT_LABELS.get(v, v) for v in variants]
    n_datasets, n_m = len(datasets), len(methods)
    fig, axes = plt.subplots(1, n_datasets,
                             figsize=(HET_WIDTH_PER_DS * n_datasets, HET_H),
                             squeeze=False)
    axes = axes[0]

    x = np.arange(len(x_labels))
    bar_w = 0.8 / n_m
    _real_labels = {"codon": "Codon", "physionet": "PhysioNet", "nhanes": "NHANES"}

    for ax, dataset in zip(axes, datasets):
        sub = hetero_per_exp[hetero_per_exp["dataset_name"] == dataset]
        for i, method in enumerate(methods):
            mdata = sub[sub["method"] == method].set_index("variant")
            vals = [mdata["rmse"].get(v, np.nan) for v in variants]
            errs = [mdata["rmse_std"].get(v, 0.0) if "rmse_std" in mdata.columns else 0.0
                    for v in variants]
            x_pos = x + (i - (n_m - 1) / 2) * bar_w
            ax.bar(x_pos, vals, width=bar_w * 0.9,
                   yerr=errs, capsize=3, error_kw={"linewidth": 1, "ecolor": "#333333"},
                   label=METHOD_LABELS.get(method, method),
                   color=METHOD_COLORS.get(method))

        if variants and variants[0] == "iid":
            # Separates the homogeneous reference from the skewed scenarios.
            ax.axvline(0.5, color="#888888", linewidth=1, linestyle=(0, (3, 3)))
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=HET_XTICK)
        ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
        ax.set_title(_real_labels.get(dataset, dataset), fontsize=HET_TITLE,
                     fontweight="bold", pad=TITLE_PAD)
        ax.tick_params(labelsize=HET_TICK)
        # No gridlines at all: the three panels carry different y-scales, so a
        # horizontal rule sits at a different value in each one and invites a
        # comparison across panels that it does not actually support.
        ax.grid(visible=False)

    axes[0].set_ylabel(NRMSE_LABEL, fontsize=HET_LABEL, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center", ncol=n_m,
                        bbox_to_anchor=(0.5, -0.03), fontsize=HET_LEGEND, frameon=True)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, "figure_hetero", out_dir, fmt)


_PROX_RE = re.compile(r"^ReMasker_prox_mu([0-9.eE+-]+)$")
_FEDOPT_RE = re.compile(r"^ReMasker_fedopt_lr([0-9.eE+-]+)$")


def parse_strategy_tag(name):
    """
    Map an Imputed/ directory name onto (family, hyperparameter).

    "ReMasker"              -> ("FedAvg", None)
    "ReMasker_prox_mu0.01"  -> ("FedProx", 0.01)
    "ReMasker_fedopt_lr0.1" -> ("FedOpt", 0.1)
    anything else           -> None

    Parsed rather than listed, so adding a new mu/lr to the sweep needs no edit
    here — it is picked up as soon as its results exist on disk.
    """
    if name == "ReMasker":
        return ("FedAvg", None)
    m = _PROX_RE.match(name)
    if m:
        return ("FedProx", float(m.group(1)))
    m = _FEDOPT_RE.match(name)
    if m:
        return ("FedOpt", float(m.group(1)))
    return None


def _annotate_strategies(hetero_per_exp):
    """Hetero rows that are ReMasker strategy runs, with family/hp/rel columns."""
    df = hetero_per_exp.copy()
    parsed = df["method"].map(parse_strategy_tag)
    df = df[parsed.notna()].copy()
    if df.empty:
        return df
    df["family"] = df["method"].map(lambda m: parse_strategy_tag(m)[0])
    df["hp"] = df["method"].map(lambda m: parse_strategy_tag(m)[1])

    # Relative to FedAvg on the same (dataset, variant). Raw NRMSE differs in
    # scale between datasets, so averaging it across a fold's two training
    # datasets would let whichever one has the larger errors pick mu by itself.
    fedavg = (df[df["family"] == "FedAvg"]
              .drop_duplicates(subset=["dataset_name", "variant"])
              .set_index(["dataset_name", "variant"])["rmse"])
    df["rel"] = [
        row.rmse / fedavg.get((row.dataset_name, row.variant), np.nan)
        for row in df.itertuples()
    ]
    return df


def select_hp_lodo(hetero_per_exp, datasets=None):
    """
    Leave-one-dataset-out selection of FedProx's mu and FedOpt's lr.

    For each held-out dataset, every candidate tag is scored on the OTHER real
    datasets only, and the winner is carried over to the held-out one — so a
    reported number is never produced by a hyperparameter that was chosen on
    the same dataset it is reported for.

    Score = mean NRMSE relative to FedAvg over the fold's training datasets and
    all scenarios; lowest wins. A candidate that did not run on every training
    dataset in the fold is excluded rather than scored on a partial average,
    which would otherwise flatter whichever tag happened to run only on the
    easier dataset.

    Returns (selection, scored):
      selection : {dataset: {"FedProx": tag_or_None, "FedOpt": tag_or_None}}
      scored    : tidy frame, one row per (held_out, family, method) candidate
    """
    cols = ["held_out", "family", "method", "hp", "score", "n_train_datasets", "selected"]
    df = _annotate_strategies(hetero_per_exp)
    if df.empty:
        return {}, pd.DataFrame(columns=cols)

    datasets = datasets or [d for d in REAL_DATASETS if d in set(df["dataset_name"])]
    selection, rows = {}, []

    for held_out in datasets:
        train = [d for d in datasets if d != held_out]
        selection[held_out] = {"FedProx": None, "FedOpt": None}
        if not train:
            continue
        sub = df[df["dataset_name"].isin(train)]

        for family in ("FedProx", "FedOpt"):
            cand = sub[sub["family"] == family]
            if cand.empty:
                continue
            score = cand.groupby("method")["rel"].mean().dropna()
            coverage = cand.groupby("method")["dataset_name"].nunique()
            score = score[coverage.reindex(score.index).fillna(0) >= len(train)]
            if score.empty:
                continue
            best = score.idxmin()
            selection[held_out][family] = best
            for method, val in score.sort_values().items():
                rows.append({
                    "held_out": held_out, "family": family, "method": method,
                    "hp": parse_strategy_tag(method)[1], "score": round(float(val), 4),
                    "n_train_datasets": len(train), "selected": method == best,
                })

    return selection, pd.DataFrame(rows, columns=cols)


def build_lodo_strategy_frame(hetero_per_exp, selection):
    """
    Collapse each dataset's selected tags onto the canonical FedAvg / FedProx /
    FedOpt names, so every panel shares one legend even though the underlying
    mu/lr differs per panel. Keeps `source_tag` for the audit trail.
    """
    out = []
    for dataset, sel in selection.items():
        sub = hetero_per_exp[hetero_per_exp["dataset_name"] == dataset]
        for canonical, tag in (("FedAvg", "ReMasker"),
                               ("FedProx", sel.get("FedProx")),
                               ("FedOpt", sel.get("FedOpt"))):
            if not tag:
                continue
            rows = sub[sub["method"] == tag].copy()
            if rows.empty:
                continue
            rows["source_tag"] = tag
            rows["method"] = canonical
            out.append(rows)
    return pd.concat(out, ignore_index=True) if out else hetero_per_exp.iloc[0:0].copy()


def make_hp_selection_table(hetero_per_exp, out_dir):
    """
    Write the LODO audit trail: which mu/lr each fold picked and on what score
    (table_hp_selection.csv), plus every candidate's per-scenario numbers in
    tidy form (table_hp_candidates.csv) so any other pivot can be built from it.
    """
    selection, scored = select_hp_lodo(hetero_per_exp)
    if scored.empty:
        print("  No FedProx/FedOpt candidates found — skipping HP selection table.")
        return selection

    out_dir.mkdir(parents=True, exist_ok=True)

    sel_path = out_dir / "table_hp_selection.csv"
    scored.to_csv(sel_path, index=False)
    print("  Saved -> %s" % sel_path)

    cand_path = out_dir / "table_hp_candidates.csv"
    cand_cols = ["method", "family", "hp", "dataset_name", "variant",
                 "rmse", "rmse_std", "n_seeds", "rel"]
    df = _annotate_strategies(hetero_per_exp)
    df[[c for c in cand_cols if c in df.columns]].to_csv(cand_path, index=False)
    print("  Saved -> %s" % cand_path)

    for dataset, sel in selection.items():
        chosen = ", ".join("%s=%s" % (STRATEGY_LABELS.get(fam, fam),
                                      parse_strategy_tag(tag)[1])
                           for fam, tag in sel.items() if tag) or "nothing selected"
        print("    %s: %s  (chosen on the other %d dataset(s))"
              % (dataset, chosen, len(selection) - 1))

    return selection


def _seed_caption_clause(hetero_strat):
    """
    The caption sentence describing the +/- values, derived from the data.

    Derived from n_seeds rather than asserted: a hardcoded wording ("the
    Homogeneous row is a single run, so it has no standard deviation") silently
    became false once the seeded 'iid' variant replaced the single grid cell as
    the homogeneous reference, and the caption claimed no error bars while the
    table printed them.
    """
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    seeds = sorted({int(n) for n in hetero_strat["n_seeds"]}) if not hetero_strat.empty else []
    if not seeds:
        return "Cells report NRMSE."

    if len(seeds) == 1:
        n = seeds[0]
        draws = ("a single seed" if n == 1
                 else f"{words.get(n, n)} independent seeds")
    else:
        draws = f"{min(seeds)}--{max(seeds)} independent seeds"

    if seeds == [1]:
        return f"All cells report NRMSE from {draws}."
    return f"All cells report mean $\\pm$ standard deviation across {draws}."


def make_strategy_table(hetero_per_exp, out_dir):
    """
    FedAvg vs FedProx vs FedAdam (STRATEGY_ORDER), the same runs as
    figure_strategy_hetero.png, as a table.

    Rows = (dataset, scenario). Columns = strategy x {Val, Feat}, mirroring
    Table 1: Val is NRMSE over randomly-missing cells (MR), Feat over
    entirely-missing features (MFR). The figure plots the combined metric;
    this table breaks it out, since the two missingness types are what the
    imputers actually differ on.

    Every cell shows mean +/- std across the --hetero_seeds draws (the +/- is
    omitted only where a variant ran at a single seed), the homogeneous row
    included, since it comes from the seeded 'iid' Hetero variant.

    Best per (row, sub-column) marked bold in LaTeX / * in CSV; second
    underlined / +.
    """
    methods = [m for m in STRATEGY_ORDER
               if not hetero_per_exp.empty and m in set(hetero_per_exp["method"])]
    if not methods:
        print("  No strategy-comparison results found — skipping strategy table.")
        return

    hetero_strat = hetero_per_exp[hetero_per_exp["method"].isin(methods)] if not hetero_per_exp.empty else hetero_per_exp

    hetero_datasets = set(hetero_strat["dataset_name"].unique()) if not hetero_strat.empty else set()
    datasets = [d for d in DATASET_ORDER if d in hetero_datasets]
    _real_labels = {"codon": "Codon", "physionet": "PhysioNet", "nhanes": "NHANES"}

    variants = [v for v in VARIANT_ORDER if not hetero_strat.empty and v in hetero_strat["variant"].unique()]
    if not hetero_strat.empty:
        variants += [v for v in sorted(hetero_strat["variant"].unique()) if v not in variants]
    variant_labels = [VARIANT_LABELS.get(v, v).replace("\n", " ") for v in variants]
    scenarios = variant_labels

    # Val -> the MR split, Feat -> the MFR split (named as in Table 1).
    SUBS = [("Val", "rmse_mr", "rmse_mr_std"), ("Feat", "rmse_mfr", "rmse_mfr_std")]

    # (dataset, scenario) -> {(method, sub): (value, std_or_None)}
    lookup = {}
    for d in datasets:
        h = hetero_strat[hetero_strat["dataset_name"] == d] if not hetero_strat.empty else hetero_strat
        for v, label in zip(variants, variant_labels):
            hv = h[h["variant"] == v] if not h.empty else h
            cell = {}
            for _, row in hv.iterrows():
                for sub, col, std_col in SUBS:
                    std = row[std_col] if row["n_seeds"] > 1 else None
                    cell[(row["method"], sub)] = (row[col], std)
            lookup[(d, label)] = cell

    rows = [(d, s_) for d in datasets for s_ in scenarios]

    def _fmt(v, std, latex=False):
        if pd.isna(v):
            return None
        if std is None or pd.isna(std):
            return f"{v:.3f}"
        pm = r"$\pm$" if latex else "±"
        return f"{v:.3f} {pm} {std:.3f}"

    # --- best / second per (row, sub) ---
    # Ranked on the *displayed* precision, and each tier is a set rather than a
    # single method. Ranking on raw values would bold one of two cells the
    # reader sees as identical, asserting an ordering at the fourth decimal
    # while the seed standard deviation printed beside it is an order of
    # magnitude larger.
    best, second = {}, {}
    for key in rows:
        for sub, _col, _std in SUBS:
            vals = {m: lookup.get(key, {}).get((m, sub), (np.nan, None))[0] for m in methods}
            vals = {m: v for m, v in vals.items() if not pd.isna(v)}
            if not vals:
                continue
            shown = {m: round(v, 3) for m, v in vals.items()}
            tiers = sorted(set(shown.values()))
            top = {m for m, v in shown.items() if v == tiers[0]}
            runner = ({m for m, v in shown.items() if v == tiers[1]}
                      if len(tiers) > 1 else set())
            # Underline the runners-up only when the best is unique. If two
            # strategies share the best value, whatever remains is simply last,
            # and underlining it would present it as noteworthy. A tie *for*
            # second is different: those entries really are joint second, so
            # both are underlined.
            if len(top) > 1:
                runner = set()
            best[(key, sub)] = top
            second[(key, sub)] = runner

    # --- CSV ---
    csv_rows = []
    for key in rows:
        d, s_ = key
        row_vals = lookup.get(key, {})
        row = {"Dataset": _real_labels.get(d, d), "Scenario": s_}
        for m in methods:
            for sub, _col, _std in SUBS:
                v, std = row_vals.get((m, sub), (np.nan, None))
                cell = _fmt(v, std)
                if cell is None:
                    row[f"{STRATEGY_LABELS.get(m, m)} {sub}"] = "—"
                    continue
                if m in best.get((key, sub), ()):
                    cell += "*"
                elif m in second.get((key, sub), ()):
                    cell += "+"
                row[f"{STRATEGY_LABELS.get(m, m)} {sub}"] = cell
        csv_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "table_strategy.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  Saved -> {csv_path}")

    # --- LaTeX ---
    n_m = len(methods)
    col_spec = "ll" + "cc" * n_m
    lines = [
        r"\begin{table*}[tp]",
        r"\centering",
        r"\caption{Normalized Root Mean Squared Error (NRMSE) for the FedAvg, FedProx, and "
        r"FedAdam federated optimization strategies (Fed-ReMasker), evaluated on each real-world "
        r"dataset under the Homogeneous condition and four client-heterogeneity "
        r"scenarios. The Val and Feat columns report the NRMSE on value-level "
        r"($\text{NRMSE}_\text{val}$) and feature-level ($\text{NRMSE}_\text{feat}$) missing "
        r"positions, respectively. " + _seed_caption_clause(hetero_strat) +
        r" \textbf{Bold}: best per row; \underline{underline}: second best. "
        r"Bold and underline are applied within each position type separately, "
        r"and strategies that tie at the reported precision receive the same "
        r"mark.}",
        r"\label{tab2}",
        r"\setlength{\tabcolsep}{3pt}",
        r"{\footnotesize",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Row 1: strategy names spanning their two sub-columns
    lines.append(r"\textbf{Dataset} & \textbf{Scenario} & "
                 + " & ".join(r"\multicolumn{2}{c}{\textbf{" + STRATEGY_LABELS.get(m, m) + "}}"
                              for m in methods) + r" \\")
    lines.append(" ".join(r"\cmidrule(lr){" + f"{3 + i*2}-{4 + i*2}" + "}"
                          for i in range(n_m)))
    # Row 2: Val / Feat sub-headers
    lines.append(r" &  & " + " & ".join(["Val & Feat"] * n_m) + r" \\")
    lines.append(r"\midrule")

    for d in datasets:
        d_label = _real_labels.get(d, d.replace("_", r"\_"))
        for s_ in scenarios:
            key = (d, s_)
            row_vals = lookup.get(key, {})
            cells = []
            for m in methods:
                for sub, _col, _std in SUBS:
                    v, std = row_vals.get((m, sub), (np.nan, None))
                    cell = _fmt(v, std, latex=True)
                    if cell is None:
                        cells.append("—")
                        continue
                    if m in best.get((key, sub), ()):
                        cell = r"\textbf{" + cell + "}"
                    elif m in second.get((key, sub), ()):
                        cell = r"\underline{" + cell + "}"
                    cells.append(cell)
            lines.append(d_label + " & " + s_ + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]

    tex_path = out_dir / "table_strategy.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved -> {tex_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, name, out_dir, fmt, dpi=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=dpi or SAVE_DPI)
    plt.close(fig)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global SAVE_DPI

    parser = argparse.ArgumentParser(description="Plot imputation benchmark results.")
    parser.add_argument("--input", default=None,
                        help="Input CSV (default: Results/imputation_results.csv, or "
                             "..._grid.csv / ..._hetero.csv when --scope is grid/hetero)")
    parser.add_argument("--outdir", default="Results/figures")
    parser.add_argument("--format", default="png", choices=["png", "svg"])
    parser.add_argument("--dpi", type=int, default=SAVE_DPI,
                        help=f"Raster resolution for --format png (default: {SAVE_DPI})")
    parser.add_argument("--scope", default="both", choices=["grid", "hetero", "both"],
                        help="Render the regular grid table/figure, the Hetero figure, or both (default)")
    args = parser.parse_args()

    SAVE_DPI = args.dpi

    input_path = args.input
    if input_path is None:
        suffix = "" if args.scope == "both" else f"_{args.scope}"
        input_path = f"Results/imputation_results{suffix}.csv"

    df = _load(input_path)
    out_dir = Path(args.outdir)

    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Methods : {sorted(df['method'].unique())}")
    print(f"Datasets: {sorted(df['dataset_name'].unique())}")
    print()

    per_exp = _to_per_experiment(df)
    per_ds = _to_per_dataset(per_exp)
    hetero_per_exp = _to_hetero_per_experiment(df)

    # The homogeneous reference is the 'iid' Hetero variant: drawn once per
    # --hetero_seeds like every other scenario, so it carries a real error bar.
    if not hetero_per_exp.empty and "iid" not in set(hetero_per_exp["variant"]):
        print("WARNING: no 'iid' Hetero variant found — the homogeneous "
              "reference column will be missing. Generate it with "
              "'python Datasets/create_datasets.py --hetero --only_hetero'.")

    if args.scope in ("grid", "both"):
        print("Table ...")
        make_table(per_ds, out_dir)

        print("Comparison figure (one row per dataset) ...")
        fig_comparison(per_exp, out_dir, args.format)

    if args.scope in ("hetero", "both"):
        # Only the base methods (METHOD_ORDER) — ReMasker here is plain FedAvg;
        # the tagged strategy runs go to the strategy table below.
        print("Heterogeneity figure (5 methods) ...")
        fig_hetero(hetero_per_exp, out_dir, args.format)

        print("Hyperparameter selection (leave-one-dataset-out) ...")
        selection = make_hp_selection_table(hetero_per_exp, out_dir)

        print("Strategy comparison table (FedAvg/FedProx/FedAdam) ...")
        lodo_frame = build_lodo_strategy_frame(hetero_per_exp, selection or {})
        make_strategy_table(lodo_frame, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
