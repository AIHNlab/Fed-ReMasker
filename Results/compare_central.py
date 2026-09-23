"""
Centralized reference vs the federated model: the cost of federation.

    python Results/compare_central.py
    python Results/compare_central.py --scope grid
    python Results/compare_central.py --out Results/figures/table_central.csv

Both arms see identical missingness masks (the centralized run pools the same
client files) and an identical compute budget, so the difference isolates
federation itself. A positive gap means the centralized model is better.

Two views:

  --scope hetero   per (dataset, scenario). Also reports the iid -> non_iid
                   degradation for each arm, which separates difficulty that is
                   intrinsic to a value-correlated mask from difficulty that
                   federation introduces: if the centralized arm degrades just
                   as much, federation is not the cause.

  --scope grid     the gap as a function of MFR. This is the one that speaks to
                   the paper's subject -- the gap at MFR=0 is the pure
                   data-pooling advantage, and its growth is the cost
                   attributable to clients holding disjoint feature sets.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from plot_results import _to_hetero_per_experiment, _to_per_experiment  # noqa: E402

METRICS = [("rmse", "All"), ("rmse_mr", "Val"), ("rmse_mfr", "Feat")]


def load(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).drop_duplicates()
    df["group"] = df["dataset_name"].str.extract(r"^(linear|nn)")[0].fillna("real")
    return df


def _check(df, ref, cen):
    have = set(df["method"])
    for m in (ref, cen):
        if m not in have:
            raise SystemExit(
                f"{m!r} not found in the input. Methods present: {sorted(have)}\n"
                "Run evaluate_imputation.py after the centralized imputation "
                "(it discovers the tag from Imputed/ automatically)."
            )


def hetero_view(df, ref, cen):
    h = _to_hetero_per_experiment(df[df["method"].isin([ref, cen])])
    rows = []
    for metric, label in METRICS:
        p = h.pivot_table(index=["dataset_name", "variant"], columns="method", values=metric)
        if ref not in p.columns or cen not in p.columns:
            continue
        p = p.dropna()
        for (ds, var), r in p.iterrows():
            rows.append({"scope": "hetero", "dataset": ds, "condition": var,
                         "metric": label, "federated": r[ref], "centralized": r[cen],
                         "gap": r[ref] - r[cen],
                         "rel_pct": 100 * (r[ref] - r[cen]) / r[ref]})
    return pd.DataFrame(rows)


def hetero_per_seed(df, ref, cen, metric="rmse"):
    """
    The gap on each individual (dataset, condition, seed) draw.

    hetero_view() averages the seeds away before reporting, so its percentages
    are condition-level. A claim about the draws themselves -- that every one of
    a dataset's draws favours the same arm, say -- cannot be read off it, and
    the condition means are not the endpoints of the draw range. This keeps the
    draws separate so the sign consistency and the actual spread are visible.
    """
    h = df[(df["is_hetero"] == True) &  # noqa: E712
           (df["method"].isin([ref, cen]))]
    if h.empty:
        return pd.DataFrame()

    def _wrmse(g):
        ok = g[g["n_missing_numeric"] > 0]
        if ok.empty:
            return np.nan
        return float(np.sqrt((ok["n_missing_numeric"] * ok[metric] ** 2).sum()
                             / ok["n_missing_numeric"].sum()))

    per = (h.groupby(["method", "dataset_name", "variant"])
             .apply(_wrmse, include_groups=False).rename("v").reset_index())
    p = per.pivot_table(index=["dataset_name", "variant"], columns="method",
                        values="v").dropna()
    if ref not in p.columns or cen not in p.columns:
        return pd.DataFrame()
    p = p.reset_index()
    p["gap"] = p[ref] - p[cen]
    p["rel_pct"] = 100 * p["gap"] / p[ref]
    return p


def noniid_penalty(df, ref, cen):
    """iid -> non_iid degradation for each arm, per dataset and metric."""
    h = _to_hetero_per_experiment(df[df["method"].isin([ref, cen])])
    rows = []
    for metric, label in METRICS:
        p = h.pivot_table(index=["dataset_name", "variant"], columns="method", values=metric)
        if ref not in p.columns or cen not in p.columns:
            continue
        for ds in sorted({d for d, _ in p.index}):
            if (ds, "iid") not in p.index or (ds, "non_iid") not in p.index:
                continue
            r = {}
            for arm, name in ((ref, "federated"), (cen, "centralized")):
                a, b = p.loc[(ds, "iid"), arm], p.loc[(ds, "non_iid"), arm]
                if pd.isna(a) or pd.isna(b):
                    r = None
                    break
                r[f"{name}_iid"], r[f"{name}_noniid"] = a, b
                r[f"{name}_penalty"] = b - a
            if r:
                rows.append({"dataset": ds, "metric": label, **r})
    return pd.DataFrame(rows)


def _paired_grid(df, ref, cen, metric):
    """Cells where both arms ran, one row per (dataset, MFR, MR, K)."""
    per = _to_per_experiment(df[df["method"].isin([ref, cen])])
    p = per.pivot_table(index=["dataset_name", "MFR", "MR", "N_clients"],
                        columns="method", values=metric)
    if ref not in p.columns or cen not in p.columns:
        return None
    p = p.dropna()
    return None if p.empty else p.assign(gap=p[ref] - p[cen]).reset_index()


def grid_view(df, ref, cen, hold_k=5):
    """
    Gap per MFR level, at a single client count.

    K is held fixed because the centralized arm swept MFR only at K=5; the
    K=3/K=10 cells exist at one (MFR, MR) point alone. Averaging them in would
    give that MFR level a different cell mix from its neighbours, so the levels
    would no longer be comparable to each other.
    """
    rows = []
    for metric, label in METRICS:
        g = _paired_grid(df, ref, cen, metric)
        if g is None:
            continue
        g = g[g["N_clients"] == hold_k]
        if g.empty:
            continue
        for mfr, s in g.groupby("MFR"):
            rows.append({"scope": f"grid (K={hold_k})", "metric": label, "MFR": mfr,
                         "n_cells": len(s),
                         "federated": s[ref].mean(), "centralized": s[cen].mean(),
                         "gap": s["gap"].mean(),
                         "rel_pct": 100 * s["gap"].mean() / s[ref].mean()})
    return pd.DataFrame(rows)


def k_view(df, ref, cen):
    """
    Gap per client count, restricted to the (MFR, MR) points where more than one
    K was run in both arms -- otherwise the K levels would differ in which cells
    they average, which is the very confound this view exists to avoid.
    """
    rows = []
    for metric, label in METRICS:
        g = _paired_grid(df, ref, cen, metric)
        if g is None:
            continue
        n_k = g.groupby(["MFR", "MR"])["N_clients"].nunique()
        keep = set(n_k[n_k > 1].index)
        if not keep:
            continue
        g = g[[(m, r) in keep for m, r in zip(g["MFR"], g["MR"])]]
        for k, s in g.groupby("N_clients"):
            rows.append({"scope": "grid by K", "metric": label, "K": int(k),
                         "MFR": sorted({m for m, _ in keep}),
                         "MR": sorted({r for _, r in keep}),
                         "n_cells": len(s),
                         "federated": s[ref].mean(), "centralized": s[cen].mean(),
                         "gap": s["gap"].mean(),
                         "rel_pct": 100 * s["gap"].mean() / s[ref].mean()})
    return pd.DataFrame(rows)


DATASET_LABELS = {"codon": "Codon", "nhanes": "NHANES", "physionet": "PhysioNet"}


def write_tex(hetero_t, grid_t, k_t, out, hold_k=5, metric="All", decimals=3):
    """
    The three views as one single-column table: gap vs MFR, gap vs K, and the
    per-dataset heterogeneity conditions.

    Only one metric is emitted (default "All"). Val and Feat track it closely
    enough that three panels would state the same result three times, and the
    table has to survive in one IEEE column -- hence four columns, with the
    cell counts moved into the caption and the absolute gap dropped (it is
    federated - centralized, and both are shown).
    """
    def esc(s): return str(s).replace("_", r"\_")
    rows = []

    if grid_t is not None and not grid_t.empty:
        g = grid_t[grid_t["metric"] == metric].sort_values("MFR")
        if not g.empty:
            rows.append((None, rf"(a) Feature-missingness rate ($K={hold_k}$)"))
            for _, r in g.iterrows():
                rows.append((rf"MFR $={r.MFR:g}$", r))

    if k_t is not None and not k_t.empty:
        k = k_t[k_t["metric"] == metric].sort_values("K")
        if not k.empty:
            mfr = ", ".join(f"{v:g}" for v in k.iloc[0]["MFR"])
            mr = ", ".join(f"{v:g}" for v in k.iloc[0]["MR"])
            rows.append((None, rf"(b) Client count (MFR $={mfr}$, MR $={mr}$)"))
            for _, r in k.iterrows():
                rows.append((rf"$K={int(r.K)}$", r))

    if hetero_t is not None and not hetero_t.empty:
        h = hetero_t[hetero_t["metric"] == metric]
        if not h.empty:
            rows.append((None, rf"(c) Heterogeneity (real datasets, $K={hold_k}$)"))
            for _, r in h.iterrows():
                ds = DATASET_LABELS.get(r.dataset, esc(r.dataset))
                cond = "Homogeneous" if r.condition == "iid" else "non-IID"
                rows.append((f"{ds}, {cond}", r))

    if not rows:
        print(f"No {metric!r} rows to write -- skipping {out}")
        return

    L = [
        r"\begin{table}[tp]",
        r"\centering",
        r"\caption{Cost of federation. Fed-ReMasker"
        r" versus centralized ReMasker trained on the pooled client data, "
        r"using the same underlying client-level missingness masks and an identical number of data passes. "
        r"$\Delta$\% is the relative NRMSE gap; positive values favor centralization. "
        r"Values are NRMSE over all missing positions.}",
        r"\label{tab:central}",
        r"\begin{tabular}{lccr}",
        r"\toprule",
        r" & Federated & Centralized & $\Delta$\% \\",
        r"\midrule",
    ]
    first = True
    for label, payload in rows:
        if label is None:
            if not first:
                L.append(r"\addlinespace")
            L.append(r"\multicolumn{4}{l}{\textit{" + payload + r"}} \\")
            first = False
            continue
        L.append(rf"\quad {label} & {payload.federated:.{decimals}f} & "
                 rf"{payload.centralized:.{decimals}f} & ${payload.rel_pct:+.1f}$ \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", default=["Results/imputation_results.csv"],
                    help="One or more per-client result CSVs; they are concatenated.")
    ap.add_argument("--reference", default="ReMasker", help="The federated arm")
    ap.add_argument("--centralized", default="ReMasker_central", help="The pooled arm")
    ap.add_argument("--scope", default="both", choices=["hetero", "grid", "both"])
    ap.add_argument("--hold-k", type=int, default=5,
                    help="Client count at which the MFR breakdown is reported "
                         "(default: 5, the only K where the centralized arm swept MFR)")
    ap.add_argument("--out", default=None,
                    help="Optional CSV path for the tables. A LaTeX version of "
                         "the same tables is written alongside it, as .tex")
    ap.add_argument("--tex-metric", default="All", choices=["All", "Val", "Feat"],
                    help="Which metric the .tex table reports (default: All). "
                         "One only -- the three track each other closely, and "
                         "the table has to fit a single column.")
    ap.add_argument("--tex-decimals", type=int, default=4,
                    help="Decimal places for the NRMSE columns of the .tex table "
                         "(default: 4). Deliberately one more than "
                         "table_results.tex: that table separates methods "
                         "(differences ~0.1), this one separates two arms of the "
                         "same method (gaps ~0.002). At 3 dp several rows round "
                         "to two identical values beside a non-zero Delta%%, "
                         "which reads as an error in the table.")
    args = ap.parse_args()

    df = load(args.input)
    _check(df, args.reference, args.centralized)
    print(f"{args.centralized} vs {args.reference}   "
          f"(positive gap = centralized is better)\n")

    # The single number for the whole comparison, before the breakdowns. The
    # per-MFR and per-K views each cover part of the grid and overlap at their
    # shared cell, so neither can be averaged into an overall figure -- this
    # pools every cell where both arms ran, once each.
    if args.scope in ("grid", "both"):
        overall = _paired_grid(df, args.reference, args.centralized, "rmse")
        if overall is not None and not overall.empty:
            rel = 100 * overall["gap"].mean() / overall[args.reference].mean()
            by_k = ", ".join(
                f"K={int(k)}: {len(s)}"
                for k, s in overall.groupby("N_clients"))
            print(f"=== Overall: {rel:+.1f}% over the {len(overall)} "
                  f"configurations where both arms ran ({by_k}) ===\n")

    tables = []
    hetero_t = grid_t = kt = None
    if args.scope in ("hetero", "both"):
        t = hetero_t = hetero_view(df, args.reference, args.centralized)
        if not t.empty:
            tables.append(t)
            for label in ("All", "Val", "Feat"):
                s = t[t["metric"] == label]
                if s.empty:
                    continue
                print(f"=== Heterogeneity conditions, {label} ===")
                for _, r in s.iterrows():
                    print(f"  {r.dataset:<11}{r.condition:<9}"
                          f"fed={r.federated:.4f}  central={r.centralized:.4f}  "
                          f"gap={r.gap:+.4f} ({r.rel_pct:+.1f}%)")
                print(f"  mean gap over {len(s)} conditions: {s.gap.mean():+.4f} "
                      f"({s.rel_pct.mean():+.1f}%)\n")

            seeds = hetero_per_seed(df, args.reference, args.centralized)
            if not seeds.empty:
                print("=== Per-draw, by dataset (All) ===")
                print("    (the condition means above average the seeds away; "
                      "these are the individual draws)")
                for ds, g in seeds.groupby("dataset_name"):
                    n_pos = int((g.gap > 0).sum())
                    sign = ("all favour centralizing" if n_pos == len(g)
                            else "all favour federating" if n_pos == 0
                            else "mixed")
                    print(f"  {ds:<11}{n_pos}/{len(g)} draws positive -- {sign}; "
                          f"range {g.rel_pct.min():+.1f}% .. {g.rel_pct.max():+.1f}%")
                    for _, r in g.sort_values("variant").iterrows():
                        print(f"      {r.variant:<18}{r.gap:+.4f}  ({r.rel_pct:+.1f}%)")
                print()

            pen = noniid_penalty(df, args.reference, args.centralized)
            if not pen.empty:
                print("=== iid -> non_iid degradation, both arms ===")
                print("    (similar penalties => the difficulty is intrinsic to the "
                      "mask, not caused by federation)")
                for label in ("All", "Val", "Feat"):
                    s = pen[pen["metric"] == label]
                    if s.empty:
                        continue
                    print(f"  {label}:")
                    for _, r in s.iterrows():
                        print(f"    {r.dataset:<11}federated {r.federated_penalty:+.4f}"
                              f"    centralized {r.centralized_penalty:+.4f}")
                print()

    if args.scope in ("grid", "both"):
        t = grid_t = grid_view(df, args.reference, args.centralized, args.hold_k)
        if t.empty:
            print("=== Grid: no cells where both arms ran ===\n")
        else:
            tables.append(t)
            print(f"=== Grid: gap by MFR (K={args.hold_k}) ===")
            for label in ("All", "Val", "Feat"):
                s = t[t["metric"] == label]
                if s.empty:
                    continue
                print(f"  {label}:")
                for _, r in s.iterrows():
                    print(f"    MFR={r.MFR:<5} n={int(r.n_cells):<4} "
                          f"fed={r.federated:.4f}  central={r.centralized:.4f}  "
                          f"gap={r.gap:+.4f} ({r.rel_pct:+.1f}%)")
            print()

        kt = k_view(df, args.reference, args.centralized)
        if not kt.empty:
            tables.append(kt)
            mfr = kt.iloc[0]["MFR"]
            mr = kt.iloc[0]["MR"]
            print(f"=== Grid: gap by K (at MFR={mfr}, MR={mr}) ===")
            print("    (a flat centralized column means K changes only the mask "
                  "geometry, which a row-wise imputer does not see)")
            for label in ("All", "Val", "Feat"):
                s = kt[kt["metric"] == label]
                if s.empty:
                    continue
                print(f"  {label}:")
                for _, r in s.iterrows():
                    print(f"    K={r.K:<4} n={int(r.n_cells):<4} "
                          f"fed={r.federated:.4f}  central={r.centralized:.4f}  "
                          f"gap={r.gap:+.4f} ({r.rel_pct:+.1f}%)")
            print()

    if args.out and tables:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(tables, ignore_index=True).to_csv(out, index=False)
        print(f"Saved -> {out}")
        write_tex(hetero_t, grid_t, kt, out.with_suffix(".tex"),
                  hold_k=args.hold_k, metric=args.tex_metric,
                  decimals=args.tex_decimals)


if __name__ == "__main__":
    main()
