"""
Paired comparison of each baseline against a reference method.

    python Results/paired_tests.py
    python Results/paired_tests.py --reference ReMasker --metric rmse
    python Results/paired_tests.py --out Results/figures/table_paired.csv

Every method ran the same grid scenarios, so the comparison can be paired:
for each (dataset, MFR, MR, K) cell, d = NRMSE_baseline - NRMSE_reference.
Positive d means the reference won that scenario.

This is the right test for a single-seed grid. Each cell is one shared
experimental condition rather than an independent replicate of one condition,
so a confidence interval on any single method's mean would not be meaningful --
but the *difference* between two methods on the same cell is, because the
scenario is held fixed and only the method varies. Hence a paired t interval on
d, not a per-method interval.

Reports, per baseline: mean difference, its 95% paired-t CI, the fraction of
scenarios the reference wins, and the paired t-test p-value.
"""
import argparse
import fnmatch
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

METHOD_ORDER = ["Mean", "MIWAE", "CAFE", "FedHF", "ReMasker"]

# One paired unit: a (dataset, MFR, MR, K) cell on the grid, or a
# (dataset, variant) scenario under hetero, where variant keeps its _seedN
# suffix so the three seeds count separately (3 x 4 x 3 = 36).
#
# 'iid' is excluded from hetero by default: it is the homogeneous reference,
# not a heterogeneity condition, so counting it would make "across N
# heterogeneity scenarios" false for a quarter of them. --include-iid restores
# it (45 units) when the reference belongs in the average.
PAIR_KEYS = {
    "grid": ["dataset_name", "MFR", "MR", "N_clients"],
    "hetero": ["dataset_name", "variant"],
}
EXPECTED_N = {"grid": "324 = 9 datasets x 4 MFR x 3 MR x 3 K",
              "hetero": "36 = 3 datasets x 4 heterogeneity scenarios x 3 seeds"}
HETERO_WITH_IID_N = "45 = 3 datasets x 5 conditions (incl. IID) x 3 seeds"
_SEED_SUFFIX_RE = re.compile(r"_seed\d+$")


def expected_n(scope, metric, common_support=False, include_iid=False):
    """
    What the unit count should be, which the feature-level metric changes.

    At MFR=0 no feature is entirely missing, so rmse_mfr is undefined for that
    quarter of the grid and those cells drop out of the pairing. That is
    correct rather than a gap in the runs -- but the count differs from the
    other metrics, so say why instead of letting it look like missing results.
    """
    if scope == "grid" and (metric == "rmse_mfr" or common_support):
        return ("243 = 9 datasets x 3 MFR(>0) x 3 MR x 3 K; the MFR=0 cells "
                "have no feature-level missingness and are excluded")
    if scope == "hetero" and include_iid:
        return HETERO_WITH_IID_N
    return EXPECTED_N[scope]


def per_experiment(df, metric, scope, common_support=False, include_iid=False):
    """Weighted NRMSE across clients -> one row per (method, paired unit)."""
    n_col = {"rmse": "n_missing_numeric",
             "rmse_mr": "n_missing_numeric_mr",
             "rmse_mfr": "n_missing_numeric_mfr"}[metric]

    def _wrmse(g):
        valid = g[n_col] > 0
        if not valid.any():
            return np.nan
        return float(np.sqrt(
            (g.loc[valid, n_col] * g.loc[valid, metric] ** 2).sum()
            / g.loc[valid, n_col].sum()
        ))

    is_h = df["is_hetero"].astype(bool)
    sub = df[is_h] if scope == "hetero" else df[~is_h]
    if scope == "hetero" and not include_iid:
        base = sub["variant"].astype(str).str.replace(_SEED_SUFFIX_RE, "", regex=True)
        sub = sub[base != "iid"]
    if common_support and scope == "grid":
        # Drop MFR=0, where feature-level error is undefined, so Val and Feat
        # cover the same 243 cells. Only needed when comparing the two metrics
        # to each other; within one metric the pairing is already exact.
        sub = sub[sub["MFR"] > 0]
    if sub.empty:
        raise SystemExit(f"No {scope} rows in the input — nothing to pair.")
    return (sub.groupby(["method"] + PAIR_KEYS[scope])
               .apply(_wrmse, include_groups=False)
               .rename(metric)
               .reset_index())


def baselines_for(per_exp, reference, include_tags, exclude):
    """
    Which methods to compare against the reference.

    Defaults to the base methods only. Tagged runs (ReMasker_prox_mu*,
    ReMasker_fedopt_lr*) are ablation arms of the reference itself rather than
    competing baselines, and they only ever ran a couple of grid cells -- so
    including them mixes 324-scenario rows with 3-scenario rows in one table,
    where the underpowered rows read as if they meant something.
    """
    present = set(per_exp["method"])
    names = [m for m in METHOD_ORDER if m != reference and m in present]
    if include_tags:
        names += sorted(present - set(METHOD_ORDER) - {reference})
    return [n for n in names
            if not any(fnmatch.fnmatch(n, pat) for pat in exclude)]


def compare(per_exp, reference, metric, others, scope, cluster=False):
    """
    Paired comparison of each baseline against the reference.

    `cluster` chooses the unit of analysis for the interval and p-value.
    Scenarios from one dataset share its features, difficulty and partition
    seed, so they are not independent draws: a t-interval over all of them
    understates the standard error and reads as more precise than the design
    supports. With cluster=True each dataset contributes one observation --
    the mean of its own paired differences -- so n is the number of datasets.
    The design is balanced, so the point estimate is unchanged either way;
    only the interval widens.

    Win rates and counts are descriptive and unaffected, so they are reported
    identically under both.
    """
    keys = PAIR_KEYS[scope]
    ref = per_exp[per_exp["method"] == reference][keys + [metric]]
    if ref.empty:
        raise SystemExit(f"Reference method {reference!r} has no grid results.")

    rows = []
    for method in others:
        base = per_exp[per_exp["method"] == method][keys + [metric]]
        if base.empty:
            continue
        # Inner join: only units where BOTH ran, so the pairing is exact.
        merged = base.merge(ref, on=keys, suffixes=("_base", "_ref")).dropna()
        if len(merged) < 2:
            continue
        d = merged[f"{metric}_base"].to_numpy() - merged[f"{metric}_ref"].to_numpy()
        if cluster:
            units = (merged.assign(_d=d).groupby("dataset_name")["_d"]
                     .mean().to_numpy())
        else:
            units = d
        lo, hi = st.t.interval(0.95, len(units) - 1,
                               loc=units.mean(), scale=st.sem(units))
        p = float(st.ttest_1samp(units, 0.0).pvalue)
        rows.append({
            "baseline": method,
            "n_scenarios": len(d),
            "n_units": len(units),
            "mean_baseline": merged[f"{metric}_base"].mean(),
            "mean_reference": merged[f"{metric}_ref"].mean(),
            "mean_d": d.mean(),
            "ci_low": lo,
            "ci_high": hi,
            "win_rate": float((d > 0).mean()),
            "p_value": float(p),
        })
    return pd.DataFrame(rows)


def outright_win_rate(per_exp, reference, others, metric, scope):
    """
    Fraction of units where the reference is the strict minimum across ALL
    compared methods at once.

    Distinct from the pairwise win rates, and always <= the smallest of them:
    a scenario counts here only if the reference beats every baseline on it.
    This is the statistic behind a claim of the form "achieves the lowest error
    in XX% of scenarios" -- quoting a pairwise rate there would overstate it.

    Restricted to units where every method ran, so the denominator is a set of
    scenarios on which the comparison is actually complete.

    Only the base methods count, even under --include-tags. A tagged run is
    either an ablation arm of the reference or a reference condition such as a
    centralized oracle; folding those into "lowest of all methods" silently
    changes what the number claims, and it is the number that ends up quoted as
    "achieves the lowest error in XX% of scenarios".
    """
    others = [m for m in others if m in METHOD_ORDER]
    if not others:
        return 0, 0, []
    keys = PAIR_KEYS[scope]
    wide = None
    for m in [reference] + list(others):
        col = per_exp[per_exp["method"] == m][keys + [metric]].rename(
            columns={metric: m})
        wide = col if wide is None else wide.merge(col, on=keys)
    wide = wide.dropna()
    if wide.empty:
        return 0, 0, []
    vals = wide[[reference] + list(others)].to_numpy()
    wins = vals[:, 0] < vals[:, 1:].min(axis=1)
    lost = wide.loc[~wins, keys].to_dict("records")
    return int(wins.sum()), len(wide), lost, wide.loc[~wins].assign(
        best_other=wide.loc[~wins, list(others)].min(axis=1),
        winner=wide.loc[~wins, list(others)].idxmin(axis=1),
    )


def exception_profile(lost_df, reference, keys, others=()):
    """
    How bad are the scenarios the reference does not win, and where are they?

    A win rate on its own invites the question a reader should ask: are the
    remaining few catastrophic or negligible? This answers it with the size of
    each shortfall and how they distribute over the grid axes, which is the
    form the paper quotes ("at most X NRMSE, N of them at K=10").
    """
    if lost_df.empty:
        return None
    d = lost_df[reference] - lost_df["best_other"]
    out = {"n": len(lost_df),
           "max_deficit": float(d.max()),
           "median_deficit": float(d.median()),
           "max_pct": float(100 * (d / lost_df["best_other"]).max()),
           "median_pct": float(100 * (d / lost_df["best_other"]).median())}
    # Which baseline beats the reference here -- not which one is lowest. The
    # two differ wherever more than one baseline is ahead, and "baseline X
    # accounts for N of the exceptions" means the former; counting argmin
    # instead silently drops the scenarios where another baseline was lower
    # still. The counts therefore sum to more than n when they overlap.
    beats = {m: int((lost_df[m] < lost_df[reference]).sum())
             for m in others if m in lost_df.columns}
    if beats:
        out["by_baseline_beating_reference"] = {k: v for k, v in beats.items() if v}
    for k in keys:
        if k in lost_df.columns and lost_df[k].nunique() > 1 or k == "dataset_name":
            out[f"by_{k}"] = lost_df[k].value_counts().to_dict()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="Results/imputation_results.csv")
    ap.add_argument("--reference", default="ReMasker")
    ap.add_argument("--metric", default="rmse",
                    choices=["rmse", "rmse_mr", "rmse_mfr"],
                    help="rmse = combined over all missing positions (default)")
    ap.add_argument("--scope", default="grid", choices=["grid", "hetero"],
                    help="'grid' pairs on (dataset, MFR, MR, K) cells (default); "
                         "'hetero' pairs on (dataset, scenario, seed) draws, which "
                         "has genuine seed replication.")
    ap.add_argument("--include-iid", action="store_true",
                    help="Include the homogeneous 'iid' draws in the hetero scope "
                         "(45 units instead of 36). Off by default: 'iid' is the "
                         "reference condition, not a heterogeneity scenario.")
    ap.add_argument("--common-support", action="store_true",
                    help="Restrict the grid to MFR>0 so every metric is averaged "
                         "over the same 243 cells. Use when comparing Val against "
                         "Feat; unnecessary within a single metric.")
    ap.add_argument("--cluster", action="store_true",
                    help="Compute the interval and p-value across datasets "
                         "rather than across scenarios, using each dataset's "
                         "mean paired difference as one observation. Scenarios "
                         "from one dataset are not independent, so the default "
                         "interval is narrower than the design supports; the "
                         "point estimate is unchanged either way.")
    ap.add_argument("--out", default=None, help="Optional CSV path for the table")
    ap.add_argument("--exclude", nargs="*", default=["*_old"], metavar="PATTERN",
                    help="Skip methods matching these fnmatch patterns "
                         "(default: '*_old' -- superseded runs).")
    ap.add_argument("--include-tags", action="store_true",
                    help="Also compare tagged runs (ReMasker_prox_mu*, etc.). "
                         "Off by default: they are ablation arms of the reference, "
                         "and ran only a few grid cells, so their rows would be "
                         "paired over far fewer scenarios than the baselines'.")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"{path} not found — run evaluate_imputation.py first.")

    df = pd.read_csv(path)
    per_exp = per_experiment(df, args.metric, args.scope, args.common_support,
                             args.include_iid)
    others = baselines_for(per_exp, args.reference, args.include_tags, args.exclude)
    skipped = sorted(set(per_exp["method"]) - set(others) - {args.reference})
    res = compare(per_exp, args.reference, args.metric, others, args.scope,
                  cluster=args.cluster)
    if res.empty:
        raise SystemExit("No baseline shared any grid scenario with the reference.")

    print(f"Reference: {args.reference}   metric: {args.metric}   "
          f"scope: {args.scope}   source: {path}")
    print(f"d = NRMSE_baseline - NRMSE_{args.reference}, paired per "
          f"{'(dataset, scenario, seed) draw' if args.scope == 'hetero' else '(dataset, MFR, MR, K) cell'}"
          f"; d > 0 means {args.reference} wins.")
    if skipped:
        print(f"Not compared: {', '.join(skipped)}")
    print()

    show = res.copy()
    show["95% CI"] = [f"[{lo:+.4f}, {hi:+.4f}]"
                      for lo, hi in zip(res["ci_low"], res["ci_high"])]
    show["win_rate"] = (res["win_rate"] * 100).map("{:.1f}%".format)
    show["p_value"] = res["p_value"].map(
        lambda v: "<1e-300" if v == 0 else f"{v:.3g}")
    for c in ["mean_baseline", "mean_reference", "mean_d"]:
        show[c] = res[c].map("{:.4f}".format)
    print(show[["baseline", "n_scenarios", "mean_baseline", "mean_reference",
                "mean_d", "95% CI", "win_rate", "p_value"]].to_string(index=False))

    won, total, lost, lost_df = outright_win_rate(per_exp, args.reference,
                                                  others, args.metric, args.scope)
    dropped = [m for m in others if m not in METHOD_ORDER]
    if total and dropped:
        print(f"\n(outright counts base methods only; excluded from it: "
              f"{', '.join(dropped)})")
    if total:
        n_pool = len([m for m in others if m in METHOD_ORDER]) + 1
        print(f"\nOutright: {args.reference} is the lowest of all "
              f"{n_pool} methods on {won}/{total} units "
              f"({100 * won / total:.1f}%).")
        if lost and len(lost) <= 12:
            for row in lost:
                print("    lost: " + ", ".join(f"{k}={v}" for k, v in row.items()))
        prof = exception_profile(lost_df, args.reference, PAIR_KEYS[args.scope],
                                 [m for m in others if m in METHOD_ORDER])
        if prof:
            print(f"\n  The {prof['n']} exception"
                  f"{'' if prof['n'] == 1 else 's'}: deficit at most "
                  f"{prof['max_deficit']:.4f} NRMSE ({prof['max_pct']:.1f}%), "
                  f"median {prof['median_deficit']:.4f} "
                  f"({prof['median_pct']:.1f}%).")
            for k, v in prof.items():
                if k.startswith("by_"):
                    print(f"    by {k[3:]:<12} "
                          + "  ".join(f"{a}:{b}" for a, b in sorted(v.items(),
                                                                    key=str)))

    n = res["n_scenarios"].max()
    print(f"\nPaired over {n} units "
          f"(expected {expected_n(args.scope, args.metric, args.common_support, args.include_iid)}).")
    if res["n_scenarios"].nunique() > 1:
        print("NOTE: rows differ in scenario count, so `mean_reference` differs "
              "between them — each is the reference's mean over that row's own "
              "paired cells, which is what keeps the pairing valid. Rows with "
              "few scenarios have little power; do not read them as null results.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.out, index=False)
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
