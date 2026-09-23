import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Imputation"))

import numpy as np
import pandas as pd
import fnmatch
import glob
import argparse
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from log_utils import log
from data_utils import compute_col_ranges

# The five methods the benchmark compares. Tagged runs on disk
# (ReMasker_prox_mu*, ReMasker_central, ...) are ablation arms or reference
# conditions, not competitors, so any statement of the form "the ranking of the
# methods" has to exclude them or it silently means something else.
BASE_METHODS = ("Mean", "MIWAE", "ReMasker", "CAFE", "FedHF")


def evaluate_imputation(input_root="Datasets", method="Mean", scope="both"):
    """
    Evaluate imputation quality for one method across all datasets / splits.

    Returns a tidy DataFrame — one row per client file — with columns:
        method, dataset_type, dataset_name, is_hetero, variant, MFR, MR, N_clients, client,
        rmse        — normalised RMSE over ALL missing numeric positions
        rmse_mr     — normalised RMSE on randomly-missing cells only (MR)
        rmse_mfr    — normalised RMSE on fully-missing features only (MFR)
        n_missing_numeric

    Non-numeric (categorical) columns are skipped entirely.

    RMSE is normalised by the global observed range of each feature (computed
    from all client missing files in the group, matching impute_missing.py).

    Rows come from either the regular MFR/MR/N_clients grid (is_hetero=False,
    variant=NaN) or the Hetero client-heterogeneity variants created by
    create_datasets.py --hetero (is_hetero=True, MFR/MR/N_clients=NaN,
    variant holds the scenario name e.g. "features_imbalanced"). `scope`
    controls which of the two are scanned: "grid", "hetero", or "both" (default).
    """
    imputed_files = []
    if scope in ("grid", "both"):
        imputed_files += glob.glob(f"{input_root}/*/*/Imputed/{method}/*/*/*/*.csv")
    if scope in ("hetero", "both"):
        imputed_files += glob.glob(f"{input_root}/*/*/Imputed/{method}/Hetero/*/*.csv")
    imputed_files = sorted(imputed_files)
    input_path = Path(input_root).resolve()
    rows = []
    col_range_cache = {}  # keyed by group tuple — avoid re-reading N×N files

    for imp_file in tqdm(imputed_files, desc=f"Evaluating {method}", unit="file"):
        imp_path = Path(imp_file).resolve()
        rel = imp_path.relative_to(input_path)

        dataset_type = rel.parts[0]
        dataset_name = rel.parts[1]
        is_hetero = rel.parts[4] == "Hetero"
        if is_hetero:
            variant = rel.parts[5]
            cond_parts = ("Hetero", variant)
        else:
            variant = None
            cond_parts = (rel.parts[4], rel.parts[5], rel.parts[6])
        filename = Path(rel.parts[-1]).stem.replace("_imputed", "") + ".csv"
        client_id = Path(rel.parts[-1]).stem.replace("_imputed", "")

        gt_file = input_path / dataset_type / dataset_name / "Original" / Path(*cond_parts) / filename
        missing_file = input_path / dataset_type / dataset_name / "Missing" / Path(*cond_parts) / filename

        if not gt_file.exists():
            print(f"  Warning: GT not found — {gt_file}")
            continue
        if not missing_file.exists():
            print(f"  Warning: missing file not found — {missing_file}")
            continue

        try:
            df_gt = pd.read_csv(gt_file)
            df_imp = pd.read_csv(imp_file)
            df_miss = pd.read_csv(missing_file)
        except (OSError, pd.errors.ParserError) as e:
            # One truncated file (an interrupted run, a transient I/O error)
            # should not lose an evaluation already hundreds of files in.
            print(f"  Warning: failed to read a file for {imp_file}: {e}")
            continue
        mask = df_miss.isnull()

        # --- global observed ranges (cached per group) ---
        group_key = (dataset_type, dataset_name) + cond_parts
        if group_key not in col_range_cache:
            missing_dir = input_path / dataset_type / dataset_name / "Missing" / Path(*cond_parts)
            col_range_cache[group_key] = compute_col_ranges(missing_dir.glob("*.csv"))
        col_range = col_range_cache[group_key]

        # --- identify MFR columns (entirely NaN) vs MR cells (partially NaN) ---
        mfr_cols = {col for col in df_miss.columns if df_miss[col].isna().all()}
        mr_cols = {col for col in df_miss.columns
                   if df_miss[col].isna().any() and col not in mfr_cols}

        sq_all, sq_mr, sq_mfr = [], [], []
        wd_all, wd_mr, wd_mfr = [], [], []
        n_miss_num = n_miss_num_mr = n_miss_num_mfr = 0

        for col in df_gt.columns:
            if not np.issubdtype(df_gt[col].dtype, np.number):
                continue  # skip non-numeric (categorical) columns entirely
            col_mask = mask[col]
            if col_mask.sum() == 0:
                continue

            true_vals = df_gt.loc[col_mask, col]
            pred_vals = df_imp.loc[col_mask, col]

            se = ((true_vals - pred_vals) / col_range.get(col, 1.0)) ** 2
            sq_all.append(se.values)

            w_dist = wasserstein_distance(true_vals, pred_vals) / col_range.get(col, 1.0)
            wd_all.append(w_dist)

            n_miss_num += int(col_mask.sum())
            if col in mfr_cols:
                sq_mfr.append(se.values)
                wd_mfr.append(w_dist)
                n_miss_num_mfr += int(col_mask.sum())
            elif col in mr_cols:
                sq_mr.append(se.values)
                wd_mr.append(w_dist)
                n_miss_num_mr += int(col_mask.sum())

        def _rmse(parts):
            return float(np.sqrt(np.concatenate(parts).mean())) if parts else np.nan

        def _mean_wd(parts):
            return float(np.mean(parts)) if parts else np.nan

        rows.append({
            "method": method,
            "dataset_type": dataset_type,
            "dataset_name": dataset_name,
            "is_hetero": is_hetero,
            "variant": variant,
            "MFR": np.nan if is_hetero else float(cond_parts[0].split("_")[1]),
            "MR": np.nan if is_hetero else float(cond_parts[1].split("_")[1]),
            "N_clients": np.nan if is_hetero else int(cond_parts[2].lstrip("N")),
            "client": client_id,
            "rmse": _rmse(sq_all),
            "rmse_mr": _rmse(sq_mr),
            "rmse_mfr": _rmse(sq_mfr),
            "wd": _mean_wd(wd_all),
            "wd_mr": _mean_wd(wd_mr),
            "wd_mfr": _mean_wd(wd_mfr),
            "n_missing_numeric": n_miss_num,
            "n_missing_numeric_mr": n_miss_num_mr,
            "n_missing_numeric_mfr": n_miss_num_mfr,
        })

    columns = [
        "method", "dataset_type", "dataset_name", "is_hetero", "variant",
        "MFR", "MR", "N_clients", "client", "rmse", "rmse_mr", "rmse_mfr",
        "wd", "wd_mr", "wd_mfr", "n_missing_numeric", "n_missing_numeric_mr", "n_missing_numeric_mfr",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df["is_hetero"] = df["is_hetero"].astype(bool)  # explicit dtype: an empty `rows` would
    # otherwise default every column (including this one) to `object`, and `~` on a non-bool
    # empty column corrupts the frame's structure downstream in aggregate_results().
    return df


def discover_methods(input_root="Datasets", exclude=()):
    """
    Every method/tag that actually has imputed output on disk, read from the
    `Imputed/<name>/` directory names.

    `exclude` is a list of fnmatch patterns (e.g. "*_old") dropped from the
    result — superseded runs kept on disk for reference would otherwise be
    evaluated in full, costing a few thousand file reads each for rows no
    figure plots.

    Replaces a hardcoded list: --tag runs land in `Imputed/<Method>_<tag>/`
    with arbitrary names (ReMasker_prox_mu0.001, ReMasker_fedopt_lr0.1, ...),
    so any fixed list silently skips every ablation arm it wasn't updated for
    while reporting "No results found" for names that no longer exist.

    Sorted so the base methods come first in METHOD_ORDER order, then tagged
    variants alphabetically — keeps the console summary readable.
    """
    base_order = ["Mean", "MIWAE", "ReMasker", "CAFE", "FedHF"]
    names = {Path(d).name for d in glob.glob(f"{input_root}/*/*/Imputed/*")
             if Path(d).is_dir()}
    if exclude:
        names = {n for n in names
                 if not any(fnmatch.fnmatch(n, pat) for pat in exclude)}
    ordered = [m for m in base_order if m in names]
    ordered += sorted(n for n in names if n not in base_order)
    return ordered


def _weighted_rmse(rmse_vals, n_vals):
    valid = n_vals > 0
    if not valid.any():
        return np.nan
    return float(np.sqrt((n_vals[valid] * rmse_vals[valid] ** 2).sum() / n_vals[valid].sum()))


def _macro_rmse(rmse_vals, n_vals):
    """
    Unweighted mean of the per-client NRMSE — every client counts once,
    regardless of how many missing entries it holds.

    The weighted form above is the paper's metric: it answers "what is the
    average error per imputed value". This one answers "per client", which is
    the question that matters when client sizes differ by several fold. Kept
    alongside so the choice of weighting can be shown not to drive any result
    rather than merely asserted.
    """
    valid = n_vals > 0
    return float(rmse_vals[valid].mean()) if valid.any() else np.nan


def _weighted_mean(vals, n_vals):
    valid = n_vals > 0
    if not valid.any():
        return np.nan
    return float((n_vals[valid] * vals[valid]).sum() / n_vals[valid].sum())


def aggregate_results(combined):
    """
    Given a per-client DataFrame, return two aggregated DataFrames:
        - per_experiment : weighted RMSE across clients per (method, dataset, MFR, MR, N_clients),
                           weighted by number of missing values so larger clients contribute more.
        - per_dataset    : simple mean across experiments — each (MFR, MR, N_clients) combo
                           gets equal weight regardless of client count or missing value volume.
    Hetero rows (is_hetero=True, no MFR/MR/N_clients) are excluded here —
    see aggregate_hetero_results.
    """
    combined = combined[~combined["is_hetero"]]

    per_experiment = (
        combined
        .groupby(["method", "dataset_name", "MFR", "MR", "N_clients"])
        .apply(lambda g: pd.Series({
            "rmse": _weighted_rmse(g["rmse"], g["n_missing_numeric"]),
            "rmse_mr": _weighted_rmse(g["rmse_mr"], g["n_missing_numeric_mr"]),
            "rmse_mfr": _weighted_rmse(g["rmse_mfr"], g["n_missing_numeric_mfr"]),
            "rmse_macro": _macro_rmse(g["rmse"], g["n_missing_numeric"]),
            "wd": _weighted_mean(g["wd"], g["n_missing_numeric"]),
            "wd_mr": _weighted_mean(g["wd_mr"], g["n_missing_numeric_mr"]),
            "wd_mfr": _weighted_mean(g["wd_mfr"], g["n_missing_numeric_mfr"]),
        }), include_groups=False)
        .round(4)
        .reset_index()
    )

    metric_cols = ["rmse", "rmse_mr", "rmse_mfr", "rmse_macro",
                   "wd", "wd_mr", "wd_mfr"]
    per_dataset = (
        per_experiment
        .groupby(["method", "dataset_name"])
        [metric_cols]
        .mean()
        .round(4)
        .reset_index()
    )

    return per_experiment, per_dataset


def aggregate_hetero_results(combined):
    """
    Weighted RMSE across clients per (method, dataset_name, variant) for the
    Hetero client-heterogeneity scenarios — mirrors aggregate_results but
    keyed by 'variant' instead of MFR/MR/N_clients.
    """
    hetero = combined[combined["is_hetero"]]
    if hetero.empty:
        return pd.DataFrame(columns=[
            "method", "dataset_name", "variant",
            "rmse", "rmse_mr", "rmse_mfr", "rmse_macro", "wd", "wd_mr", "wd_mfr",
        ])

    return (
        hetero
        .groupby(["method", "dataset_name", "variant"])
        .apply(lambda g: pd.Series({
            "rmse": _weighted_rmse(g["rmse"], g["n_missing_numeric"]),
            "rmse_mr": _weighted_rmse(g["rmse_mr"], g["n_missing_numeric_mr"]),
            "rmse_mfr": _weighted_rmse(g["rmse_mfr"], g["n_missing_numeric_mfr"]),
            "rmse_macro": _macro_rmse(g["rmse"], g["n_missing_numeric"]),
            "wd": _weighted_mean(g["wd"], g["n_missing_numeric"]),
            "wd_mr": _weighted_mean(g["wd_mr"], g["n_missing_numeric_mr"]),
            "wd_mfr": _weighted_mean(g["wd_mfr"], g["n_missing_numeric_mfr"]),
        }), include_groups=False)
        .round(4)
        .reset_index()
    )


def run_all(
    methods=BASE_METHODS,
    input_root="Datasets",
    output_path="Results/imputation_results.xlsx",
    scope="both",
):
    """
    Evaluate all methods and save results to an Excel file with three sheets:
        - 'per_client'     : one row per client file (raw results)
        - 'per_experiment' : mean across clients per (method, dataset, MFR, MR, N_clients)
        - 'per_dataset'    : two-step mean across experiments — equal weight per experiment
    A CSV copy of the per-client sheet is also saved alongside.

    `scope` ("grid", "hetero", or "both") restricts which imputed results are
    scanned — see evaluate_imputation().
    """
    all_dfs = []
    for method in methods:
        log(f"Evaluating {method} ...")
        df = evaluate_imputation(input_root=input_root, method=method, scope=scope)
        msg = f"  {len(df)} client files evaluated." if not df.empty else "  No results found."
        log(msg)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    per_experiment, per_dataset = aggregate_results(combined)
    hetero_per_experiment = aggregate_hetero_results(combined)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sheets = ["per_client"]
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="per_client", index=False)
        # Skip sheets that would be empty for this scope (e.g. per_experiment/per_dataset
        # are grid-only and hetero_per_experiment is hetero-only — a --scope hetero or
        # --scope grid file otherwise ends up with confusing, uselessly-empty sheets).
        if not per_experiment.empty:
            per_experiment.to_excel(writer, sheet_name="per_experiment", index=False)
            sheets.append("per_experiment")
        if not per_dataset.empty:
            per_dataset.to_excel(writer, sheet_name="per_dataset", index=False)
            sheets.append("per_dataset")
        if not hetero_per_experiment.empty:
            hetero_per_experiment.to_excel(writer, sheet_name="hetero_per_experiment", index=False)
            sheets.append("hetero_per_experiment")

    combined.to_csv(out.with_suffix(".csv"), index=False)

    log(f"Saved -> {out} (sheets: {', '.join(sheets)})")
    log(f"Saved -> {out.with_suffix('.csv')} (per-client CSV)")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default=None,
                        help="Evaluate one method or tagged variant e.g. CAFE, CAFE_v2 (default: all)")
    parser.add_argument("--input_root", type=str, default="Datasets")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: Results/imputation_results.xlsx, or "
                             "..._grid.xlsx / ..._hetero.xlsx when --scope is grid/hetero)")
    parser.add_argument("--scope", type=str, default="both", choices=["grid", "hetero", "both"],
                        help="Evaluate the regular MFR/MR/N grid, the Hetero variants, or both (default)")
    parser.add_argument("--exclude", nargs="*", default=["*_old"], metavar="PATTERN",
                        help="Skip discovered methods matching these fnmatch patterns "
                             "(default: '*_old'). Pass --exclude with no values to "
                             "evaluate everything on disk. Ignored when --method is given.")
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        suffix = "" if args.scope == "both" else f"_{args.scope}"
        output_path = f"Results/imputation_results{suffix}.xlsx"

    if args.method:
        methods = [args.method]
    else:
        methods = discover_methods(args.input_root, exclude=args.exclude)
        if not methods:
            raise SystemExit(
                f"No Imputed/ directories found under {args.input_root}/*/*/ — "
                "nothing to evaluate. Run Imputation/impute_missing.py first."
            )
        log(f"Discovered {len(methods)} method(s) on disk: {', '.join(methods)}")
        skipped = sorted(set(discover_methods(args.input_root)) - set(methods))
        if skipped:
            log(f"Skipped {len(skipped)} by --exclude: {', '.join(skipped)}")
    df = run_all(methods=methods, input_root=args.input_root, output_path=output_path, scope=args.scope)

    if args.scope in ("grid", "both"):
        _, per_dataset = aggregate_results(df)
        print("\n=== Per-dataset mean RMSE (equal weight per experiment) ===")
        print(per_dataset.to_string(index=False))
    if args.scope in ("hetero", "both"):
        hetero_summary = aggregate_hetero_results(df)
        print("\n=== Per-(method, dataset, variant) weighted RMSE (Hetero) ===")
        print(hetero_summary.to_string(index=False) if not hetero_summary.empty else "  No Hetero results found.")

        # Does client-size weighting drive the result? rmse pools every entry,
        # so large clients weigh more; rmse_macro averages per client. Printed
        # because the paper claims an answer, which should not need a script.
        if not hetero_summary.empty:
            h = hetero_summary[hetero_summary["method"].isin(BASE_METHODS)].copy()
            h["condition"] = h["variant"].str.replace(r"_seed\d+$", "",
                                                      regex=True)
            per = (h.groupby(["condition", "method"])[["rmse", "rmse_macro"]]
                     .mean())
            per["shift"] = per["rmse_macro"] - per["rmse"]
            flips = [c for c, g in per.groupby("condition")
                     if list(g["rmse"].sort_values().index)
                     != list(g["rmse_macro"].sort_values().index)]
            worst = per["shift"].abs().idxmax()
            print("\n=== Equal-client (macro) sensitivity ===")
            print(f"  Largest shift: {per.loc[worst, 'shift']:+.4f} NRMSE "
                  f"({worst[1]} under {worst[0]})")
            print("  Method ranking under macro vs pooled: "
                  + ("unchanged in all "
                     f"{per.index.get_level_values(0).nunique()} conditions"
                     if not flips else f"DIFFERS in {', '.join(flips)}"))
