import numpy as np
import pandas as pd


def get_hp(args, key, defaults):
    """Return args.key if set (e.g. via --hp KEY=VALUE), else fall back to defaults[key]."""
    return getattr(args, key, defaults[key])


def log_hps(args, defaults, method_name):
    """
    Print the effective hyperparameters for a method, showing which values
    come from defaults and which were overridden via --hp.
    """
    overrides = {k for k in defaults if getattr(args, k, defaults[k]) != defaults[k]}
    print(f"\n  [{method_name}] Hyperparameters:")
    for k, default_val in defaults.items():
        effective = getattr(args, k, default_val)
        tag = "  <- override" if k in overrides else ""
        print(f"    {k} = {effective}{tag}")
    print()


# ---------------------------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------------------------

def verify_column_consistency(df_dict):
    """Check all DataFrames have identical columns and no column is 100% missing."""
    if not df_dict:
        print("ERROR: DataFrame dictionary is empty.")
        return False

    filenames = list(df_dict.keys())
    reference_cols = df_dict[filenames[0]].columns
    globally_empty = set(df_dict[filenames[0]].columns[df_dict[filenames[0]].isnull().all()])

    mismatched = []
    for name, df in df_dict.items():
        if not reference_cols.equals(df.columns):
            mismatched.append(name)
            continue
        globally_empty &= set(df.columns[df.isnull().all()])

    if mismatched:
        print(f"ERROR: Mismatch in {len(mismatched)} files: {mismatched}")
        return False

    if globally_empty:
        print(f"WARNING: columns 100% empty in every file: {list(globally_empty)}")
        return False

    return True


def read_dataset(files):
    """Load client CSVs, auto-detect and encode categorical columns."""
    raw_data_dict = {f: pd.read_csv(f) for f in files}
    if not verify_column_consistency(raw_data_dict):
        return None, None, None, None

    all_dfs = list(raw_data_dict.values())
    all_cols = all_dfs[0].columns.copy()

    global_categories = {}
    cat_features, vocab_sizes = [], []

    for col_idx, col in enumerate(all_cols):
        combined_col = pd.concat([df[col] for df in all_dfs if col in df.columns])
        if combined_col.dropna().dtype == object:
            global_categories[col] = combined_col.dropna().unique()
            cat_features.append(col_idx)
            vocab_sizes.append(len(global_categories[col]))

    processed = {}
    for filename, df_raw in raw_data_dict.items():
        proc = {}
        for col in df_raw.columns:
            if col in global_categories:
                cat_col = pd.Categorical(df_raw[col], categories=global_categories[col])
                proc[col] = pd.Series(cat_col.codes).replace(-1, np.nan)
            else:
                proc[col] = pd.to_numeric(df_raw[col], errors="coerce")
        processed[filename] = pd.DataFrame(proc)

    return processed, cat_features, vocab_sizes, global_categories


# ---------------------------------------------------------------------------
# Statistics and normalisation
# ---------------------------------------------------------------------------

def _compute_local_stats(df, cat_col_set):
    """
    Compute per-column summary statistics for a single client.
    This is what each client would run locally and send to the server.
    """
    local = {}
    for col in df.columns:
        observed = df[col].dropna()
        n = len(observed)
        if col in cat_col_set:
            local[col] = {"categorical": True, "n": n,
                          "counts": observed.value_counts() if n > 0 else pd.Series(dtype=float)}
        else:
            local[col] = {"categorical": False, "n": n,
                          "sum": float(observed.sum()) if n > 0 else None,
                          "min": float(observed.min()) if n > 0 else None,
                          "max": float(observed.max()) if n > 0 else None}
    return local


def _aggregate_global_stats(local_stats_list):
    """
    Aggregate per-client summary statistics into global stats.
    Only sees summary statistics -- no raw data.
    """
    cols = list(local_stats_list[0].keys())
    stats = {}

    for col in cols:
        col_stats = [ls[col] for ls in local_stats_list]
        is_cat = col_stats[0]["categorical"]
        total_n = sum(s["n"] for s in col_stats)

        if total_n == 0:
            stats[col] = {"categorical": is_cat}
            continue

        if is_cat:
            counts = pd.concat(
                [s["counts"] for s in col_stats if s["n"] > 0]
            ).groupby(level=0).sum()
            stats[col] = {"categorical": True, "mode": float(counts.idxmax())}
        else:
            valid = [s for s in col_stats if s["n"] > 0]
            global_mean = sum(s["sum"] for s in valid) / total_n
            global_min = min(s["min"] for s in valid)
            global_max = max(s["max"] for s in valid)
            scale = (global_max - global_min) or 1.0
            stats[col] = {
                "categorical": False,
                "mean": global_mean,
                "min": global_min,
                "scale": scale,
            }

    return stats


def compute_global_stats(dfs_dict, cat_col_set):
    """
    Compute global stats in a federated manner:
      1. Each client computes local summary statistics (no raw data leaves the client).
      2. The server aggregates the local stats into global stats.
    """
    local_stats_list = [_compute_local_stats(df, cat_col_set) for df in dfs_dict.values()]
    stats = _aggregate_global_stats(local_stats_list)
    return stats, len(list(dfs_dict.values())[0].columns)


def compute_col_ranges(missing_files):
    """
    Compute per-column observed range (max - min) from a list of client CSV paths.
    Used by evaluate_imputation.py to normalise RMSE on the same scale as imputation.
    Returns {col: range} where range > 0 for numeric columns.
    """
    combined = pd.concat([pd.read_csv(f) for f in sorted(missing_files)], ignore_index=True)
    col_range = {}
    for col in combined.columns:
        if np.issubdtype(combined[col].dtype, np.number):
            observed = combined[col].dropna()
            r = observed.max() - observed.min()
            col_range[col] = r if r > 0 else 1.0
    return col_range


def normalize(dfs_dict, global_stats):
    """Normalise numeric columns to [0, 1] using global min/max."""
    out = {}
    for name, df in dfs_dict.items():
        df_norm = df.copy()
        for col in df_norm.columns:
            if not global_stats[col]["categorical"]:
                df_norm[col] = (df_norm[col] - global_stats[col]["min"]) / global_stats[col]["scale"]
        out[name] = df_norm
    return out


def unnormalize_and_decode(dfs_dict_imputed, global_stats, encodings):
    """Reverse normalisation and decode integer-coded categoricals."""
    out = {}
    for name, df_imp in dfs_dict_imputed.items():
        df_out = df_imp.copy()
        for col in df_imp.columns:
            if global_stats[col]["categorical"]:
                uniques = encodings[col]

                def map_back(val, u=uniques):
                    if pd.isna(val):
                        return np.nan
                    idx = int(round(val))
                    return u[idx] if 0 <= idx < len(u) else np.nan
                df_out[col] = df_out[col].apply(map_back)
            else:
                df_out[col] = df_out[col] * global_stats[col]["scale"] + global_stats[col]["min"]
        out[name] = df_out
    return out
