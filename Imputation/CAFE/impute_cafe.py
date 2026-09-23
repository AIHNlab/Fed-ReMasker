"""
Fed-CAFE imputation -- federated ICE imputation with Cafe Averaging.

Implements Algorithm 1 + Algorithm 2 from:
    "Cafe: Improved Federated Data Imputation by Leveraging Missing Data
    Heterogeneity", Min et al., IEEE TKDE 2025.

Default hyperparameters match the paper's reported best settings:
    alpha = 0.95, beta = 4, gamma = 0.05, global_rounds = 20
"""

import numpy as np
from tqdm import trange, tqdm

from CAFE.ice_imputer import ICEImputer
from CAFE.aggregation_cafe import cafe_avg, fedmechclw
from data_utils import get_hp

# Paper S.IV: "alpha = 0.95, gamma = 0.05, beta = 4 is applicable across all datasets"
# Paper S.IV: "20 rounds"
# Min et al. (2025), Section IV, except where noted.
_DEFAULTS = dict(
    global_rounds=60,       # paper: 20 -- raised to match the other methods, so
                            # the comparison is at an equal round budget

    aggregation="cafe",     # "cafe" = Algorithm 2  |  "fedmechclw" = cluster variant
    alpha=0.95,             # alpha: complementarity vs sample-size balance
    beta=4,                 # beta: normalisation exponent
    gamma=0.05,             # gamma: self-model weight
    tol=1e-3,               # convergence tolerance (L2 norm)
)


def _get(args, key):
    return get_hp(args, key, _DEFAULTS)


def impute_cafe_file(dfs_dict_norm, args, global_stats=None):
    """
    Federated CAFE (ICE-style) imputation matching the paper's Algorithm 1.

    Parameters
    ----------
    dfs_dict_norm : dict {filename: DataFrame}
        One normalised DataFrame per client (NaN marks missing values).
        Numeric columns should be in [0, 1]; categorical columns contain
        integer codes (0 … n_classes-1).
    args : argparse.Namespace or any object with attributes
        Hyperparameters — see _DEFAULTS for supported keys.
    global_stats : dict {col: {categorical, mean, min, scale, mode}}
        Output of compute_global_stats() in impute_missing.py.

    Returns
    -------
    dict {filename: DataFrame}  — same structure, NaN positions filled.
    """
    # --- hyperparameters ---
    from data_utils import log_hps
    log_hps(args, _DEFAULTS, "Fed-CAFE")
    global_rounds = _get(args, "global_rounds")
    aggregation = _get(args, "aggregation")
    cafe_params = {
        "alpha": _get(args, "alpha"),
        "beta": _get(args, "beta"),
        "gamma": _get(args, "gamma"),
    }
    tol = _get(args, "tol")

    filenames = list(dfs_dict_norm.keys())
    columns = list(list(dfs_dict_norm.values())[0].columns)
    n_features = len(columns)

    # --- identify categorical column indices ---
    cat_col_indices = set()
    if global_stats is not None:
        for j, col in enumerate(columns):
            if global_stats[col]["categorical"]:
                cat_col_indices.add(j)

    # --- convert DataFrames to numpy (NaN preserved) ---
    client_X: dict = {}
    client_mask: dict = {}
    for fname, df in dfs_dict_norm.items():
        X = df.values.astype(np.float64)
        client_X[fname] = X
        client_mask[fname] = np.isnan(X)

    # --- global clip ranges ---
    all_X = np.concatenate(list(client_X.values()), axis=0)
    features_min = np.nanmin(all_X, axis=0)
    features_max = np.nanmax(all_X, axis=0)
    for j in range(n_features):
        if j not in cat_col_indices:
            features_min[j], features_max[j] = 0.0, 1.0   # already normalised

    # --- one ICEImputer per client (Algorithm 1 setup) ---
    imputers: dict = {}
    for fname in filenames:
        imp = ICEImputer(client_mask[fname], cat_col_indices=cat_col_indices, clip=True)
        imp.features_min = features_min
        imp.features_max = features_max
        imputers[fname] = imp

    # --- initial fill: federated mean/mode (Algorithm 1 line 5) ---
    for fname in filenames:
        client_X[fname] = imputers[fname].initial_impute(
            client_X[fname], global_stats, columns
        )

    # --- visit order: ascending fraction of missing values ---
    combined_mask = np.concatenate([client_mask[f] for f in filenames], axis=0)
    frac_missing = combined_mask.mean(axis=0)
    visit_order = [j for j in np.argsort(frac_missing) if frac_missing[j] > 0]

    if not visit_order:
        return _build_output(dfs_dict_norm, client_X, client_mask, columns)

    # --- iterative imputation (Algorithm 1 lines 6–20) ---
    prev_X = {f: client_X[f].copy() for f in filenames}

    for _ in trange(global_rounds, desc="Fed-CAFE round", colour="cyan"):
        tqdm.write(f"  [CAFE] Round {_+1}/{global_rounds}")
        for feat_idx in visit_order:           # Algorithm 1 line 7: for each f ∈ F

            # Algorithm 1 lines 8–12: fit local imputation + mechanism models
            fit_results: dict = {}
            for fname in filenames:
                w, info, ms = imputers[fname].fit_feature(client_X[fname], feat_idx)
                fit_results[fname] = (w, info, ms)

            # clients that can actually fit (feature not entirely missing)
            valid = [f for f in filenames if fit_results[f][0] is not None]
            if not valid:
                continue

            # Algorithm 1 lines 13–14: Cafe averaging on the server
            global_w = _aggregate(valid, fit_results, aggregation, cafe_params)

            # MFR clients (entirely-missing feature) get the plain FedAvg of
            # local weights — the globally averaged model before personalisation,
            # weighted by each valid client's number of observable samples.
            if len(valid) < len(filenames):
                sizes = np.array(
                    [fit_results[f][1]["sample_size"] for f in valid], dtype=float
                )
                sizes /= max(sizes.sum(), 1e-8)
                fedavg_w = np.average(
                    [fit_results[f][0] for f in valid], axis=0, weights=sizes
                )
                for fname in filenames:
                    if fname not in global_w:
                        global_w[fname] = fedavg_w

            # Algorithm 1 lines 15–18: update imputations
            for fname in filenames:
                client_X[fname] = imputers[fname].transform_feature(
                    client_X[fname], feat_idx, global_w[fname]
                )

        # convergence check
        max_diff = max(
            np.linalg.norm(client_X[f] - prev_X[f]) for f in filenames
        )
        prev_X = {f: client_X[f].copy() for f in filenames}
        if max_diff < tol:
            break

    return _build_output(dfs_dict_norm, client_X, client_mask, columns)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate(valid, fit_results, aggregation, cafe_params):
    """Run Cafe averaging and return {fname: personalised_weights}."""
    if len(valid) == 1:
        return {valid[0]: fit_results[valid[0]][0]}

    w_dict = {f: fit_results[f][0] for f in valid}
    info_dict = {f: fit_results[f][1] for f in valid}
    ms_dict = {f: fit_results[f][2] for f in valid}

    try:
        if aggregation == "fedmechclw":
            agg_list = fedmechclw(w_dict, info_dict, ms_dict, cafe_params)
        else:                           # default: paper's Algorithm 2
            agg_list = cafe_avg(w_dict, info_dict, ms_dict, cafe_params)
    except Exception:
        # fallback: uniform average
        avg = np.mean([w_dict[f] for f in valid], axis=0)
        agg_list = [avg] * len(valid)

    return {f: agg_list[i] for i, f in enumerate(valid)}


def _build_output(dfs_dict_norm, client_X, client_mask, columns):
    """Write imputed values back into copies of the original DataFrames."""
    dfs_dict_imputed = {}
    for fname, df in dfs_dict_norm.items():
        df_out = df.copy()
        mask = client_mask[fname]
        X_imp = client_X[fname]
        for j, col in enumerate(columns):
            missing_rows = np.where(mask[:, j])[0]
            if len(missing_rows) > 0:
                df_out.iloc[missing_rows, j] = X_imp[missing_rows, j]
        dfs_dict_imputed[fname] = df_out
    return dfs_dict_imputed
