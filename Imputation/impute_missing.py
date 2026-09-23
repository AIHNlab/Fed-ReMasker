import os
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
# Must set CUDA_VISIBLE_DEVICES BEFORE any PyTorch imports, because DEVICE is
# resolved at module-load time inside each method file.
if '--no_gpu' in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import re
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict
import glob
import argparse
from tqdm import tqdm

from data_utils import read_dataset, compute_global_stats, normalize, unnormalize_and_decode
from log_utils import log, setup_log_file
from Mean.impute_mean import impute_mean_file
from Remasker.impute_remasker import impute_remasker_file
from Miwae.impute_miwae import impute_miwae_file
from CAFE.impute_cafe import impute_cafe_file
from FedHF.impute_fedhf import impute_fedhf_file


def create_args_parser():
    parser = argparse.ArgumentParser(description="Federated imputation benchmark.")
    parser.add_argument("--method", type=str, default="Mean",
                        choices=["Mean", "ReMasker", "MIWAE", "CAFE", "FedHF"])
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional suffix to distinguish HP variants, e.g. --tag v2 "
                             "saves under 'CAFE_v2' so previous results are preserved")
    parser.add_argument("--cat_features", type=int, nargs="+", default=[])
    parser.add_argument("--categorical_vocab_sizes", type=int, nargs="+", default=[])
    parser.add_argument("--data_dim", type=int, default=None)
    parser.add_argument("--log_file", type=str, default=None,
                        help="Save console output to this file (e.g. logs/cafe.log)")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Force CPU even if a GPU is available")
    parser.add_argument("--dataset_type", type=str, default="both",
                        choices=["synthetic", "real", "both"],
                        help="Which datasets to impute (default: both)")
    parser.add_argument("--para", action="store_true",
                        help="Enable multiprocessing for datasets")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Number of datasets to process in parallel (requires --para)")
    parser.add_argument("--hp", action="append", metavar="KEY=VALUE", default=[],
                        help="Override any method hyperparameter, e.g. "
                             "--hp global_rounds=60 --hp local_epochs=5")
    parser.add_argument("--hetero", action="store_true",
                        help="Impute the Hetero variants instead of the MFR/MR/N grid")
    parser.add_argument("--variants", type=str, nargs="+", default=None,
                        metavar="NAME",
                        help="With --hetero, restrict to these scenario names "
                             "(seed suffix stripped), e.g. --variants iid non_iid. "
                             "Default: every Hetero variant on disk.")
    parser.add_argument("--centralized", action="store_true",
                        help="Pool every client's rows into one dataset and impute it "
                             "as a single client -- the centralized reference. The "
                             "missing mask is unchanged (it is the union of the "
                             "clients'), so NRMSE stays comparable cell for cell. "
                             "Use with --tag, e.g. --tag central.")
    parser.add_argument("--mfr", type=float, default=None,
                        help="Restrict the MFR/MR/N grid to one MFR value, e.g. 0.2 "
                             "(reduces compute for ablation runs; ignored with --hetero)")
    parser.add_argument("--mr", type=float, default=None,
                        help="Restrict the MFR/MR/N grid to one MR value, e.g. 0.3")
    parser.add_argument("--n_clients", type=int, default=None,
                        help="Restrict the MFR/MR/N grid to one N_clients value, e.g. 5")
    args = parser.parse_args()
    # Inject --hp overrides into args so _get(args, key) picks them up automatically
    for kv in args.hp:
        k, v = kv.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        setattr(args, k, v)
    return args


def _output_path(input_file, input_root, method):
    rel = Path(input_file).relative_to(input_root)
    return (Path(input_root) / Path(*rel.parts[0:2]) / "Imputed" / method / Path(*rel.parts[3:-1]) / (Path(input_file).stem + "_imputed.csv"))


def _all_imputed(files, input_root, method):
    return all(_output_path(f, input_root, method).exists() for f in files)


def _process_single_dataset_group(group_dir, files, base_args, input_root, method, label):
    """Standalone worker function to process one dataset group."""

    dfs_dict, cat_features, vocab_sizes, global_categories = read_dataset(files)
    if dfs_dict is None:
        return group_dir, "Skipped (Read error)"

    # CRITICAL: Create a thread-local copy of args to prevent race conditions
    local_args = copy.deepcopy(base_args)
    local_args.cat_features = cat_features
    local_args.categorical_vocab_sizes = vocab_sizes

    global_stats, local_args.data_dim = compute_global_stats(dfs_dict, set(global_categories.keys()))

    output_paths = {}
    for f in files:
        out = _output_path(f, input_root, label)
        out.parent.mkdir(parents=True, exist_ok=True)
        output_paths[f] = out

    dfs_norm = normalize(dfs_dict, global_stats)

    # --- centralized reference -------------------------------------------
    # Concatenate every client's rows into one frame and hand the method a
    # single pseudo-client. Federation is then the only thing that differs:
    # the missing mask is the union of the clients' masks, i.e. bit-identical
    # to the federated run on this same group, so the two are comparable cell
    # for cell without re-deriving anything.
    #
    # The compute budget matches by construction rather than by tuning. The
    # federated run does `local_epochs` passes per client per round, and the
    # clients partition the data, so each round costs `local_epochs` passes
    # over the full dataset -- 60 x 5 = 300. With one pooled client the same
    # loop does 60 x 5 = 300 passes over that same full dataset.
    centralized = getattr(base_args, "centralized", False)
    if centralized:
        _names = list(dfs_norm.keys())
        _counts = [len(dfs_norm[n]) for n in _names]
        dfs_norm = {_names[0]: pd.concat([dfs_norm[n] for n in _names],
                                         ignore_index=True)}

    # Dispatch to the chosen imputation method
    if method == "Mean":
        dfs_imputed = impute_mean_file(dfs_norm, local_args, global_stats)
    elif method == "ReMasker":
        dfs_imputed = impute_remasker_file(dfs_norm, local_args, global_stats)
    elif method == "MIWAE":
        dfs_imputed = impute_miwae_file(dfs_norm, local_args, global_stats)
    elif method == "CAFE":
        dfs_imputed = impute_cafe_file(dfs_norm, local_args, global_stats)
    elif method == "FedHF":
        dfs_imputed = impute_fedhf_file(dfs_norm, local_args, global_stats)
    else:
        raise ValueError(f"Unknown method: {method}")

    if centralized:
        # Split the pooled result back along the original client boundaries so
        # the output files line up with Missing/ and Original/ exactly as a
        # federated run's would -- evaluation and plotting then need no
        # special-casing for this arm.
        _pooled = dfs_imputed[_names[0]]
        _start = 0
        dfs_imputed = {}
        for _n, _c in zip(_names, _counts):
            dfs_imputed[_n] = _pooled.iloc[_start:_start + _c].reset_index(drop=True)
            _start += _c

    dfs_out = unnormalize_and_decode(dfs_imputed, global_stats, global_categories)
    for name, df in dfs_out.items():
        df.to_csv(output_paths[name], index=False)

    return group_dir, "Success"


def impute_all(args, input_root="Datasets", method="Mean", label=None):
    """
    method : base method name used for routing (Mean, MIWAE, ReMasker, CAFE)
    label  : output folder name (defaults to method; set to e.g. 'CAFE_v2' when --tag is used)
    """
    label = label or method
    missing_files = []
    subpath = "Missing/Hetero/*/*.csv" if args.hetero else "Missing/*/*/*/*.csv"
    if args.dataset_type in ("synthetic", "both"):
        missing_files += glob.glob(f"{input_root}/Synthetic/*/{subpath}")
    if args.dataset_type in ("real", "both"):
        missing_files += glob.glob(f"{input_root}/Real/*/{subpath}")

    # Optional single-point restriction of the MFR/MR/N grid — for ablation runs
    # (FedProx/FedOpt/rounds sweeps etc.) that only need one representative
    # condition rather than the full 324-group grid. No-op when unset (default)
    # or under --hetero, whose paths don't have MFR_x/MR_y/Nz components.
    mfr, mr, n_clients = getattr(args, "mfr", None), getattr(args, "mr", None), getattr(args, "n_clients", None)
    if not args.hetero and (mfr is not None or mr is not None or n_clients is not None):
        def _matches(f):
            parts = Path(f).relative_to(input_root).parts  # [..., "Missing", "MFR_x", "MR_y", "Nz", file]
            if mfr is not None and not np.isclose(float(parts[3].split("_")[1]), mfr):
                return False
            if mr is not None and not np.isclose(float(parts[4].split("_")[1]), mr):
                return False
            if n_clients is not None and int(parts[5].lstrip("N")) != n_clients:
                return False
            return True
        missing_files = [f for f in missing_files if _matches(f)]

    # Optional restriction of --hetero to named scenarios. The seed suffix is
    # stripped before matching, so "--variants non_iid" selects all of its
    # --hetero_seeds draws. Useful for the centralized reference, which only
    # needs the scenarios that pooling does not erase: quantity skew and
    # per-client MFR spread leave nothing behind once the rows are merged,
    # whereas non_iid's mask is value-correlated and survives.
    variants = getattr(args, "variants", None)
    if args.hetero and variants:
        wanted = set(variants)
        def _variant_matches(f):
            parts = Path(f).relative_to(input_root).parts  # [..., "Missing", "Hetero", "<variant>_seedN", file]
            return re.sub(r"_seed\d+$", "", parts[4]) in wanted
        missing_files = [f for f in missing_files if _variant_matches(f)]
        if not missing_files:
            log(f"[WARN] No Hetero files matched --variants {' '.join(variants)}")

    groups = defaultdict(list)
    for f in missing_files:
        groups[str(Path(f).parent)].append(f)

    # --- Pre-filter groups before processing so TQDM progress is accurate ---
    filtered_groups = {}
    for group_dir, files in groups.items():
        if _all_imputed(files, input_root, label):
            continue
        filtered_groups[group_dir] = files

    # --- CONDITIONAL EXECUTION LOGIC ---
    if args.para:
        log(f"Found {len(filtered_groups)} datasets to process using {args.max_workers} workers.")
        # Execute in Parallel using PROCESSES, not threads
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_dataset_group,
                    group_dir, files, args, input_root, method, label
                ): group_dir
                for group_dir, files in filtered_groups.items()
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Datasets ({label})"):
                group_dir = futures[future]
                try:
                    processed_dir, status = future.result()
                    log(f"[{status}] -> {processed_dir}")
                except Exception as e:
                    log(f"[ERROR] -> Failed processing {group_dir}: {e}")
    else:
        log(f"Found {len(filtered_groups)} datasets to process sequentially.")
        # Execute in Sequential Order
        for group_dir, files in tqdm(filtered_groups.items(), desc=f"Datasets ({label})"):
            try:
                processed_dir, status = _process_single_dataset_group(
                    group_dir, files, args, input_root, method, label
                )
                log(f"[{status}] -> {processed_dir}")
            except Exception as e:
                log(f"[ERROR] -> Failed processing {group_dir}: {e}")


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    args = create_args_parser()
    if args.log_file:
        setup_log_file(args.log_file)
    label = f"{args.method}_{args.tag}" if args.tag else args.method
    impute_all(args, method=args.method, label=label)
