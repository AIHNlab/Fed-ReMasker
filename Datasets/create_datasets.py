from cdt.data import AcyclicGraphGenerator
import pandas as pd
import os
import argparse
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def split_clients(df, n_clients, seed=42):
    """Unevenly split df into n_clients, each with at least K/(2N) samples."""
    np.random.seed(seed)
    K = len(df)
    min_per_client = K // (2 * n_clients)

    indices = np.random.permutation(K)
    splits = []
    start = 0
    for i in range(n_clients):
        splits.append(indices[start:start + min_per_client].tolist())
        start += min_per_client

    remaining = indices[start:]
    for idx in remaining:
        target = np.random.randint(0, n_clients)
        splits[target].append(idx)

    return [df.iloc[client_idx].reset_index(drop=True) for client_idx in splits]


def assign_missing_features(n_clients, col_names, col_ratio, seed=42):
    """
    Assign columns to null per client ensuring no column is nulled by ALL clients.
    For each column one random client is designated as its protector (must keep it).
    """
    if col_ratio == 0:
        return [np.array([]) for _ in range(n_clients)]

    rng = np.random.default_rng(seed)
    n_cols = len(col_names)
    n_drop = math.ceil(n_cols * col_ratio)

    protectors = rng.integers(0, n_clients, size=n_cols)

    assignments = []
    for i in range(n_clients):
        available = np.where(protectors != i)[0]
        if len(available) >= n_drop:
            chosen = rng.choice(available, n_drop, replace=False)
        else:
            chosen = available
        assignments.append(col_names[chosen])

    return assignments


def introduce_missing_features(df, cols_to_null):
    """Set the given columns entirely to NaN."""
    if len(cols_to_null) == 0:
        return df.copy()
    df_out = df.copy()
    df_out[cols_to_null] = np.nan
    return df_out


def introduce_missingness(df, missing_ratio, seed=42):
    """Introduce random missing values at the given ratio."""
    if missing_ratio == 0:
        return df.copy()

    np.random.seed(seed)
    df_missing = df.copy()
    n_total = df_missing.size
    n_missing = int(n_total * missing_ratio)
    missing_indices = np.random.choice(n_total, n_missing, replace=False)
    flat_values = df_missing.to_numpy().flatten().copy().astype(np.float64)
    flat_values[missing_indices] = np.nan
    return pd.DataFrame(flat_values.reshape(df_missing.shape), columns=df_missing.columns)


def create_missing_splits(df, dataset_dir, missing_ratios, n_clients, missing_feature_ratios, seed=42):
    """Create all MFR × MR × N combinations and save client CSVs."""
    for mfr in missing_feature_ratios:
        for mr in missing_ratios:
            for n in n_clients:
                client_dir_missing = os.path.join(dataset_dir, "Missing", f"MFR_{mfr}", f"MR_{mr}", f"N{n}")
                client_dir_original = os.path.join(dataset_dir, "Original", f"MFR_{mfr}", f"MR_{mr}", f"N{n}")
                os.makedirs(client_dir_missing, exist_ok=True)
                os.makedirs(client_dir_original, exist_ok=True)

                splits = split_clients(df, n, seed=seed)
                null_assignments = assign_missing_features(n, np.array(df.columns), mfr, seed=seed)

                for i, client_df in enumerate(splits):
                    client_df_cols = introduce_missing_features(client_df, null_assignments[i])
                    client_df_missing = introduce_missingness(client_df_cols, mr, seed=seed + i)

                    client_df.to_csv(os.path.join(client_dir_original, f"client_{i+1}.csv"), index=False)
                    client_df_missing.to_csv(os.path.join(client_dir_missing, f"client_{i+1}.csv"), index=False)


def create_real_dataset(args, base_dir="Datasets/Real"):
    """
    Create MFR/MR/N splits for preprocessed real datasets.
    Expects data.csv to already exist (run preprocess_real.py first).
    """
    os.makedirs(base_dir, exist_ok=True)

    for dataset_dir in sorted(os.listdir(base_dir)):
        full_dir = os.path.join(base_dir, dataset_dir)
        data_path = os.path.join(full_dir, "Original", "data.csv")
        if not os.path.exists(data_path):
            print(f"Skipping {dataset_dir} — no data.csv found (run preprocess_real.py first)")
            continue

        print(f"Creating splits for real dataset: {dataset_dir}")
        df = pd.read_csv(data_path)
        print(f"  Loaded {df.shape}")

        create_missing_splits(
            df, full_dir,
            args.missing_ratios, args.n_clients, args.missing_feature_ratios,
            seed=args.seed,
        )


def create_synthetic_dataset(args, base_dir="Datasets/Synthetic", mechanism="linear"):
    os.makedirs(base_dir, exist_ok=True)

    for d in args.n_variables:
        name = f"{mechanism}_d{d}"
        dataset_dir = os.path.join(base_dir, name)
        original_dir = os.path.join(dataset_dir, "Original")
        data_path = os.path.join(original_dir, "data.csv")

        os.makedirs(original_dir, exist_ok=True)

        if os.path.exists(data_path):
            print(f"{name} already exists. Loading from disk.")
            df = pd.read_csv(data_path)
        else:
            generator = AcyclicGraphGenerator(
                nodes=d,
                npoints=args.n_samples,
                causal_mechanism=mechanism,
                noise="gaussian",
                noise_coeff=args.noise,
            )
            df, _ = generator.generate()
            df.to_csv(data_path, index=False)
            print(f"Generated {name}: {df.shape}")

        create_missing_splits(
            df, dataset_dir,
            args.missing_ratios, args.n_clients, args.missing_feature_ratios,
            seed=args.seed,
        )


def compute_gini(sizes):
    """
    Gini coefficient of client sample sizes.

    Standard formula, for x sorted ascending:
        G = 2 * sum(i * x_i) / (n * sum(x))  -  (n + 1) / n

    Range: 0.0 (all clients equal) to (n-1)/n (one client holds everything).
    """
    x = np.sort(np.asarray(sizes, dtype=float))
    n = x.size
    total = x.sum()
    if n <= 1 or total <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * total) - (n + 1) / n)


def _geometric_sizes(N, n_clients, ratio, min_size):
    """
    Deterministic client sizes with geometrically decaying proportions.

    Client i gets weight ratio**i, so ratio=1.0 is uniform and smaller
    ratios concentrate mass in the first clients. Uses largest-remainder
    rounding so the sizes sum to exactly N, then enforces min_size.
    """
    weights = ratio ** np.arange(n_clients, dtype=float)
    exact = weights / weights.sum() * N

    sizes = np.floor(exact).astype(np.int64)
    remainder = N - int(sizes.sum())
    if remainder > 0:
        # Hand the leftover units to the largest fractional parts.
        order = np.argsort(-(exact - sizes))
        sizes[order[:remainder]] += 1

    # Enforce the floor, then claw the deficit back from the biggest clients.
    sizes = np.maximum(sizes, min_size)
    overflow = int(sizes.sum()) - N
    while overflow > 0:
        donor = int(np.argmax(sizes))
        if sizes[donor] <= min_size:
            raise ValueError(
                f"Cannot fit {n_clients} clients with min_size={min_size} into N={N}."
            )
        take = min(overflow, sizes[donor] - min_size)
        sizes[donor] -= take
        overflow -= take

    return sizes


def split_clients_imbalanced(df, n_clients, gini_target=0.2, seed=42,
                             min_size=1, tol=1e-4, max_iter=200):
    """
    Split df into n_clients whose sizes have the requested Gini.

    Sizes are generated deterministically from a geometric profile, and the
    decay ratio is found by bisection: Gini decreases monotonically as the
    ratio rises toward 1.0, so bisection is well posed. Only the assignment
    of rows to clients is random.

    Returns:
        (client_data, actual_gini, sizes) with sizes sorted descending.
    """
    N = len(df)
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1")
    if n_clients * min_size > N:
        raise ValueError(
            f"n_clients={n_clients} x min_size={min_size} exceeds N={N}."
        )

    gini_max = (n_clients - 1) / n_clients
    if gini_target >= gini_max:
        raise ValueError(
            f"gini_target={gini_target} is unreachable with {n_clients} clients "
            f"(theoretical max is {gini_max:.3f})."
        )

    if gini_target <= tol:
        sizes = _geometric_sizes(N, n_clients, 1.0, min_size)
    else:
        # ratio -> 0 gives maximum imbalance, ratio = 1 gives uniform.
        lo, hi = 1e-6, 1.0
        sizes = _geometric_sizes(N, n_clients, hi, min_size)
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            sizes = _geometric_sizes(N, n_clients, mid, min_size)
            g = compute_gini(sizes)
            if abs(g - gini_target) <= tol:
                break
            if g > gini_target:
                lo = mid   # too unequal, raise the ratio
            else:
                hi = mid   # too equal, lower the ratio

    sizes = np.sort(sizes)[::-1]

    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)

    client_data = []
    start = 0
    for size in sizes:
        client_idx = indices[start:start + size]
        client_data.append(df.iloc[client_idx].reset_index(drop=True))
        start += size

    return client_data, compute_gini(sizes), [int(s) for s in sizes]


def assign_missing_features_heterogeneous(n_clients, col_names, mfr_per_client, seed=42):
    """Assign columns with different MFR per client."""
    rng = np.random.default_rng(seed)
    n_cols = len(col_names)

    protectors = rng.integers(0, n_clients, size=n_cols)

    assignments = []
    for client_id in range(n_clients):
        mfr = mfr_per_client[client_id]

        if mfr == 0:
            assignments.append(np.array([]))
            continue

        # Compute number of features to drop based on TOTAL features
        n_to_drop = math.ceil(n_cols * mfr)

        # Can only drop features this client doesn't protect
        available = np.where(protectors != client_id)[0]

        if len(available) >= n_to_drop:
            chosen_indices = rng.choice(available, n_to_drop, replace=False)
        else:
            # If not enough available features, drop all available (protector prevents dropping protected)
            chosen_indices = available

        assignments.append(col_names[chosen_indices])

    return assignments


def split_clients_clustering_noniid(df, n_clients, n_clusters=None, seed=42):
    """Non-IID split: cluster on standardized features, assign clusters to clients
    greedily (largest cluster to smallest client) so sizes stay comparable."""
    if n_clusters is None:
        n_clusters = 2 * n_clients

    X = StandardScaler().fit_transform(df.to_numpy(dtype=float))
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(X)

    # Biggest clusters first, each handed to whichever client is currently smallest.
    order = np.argsort(-np.bincount(labels, minlength=n_clusters))
    client_clusters = [[] for _ in range(n_clients)]
    client_sizes = np.zeros(n_clients, dtype=int)
    for c in order:
        k = int(np.argmin(client_sizes))
        client_clusters[k].append(c)
        client_sizes[k] += int(np.sum(labels == c))

    client_data = [df[np.isin(labels, cs)].reset_index(drop=True) for cs in client_clusters]
    print(f"  {n_clusters} clusters on standardized features, sizes={[len(c) for c in client_data]}")
    return client_data, n_clusters


def create_heterogeneous_scenario(dataset_dir, mr, mfr, n_clients, variant_type,
                                  variant_config, seed=42, seed_tag=None):
    """
    Create ONE heterogeneous scenario for a specific dataset.
    Saves to: {dataset_dir}/Hetero/{variant_type}/client_{i}.csv

    Args:
        dataset_dir: Path to dataset (e.g., "Datasets/Synthetic/linear_d50")
        n_clients: Number of clients (e.g., 5)
        variant_type: "iid", "clients_imbalanced", "features_imbalanced", or "non_iid"
        variant_config: dict with variant parameters
            - For "clients_imbalanced": {"gini": 0.2}
            - For "features_imbalanced": {"mfr_per_client": [0.1, 0.15, 0.2, 0.3, 0.4]}
            - For "non_iid": {} (empty dict)
            - For "iid": {} (empty dict) — the homogeneous reference
        seed: Random seed
        seed_tag: optional label appended to the variant folder name (e.g.
            "seed43"), so repeated draws of the same scenario at different
            seeds land in separate output dirs instead of overwriting each
            other. Leave None for a single-draw run (original behaviour).

    Returns:
        None (saves to disk)
    """

    # Load baseline data
    data_path = os.path.join(dataset_dir, "Original", "data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    col_names = np.array(df.columns)

    # Create splits and feature assignments based on variant type
    if variant_type == "clients_imbalanced":
        gini = variant_config["gini"]
        variant_name = f"{variant_type}_gini{gini}"
        print(f"Creating {variant_type} variant: Gini={gini}, n_clients={n_clients}")
        splits = split_clients_imbalanced(df, n_clients, gini_target=gini, seed=seed)
        splits, actual_gini, sizes = splits
        null_assignments = assign_missing_features(n_clients, col_names, mfr, seed=seed)
        print(f"  Target Gini={gini:.2f}, actual={actual_gini:.2f}, sizes={sizes}")

    elif variant_type == "features_imbalanced":
        variant_name = variant_type
        mfr_per_client = variant_config["mfr_per_client"]
        print(f"Creating {variant_type} variant: MFR_per_client={mfr_per_client}, n_clients={n_clients}")
        splits = split_clients(df, n_clients, seed=seed)
        null_assignments = assign_missing_features_heterogeneous(n_clients, col_names, mfr_per_client, seed=seed)
        print("Per-client MFR:")
        for i, mfr_val in enumerate(mfr_per_client):
            print(f"    Client {i}: MFR={mfr_val:.2f}, {len(null_assignments[i])}/{len(col_names)} features null")

    elif variant_type == "iid":
        # The homogeneous reference point, generated as a Hetero variant so it
        # goes through the same seeded-draw machinery as the skewed scenarios:
        # random client split, uniform MFR, same MR/MFR/N as everything else.
        # Without this the "Homogeneous (IID)" bar would be the single
        # MFR/MR/N grid cell — one partition draw with no seed spread, and so
        # no error bar to compare the scenarios' error bars against.
        variant_name = variant_type
        print(f"Creating {variant_type} variant: uniform MFR={mfr}, n_clients={n_clients}")
        splits = split_clients(df, n_clients, seed=seed)
        null_assignments = assign_missing_features(n_clients, col_names, mfr, seed=seed)

    elif variant_type == "non_iid":
        variant_name = variant_type
        print(f"Creating {variant_type} variant: clustering-based, n_clients={n_clients}")
        splits, n_clusters = split_clients_clustering_noniid(df, n_clients, seed=seed)
        null_assignments = assign_missing_features(n_clients, col_names, mfr, seed=seed)
    else:
        raise ValueError(f"Unknown variant_type: {variant_type}. Use 'iid', 'clients_imbalanced', 'features_imbalanced', or 'non_iid'.")

    if seed_tag:
        variant_name = f"{variant_name}_{seed_tag}"

    client_dir_missing = os.path.join(dataset_dir, "Missing/Hetero", variant_name)
    client_dir_original = os.path.join(dataset_dir, "Original/Hetero", variant_name)

    os.makedirs(client_dir_missing, exist_ok=True)
    os.makedirs(client_dir_original, exist_ok=True)

    # Save client files
    for i, client_df in enumerate(splits):
        client_df_cols = introduce_missing_features(client_df, null_assignments[i])
        client_df_missing = introduce_missingness(client_df_cols, mr, seed=seed + i)

        client_df.to_csv(os.path.join(client_dir_original, f"client_{i+1}.csv"), index=False)
        client_df_missing.to_csv(os.path.join(client_dir_missing, f"client_{i+1}.csv"), index=False)

    print(f"✓ Saved to {client_dir_missing}")


def create_all_hetero_variants(datasets, mr, mfr, n_clients, gini_levels=None,
                               mfr_per_client=None, seeds=(42,)):
    """
    Create all heterogeneous variants for given datasets, repeated once per
    seed in `seeds`. With more than one seed, each variant gets a distinct
    "..._seedN" output dir (see create_heterogeneous_scenario) so downstream
    evaluation can average across independent client-partition draws instead
    of reporting a single random split's RMSE as if it were the true effect.
    """
    for dataset in datasets:
        dataset_dir = f"Datasets/{dataset}"

        for seed in seeds:
            seed_tag = f"seed{seed}" if len(seeds) > 1 else None

            create_heterogeneous_scenario(
                dataset_dir, mr, mfr, n_clients,
                variant_type="iid",
                variant_config={},
                seed=seed, seed_tag=seed_tag,
            )

            for gini in (gini_levels or []):
                create_heterogeneous_scenario(
                    dataset_dir, mr, mfr, n_clients,
                    variant_type="clients_imbalanced",
                    variant_config={"gini": gini},
                    seed=seed, seed_tag=seed_tag,
                )

            create_heterogeneous_scenario(
                dataset_dir, mr, mfr, n_clients,
                variant_type="features_imbalanced",
                variant_config={"mfr_per_client": mfr_per_client},
                seed=seed, seed_tag=seed_tag,
            )

            create_heterogeneous_scenario(
                dataset_dir, mr, mfr, n_clients,
                variant_type="non_iid",
                variant_config={},
                seed=seed, seed_tag=seed_tag,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic federated imputation datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_variables", type=int, nargs="+", default=[20, 50, 100],
                        help="Feature counts for synthetic datasets")
    parser.add_argument("--n_clients", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--missing_ratios", type=float, nargs="+", default=[0.1, 0.3, 0.5],
                        help="Missing value ratios (MR)")
    parser.add_argument("--missing_feature_ratios", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3],
                        help="Missing feature ratios per client (MFR)")
    parser.add_argument("--noise", type=float, default=0.4)
    parser.add_argument("--real", action="store_true",
                        help="Also create splits for real datasets in Datasets/Real/")
    parser.add_argument("--hetero", action="store_true",
                        help="Also create splits for heterogeneous datasets")
    parser.add_argument("--hetero_seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Random seeds for Hetero client-partition draws — "
                             "more than one seed lets evaluation report mean/std "
                             "instead of a single noisy split")
    parser.add_argument("--only_hetero", action="store_true",
                        help="Skip the synthetic/real MFR-MR-N grid entirely and only "
                             "(re)create Hetero variants. Requires --hetero and expects "
                             "Real/<dataset>/Original/data.csv to already exist (run "
                             "preprocess_real.py, or a prior --real run, first)")
    args = parser.parse_args()

    if not args.only_hetero:
        create_synthetic_dataset(args, mechanism="linear")
        create_synthetic_dataset(args, mechanism="nn")
        if args.real:
            create_real_dataset(args)
    if args.hetero:
        create_all_hetero_variants(
            ["Real/codon", "Real/nhanes", "Real/physionet"],
            mr=0.3,
            mfr=0.2,
            n_clients=5,
            gini_levels=[0.2, 0.35],
            mfr_per_client=[0.1, 0.15, 0.2, 0.25, 0.3],
            seeds=args.hetero_seeds,
        )
