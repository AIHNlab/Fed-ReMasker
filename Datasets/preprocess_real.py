"""
Preprocess real datasets for the imputation benchmark.

Normalisation is intentionally omitted here: data_utils.normalize() handles
per-column min-max scaling from observed values right before each imputation call.

Run from Fed-ReMasker-benchmark root:
    python Datasets/preprocess_real.py
    python Datasets/preprocess_real.py --datasets physionet
    python Datasets/preprocess_real.py --max-samples 10000
"""

import argparse
import urllib.request
import zipfile
import tarfile
import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
import requests


BASE = Path("Datasets/Real")


# ---------------------------------------------------------------------------
# Dataset-specific preprocessing
# ---------------------------------------------------------------------------

def preprocess_codon(args):
    """
    Codon usage frequencies (UCI).
    Downloads automatically from the UCI ML Repository if not already present.
    Drops metadata columns and target (Kingdom).
    Filters out 'plm' kingdom entries.
    Keeps only DNAtype=0 (genomic) to avoid duplicate species entries.
    """
    raw_dir = BASE / "codon" / "Raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "data_raw.csv"

    zip_path = raw_dir / "codon_usage.zip"
    if not zip_path.exists():
        url = "https://archive.ics.uci.edu/static/public/577/codon+usage.zip"
        print(f"\n[Codon] Downloading dataset from {url} ...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print("  Download complete.")
        except Exception as e:
            raise RuntimeError(f"Could not download Codon dataset: {e}")
    if not path.exists():
        print(f"[Codon] Extracting {zip_path.name} ...")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                if not csv_names:
                    raise RuntimeError("No CSV found inside codon+usage.zip")
                z.extract(csv_names[0], raw_dir)
                (raw_dir / csv_names[0]).rename(path)
            print(f"  Extracted to {path}")
        except Exception as e:
            raise RuntimeError(f"Could not extract Codon dataset: {e}")

    print(f"\n[Codon] Loading {path} ...")
    df = pd.read_csv(path, sep=",", low_memory=False)
    print(f"  Raw shape: {df.shape}")

    df = df.dropna()
    df = df.replace("non-B hepatitis virus", 0).replace("12;I", 0).replace("-", 0)

    target_col = "Kingdom"
    df = df[df[target_col] != "plm"]

    # Keep only genomic DNA (DNAtype=0) — one row per species, homogeneous
    if "DNAtype" in df.columns:
        df = df[df["DNAtype"] == 0]

    df.drop(columns=["SpeciesID", "Ncodons", "SpeciesName", "DNAtype", target_col],
            inplace=True, errors="ignore")

    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    df.columns = df.columns.astype(str)

    out_path = BASE / "codon" / "Original" / "data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved {df.shape} -> {out_path}")
    return df


def preprocess_physionet(args):
    """
    PhysioNet Challenge 2012 (Set-A + Set-B + Set-C, ~12,000 ICU patients).
    Downloads automatically — no authentication required.
    Pivots from long format (Time, Parameter, Value) to wide format (one row per patient).
    Drops identifier/categorical columns (RecordID, MechVent, Gender, ICUType).
    Replaces -1 (PhysioNet sentinel for missing) with NaN.
    Drops columns with >20% missing, then keeps only fully observed rows.
    """
    raw_dir = BASE / "physionet" / "Raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    sets = {
        "set-a": ("https://physionet.org/files/challenge-2012/1.0.0/set-a.zip", "zip"),
        "set-b": ("https://physionet.org/files/challenge-2012/1.0.0/set-b.zip", "zip"),
        "set-c": ("https://physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz", "tar.gz"),
    }
    for set_name, (url, fmt) in sets.items():
        archive_path = raw_dir / f"{set_name}.{fmt}"
        extract_dir = raw_dir / set_name
        if not archive_path.exists():
            print(f"\n[PhysioNet] Downloading {set_name} from {url} ...")
            try:
                urllib.request.urlretrieve(url, archive_path)
                print("  Download complete.")
            except Exception as e:
                print(f"  WARNING: Could not download {set_name} ({e}) — skipping.")
                continue
        if not extract_dir.exists():
            print(f"[PhysioNet] Extracting {set_name} ...")
            if fmt == "zip":
                with zipfile.ZipFile(archive_path, "r") as z:
                    z.extractall(raw_dir)
            else:
                with tarfile.open(archive_path, "r:gz") as t:
                    t.extractall(raw_dir)

    files = []
    for set_name in ["set-a", "set-b", "set-c"]:
        set_dir = raw_dir / set_name
        if set_dir.exists():
            files += sorted(set_dir.glob("*.txt"))
    print(f"\n[PhysioNet] Loading {len(files)} patient records ...")

    samples = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df = df.sort_values("Time")
            df = df.drop_duplicates("Parameter", keep="last")
            pivot = df.pivot_table(index=None, columns="Parameter",
                                   values="Value", aggfunc="last")
            samples.append(pivot)
        except Exception:
            continue

    data = pd.concat(samples, ignore_index=True)
    print(f"  Raw shape: {data.shape}")

    # -1 is the PhysioNet sentinel for "not recorded"
    data.replace(-1, np.nan, inplace=True)

    # Drop identifier and categorical columns
    drop_always = ["RecordID", "MechVent", "Gender", "ICUType"]
    data.drop(columns=drop_always, inplace=True, errors="ignore")

    # Drop columns with >20% missing
    col_ms = data.isnull().mean()
    drop_cols = col_ms[col_ms > 0.20].index.tolist()
    print(f"  Dropping {len(drop_cols)} high-missingness columns (>20%): {drop_cols}")
    data.drop(columns=drop_cols, inplace=True)

    # Keep only fully observed rows
    data = data.dropna().reset_index(drop=True)
    print(f"  After keeping only complete rows: {data.shape}")

    data.columns = data.columns.astype(str)

    out_path = BASE / "physionet" / "Original" / "data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False)
    print(f"  Saved {data.shape} -> {out_path}")
    return data


def preprocess_nhanes(args):
    """
    NHANES (2003–2018), 8 survey cycles × 10 modules.
    Downloads XPT files from CDC, merges by participant ID (SEQN).
    Drops survey weights, design variables, identifiers, and categorical-coded numerics.
    Drops columns with >30% missing, then keeps only fully observed rows.
    """

    out_dir = BASE / "nhanes" / "Original"
    xpt_dir = BASE / "nhanes" / "Raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    xpt_dir.mkdir(parents=True, exist_ok=True)

    max_missing = 0.3
    min_unique_values = 10

    cycles = {
        "2003-2004": {"year": "2003", "suffix": "C"},
        "2005-2006": {"year": "2005", "suffix": "D"},
        "2007-2008": {"year": "2007", "suffix": "E"},
        "2009-2010": {"year": "2009", "suffix": "F"},
        "2011-2012": {"year": "2011", "suffix": "G"},
        "2013-2014": {"year": "2013", "suffix": "H"},
        "2015-2016": {"year": "2015", "suffix": "I"},
        "2017-2018": {"year": "2017", "suffix": "J"},
    }

    modules = ["DEMO", "BMX", "BPX", "TCHOL", "HDL", "TRIGLY", "GLU", "GHB", "CBC", "BIOPRO"]

    print(f"\n[NHANES] Loading {len(cycles)} cycles × {len(modules)} modules ...")

    def download_xpt(year, fname, path):
        url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{fname}.xpt"

        if path.exists():
            return True

        try:
            r = requests.get(url, timeout=60)
        except Exception:
            return False

        if r.status_code != 200:
            return False

        # Avoid saving HTML error pages.
        start = r.content[:300].upper()
        if b"HEADER RECORD" not in start and b"LIBRARY" not in start:
            return False

        path.write_bytes(r.content)
        return True

    def clean_module(df):
        df = df.loc[:, ~df.columns.duplicated()].copy()

        drop_cols = []

        for col in df.columns:
            upper = col.upper()

            if col in ["SEQN", "cycle"]:
                continue

            # survey weights
            if upper.startswith("WT"):
                drop_cols.append(col)
                continue

            # survey design variables
            if upper in ["SDMVPSU", "SDMVSTRA"]:
                drop_cols.append(col)
                continue

            # admin/status/comment/code variables
            if any(p in upper for p in ["STAT", "STATUS", "COMMENT", "COMM", "CODE"]):
                drop_cols.append(col)
                continue

        return df.drop(columns=drop_cols, errors="ignore")

    def safe_merge(left, right):
        keys = ["SEQN", "cycle"]

        overlap = set(left.columns).intersection(right.columns) - set(keys)
        if overlap:
            right = right.drop(columns=list(overlap), errors="ignore")

        return pd.merge(left, right, on=keys, how="outer")

    all_cycles = []

    for cycle_name, info in cycles.items():
        print(f"  Loading {cycle_name}")

        year = info["year"]
        suffix = info["suffix"]
        dfs = []

        for module in modules:
            fname = f"{module}_{suffix}"
            path = xpt_dir / f"{fname}.xpt"

            if not download_xpt(year, fname, path):
                continue

            try:
                d = pd.read_sas(path, format="xport")
                d["cycle"] = cycle_name

                if "SEQN" not in d.columns:
                    continue

                d = clean_module(d)
                dfs.append(d)

            except Exception:
                continue

        if len(dfs) == 0:
            continue

        merged = reduce(safe_merge, dfs)
        all_cycles.append(merged)

    if len(all_cycles) == 0:
        raise RuntimeError("No NHANES files loaded.")

    df = pd.concat(all_cycles, ignore_index=True)
    print(f"  Raw shape: {df.shape}")

    # Drop identifiers, survey metadata, and known categorical-range columns
    drop_always = ["SEQN", "cycle", "SDDSRVYR",
                   "PEASCTM1",   # blood pressure time in seconds
                   "INDHHIN2",   # household income (range/category)
                   "INDFMIN2"]   # family income (range/category)
    df = df.drop(columns=drop_always, errors="ignore")

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")

    # Drop low-cardinality columns (binary/categorical-coded numerics)
    df = df.loc[:, df.nunique(dropna=True) > min_unique_values]
    print(f"  After low-cardinality filter (>{min_unique_values} unique): {df.shape}")

    # Drop high-missingness columns
    df = df.loc[:, df.isna().mean() <= max_missing]
    print(f"  After missingness filter (<={max_missing:.0%}): {df.shape}")

    # Keep only fully observed rows
    df = df.dropna().reset_index(drop=True)
    print(f"  After keeping only complete rows: {df.shape}")

    assert df.isna().sum().sum() == 0, "Final dataset still has missing values."

    df.columns = df.columns.astype(str)

    out_path = out_dir / "data.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {df.shape} -> {out_path}")
    return df
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess real datasets for the imputation benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["codon", "physionet", "nhanes"],
        choices=["codon", "physionet", "nhanes"],
        help="Which datasets to preprocess.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if "codon" in args.datasets:
        preprocess_codon(args)
    if "physionet" in args.datasets:
        preprocess_physionet(args)
    if "nhanes" in args.datasets:
        preprocess_nhanes(args)

    print("\nDone. Run python Datasets/create_datasets.py --real to generate splits.")
