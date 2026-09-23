"""
Compare the FedProx mu / FedOpt lr candidates against each other.

    python Results/compare_hp.py                  # all three views
    python Results/compare_hp.py --dataset codon  # absolute NRMSE for one dataset
    python Results/compare_hp.py --metric rmse    # absolute instead of relative

Reads Results/figures/table_hp_candidates.csv, written by plot_results.py
--scope hetero/both. `rel` is NRMSE divided by FedAvg's on the same
(dataset, scenario): below 1.0 beat FedAvg, above 1.0 lost to it.

This is the descriptive view of the sweep. Which value is actually *selected*
per dataset is a separate question, answered by the leave-one-dataset-out
procedure in table_hp_selection.csv -- a config can look best overall here and
still not be chosen for a given dataset, because its fold excludes that
dataset's own results.
"""
import argparse
from pathlib import Path

import pandas as pd


def load(path):
    df = pd.read_csv(path)
    # FedAvg has no hyperparameter; pivot_table would drop those rows if NaN
    # stayed in the index, losing the 1.0 reference line the table is read against.
    df["config"] = df["family"] + df["hp"].map(
        lambda v: "" if pd.isna(v) else f" {v:g}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="Results/figures/table_hp_candidates.csv")
    ap.add_argument("--dataset", default=None,
                    help="Restrict the per-scenario view to one dataset")
    ap.add_argument("--metric", default="rel", choices=["rel", "rmse"],
                    help="rel = ratio to FedAvg (default); rmse = absolute NRMSE")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"{path} not found — run plot_results.py --scope hetero first.")
    df = load(path)

    unit = "relative to FedAvg" if args.metric == "rel" else "absolute NRMSE"

    print(f"=== Per dataset ({unit}) ===")
    per_ds = df.pivot_table(index="config", columns="dataset_name", values=args.metric)
    per_ds["mean"] = per_ds.mean(axis=1)
    print(per_ds.round(4).sort_values("mean").to_string())

    print(f"\n=== Per scenario ({unit}) ===")
    sub = df if args.dataset is None else df[df["dataset_name"] == args.dataset]
    if sub.empty:
        raise SystemExit(f"No rows for dataset {args.dataset!r}.")
    if args.dataset:
        print(f"(dataset: {args.dataset})")
    print(sub.pivot_table(index="config", columns="variant", values=args.metric)
             .round(4).to_string())

    print("\n=== Spread across candidates, per family ===")
    for family, g in df[df["family"] != "FedAvg"].groupby("family"):
        best = g.groupby("config")[args.metric].mean().sort_values()
        span = best.iloc[-1] - best.iloc[0]
        print(f"  {family}: best {best.index[0]} ({best.iloc[0]:.4f}), "
              f"worst {best.index[-1]} ({best.iloc[-1]:.4f}), span {span:.4f}")


if __name__ == "__main__":
    main()
