"""
Report what is still missing, without imputing anything.

    python check_coverage.py                 # per-config summary of remaining work
    python check_coverage.py --missing       # also list every incomplete group
    python check_coverage.py --datasets      # what create_datasets.py has produced

Uses impute_missing.py's own `_output_path`, so "done" here means exactly what
the imputer would skip on the next run — not an approximation of it.

The config list mirrors run_sweep.sh, and honours the same environment
variables (METHODS, PROX_MUS, FEDOPT_LRS), so the two never drift apart.
"""
import argparse
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "Imputation"))
from impute_missing import _output_path  # noqa: E402

METHODS = os.environ.get("METHODS", "Mean MIWAE CAFE FedHF ReMasker").split()
PROX_MUS = os.environ.get("PROX_MUS", "0.001 0.01 0.1 1.0").split()
FEDOPT_LRS = os.environ.get("FEDOPT_LRS", "0.001 0.01 0.1 1.0").split()


def _groups(input_root, hetero, dataset_type="both"):
    """{group_dir: [client files]} — the same globbing impute_all() does."""
    subpath = "Missing/Hetero/*/*.csv" if hetero else "Missing/*/*/*/*.csv"
    files = []
    if dataset_type in ("synthetic", "both"):
        files += glob.glob(f"{input_root}/Synthetic/*/{subpath}")
    if dataset_type in ("real", "both"):
        files += glob.glob(f"{input_root}/Real/*/{subpath}")

    groups = defaultdict(list)
    for f in files:
        groups[str(Path(f).parent)].append(f)
    return groups


def _status(groups, input_root, label):
    """(done, partial, todo) group dirs for one method label."""
    done, partial, todo = [], [], []
    for group_dir, files in sorted(groups.items()):
        n = sum(_output_path(f, input_root, label).exists() for f in files)
        if n == len(files):
            done.append(group_dir)
        elif n == 0:
            todo.append(group_dir)
        else:
            partial.append((group_dir, n, len(files)))
    return done, partial, todo


def configs():
    """(label, hetero, dataset_type) for every run_sweep.sh step.

    The strategy tags are Hetero-only. They used to also run the single
    MFR_0.2/MR_0.3/N5 grid cell to supply the "Homogeneous (IID)" reference
    bar, but the 'iid' Hetero variant is byte-identical to that cell at
    seed 42 (and adds seeds 43/44), so those runs were duplicating work the
    Hetero pass already does.
    """
    out = []
    for m in METHODS:
        out.append((m, True, "real"))
        out.append((m, False, "both"))
    for mu in PROX_MUS:
        out.append((f"ReMasker_prox_mu{mu}", True, "real"))
    for lr in FEDOPT_LRS:
        out.append((f"ReMasker_fedopt_lr{lr}", True, "real"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_root", default="Datasets")
    ap.add_argument("--missing", action="store_true",
                    help="List every incomplete group, not just the counts")
    ap.add_argument("--datasets", action="store_true",
                    help="Report which dataset splits exist, then exit")
    args = ap.parse_args()

    if args.datasets:
        print("Hetero variants present (Missing/Hetero/*):")
        seen = defaultdict(set)
        for d in sorted(glob.glob(f"{args.input_root}/*/*/Missing/Hetero/*")):
            p = Path(d)
            seen[f"{p.parts[-5]}/{p.parts[-4]}"].add(p.name)
        for ds, variants in sorted(seen.items()):
            print(f"  {ds}: {len(variants)} -> {', '.join(sorted(variants))}")
        if not seen:
            print("  (none — run create_datasets.py --hetero --only_hetero)")
        print("\nGrid cells present (Missing/MFR_*/MR_*/N*):")
        grid = defaultdict(int)
        for d in sorted(glob.glob(f"{args.input_root}/*/*/Missing/MFR_*/MR_*/N*")):
            p = Path(d)
            grid[f"{p.parts[-6]}/{p.parts[-5]}"] += 1
        for ds, n in sorted(grid.items()):
            print(f"  {ds}: {n} cells")
        if not grid:
            print("  (none)")
        return

    # Cache group listings — the few distinct shapes are reused across configs.
    cache = {}
    rows, total_todo = [], 0

    for label, hetero, dstype in configs():
        key = (hetero, dstype)
        if key not in cache:
            cache[key] = _groups(args.input_root, hetero, dstype)
        groups = cache[key]
        done, partial, todo = _status(groups, args.input_root, label)
        remaining = len(todo) + len(partial)
        total_todo += remaining
        scope = "hetero" if hetero else "grid"
        rows.append((label, scope, len(done), len(partial), len(todo), len(groups)))

        if args.missing and remaining:
            print(f"\n--- {label} [{scope}]: {remaining} group(s) outstanding")
            for g in todo:
                print(f"    MISSING  {g}")
            for g, n, tot in partial:
                print(f"    PARTIAL  {g}  ({n}/{tot} clients)")

    width = max(len(r[0]) for r in rows) + 2
    print(f"\n{'CONFIG':<{width}}{'SCOPE':<18}{'DONE':>7}{'PARTIAL':>9}{'TODO':>7}{'TOTAL':>7}")
    print("-" * (width + 48))
    for label, scope, done, part, todo, tot in rows:
        flag = "" if (part + todo) == 0 else "  <-"
        print(f"{label:<{width}}{scope:<18}{done:>7}{part:>9}{todo:>7}{tot:>7}{flag}")

    print("-" * (width + 48))
    if total_todo == 0:
        print("Everything in the sweep is imputed. Next: ./run_sweep.sh evaluate")
    else:
        print(f"{total_todo} dataset group(s) still to impute. ./run_sweep.sh will "
              f"pick up exactly these and skip the rest.")
        if not args.missing:
            print("Re-run with --missing to list them.")


if __name__ == "__main__":
    main()
