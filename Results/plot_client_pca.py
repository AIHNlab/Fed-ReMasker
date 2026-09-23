"""
PCA projection of the client partitions, one panel per Hetero scenario.

    python Results/plot_client_pca.py
    python Results/plot_client_pca.py --dataset nhanes --variants iid non_iid
    python Results/plot_client_pca.py --style camera

Several datasets stack into one grid -- a row each, scenarios as columns, with a
single shared legend. Each row keeps its own PCA basis and its own explained
variance, since the datasets do not share features:

    python Results/plot_client_pca.py --dataset codon nhanes physionet \
        --style camera --format svg

Illustrates what each partitioning scheme does to the feature distribution:
in the IID panel the client colours are interleaved, in the non-IID panel they
occupy distinct regions.

This is a description of the experimental setup, not evidence about it. The
non-IID split is *built* by running KMeans on these same standardised features
(create_datasets.py: split_clients_clustering_noniid), so separation in the
projection is by construction. Its purpose in a paper is to make the
manipulation legible -- and, more usefully, to show why non-IID is the one
scenario whose missingness pattern correlates with the feature values: which
features a row loses depends on its client, and its client depends on where it
sits in this space.

PCA rather than t-SNE deliberately: the basis is deterministic, shared across
panels so they are comparable, and the axes carry an explained-variance figure.
t-SNE's cluster separation and inter-cluster distances are artifacts of
perplexity and initialisation, so it cannot support even this modest claim.

Reads Original/Hetero/ (the complete data), so it needs no imputation results.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

VARIANT_TITLES = {
    "iid": "Homogeneous",
    "non_iid": "Feature-distribution skew (non-IID)",
    "clients_imbalanced_gini0.2": "Quantity skew (Gini 0.2)",
    "clients_imbalanced_gini0.35": "Quantity skew (Gini 0.35)",
    "features_imbalanced": "Missingness skew (MFR)",
}
DATASET_TITLES = {"codon": "Codon", "physionet": "PhysioNet", "nhanes": "NHANES"}
CLIENT_COLORS = ["#4c8cbf", "#e08c3a", "#d94f3d", "#5ab46e", "#8e44ad",
                 "#2e86c1", "#9e9e9e", "#c0392b", "#16a085", "#7f8c8d"]


def load_variant(root, dataset, variant, seed):
    d = Path(root) / "Real" / dataset / "Original" / "Hetero" / f"{variant}_seed{seed}"
    if not d.is_dir():
        # Common cause: a machine that only holds Missing/ (imputation inputs)
        # because Original/ was never copied -- imputation does not need it,
        # only evaluation and this figure do. Say so, and offer the cheap
        # targeted regeneration rather than the full --only_hetero rebuild.
        missing_twin = d.parent.parent.parent / "Missing" / "Hetero" / d.name
        hint = ("Regenerate just this variant with:\n"
                "  python -c \"import sys; sys.path.insert(0,'Datasets'); "
                "from create_datasets import create_heterogeneous_scenario as c; "
                f"c('Datasets/Real/{dataset}', mr=0.3, mfr=0.2, n_clients=5, "
                f"variant_type='{variant}', variant_config={{}}, seed={seed}, "
                f"seed_tag='seed{seed}')\"")
        if missing_twin.is_dir():
            hint = (f"{missing_twin} exists but {d.name} does not under Original/. "
                    "This machine has the imputation inputs but not the ground "
                    "truth. Copy Original/Hetero/ across, or regenerate it "
                    f"(deterministic):\n{hint}")
        raise SystemExit(f"{d} not found.\n{hint}")
    files = sorted(d.glob("client_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
    return [pd.read_csv(f) for f in files]


def fit_basis(client_dfs):
    """
    One PCA basis per dataset, fitted on that dataset's pooled rows.

    Every variant partitions the *same* rows, so pooling any of them gives the
    same matrix -- fitting once keeps a dataset's panels on identical axes,
    without which their spreads could not be compared by eye. The basis cannot
    be shared *across* datasets: they have different features, so their
    components are not the same quantity.
    """
    pooled = pd.concat(client_dfs, ignore_index=True).to_numpy(dtype=float)
    scaler = StandardScaler().fit(pooled)
    pca = PCA(n_components=2, random_state=0).fit(scaler.transform(pooled))
    return scaler, pca, pca.explained_variance_ratio_ * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_root", default="Datasets")
    ap.add_argument("--dataset", nargs="+", default=["codon"],
                    help="One dataset, or several to stack as rows of one grid "
                         "(e.g. --dataset codon nhanes physionet)")
    ap.add_argument("--variants", nargs="+", default=["iid", "non_iid"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="Results/figures")
    ap.add_argument("--format", default="png", choices=["png", "svg"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--max-points", type=int, default=4000,
                    help="Subsample per panel for legibility (0 = all points)")
    ap.add_argument("--style", default="default", choices=["default", "camera"],
                    help="'camera' = IEEE double-column width (7.16in, 8pt)")
    ap.add_argument("--clip-percentile", type=float, default=99.5,
                    help="Axis limits span this percentile range rather than "
                         "the extremes, so a few far-out points cannot squash "
                         "the rest (default: 99.5, i.e. 0.5--99.5). Use 100 "
                         "for the full range.")
    ap.add_argument("--font-size", type=float, default=None,
                    help="Override the style's font size "
                         "(default: 13 for 'default', 8 for 'camera')")
    ap.add_argument("--panel-aspect", type=float, default=None,
                    help="Panel height / width (default: 0.85 for a single "
                         "dataset, 0.7 for a grid, which keeps three rows on "
                         "one page)")
    args = ap.parse_args()

    rows = []
    for ds in args.dataset:
        panels = {v: load_variant(args.input_root, ds, v, args.seed)
                  for v in args.variants}
        scaler, pca, ev = fit_basis(panels[args.variants[0]])
        rows.append((ds, panels, scaler, pca, ev))

    nrows, ncols = len(rows), len(args.variants)
    grid = nrows > 1
    aspect = args.panel_aspect if args.panel_aspect else (0.7 if grid else 0.85)

    if args.style == "camera":
        fs, figw, msize = 8, 7.16, 1.5
        fs = args.font_size or fs
        plt.rcParams.update({"font.family": "serif",
                             "font.serif": ["Times New Roman", "Liberation Serif",
                                            "Nimbus Roman", "DejaVu Serif"],
                             "font.size": fs})
    else:
        fs, figw, msize = args.font_size or 13, 6.0 * ncols, 3.0

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figw, figw / ncols * aspect * nrows),
                             squeeze=False)

    for r, (ds, panels, scaler, pca, ev) in enumerate(rows):
        row_axes = axes[r]
        drawn = []
        for c, variant in enumerate(args.variants):
            ax = row_axes[c]
            for i, cdf in enumerate(panels[variant]):
                X = pca.transform(scaler.transform(cdf.to_numpy(dtype=float)))
                cap = args.max_points // len(panels[variant]) if args.max_points else 0
                if cap and len(X) > cap:
                    X = X[rng.choice(len(X), cap, replace=False)]
                drawn.append(X)
                ax.scatter(X[:, 0], X[:, 1], s=msize, alpha=0.5, linewidths=0,
                           color=CLIENT_COLORS[i % len(CLIENT_COLORS)],
                           label=f"Client {i + 1}")
            # Column headings once, at the top: they name the scenario, which is
            # the same down every row.
            if r == 0:
                ax.set_title(VARIANT_TITLES.get(variant, variant),
                             fontsize=fs, fontweight="bold")
            # The explained variance differs per dataset, so the axis label
            # belongs on every row, not only the bottom one.
            ax.set_xlabel(f"PC1 ({ev[0]:.1f}% variance)" if grid
                          else f"First principal component ({ev[0]:.1f}% variance)",
                          fontsize=fs)
            ax.tick_params(labelsize=fs)

        # Share limits within the row only: one basis per dataset, so equal
        # limits are what make the panels comparable. Across rows the axes are
        # different quantities and forcing them to match would mislead.
        #
        # Limits come from a percentile range, not the extremes: NHANES has a
        # handful of points an order of magnitude out, and letting them set the
        # range squashes every other point into a flat band, hiding the very
        # thing the figure exists to show. Outliers are still drawn, just
        # outside the view, and the count is reported.
        P = np.concatenate(drawn)
        q = args.clip_percentile
        xlo, xhi = np.percentile(P[:, 0], [100 - q, q])
        ylo, yhi = np.percentile(P[:, 1], [100 - q, q])
        padx, pady = 0.05 * (xhi - xlo), 0.05 * (yhi - ylo)
        xlo, xhi, ylo, yhi = xlo - padx, xhi + padx, ylo - pady, yhi + pady
        hidden = int(((P[:, 0] < xlo) | (P[:, 0] > xhi) |
                      (P[:, 1] < ylo) | (P[:, 1] > yhi)).sum())
        if hidden:
            print(f"  {ds}: {hidden} of {len(P)} plotted points "
                  f"({100 * hidden / len(P):.1f}%) fall outside the "
                  f"{q:g}th-percentile view")
        for c, ax in enumerate(row_axes):
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            if c:
                ax.tick_params(labelleft=False)

        # Plain weight: the bold dataset name is a separate Text added after
        # layout, since one Text cannot mix weights.
        row_axes[0].set_ylabel(
            f"PC2 ({ev[1]:.1f}% variance)" if grid
            else f"Second principal component ({ev[1]:.1f}% variance)",
            fontsize=fs)

    handles, labels = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                     fontsize=fs, frameon=True,
                     markerscale=4 if args.style == "camera" else 3,
                     bbox_to_anchor=(0.5, -0.02))
    for t in leg.get_texts():
        t.set_fontweight("bold")

    # A grid names each dataset on its own row, so a single overall title would
    # have nothing left to say.
    if not grid:
        fig.suptitle(DATASET_TITLES.get(args.dataset[0], args.dataset[0]),
                     fontsize=fs + 1, fontweight="bold")
    # savefig's tight bbox takes in the legend, so the bottom margin only has to
    # clear the last row's label; and within a row only the left panel keeps y
    # tick labels, so the default gutter is holding nothing.
    fig.tight_layout(rect=[0, 0.03 if grid else 0.08, 1, 1 if grid else 0.97],
                     w_pad=0.3 if grid else None)

    # One x-label per row, not one per panel: a row's panels share a basis, so
    # the text is identical. After tight_layout, so the per-axis labels have
    # already reserved the space the centred one moves into.
    if grid:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        for row_axes, (ds, *_) in zip(axes, rows):
            if ncols > 1:
                label = row_axes[0].xaxis.label
                text = label.get_text()
                y = inv.transform((0, label.get_window_extent(renderer).y0))[1]
                for ax in row_axes:
                    ax.set_xlabel("")
                x = (row_axes[0].get_position().x0 +
                     row_axes[-1].get_position().x1) / 2
                fig.text(x, y, text, ha="center", va="bottom", fontsize=fs)

            # Placed relative to the y-label's drawn extent, so it clears the
            # tick numbers whatever their width.
            ylab = row_axes[0].yaxis.label
            pos = row_axes[0].get_position()
            fig.text(inv.transform((ylab.get_window_extent(renderer).x0, 0))[0] - 0.012,
                     (pos.y0 + pos.y1) / 2, DATASET_TITLES.get(ds, ds),
                     ha="center", va="center", rotation=90,
                     fontsize=fs + 1, fontweight="bold")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    name = "all" if grid else args.dataset[0]
    path = out / f"figure_client_pca_{name}.{args.format}"
    fig.savefig(path, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"Saved -> {path}")
    for ds, _, _, _, ev in rows:
        print(f"  {ds}: PC1 {ev[0]:.1f}% + PC2 {ev[1]:.1f}% "
              f"= {ev[:2].sum():.1f}% of variance")


if __name__ == "__main__":
    main()
