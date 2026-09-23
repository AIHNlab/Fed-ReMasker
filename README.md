# Federated Imputation Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-2609.28105-b31b1b.svg)](https://arxiv.org/abs/2609.28105)

Code for **Fed-ReMasker: Federated Tabular Imputation under Feature-Level Missingness**
([arXiv:2609.28105](https://arxiv.org/abs/2609.28105)).

Benchmark of federated missing-data imputation under **feature-level missingness** —
entire features absent at some clients — across six synthetic causal datasets and
three real ones (Codon, PhysioNet, NHANES).

Methods: Mean, Fed-MIWAE, Fed-ReMasker, Fed-CAFE, Fed-HF, compared over a
MFR × MVR × K grid, four client-heterogeneity conditions, three server-side
optimization strategies (FedAvg / FedProx / FedAdam), and a centralized reference.


## Installation

```bash
conda create -n fedremasker python=3.10.12
conda activate fedremasker
pip install -r requirements.txt
```


## Repository structure

```plaintext
.
├── Datasets/
│   ├── preprocess_real.py       # Raw downloads -> numeric feature matrices
│   ├── create_datasets.py       # Missing/Original splits, grid and heterogeneity
│   ├── Synthetic/{linear,nn}_d{20,50,100}/
│   └── Real/{codon,physionet,nhanes}/
│         └── {Original,Missing,Imputed}/...
│
├── Imputation/
│   ├── impute_missing.py        # Single entry point — run with --method
│   ├── aggregation.py           # FedAvg / FedProx / FedAdam
│   ├── data_utils.py            # Normalisation, global stats, column ranges
│   └── {Mean,Miwae,Remasker,CAFE,FedHF}/
│
├── Results/
│   ├── evaluate_imputation.py   # Imputed CSVs -> NRMSE, per client and aggregated
│   ├── plot_results.py          # Tables I-II, Figures 1-2
│   ├── plot_client_pca.py       # Figure 3
│   ├── compare_central.py       # Table III
│   ├── paired_tests.py          # Paired margins, intervals, win rates
│   └── compare_hp.py            # Inspect the LODO hyperparameter candidates
│
├── LICENSE
├── run_sweep.sh                 # Every configuration the paper needs, in stages
└── check_coverage.py            # Report what is still missing, without imputing
```

Generated data and results (`Missing/`, `Imputed/`, `Original/MFR_*/`, figures,
logs) are gitignored — they are reproducible from the sources above.


## Reproducing the paper

```bash
# 1. Data
python Datasets/preprocess_real.py
python Datasets/create_datasets.py                          # synthetic + real grid
python Datasets/create_datasets.py --hetero --only_hetero   # heterogeneity conditions

# 2. Imputation — all methods, all scenarios, plus strategy and centralized arms
./run_sweep.sh

# 3. Evaluation — discovers every method on disk, writes one per-client file
python Results/evaluate_imputation.py --scope both

# 4. Figures and tables
python Results/plot_results.py --scope both
python Results/plot_client_pca.py --dataset codon nhanes physionet
python Results/compare_central.py --out Results/figures/table_central.csv

# 5. Reported statistics
python Results/paired_tests.py --scope grid   --metric rmse     --cluster
python Results/paired_tests.py --scope grid   --metric rmse_mr  --cluster
python Results/paired_tests.py --scope grid   --metric rmse_mfr --cluster
python Results/paired_tests.py --scope hetero --metric rmse_mr
python Results/paired_tests.py --scope hetero --metric rmse_mfr
python Results/paired_tests.py --scope hetero --metric rmse --include-iid --include-tags \
    --cluster --exclude "ReMasker_prox*" "ReMasker_fedopt*"
```

Step 3 writes `Results/imputation_results.csv`, which every later command reads by
default; `--input` overrides it.

| Paper | File | Produced by |
|---|---|---|
| Table I — NRMSE per dataset × method | `figures/table_results.tex` | `plot_results.py --scope grid` |
| Table II — FedAvg / FedProx / FedAdam | `figures/table_strategy.tex` | `plot_results.py --scope hetero` |
| Table III — cost of federation | `figures/table_central.tex` | `compare_central.py` |
| Fig. 1 — NRMSE vs MVR / MFR / K | `figures/figure_comparison.png` | `plot_results.py --scope grid` |
| Fig. 2 — heterogeneity conditions | `figures/figure_hetero.png` | `plot_results.py --scope hetero` |
| Fig. 3 — client partitions in PCA space | `figures/figure_client_pca_all.png` | `plot_client_pca.py --dataset codon nhanes physionet` |

In the exception block printed under each outright win rate, a baseline is counted
once per scenario in which it beats the reference — so those counts overlap, and
can sum to more than the number of exceptions. Step 3 prints the equal-client
sensitivity check.


## Datasets

### Data sources and licenses

The three real datasets are redistributed here in preprocessed form
(`Datasets/Real/<name>/Original/data.csv`); `preprocess_real.py` rebuilds each one
from its original source. All three permit redistribution with attribution.

| Dataset | Source | License | Citation |
|---|---|---|---|
| `codon` | [UCI ML Repository, dataset 577](https://archive.ics.uci.edu/dataset/577/codon+usage) | CC BY 4.0 | Hallee, L. & Khomtchouk, B. (2020). *Codon usage* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5KP6B> |
| `physionet` | [PhysioNet/CinC Challenge 2012, v1.0.0](https://physionet.org/content/challenge-2012/1.0.0/) — open access | ODC-By 1.0 | Silva, I., Moody, G., Scott, D. J., Celi, L. A., & Mark, R. G. (2012). *Predicting In-Hospital Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012*. Computing in Cardiology, 39, 245-248. |
| `nhanes` | [CDC/NCHS NHANES public-use files](https://wwwn.cdc.gov/nchs/nhanes/), cycles 2003-2004 through 2017-2018 | U.S. Government work, no copyright (17 U.S.C. 105) | National Center for Health Statistics, Centers for Disease Control and Prevention. *National Health and Nutrition Examination Survey*. |

NHANES data are used for statistical analysis only, in line with the NCHS data user
agreement: identifiers (`SEQN`) and survey design variables are dropped during
preprocessing, and no attempt is made to identify individuals.

Synthetic data comes from `cdt.data.AcyclicGraphGenerator`, which samples a random
DAG and propagates values along its edges — `linear` (weighted sums, so Ridge-based
CAFE is well specified) and `nn` (small neural networks, so linear imputers are
misspecified).

The six synthetic datasets ship as `Datasets/Synthetic/<name>/Original/data.csv`.
`AcyclicGraphGenerator` is not seeded, so regenerating them on a machine where those
files are absent draws a different DAG and different data; the CSVs tracked in this
repository are the canonical data behind the reported results, and
`create_datasets.py` loads them from disk whenever they are present.


### Grid

`create_datasets.py` builds every combination of:

- **MFR** — features entirely missing per client: 0.0, 0.1, 0.2, 0.3
- **MVR** — random cell-level missingness: 0.1, 0.3, 0.5 (`--missing_ratios`, `MR_*` on disk)
- **K** — number of clients: 3, 5, 10

for 324 scenarios over the nine datasets. No column is ever nulled by *every*
client: each is assigned a protector client that must retain it.

```bash
python Datasets/create_datasets.py --real                 # real datasets only
python Datasets/create_datasets.py --n_variables 20 50    # narrow the synthetic sweep
```

### Heterogeneity conditions

Four conditions on the three real datasets, holding MFR=0.2, MVR=0.3, K=5 fixed so
any effect is attributable to the partition rather than to average difficulty:

| Variant | What differs |
|---|---|
| `iid` | Nothing — the homogeneous reference, drawn per seed like the others so it carries an error bar |
| `clients_imbalanced_gini{0.2,0.35}` | Client sample sizes skewed to a target Gini coefficient |
| `features_imbalanced` | Per-client MFR varies (0.1 … 0.3, mean 0.2) instead of being uniform |
| `non_iid` | Clients are KMeans clusters on standardised features, not random splits |

```bash
python Datasets/create_datasets.py --hetero --only_hetero
```

Each is generated once per seed in `--hetero_seeds` (default `42 43 44`), tagged
`<variant>_seed<N>`. At a given seed, every condition except `features_imbalanced`
removes the same features from each client index, so comparisons isolate the
partition.


## Running imputation

`run_sweep.sh` covers everything the paper needs. It is safe to re-run and safe to
interrupt — `impute_missing.py` skips any group already fully imputed, so a second
run picks up only what is missing.

```bash
./run_sweep.sh                 # everything
./run_sweep.sh hetero          # 5 methods, heterogeneity conditions
./run_sweep.sh strategy        # FedProx / FedAdam sweep
./run_sweep.sh grid            # 5 methods, full grid (the expensive stage)
./run_sweep.sh central         # centralized reference
DRY_RUN=1 ./run_sweep.sh       # print the commands without running them
```

Tunable through the `GPU`, `WORKERS`, `METHODS`, `PROX_MUS`, `FEDOPT_LRS`
environment variables. `python check_coverage.py` reports what is still missing
without imputing anything.

Single runs, for one method or one ablation:

```bash
python Imputation/impute_missing.py --method ReMasker
python Imputation/impute_missing.py --method ReMasker --hetero --dataset_type real
python Imputation/impute_missing.py --method ReMasker --tag prox --hp strategy=fedprox --hp prox_mu=0.01
```

`--help` documents the rest (`--dataset_type`, `--no_gpu`, `--log_file`, `--para`,
`--mfr/--mr/--n_clients` to narrow the grid, `--centralized`).

**Never overwrite a baseline run.** Any hyperparameter variant goes under a
`--tag`, which writes to `Imputed/<Method>_<tag>/`. Evaluation and plotting glob
`Imputed/*/`, so a new tag is picked up as soon as it has results — no list to keep
in sync, and `--exclude` drops any you don't want compared.

Hyperparameter defaults are the values reported in each method's original paper,
held fixed across datasets, and echoed at the start of every run. Override any of
them with `--hp KEY=VALUE` rather than editing the defaults:

```bash
python Imputation/list_hyperparameters.py              # every method, with sources
python Imputation/list_hyperparameters.py --markdown   # same, as tables
```

That reads the `_DEFAULTS` dict at the top of each
`Imputation/<Method>/impute_<method>.py`, so it is always what the code runs.

Fed-ReMasker's server-side strategy is selected the same way:

| `--hp` | Values | Meaning |
|---|---|---|
| `strategy` | `fedavg` (default) \| `fedprox` \| `fedopt` | FedProx adds a proximal term to the local loss; FedOpt takes an Adam-style server step on the aggregated update |
| `prox_mu` | float | FedProx proximal strength, used only when `strategy=fedprox` |
| `server_lr` | float | FedOpt server learning rate, used only when `strategy=fedopt` |
| `server_opt_variant` | `adam` (default) \| `yogi` \| `adagrad` | Instantiation of the FedOpt framework. The default is FedAdam, which is what the tables report; a tag does not record this value, so if you sweep it, put the variant in the tag and add a matching entry to `STRATEGY_LABELS` in `plot_results.py` |


## Evaluation

```bash
python Results/evaluate_imputation.py --scope both     # default; grid | hetero also valid
```

Reads the imputed CSVs off disk — no training — and computes NRMSE for every
method and tag it finds. NRMSE is normalised per feature by the **observed** range
pooled across that scenario's client files, never by the clean ground truth, so it
is identical for every method compared within a scenario.

Output goes to `Results/imputation_results.xlsx` (`_grid`/`_hetero` suffix when
`--scope` isn't `both`), with a CSV copy of the per-client sheet alongside:

| Sheet | Contents |
|---|---|
| `per_client` | One row per client file — what every analysis script reads |
| `per_experiment` | Pooled across clients per (method, dataset, MFR, MVR, K). Grid only |
| `per_dataset` | Mean across experiments, so each cell counts equally regardless of client count. Grid only |
| `hetero_per_experiment` | Pooled per (method, dataset, variant). Heterogeneity only |

Both aggregated sheets carry `rmse_macro` beside `rmse`: `rmse` pools every missing
entry, so larger clients weigh more; `rmse_macro` averages the per-client NRMSE, so
every client counts once. The paper reports `rmse` — `rmse_macro` exists to show
that the weighting choice does not drive any result.

### Choosing the FedProx / FedAdam hyperparameter

`table_strategy` does not report FedProx and FedAdam at a hyperparameter tuned on
the dataset being reported. For each real dataset, mu and the server learning rate
are chosen using **only the other two** (leave-one-dataset-out, 3 folds), scoring
candidates by NRMSE relative to FedAvg on the same scenario. The selection is
written to `figures/table_hp_selection.csv` and echoed to the console; the scoring
rationale is documented in `select_hp_lodo` in `plot_results.py`.

### Centralized reference

`compare_central.py` compares the federated arm against `ReMasker_central`, a
ReMasker trained on the pooled client files (`run_sweep.sh central`). Both arms see
the same underlying client-level masks and an identical epoch budget, so the
difference isolates federation. A **positive gap means centralizing is better**.

It opens with the overall gap across every configuration where both arms ran, then
breaks it down three ways — by MFR at K=5, by K at one (MFR, MVR) point, and per
(dataset, heterogeneity condition) with each arm's own iid→non-IID degradation.
Those three go to the CSV and become the blocks of the `.tex` table; `--tex-metric`
and `--tex-decimals` control the LaTeX output.

Two further blocks print but are not tabulated: the individual (dataset, condition,
seed) scenarios behind the heterogeneity means, which is where a claim about sign
consistency across seeds has to be read from, and each arm's iid→non-IID
degradation. The centralized arm deliberately covers only part of the grid;
`stage_central` in `run_sweep.sh` explains which part and why.


## References

- Fed-MIWAE: Balelli et al., *Federated Imputation of Incomplete Data via Deep Generative Models*, MICCAI 2023.
- ReMasker: Du et al., *Imputing Tabular Data with Masked Autoencoding*, arXiv 2023.
- CAFE: Min et al., *Cafe: Improved Federated Data Imputation by Leveraging Missing Data Heterogeneity*, IEEE TKDE 2025.
- Fed-HF: Hocine et al., *Federated Imputation under Heterogeneous Feature Spaces*, arXiv 2025.
- FedProx: Li et al., *Federated Optimization in Heterogeneous Networks*, MLSys 2020.
- FedOpt / FedAdam: Reddi et al., *Adaptive Federated Optimization*, ICLR 2021.


## Citation

If you use this code, please cite:

```bibtex
@misc{papathanail2026fedremasker,
  title         = {Fed-ReMasker: Federated Tabular Imputation under Feature-Level Missingness},
  author        = {Papathanail, Ioannis and Poursoleymani, Rooholla and Abdur Rahman, Lubnaa and Mougiakakou, Stavroula Georgia},
  year          = {2026},
  eprint        = {2609.28105},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.28105}
}
```


## License

Copyright (c) 2026 University of Bern, ARTORG Center for Biomedical Engineering Research, Authors: Ioannis Papathanail, Rooholla Poursoleymani, Lubnaa Abdur Rahman, Stavroula Georgia Mougiakakou, on behalf of the BETTER4U Consortium

The code is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at [![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0.txt)

The datasets redistributed under `Datasets/` keep the licenses of their original
sources — see [Data sources and licenses](#data-sources-and-licenses).
