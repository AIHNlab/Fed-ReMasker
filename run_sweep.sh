#!/usr/bin/env bash
#
# Run the whole benchmark matrix in one go.
#
#   ./run_sweep.sh              # everything (hetero + strategy + grid)
#   ./run_sweep.sh hetero       # 5 base methods, Hetero scenarios only
#   ./run_sweep.sh strategy     # FedProx/FedOpt sweep over the Hetero scenarios
#   ./run_sweep.sh grid         # 5 base methods, full MFR/MR/N grid (the expensive one)
#   ./run_sweep.sh central      # centralized ReMasker reference (pooled clients)
#   ./run_sweep.sh evaluate     # evaluate + plot only, no imputation
#
# Safe to re-run and safe to interrupt: impute_missing.py skips any dataset
# group that is already fully imputed, so a second run only picks up what is
# missing. Nothing here deletes existing results.
#
# Knobs (environment variables):
#   GPU=1            CUDA_VISIBLE_DEVICES value
#   WORKERS=4        --max_workers for --para
#   METHODS="..."    base methods to run
#   PROX_MUS="..."   FedProx mu values
#   FEDOPT_LRS="..." FedOpt server_lr values
#   DRY_RUN=1        print the commands instead of running them
#
# Example:
#   tmux new -s sweep -d './run_sweep.sh 2>&1 | tee logs/sweep.log'

cd "$(dirname "$0")" || exit 1
mkdir -p logs

GPU=${GPU:-1}
WORKERS=${WORKERS:-4}
# Note: ${VAR-default}, not ${VAR:-default} -- an explicitly empty value means
# "none of these", so a family can be switched off when splitting work across
# machines (e.g. FEDOPT_LRS= runs only the FedProx arms).
METHODS=${METHODS-"Mean MIWAE CAFE FedHF ReMasker"}
PROX_MUS=${PROX_MUS-"0.001 0.01 0.1 1.0"}
FEDOPT_LRS=${FEDOPT_LRS-"0.001 0.01 0.1 1.0"}

failed=0

run() {
  # run <log-name> <impute_missing.py args...>
  local name="$1"; shift
  echo
  echo "=============================================================="
  echo "  $name"
  echo "  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=============================================================="
  if [ -n "${DRY_RUN:-}" ]; then
    echo "DRY RUN: python Imputation/impute_missing.py $* --para --max_workers $WORKERS"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$GPU" python Imputation/impute_missing.py \
    "$@" --para --max_workers "$WORKERS" --log_file "logs/${name}.log"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!! FAILED (exit $rc): $name" >&2
    failed=$((failed + 1))
  fi
  return 0   # keep the sweep going; failures are counted and reported at the end
}

stage_hetero() {
  for m in $METHODS; do
    run "${m}_hetero" --method "$m" --hetero --dataset_type real
  done
}

stage_grid() {
  for m in $METHODS; do
    run "${m}_grid" --method "$m" --dataset_type both
  done
}

stage_strategy() {
  # Hetero only. The FedAvg arm is the plain ReMasker run from the stages
  # above, and the homogeneous reference comes from the 'iid' Hetero variant
  # -- which is byte-identical to the MFR_0.2/MR_0.3/N5 grid cell at seed 42,
  # so running that cell separately would only duplicate it.
  for mu in $PROX_MUS; do
    run "prox_mu${mu}_hetero" --method ReMasker --tag "prox_mu$mu" --hetero \
      --dataset_type real --hp strategy=fedprox --hp prox_mu="$mu"
  done
  for lr in $FEDOPT_LRS; do
    run "fedopt_lr${lr}_hetero" --method ReMasker --tag "fedopt_lr$lr" --hetero \
      --dataset_type real --hp strategy=fedopt --hp server_lr="$lr"
  done
}

stage_central() {
  # Centralized reference: identical masks, identical 300-epoch budget, no
  # federation. Two axes only.
  #
  # The full MR x MFR plane at K=5: 3 MR x 4 MFR x 9 datasets = 108 groups.
  # MFR is the axis that carries the argument -- it spans 0..0.3, so the gap at
  # MFR=0 is the pure data-pooling advantage and its growth is the cost
  # specifically attributable to clients holding disjoint feature sets. Sweeping
  # MR as well costs 3x, but makes the design complete and symmetric and removes
  # any "evaluated at a single MR" caveat.
  #
  # The K axis is deliberately omitted: each client independently withholds
  # ~MFR of the features, so the pooled fraction of rows missing any given
  # feature is ~MFR regardless of K, and a row-wise imputer never observes
  # which rows share a mask. The centralized result is therefore expected to be
  # approximately K-invariant, and sweeping it would spend compute confirming
  # that.
  #
  # Hetero is restricted to iid and non_iid (18 groups, not 45). Pooling erases
  # quantity skew and per-client MFR spread outright -- shuffled rows are
  # shuffled rows -- so those three would return numbers predictable in
  # advance. non_iid is the exception: its mask is value-correlated (clients
  # are feature-space clusters) and so survives pooling, making the
  # iid/non_iid pair the control that separates intrinsic mask difficulty from
  # the cost of federating under it.
  run "central_grid" --method ReMasker --tag central --centralized \
    --n_clients 5
  run "central_hetero" --method ReMasker --tag central --centralized --hetero \
    --dataset_type real --variants iid non_iid
}

stage_evaluate() {
  echo
  echo "=============================================================="
  echo "  Evaluate + plot"
  echo "=============================================================="
  if [ -n "${DRY_RUN:-}" ]; then
    echo "DRY RUN: python Results/evaluate_imputation.py --scope both"
    echo "DRY RUN: python Results/plot_results.py --scope both"
    return 0
  fi
  python Results/evaluate_imputation.py --scope both || failed=$((failed + 1))
  python Results/plot_results.py --scope both || failed=$((failed + 1))
}

case "${1:-all}" in
  hetero)   stage_hetero ;;
  grid)     stage_grid ;;
  strategy) stage_strategy ;;
  central)  stage_central ;;
  evaluate) stage_evaluate ;;
  all)      stage_hetero; stage_strategy; stage_grid; stage_central; stage_evaluate ;;
  *)        echo "usage: $0 [all|hetero|strategy|grid|central|evaluate]" >&2; exit 2 ;;
esac

echo
if [ $failed -eq 0 ]; then
  echo "Sweep finished with no failures."
else
  echo "Sweep finished with $failed failed step(s) — search this log for '!! FAILED'."
fi
