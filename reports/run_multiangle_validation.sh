#!/bin/bash
#
# Multiangle det-vs-MC validation via individual SLURM jobs.
#
# Submits 8 independent worker jobs (4 temps x 2 grids) + 1 collect job.
# Each worker computes 3D multiangle sigma matrix (8 angle bins) for one
# (temperature, grid) combination, comparing deterministic vs MC ensemble
# (5 seeds x 1M samples).
#
# Usage:  bash reports/run_multiangle_validation.sh
#
set -euo pipefail

PROJECT=/home/itamarg/workspace_current/ComptonMatrixExact
RESULTS_DIR=${PROJECT}/reports/generated/multiangle_results

mkdir -p "${RESULTS_DIR}"

NUM_TASKS=8
WORKER_JOBS=""

echo "Submitting ${NUM_TASKS} worker jobs..."

for IDX in $(seq 0 $((NUM_TASKS - 1))); do
    JOB_ID=$(sbatch --parsable \
        --job-name="ma_w${IDX}" \
        --partition=bigrun \
        --ntasks=1 \
        --cpus-per-task=16 \
        --output="${RESULTS_DIR}/worker_${IDX}.out" \
        --error="${RESULTS_DIR}/worker_${IDX}.err" \
        --export=ALL,OMP_NUM_THREADS=16 \
        --wrap="set -euo pipefail && cd ${PROJECT} && echo \"=== Worker ${IDX} === Host: \$(hostname) Start: \$(date)\" && python3.12 reports/multiangle_validation.py --worker ${IDX} && echo \"End: \$(date)\"")
    echo "  Worker ${IDX}: job ${JOB_ID}"

    if [ -z "${WORKER_JOBS}" ]; then
        WORKER_JOBS="${JOB_ID}"
    else
        WORKER_JOBS="${WORKER_JOBS}:${JOB_ID}"
    fi
done

# ── Collect job (runs after all workers complete) ─────────────────────────
COLLECT_JOB=$(sbatch --parsable \
    --job-name="ma_collect" \
    --partition=bigrun \
    --ntasks=1 \
    --cpus-per-task=1 \
    --output="${RESULTS_DIR}/collect.out" \
    --error="${RESULTS_DIR}/collect.err" \
    --dependency="afterok:${WORKER_JOBS}" \
    --wrap="set -euo pipefail && cd ${PROJECT} && echo \"=== Collect === Host: \$(hostname) Start: \$(date)\" && python3.12 reports/multiangle_validation.py --collect && echo \"End: \$(date)\"")

echo ""
echo "Collect job:  ${COLLECT_JOB}  (depends on all ${NUM_TASKS} workers)"
echo ""
echo "Monitor:  squeue -u ${USER} | grep ma_"
