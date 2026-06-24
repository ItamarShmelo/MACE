#!/bin/bash
#
# Distributed multigroup solver validation via SLURM job array.
#
# Submits 24 independent worker tasks + 1 dependent collect job.
# Task mapping:  0-17 = sigma (6 temps x 3 grids)
#                18-20 = derivative (3 temps)
#                21-23 = mu profiles (3 temps)
#
# Usage:  bash reports/run_multigroup_validation.sh
#
set -euo pipefail

cd /home/itamarg/workspace_current/ComptonMatrixExact
mkdir -p reports/generated/results

# ── Worker array job ──────────────────────────────────────────────────────
WORKER_JOB=$(sbatch --parsable <<'WORKER_SCRIPT'
#!/bin/bash
#SBATCH --job-name=mg_worker
#SBATCH --partition=bigrun
#SBATCH --array=0-23
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=reports/generated/results/worker_%a.out
#SBATCH --error=reports/generated/results/worker_%a.err

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd /home/itamarg/workspace_current/ComptonMatrixExact

echo "=== Worker task ${SLURM_ARRAY_TASK_ID} ==="
echo "Host:  $(hostname)"
echo "Cores: ${SLURM_CPUS_PER_TASK}"
echo "Start: $(date)"

python3.12 reports/multigroup_solver_validation.py --worker ${SLURM_ARRAY_TASK_ID}

echo "End:   $(date)"
WORKER_SCRIPT
)

echo "Worker array job: ${WORKER_JOB}  (24 tasks, 16 cores each)"

# ── Collect job (runs after all workers complete) ─────────────────────────
COLLECT_JOB=$(sbatch --parsable --dependency=afterok:${WORKER_JOB} <<'COLLECT_SCRIPT'
#!/bin/bash
#SBATCH --job-name=mg_collect
#SBATCH --partition=bigrun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=reports/generated/multigroup_collect_%j.out
#SBATCH --error=reports/generated/multigroup_collect_%j.err

set -euo pipefail
cd /home/itamarg/workspace_current/ComptonMatrixExact

echo "=== Collect ==="
echo "Host:  $(hostname)"
echo "Start: $(date)"

python3.12 reports/multigroup_solver_validation.py --collect

echo "End:   $(date)"
COLLECT_SCRIPT
)

echo "Collect job:     ${COLLECT_JOB}  (depends on ${WORKER_JOB})"
echo ""
echo "Monitor:  squeue -u ${USER} | grep mg_"
