#!/bin/bash
#SBATCH --account=def-ncowan
#SBATCH --job-name=pw_gj9827d_asis
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out

# GJ 9827 d — as-is Patchwork baseline (script port of GJ_9827d.ipynb).
# Two phases in one job, each in its own environment AND its own python
# module (exoTEDRF is on 3.11, aster-env is on 3.13 because orchestral
# requires it):
#   reduce -> exoTEDRF Stages 1-3   python/3.11, ~/envs/exotedrf
#   fit    -> juliet Stages 5-6     python/3.13, ~/aster/aster-env
# The handoff is on disk, so a timed-out job can be resubmitted with
# PHASE=fit and it picks up the Stage 3 products already written.

set -euo pipefail

SCRIPT=~/aster/maiea/scripts/patchwork/patchwork_gj9827d.py
RAW_ROOT=/project/def-ncowan/wasi/jwst_raw
WORKDIR=~/scratch/patchwork/GJ_9827_d_asis
PHASE=${PHASE:-all}          # override at submit: sbatch --export=PHASE=fit ...

export CRDS_PATH=~/scratch/crds_cache
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p "$WORKDIR"

if [[ "$PHASE" == "reduce" || "$PHASE" == "all" ]]; then
    echo "=== PHASE reduce (exoTEDRF env, python/3.11) ==="
    module --force purge && module load StdEnv/2023 python/3.11
    ~/envs/exotedrf/bin/python "$SCRIPT" \
        --phase reduce --raw-root "$RAW_ROOT" --workdir "$WORKDIR"
fi

if [[ "$PHASE" == "fit" || "$PHASE" == "all" ]]; then
    echo "=== PHASE fit (aster-env, python/3.13) ==="
    module --force purge && module load StdEnv/2023 python/3.13
    source ~/aster/aster-env/bin/activate
    python "$SCRIPT" \
        --phase fit --raw-root "$RAW_ROOT" --workdir "$WORKDIR" \
        --published-csv "$WORKDIR/published_gj9827d_g395h.csv"
fi

echo "=== done; outputs in $WORKDIR ==="
