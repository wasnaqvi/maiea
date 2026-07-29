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
#   reduce -> exoTEDRF Stages 1-3   python/3.11.5, ~/envs/exotedrf
#   fit    -> juliet Stages 5-6     python/3.13.2, ~/aster/aster-env
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

# A virtualenv on Alliance clusters must be ACTIVATED, not invoked as
# <venv>/bin/python — calling the interpreter by path leaves the venv's
# site-packages off sys.path on a compute node (numpy not found).
# Each phase runs in a subshell so its module state and activation cannot
# leak into the next.

if [[ "$PHASE" == "reduce" || "$PHASE" == "all" ]]; then
    echo "=== PHASE reduce (exoTEDRF env, python/3.11.5) ==="
    (
        module purge
        # OpenCV comes from a module, not pip (Alliance ships a dummy
        # wheel that refuses to install). stcal.jump imports cv2 at module
        # level, so JumpStep cannot run without it. Load BEFORE activating
        # the venv, and never clear PYTHONPATH here -- that is how the
        # module delivers cv2.
        module load StdEnv/2023 python/3.11.5 opencv/4.12.0
        source ~/envs/exotedrf/bin/activate
        python -c "import numpy, jwst, exotedrf, cv2; print('exotedrf env OK  numpy', numpy.__version__, 'cv2', cv2.__version__)"
        python "$SCRIPT" \
            --phase reduce --raw-root "$RAW_ROOT" --workdir "$WORKDIR"
    )
fi

if [[ "$PHASE" == "fit" || "$PHASE" == "all" ]]; then
    echo "=== PHASE fit (aster-env, python/3.13.2) ==="
    (
        module purge
        module load StdEnv/2023 python/3.13.2
        source ~/aster/aster-env/bin/activate
        python -c "import numpy, juliet; print('aster-env OK')"
        python "$SCRIPT" \
            --phase fit --raw-root "$RAW_ROOT" --workdir "$WORKDIR" \
            --published-csv "$WORKDIR/published_gj9827d_g395h.csv"
    )
fi

echo "=== done; outputs in $WORKDIR ==="
