#!/bin/bash
#SBATCH --account=def-ncowan
#SBATCH --job-name=pw_gj9827d_baseline
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out

# GJ 9827 d — Patchwork baseline fit (script port of GJ_9827d.ipynb).
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
# NOTE: directory name kept as-written so the existing reduction on Fir
# (289 GB of Stage 1-3 products) is not orphaned by a rename.
WORKDIR=~/scratch/patchwork/GJ_9827_d_asis
EXOTEDRF_REPO=$HOME/exoTEDRF   # optimizer-branch checkout; shadows the installed release
PHASE=${PHASE:-all}          # override at submit: sbatch --export=PHASE=fit ...
# juliet reloads existing posteriors instead of refitting, so a rerun after
# any code/ephemeris change silently returns the OLD answers. Set FORCE_REFIT=1
# to delete previous juliet_* runs first.
FORCE_REFIT=${FORCE_REFIT:-0}
REFIT_FLAG=""; [[ "$FORCE_REFIT" == "1" ]] && REFIT_FLAG="--force-refit"

export CRDS_PATH=~/scratch/crds_cache
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
# Pin the reference context: an unpinned CRDS resolves to "latest", so two
# targets reduced weeks apart could silently use different reference files
# and no longer be a uniform survey. Matches the optimizer template.
export CRDS_CONTEXT=jwst_1322.pmap
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
        # Run exoTEDRF from the optimizer-branch checkout, so reductions and
        # optimizer sweeps share one stage-code tree.
        export PYTHONPATH="$EXOTEDRF_REPO${PYTHONPATH:+:$PYTHONPATH}"
        python -c "import numpy, jwst, exotedrf, cv2; import os; print('exotedrf env OK  numpy', numpy.__version__, 'cv2', cv2.__version__, 'from', os.path.dirname(exotedrf.__file__))"
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
            $REFIT_FLAG \
            --published-csv "$WORKDIR/published_gj9827d_g395h.csv"
    )
fi

echo "=== done; outputs in $WORKDIR ==="
