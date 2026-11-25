#!/bin/bash
# PLATON retrievals of one Patchwork planet on one H100, one model after
# another. submit_platon_survey.sh submits this for every planet; by hand:
#   cd ~/scratch && SLUG=L_98_59_d MODES=flat,eq,free sbatch --export=ALL \
#       --job-name=platon_L_98_59_d \
#       ~/aster/maiea/scripts/platon/retrieve_patchwork.sbatch
# (MODES/GASES contain commas, so pass them in the environment with
# --export=ALL, never as --export=VAR=a,b.)
# ~5-10 min per model on one H100 (L 98-59 d, 2026-11-25).
# Optional: MODE (a single model), GASES (free mode), EXTRA="--haze --force",
# REPO (code tree to run; default ~/aster/maiea).
# Submit with no venv active.
#SBATCH --job-name=platon_retrieve
#SBATCH --account=def-ncowan
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out

set -uo pipefail
: "${SLUG:?set SLUG, e.g. SLUG=L_98_59_d}"
MODES="${MODES:-${MODE:-eq}}"
GASES="${GASES:-H2O,CH4,CO2,CO,NH3,SO2,H2S}"
REPO="${REPO:-$HOME/aster/maiea}"   # or a code snapshot

nvidia-smi -L
# A GPU left full by an earlier job fails JAX start-up with
# CUDA_ERROR_OUT_OF_MEMORY (fc10512, 2026-11-25). Fail once, naming the node,
# so it can be resubmitted with --exclude=<node>.
if ! ~/bin/platon-python -c "import sys, jax; sys.exit(jax.default_backend() != 'gpu')" \
        >/dev/null 2>&1; then
    echo "GPU unusable on $(hostname); resubmit with --exclude=$(hostname -s)" >&2
    exit 3
fi
status=0
for mode in ${MODES//,/ }; do
    echo "=== $SLUG $mode $(date -Is)"
    # shellcheck disable=SC2086
    ~/bin/platon-python "$REPO/scripts/platon/retrieve_patchwork.py" \
        "$SLUG" --mode "$mode" --gases "$GASES" ${EXTRA:-} || status=1
done
exit $status
