#!/bin/bash
# One PLATON retrieval of one Patchwork planet on one H100.
#   cd ~/scratch && sbatch --job-name=platon_L_98_59_d_eq \
#       --export=ALL,SLUG=L_98_59_d,MODE=eq \
#       ~/aster/maiea/scripts/platon/retrieve_patchwork.sbatch
# ~6 min for 7 parameters on one H100 (L 98-59 d, 2026-11-25).
# Optional: EXTRA="--haze --force".
# Submit with no venv active.
#SBATCH --job-name=platon_retrieve
#SBATCH --account=def-ncowan
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail
: "${SLUG:?set SLUG, e.g. SLUG=L_98_59_d}"
MODE="${MODE:-eq}"

nvidia-smi -L
~/bin/platon-python ~/aster/maiea/scripts/platon/retrieve_patchwork.py \
    "$SLUG" --mode "$MODE" ${EXTRA:-}
