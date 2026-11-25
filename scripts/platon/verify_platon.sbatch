#!/bin/bash
# Verify the PLATON 7 env on an H100: device, data, one forward model, then
# PLATON's own transit-depth and retrieval tests.
#   cd ~/scratch && sbatch ~/aster/maiea/scripts/platon/verify_platon.sbatch
# Submit with no venv active (SLURM exports the submitting environment).
#SBATCH --job-name=platon_verify
#SBATCH --account=def-ncowan
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=platon_verify_%j.out

set -euo pipefail
PLATON_SRC="${PLATON_SRC:-/project/def-ncowan/wasi/platon}"

nvidia-smi -L
~/bin/platon-python ~/aster/maiea/scripts/platon/verify_platon.py

cd "$PLATON_SRC"
~/bin/platon-python -m pytest -q -x \
    tests/test_transit_depth_calculator.py tests/test_retrieve.py
