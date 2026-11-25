#!/bin/bash
# Submit PLATON retrievals (flat null, equilibrium, free) for every Patchwork
# planet with a combined spectrum and a row in planet_params_v17.csv.
#
#   bash ~/aster/maiea/scripts/platon/submit_platon_survey.sh          # dry run
#   bash ~/aster/maiea/scripts/platon/submit_platon_survey.sh --submit
#
# Environment:
#   ROOT   Patchwork outputs (default ~/scratch/patchwork)
#   HOLD   space-separated slugs to skip (default: the two that fail
#          verify_fits as of 2026-11-25 -- TOI_125_c, TOI_561_b)
#   MODES  default flat,eq,free;  GASES  free-mode gas list
#   FORCE=1 rerun planets whose results exist (passes --force)
#   REPO   code tree to run (default ~/aster/maiea)
#
# One H100 job per planet, models run in sequence. Skips a planet with a
# live platon_<slug> job (two jobs must never write one output directory)
# and, without FORCE, a planet whose every requested model already has a
# result.pkl. Each sbatch is wrapped in env -i so an active venv cannot
# leak into the job.

set -euo pipefail
ROOT="${ROOT:-$HOME/scratch/patchwork}"
HOLD="${HOLD:-TOI_125_c TOI_561_b}"
MODES="${MODES:-flat,eq,free}"
GASES="${GASES:-H2O,CH4,CO2,CO,NH3,SO2,H2S}"
OUT="$HOME/scratch/platon"
REPO="${REPO:-$HOME/aster/maiea}"   # code tree the jobs run
PARAMS="$REPO/scripts/patchwork/planet_params_v17.csv"
SUBMIT=0; [[ "${1:-}" == "--submit" ]] && SUBMIT=1

slugs_with_params=$(tail -n +2 "$PARAMS" | cut -d, -f1 |
                    sed 's/[ .-]/_/g')
live=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
mkdir -p "$OUT/logs"

n=0
for dir in "$ROOT"/*/; do
    slug=$(basename "$dir")
    [[ -f "$dir/combined/combined_nrs1_transmission_spectrum.csv" ]] || continue
    if [[ " $HOLD " == *" $slug "* ]]; then
        echo "hold     $slug"; continue
    fi
    if ! grep -qx "$slug" <<<"$slugs_with_params"; then
        echo "noparams $slug (add it to planet_params_v17.csv)"; continue
    fi
    if grep -qx "platon_$slug" <<<"$live"; then
        echo "running  $slug"; continue
    fi
    todo=""
    for m in ${MODES//,/ }; do
        [[ -f "$OUT/$slug/$m/result.pkl" && -z "${FORCE:-}" ]] || todo+="$m,"
    done
    todo="${todo%,}"
    if [[ -z "$todo" ]]; then
        echo "done     $slug"; continue
    fi
    extra=""; [[ -n "${FORCE:-}" ]] && extra="--force"
    echo "submit   $slug  modes=$todo"
    n=$((n + 1))
    if (( SUBMIT )); then
        # Values contain commas, which --export=VAR=... would split, so they
        # travel in the environment and the job takes --export=ALL.
        env -i HOME="$HOME" USER="$USER" SLUG="$slug" MODES="$todo" \
            GASES="$GASES" EXTRA="$extra" REPO="$REPO" \
            bash -lc "cd '$OUT/logs' && sbatch --export=ALL \
            --job-name=platon_$slug \
            '$REPO/scripts/platon/retrieve_patchwork.sbatch'"
    fi
done
(( SUBMIT )) || echo "dry run: $n job(s); rerun with --submit"
