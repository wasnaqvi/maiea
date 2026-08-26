#!/bin/bash
# Submit Patchwork survey jobs for every available planet on DRAC Fir.
#
# Copy to ~/aster/maiea/scripts/patchwork/ and run from a CLEAN login
# shell (no venv active):
#
#   chmod +x scripts/patchwork/submit_survey.sh
#   ./scripts/patchwork/submit_survey.sh                 # dry run: show the plan
#   ./scripts/patchwork/submit_survey.sh --submit        # submit REDUCE, all waves
#   ./scripts/patchwork/submit_survey.sh --submit --wave 1
#   ./scripts/patchwork/submit_survey.sh --mode fit --submit --only TOI_270_c
#
# The venv is activated only inside a SUBSHELL, for the one step that
# needs aster_toolkit (sizing the jobs and writing the sbatch files).
# Every sbatch then runs through `env -i ... bash -lc`, so the job is
# submitted from a clean environment either way: SLURM exports the
# submitting environment and `module purge` cannot undo an activation.
#
# "Available planets" means planets with a manifest in --manifest-dir
# (written by DiscoverPatchworkVisits). Raw-tree directory names are
# untrustworthy -- obsid-only folders, concatenated multi-target names,
# _nonsurvey / _index / _review / wave2_* staging dirs -- so this script
# never looks at the raw tree. If a planet is missing, run discover
# first (login node, it needs the archive):
#
#   source ~/aster/aster-env/bin/activate
#   cd ~/aster/maiea
#   python -m aster_toolkit.data_reduction.discover \
#       --raw-root /home/wasi/aster/maiea/workspace/mast/jwst_raw \
#       --manifest-dir ~/patchwork/manifests
#
# MODES
#   reduce (default)  STEPS=inspect,reduce   2 CPUs, per-target reduce sizing.
#   fit               STEPS=fit,combine FORCE_REFIT=1   4 CPUs, fit sizing.
#                     Refuses a target with no Stage 3 products. Passing
#                     the CAMPAIGN.md step-5 checklist (2 Stage 3 products
#                     per visit, segments_complete, identical
#                     exotedrf.path and crds_context) is still YOUR job --
#                     this only stops the obvious case of fitting an
#                     unreduced target.
#   contamination     STEPS=contamination    cheap; needs the combined spectrum.
#
# GUARDS
#   - Walltime and memory come from generate_survey_jobs.SIZING, the
#     single source of truth. Nothing here hand-writes them.
#   - A target whose patchwork_<slug> job is already PENDING/RUNNING is
#     skipped: two jobs writing one output directory corrupts outputs.
#   - Dry run by default; nothing is submitted without --submit.
#
# OPTIONS
#   --submit                 actually submit (default: print the plan only)
#   --mode MODE              reduce | fit | contamination   [reduce]
#   --wave 1,2               only these waves                [all]
#   --only SLUG,SLUG         only these targets, e.g. TOI_270_c
#   --skip SLUG,SLUG         skip these targets
#   --mail-user ADDR         SLURM BEGIN,END,FAIL mail
#   --manifest-dir DIR       [~/patchwork/manifests]
#   --output-root DIR        [~/scratch/patchwork]
#   --job-dir DIR            [~/patchwork/jobs]
#   --aster-repo DIR         [~/aster/maiea]
#   --aster-env DIR          [~/aster/aster-env]
#   -h, --help               this text

set -uo pipefail

MANIFEST_DIR="$HOME/patchwork/manifests"
OUTPUT_ROOT="$HOME/scratch/patchwork"
JOB_DIR="$HOME/patchwork/jobs"
ASTER_REPO="$HOME/aster/maiea"
ASTER_ENV="$HOME/aster/aster-env"
MODE="reduce"
WAVE=""
ONLY=""
SKIP=""
MAIL_USER=""
SUBMIT=0

usage() {
    # The header comment block, minus the shebang, up to the first line
    # of actual code -- so the help can never leak the script body.
    awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest-dir) MANIFEST_DIR="$2"; shift 2 ;;
        --output-root)  OUTPUT_ROOT="$2";  shift 2 ;;
        --job-dir)      JOB_DIR="$2";      shift 2 ;;
        --aster-repo)   ASTER_REPO="$2";   shift 2 ;;
        --aster-env)    ASTER_ENV="$2";    shift 2 ;;
        --mode)         MODE="$2";         shift 2 ;;
        --wave)         WAVE="$2";         shift 2 ;;
        --only)         ONLY="$2";         shift 2 ;;
        --skip)         SKIP="$2";         shift 2 ;;
        --mail-user)    MAIL_USER="$2";    shift 2 ;;
        --submit)       SUBMIT=1;          shift ;;
        -h|--help)      usage 0 ;;
        *) echo "Unknown argument: $1" >&2; usage 1 ;;
    esac
done

case "$MODE" in
    reduce)        STEPS="inspect,reduce"; CPUS=2; EXTRA_ENV="" ;;
    fit)           STEPS="fit,combine";    CPUS=4; EXTRA_ENV="FORCE_REFIT=1 " ;;
    contamination) STEPS="contamination";  CPUS=2; EXTRA_ENV="" ;;
    *) echo "Unknown --mode '$MODE' (reduce|fit|contamination)" >&2; exit 1 ;;
esac
# Stage 6.5 is seconds of emcee on ~50 channels, so it has no SIZING entry.
CONTAM_TIME="0:30:00"
CONTAM_MEM="8G"

MAIL=""
if [[ -n "$MAIL_USER" ]]; then
    MAIL=" --mail-user=$MAIL_USER --mail-type=BEGIN,END,FAIL"
fi

mkdir -p "$JOB_DIR" || exit 1

# --- plan: size every target and write its sbatch ---------------------
# aster_toolkit is only needed here. Activating inside the subshell keeps
# the caller's shell clean, which is what SLURM would otherwise inherit.
PLAN="$(mktemp)" || exit 1
trap 'rm -f "$PLAN"' EXIT

(
    # shellcheck disable=SC1091
    source "$ASTER_ENV/bin/activate" 2>/dev/null || {
        echo "Could not activate $ASTER_ENV -- pass --aster-env." >&2
        exit 1
    }
    cd "$ASTER_REPO" || exit 1
    MANIFEST_DIR="$MANIFEST_DIR" OUTPUT_ROOT="$OUTPUT_ROOT" JOB_DIR="$JOB_DIR" \
    python - <<'PY'
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts/patchwork")

from generate_survey_jobs import EXPECTED_DEPTH_PPM, SIZING
from aster_toolkit.data_reduction.survey import _slug, write_fir_slurm_script

manifest_dir = os.path.expanduser(os.environ["MANIFEST_DIR"])
output_root = os.path.expanduser(os.environ["OUTPUT_ROOT"])
job_dir = Path(os.path.expanduser(os.environ["JOB_DIR"]))

DEFAULT = ("6:00:00", "30:00:00", "48G", "24G", 4,
           "UNKNOWN TARGET - not in the sizing table, defaults used")

manifests = [m for m in sorted(glob.glob(os.path.join(manifest_dir, "*.json")))
             if "_unresolved" not in m]
if not manifests:
    print(f"NO_MANIFESTS\t{manifest_dir}")
    raise SystemExit(0)

for mpath in manifests:
    with open(mpath) as fh:
        manifest = json.load(fh)
    slug = _slug(manifest["planet_name"])
    red_t, fit_t, red_m, fit_m, wave, note = SIZING.get(slug, DEFAULT)
    script = write_fir_slurm_script(
        mpath, output_root, script_path=job_dir / f"run_{slug}.sbatch")
    depth = EXPECTED_DEPTH_PPM.get(slug)
    info = f"expect ~{depth} ppm" if depth else ""
    if note:
        info = f"{info}; {note}" if info else note
    # Tabs separate the fields, so scrub any from the free-text note.
    info = " ".join(str(info).split())
    print("\t".join([str(wave), slug, red_t, red_m, fit_t, fit_m,
                     str(script), info]))
PY
) > "$PLAN"
if [[ $? -ne 0 ]]; then
    echo "Job sizing failed -- nothing submitted." >&2
    exit 1
fi

if grep -q '^NO_MANIFESTS' "$PLAN"; then
    echo "No manifests in $(cut -f2 < "$PLAN"). Run discover first (login"
    echo "node -- it needs the archive); see the header of this script."
    exit 1
fi

# --- queued jobs, for the duplicate guard -----------------------------
QUEUED=""
if command -v squeue >/dev/null 2>&1; then
    QUEUED="$(squeue -u "$USER" -h -o '%j' 2>/dev/null)"
    if [[ $? -ne 0 ]]; then
        echo "WARNING: squeue failed; the duplicate-job guard is OFF." >&2
        QUEUED=""
    fi
fi

in_list() {  # in_list <needle> <comma-separated haystack>
    case ",$2," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

if [[ "$SUBMIT" -eq 1 ]]; then LABEL="SUBMITTING"
else LABEL="DRY RUN (pass --submit to run)"; fi

N_PLAN=0
SKIPPED=""
CURRENT_WAVE=""
FAILED=0
OUT=""

# Wave order, then slug: cheap validation targets go first.
while IFS=$'\t' read -r wave slug red_t red_m fit_t fit_m script info; do
    [[ -z "${slug:-}" ]] && continue
    if [[ -n "$ONLY" ]] && ! in_list "$slug" "$ONLY"; then continue; fi
    if [[ -n "$SKIP" ]] && in_list "$slug" "$SKIP"; then continue; fi
    if [[ -n "$WAVE" ]] && ! in_list "$wave" "$WAVE"; then continue; fi

    if [[ -n "$QUEUED" ]] && printf '%s\n' "$QUEUED" | grep -Fxq "patchwork_$slug"; then
        SKIPPED="$SKIPPED
#   $slug: a patchwork_$slug job is already queued/running -- two jobs must never write one output directory."
        continue
    fi
    if [[ "$MODE" == "fit" ]]; then
        if [[ -z "$(find "$OUTPUT_ROOT/$slug/reductions" -name '*_spectra_fullres.fits' -print -quit 2>/dev/null)" ]]; then
            SKIPPED="$SKIPPED
#   $slug: no Stage 3 products under $OUTPUT_ROOT/$slug/reductions -- run and VERIFY the reduce first."
            continue
        fi
    fi
    if [[ "$MODE" == "contamination" ]]; then
        if ! ls "$OUTPUT_ROOT/$slug/combined"/combined_*_transmission_spectrum.csv >/dev/null 2>&1; then
            SKIPPED="$SKIPPED
#   $slug: no combined spectrum under $OUTPUT_ROOT/$slug/combined -- run the fit and combine first."
            continue
        fi
    fi

    case "$MODE" in
        reduce)        TIME="$red_t"; MEM="$red_m" ;;
        fit)           TIME="$fit_t"; MEM="$fit_m" ;;
        contamination) TIME="$CONTAM_TIME"; MEM="$CONTAM_MEM" ;;
    esac

    CMD="STEPS=$STEPS ${EXTRA_ENV}sbatch --export=ALL --time=$TIME --cpus-per-task=$CPUS --mem=$MEM$MAIL $script"

    if [[ "$wave" != "$CURRENT_WAVE" ]]; then
        if [[ -n "$OUT" ]]; then OUT="$OUT
"; fi
        OUT="$OUT
# ---- wave $wave ----"
        CURRENT_WAVE="$wave"
    fi
    if [[ -n "$info" ]]; then
        OUT="$OUT

# $slug  ($info)
$CMD"
    else
        OUT="$OUT

# $slug
$CMD"
    fi
    N_PLAN=$((N_PLAN + 1))

    if [[ "$SUBMIT" -eq 1 ]]; then
        RESULT="$(env -i HOME="$HOME" USER="$USER" bash -lc "$CMD" 2>&1)"
        RC=$?
        [[ -n "$RESULT" ]] && OUT="$OUT
    $RESULT"
        if [[ $RC -ne 0 ]]; then
            OUT="$OUT
    SUBMISSION FAILED for $slug -- stopping so the failure is not repeated across every remaining target."
            FAILED=1
            break
        fi
    fi
done < <(sort -k1,1n -k2,2 "$PLAN")

echo "# Patchwork $MODE -- $N_PLAN target(s) -- $LABEL"
echo "# sbatch files in $JOB_DIR; logs land in the directory you run this from."
if [[ "$MODE" == "reduce" ]]; then
    echo "# Submit the fit only after each target passes the CAMPAIGN.md reduce checklist (step 5)."
fi
printf '%s\n' "$OUT"

if [[ -n "$SKIPPED" ]]; then
    N_SKIP="$(printf '%s' "$SKIPPED" | grep -c '^#   ')"
    echo ""
    echo "# Skipped $N_SKIP:"
    printf '%s\n' "$SKIPPED" | sed '/^$/d'
fi

if [[ "$FAILED" -eq 1 ]]; then
    exit 1
fi
if [[ "$SUBMIT" -eq 1 && "$N_PLAN" -gt 0 ]]; then
    echo ""
    echo "# $N_PLAN job(s) submitted. Watch with:"
    echo "#   squeue -u \$USER -o '%.10i %.30j %.8T %.10M %R'"
fi
exit 0
