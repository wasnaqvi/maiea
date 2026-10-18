#!/bin/bash
# Discover and REDUCE the batch-3 planets (HD 15337 b/c, HIP 9618 b,
# TOI-4336 A b, TOI-451 c/d) on DRAC Fir. Run from a CLEAN login shell
# (no venv active):
#
#   chmod +x scripts/patchwork/submit_batch3.sh
#   ./scripts/patchwork/submit_batch3.sh             # discover + dry-run plan
#   ./scripts/patchwork/submit_batch3.sh --submit    # ...then submit STEPS=inspect,reduce
#
# Steps:
#   1. Discovery on mast/jwst_raw/batch3_download ONLY, into its own manifest
#      directory (~/patchwork/manifests_batch3). The survey manifests in
#      ~/patchwork/manifests are never rewritten, so the TOI-125 b/c
#      overrides and the TOI-270 b fix are untouched.
#   2. STOPS if any visit is unresolved. Read the discovery table first: the
#      two TOI-402 visits and the two TOI-451 visits are where a wrong planet
#      would give a confidently wrong depth. Never guess; supply an override:
#        OVERRIDES="--override jw02512017001=<planet name>" ./submit_batch3.sh
#   3. Hands the new manifests to submit_survey.sh (reduce mode, 2 CPUs).
#
# Deliberately not handled:
#   - TOI-125 c (jw04126201001) is already a survey target with its own output
#     directory; it is skipped (SKIP, default TOI_125_c) so two jobs never
#     write one directory.
#   - TOI-2076 (BD+40-2790) is SUB1024B, outside the survey definition, so
#     discovery drops it.
#   - These targets are not in generate_survey_jobs.SIZING, so reduce uses the
#     DEFAULT_SIZING (6 h, 48G, 2 CPUs, wave 4). Add SIZING rows before FIT.
#
# After the reduce, run the CAMPAIGN.md step-5 checks: 2 Stage 3 products per
# visit, segments_complete true, exotedrf.path and crds_context identical to
# the rest of the survey. Fit only after that.
#
# Environment overrides: ASTER_REPO, ASTER_ENV, MANIFEST_DIR, SKIP, OVERRIDES.

set -uo pipefail

SUBMIT=0
for arg in "$@"; do
    case "$arg" in
        --submit) SUBMIT=1 ;;
        -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

ASTER_REPO="${ASTER_REPO:-$HOME/aster/maiea}"
ASTER_ENV="${ASTER_ENV:-$HOME/aster/aster-env}"
RAW_BATCH="$(readlink -f "$ASTER_REPO/workspace/mast/jwst_raw/batch3_download")"
MANIFEST_DIR="${MANIFEST_DIR:-$HOME/patchwork/manifests_batch3}"
SKIP="${SKIP:-TOI_125_c}"
OVERRIDES="${OVERRIDES:-}"

if [[ -z "$RAW_BATCH" || ! -d "$RAW_BATCH" ]]; then
    echo "batch3_download not found under $ASTER_REPO/workspace/mast/jwst_raw" >&2
    exit 1
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "A venv is active ($VIRTUAL_ENV). Run 'deactivate' first: SLURM would export it into the jobs." >&2
    exit 1
fi

echo "== Discovery: $RAW_BATCH -> $MANIFEST_DIR"
mkdir -p "$MANIFEST_DIR" || exit 1
rm -f "$MANIFEST_DIR/_unresolved.json"   # stale from a previous run
(
    # shellcheck disable=SC1091
    source "$ASTER_ENV/bin/activate" || exit 1
    cd "$ASTER_REPO" || exit 1
    # shellcheck disable=SC2086
    python -m aster_toolkit.data_reduction.discover \
        --raw-root "$RAW_BATCH" --manifest-dir "$MANIFEST_DIR" $OVERRIDES
) || { echo "Discovery failed." >&2; exit 1; }

if [[ -e "$MANIFEST_DIR/_unresolved.json" ]]; then
    echo >&2
    echo "UNRESOLVED visits in $MANIFEST_DIR/_unresolved.json. Fix with OVERRIDES and re-run;" >&2
    echo "never guess a planet name. Nothing submitted." >&2
    exit 1
fi

echo
echo "== Reduce plan (skipping: $SKIP)"
ARGS=(--manifest-dir "$MANIFEST_DIR" --skip "$SKIP" --mode reduce)
if [[ $SUBMIT -eq 1 ]]; then
    ARGS+=(--submit)
else
    echo "(dry run; re-run with --submit after reading the discovery table above)"
fi
exec "$ASTER_REPO/scripts/patchwork/submit_survey.sh" "${ARGS[@]}"
