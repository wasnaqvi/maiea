#!/usr/bin/env python
"""Submit Patchwork survey jobs for every available planet on DRAC Fir.

Run ON FIR from a login shell WITH the venv active (this script needs
aster_toolkit to size and write the sbatch files); each ``sbatch`` is
executed through ``env -i bash -lc`` so the job itself is submitted from
a CLEAN login environment, per the campaign compute discipline — SLURM
exports the submitting environment and ``module purge`` cannot undo an
activation.

    source ~/aster/aster-env/bin/activate
    cd ~/aster/maiea
    python scripts/patchwork/submit_survey.py                # dry run: show plan
    python scripts/patchwork/submit_survey.py --submit       # submit REDUCE, all waves
    python scripts/patchwork/submit_survey.py --submit --wave 1
    python scripts/patchwork/submit_survey.py --mode fit --submit --only TOI_270_c

"Available planets" means planets with a manifest in ``--manifest-dir``
(written by DiscoverPatchworkVisits). Raw-tree directory names are
untrustworthy — obsid-only folders, concatenated names, `_nonsurvey`,
`wave2_*` staging dirs — so this script never looks at the raw tree.
If a planet is missing, run discover first (login node, needs archive):

    python -m aster_toolkit.data_reduction.discover \
        --raw-root /home/wasi/aster/maiea/workspace/mast/jwst_raw \
        --manifest-dir ~/patchwork/manifests

Modes
-----
reduce (default)   STEPS=inspect,reduce  2 CPUs, per-target reduce sizing.
fit                STEPS=fit,combine FORCE_REFIT=1  4 CPUs, fit sizing.
                   Refuses per target unless Stage 3 products exist under
                   the output root. Passing the human verification
                   checklist (CAMPAIGN.md step 5: 2 Stage 3 products per
                   visit, segments_complete, identical exotedrf.path and
                   crds_context) is still YOUR job — the script only
                   stops the obvious case of fitting an unreduced target.
contamination      STEPS=contamination  cheap, needs the combined spectrum.

Guards
------
- Sizing comes from generate_survey_jobs.SIZING — the single source of
  truth. Nothing here hand-writes a walltime or memory request.
- A target with a patchwork_<slug> job already PENDING or RUNNING is
  skipped: two jobs writing one output directory corrupts outputs.
- Dry run by default; nothing is submitted without --submit.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parent))

from generate_survey_jobs import EXPECTED_DEPTH_PPM, SIZING  # noqa: E402
from aster_toolkit.data_reduction.survey import (  # noqa: E402
    _slug,
    write_fir_slurm_script,
)

# (steps, cpus, uses_reduce_sizing, extra_env)
MODES = {
    "reduce": ("inspect,reduce", 2, True, ""),
    "fit": ("fit,combine", 4, False, "FORCE_REFIT=1 "),
    "contamination": ("contamination", 2, False, ""),
}
# Stage 6.5 is seconds of emcee on ~50 channels; no entry in SIZING.
CONTAM_TIME, CONTAM_MEM = "0:30:00", "8G"
DEFAULT_SIZING = ("6:00:00", "30:00:00", "48G", "24G", 4,
                  "UNKNOWN TARGET — not in the sizing table, defaults used")


def queued_job_names() -> set[str]:
    """Names of the user's PENDING/RUNNING jobs ('' if squeue is absent,
    e.g. when dry-running this script off-cluster)."""
    if shutil.which("squeue") is None:
        return set()
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"WARNING: could not query squeue ({exc}); duplicate-job "
              "guard is OFF for this run.")
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def has_stage3_products(output_root: str, slug: str) -> bool:
    return bool(glob.glob(os.path.join(
        output_root, slug, "reductions", "**", "*_spectra_fullres.fits"),
        recursive=True))


def has_combined_spectrum(output_root: str, slug: str) -> bool:
    return bool(glob.glob(os.path.join(
        output_root, slug, "combined", "combined_*_transmission_spectrum.csv")))


def submit_clean(command: str, *, dry_run: bool) -> bool:
    """Run one sbatch command from a clean login environment.

    ``env -i HOME USER bash -lc`` is the documented wrapper: it drops the
    active venv (which SLURM would otherwise export into the job) while a
    login shell still provides the user's modules and PATH.
    """
    wrapped = ["env", "-i", f"HOME={os.environ['HOME']}",
               f"USER={os.environ.get('USER', '')}",
               "bash", "-lc", command]
    if dry_run:
        return True
    result = subprocess.run(wrapped, capture_output=True, text=True)
    for stream in (result.stdout, result.stderr):
        if stream.strip():
            print(f"    {stream.strip()}")
    return result.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Submit Patchwork jobs for all planets with a manifest.")
    ap.add_argument("--manifest-dir", default="~/patchwork/manifests")
    ap.add_argument("--output-root", default="~/scratch/patchwork")
    ap.add_argument("--job-dir", default="~/patchwork/jobs")
    ap.add_argument("--mode", choices=sorted(MODES), default="reduce")
    ap.add_argument("--wave", default="",
                    help="Comma-separated waves to include (default: all). "
                         "Waves come from the sizing table: 1=validation, "
                         "2=single-visit, 3=multi-visit, 4=large/decision.")
    ap.add_argument("--only", default="",
                    help="Comma-separated slugs to include (e.g. TOI_270_c).")
    ap.add_argument("--skip", default="", help="Comma-separated slugs to skip.")
    ap.add_argument("--mail-user", default="")
    ap.add_argument("--submit", action="store_true",
                    help="Actually submit. Without it, print the plan only.")
    args = ap.parse_args()

    manifest_dir = Path(os.path.expanduser(args.manifest_dir))
    job_dir = Path(os.path.expanduser(args.job_dir))
    output_root = os.path.expanduser(args.output_root)
    job_dir.mkdir(parents=True, exist_ok=True)

    manifests = sorted(glob.glob(str(manifest_dir / "*.json")))
    manifests = [m for m in manifests if "_unresolved" not in m]
    if not manifests:
        print(f"No manifests in {manifest_dir}. Run discover first (login "
              "node — it needs the archive); see the module docstring.")
        return 1

    waves = ({int(w) for w in args.wave.split(",") if w.strip()}
             if args.wave else set())
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    steps, cpus, use_reduce_sizing, extra_env = MODES[args.mode]
    mail = (f" --mail-user={args.mail_user} --mail-type=BEGIN,END,FAIL"
            if args.mail_user else "")
    queued = queued_job_names()

    # (wave, slug, command, note) per target, submitted in wave order.
    plan: list[tuple[int, str, str, str]] = []
    skipped: list[str] = []
    for mpath in manifests:
        with open(mpath) as fh:
            manifest = json.load(fh)
        slug = _slug(manifest["planet_name"])
        if (only and slug not in only) or slug in skip:
            continue
        sizing = SIZING.get(slug, DEFAULT_SIZING)
        red_t, fit_t, red_m, fit_m, wave, note = sizing
        if waves and wave not in waves:
            continue

        if f"patchwork_{slug}" in queued:
            skipped.append(f"{slug}: a patchwork_{slug} job is already "
                           "queued/running — two jobs must never write one "
                           "output directory.")
            continue
        if args.mode == "fit" and not has_stage3_products(output_root, slug):
            skipped.append(f"{slug}: no Stage 3 products under "
                           f"{output_root}/{slug}/reductions — run and "
                           "VERIFY the reduce first.")
            continue
        if (args.mode == "contamination"
                and not has_combined_spectrum(output_root, slug)):
            skipped.append(f"{slug}: no combined spectrum under "
                           f"{output_root}/{slug}/combined — run the fit "
                           "and combine first.")
            continue

        script = write_fir_slurm_script(
            mpath, output_root, script_path=job_dir / f"run_{slug}.sbatch")
        if args.mode == "contamination":
            time, mem = CONTAM_TIME, CONTAM_MEM
        else:
            time, mem = ((red_t, red_m) if use_reduce_sizing
                         else (fit_t, fit_m))
        cmd = (f"STEPS={steps} {extra_env}sbatch --export=ALL --time={time} "
               f"--cpus-per-task={cpus} --mem={mem}{mail} {script}")
        depth = EXPECTED_DEPTH_PPM.get(slug)
        info = (f"expect ~{depth} ppm" if depth else "")
        if note:
            info = f"{info}; {note}" if info else note
        plan.append((wave, slug, cmd, info))

    plan.sort()
    label = "SUBMITTING" if args.submit else "DRY RUN (pass --submit to run)"
    print(f"# Patchwork {args.mode} — {len(plan)} target(s) — {label}")
    print(f"# sbatch files in {job_dir}; logs land in the CWD you run "
          "this from.")
    if args.mode == "reduce":
        print("# Submit the fit only after each target passes the "
              "CAMPAIGN.md reduce checklist (step 5).")

    failures = 0
    current_wave = None
    for wave, slug, cmd, info in plan:
        if wave != current_wave:
            print(f"\n# ---- wave {wave} ----")
            current_wave = wave
        print(f"\n# {slug}" + (f"  ({info})" if info else ""))
        print(cmd)
        if not submit_clean(cmd, dry_run=not args.submit):
            failures += 1
            print(f"    SUBMISSION FAILED for {slug} — stopping so the "
                  "failure is not repeated across every remaining target.")
            break

    if skipped:
        print(f"\n# Skipped {len(skipped)}:")
        for line in skipped:
            print(f"#   {line}")
    if failures:
        return 1
    if args.submit and plan:
        print(f"\n# {len(plan)} job(s) submitted. Watch with: "
              "squeue -u $USER -o '%.10i %.30j %.8T %.10M %R'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
