#!/usr/bin/env python
"""Generate sized sbatch scripts + submission commands for the full
Patchwork G395H survey on DRAC Fir.

Run ON FIR (login node), after `git pull`:

    source ~/aster/aster-env/bin/activate
    cd ~/aster/maiea
    python scripts/patchwork/generate_survey_jobs.py \
        --manifest-dir ~/patchwork/manifests \
        --output-root ~/scratch/patchwork \
        --job-dir ~/patchwork/jobs \
        --mail-user you@example.com

It writes one sbatch per target (via survey.write_fir_slurm_script) and
prints the submission commands in the recommended order, sized per
target from the measured ~20 min per visit x detector rate (session
2026-07-29). REDUCE is submitted first, on its own; FIT only after the
reduce products verify (see CAMPAIGN.md checklist).

Walltimes are ~2x the estimate. 2 CPUs for reduce (Stage 1 is
effectively single-threaded; bigger requests only lengthen queue waits
and burn fairshare).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aster_toolkit.data_reduction.survey import (  # noqa: E402
    _slug,
    write_fir_slurm_script,
)

# slug -> (reduce_walltime, fit_walltime, reduce_mem, fit_mem, wave, note)
# Waves: 1 = cheap single-visit validation, 2 = remaining single-visit,
# 3 = multi-visit, 4 = large / needs a human decision first.
#
# MEASURED (2026-07-30, Wave 1, Stage 3 only with Stages 1-2 cached):
#   TOI-1468 c  1278 ints  5.8 G  5m14
#   LTT 3780 c  1543 ints  7.0 G  5m14
#   TOI-270 c   1763 ints  7.7 G  5m26
#   TOI-1231 b  2726 ints 11.4 G  6m00
# => Stage 3 MaxRSS ~ 0.65 G + 4 MB per integration; Stage 3 itself is
# ~5 min. The reduce_mem values below are sized for a COLD run (Stages
# 1-3), which is why they exceed these numbers.
SIZING: dict[str, tuple[str, str, str, str, int, str]] = {
    "TOI_1468_c": ("2:00:00", "15:00:00", "48G", "24G", 1, ""),
    "LTT_3780_c": ("2:00:00", "15:00:00", "48G", "24G", 1, ""),
    "TOI_270_c":  ("2:00:00", "15:00:00", "48G", "24G", 1, ""),
    "TOI_1231_b": ("2:00:00", "15:00:00", "48G", "24G", 1, ""),
    "GJ_1214_b":  ("2:00:00", "15:00:00", "48G", "24G", 2,
                   "only visit o019 usable; 13430 ppm depth = easiest sanity check"),
    "TOI_270_b":  ("2:00:00", "15:00:00", "48G", "24G", 2, "below demographic"),
    "L_98_59_d":  ("2:00:00", "15:00:00", "48G", "24G", 2, "below demographic"),
    "GJ_357_b":   ("3:00:00", "15:00:00", "48G", "24G", 2,
                   "below demographic; folder label says d, headers say b — APT check"),
    "TOI_776_b":  ("2:00:00", "15:00:00", "48G", "24G", 2, ""),
    "TOI_836_01": ("4:00:00", "15:00:00", "48G", "24G", 2,
                   "keep the .01 designation — 'TOI-836 c' is not in pscomppars"),
    "GJ_9827_d":  ("3:00:00", "30:00:00", "48G", "24G", 3,
                   "REDO of the PyPI-reduced test run, now on the optimizer branch"),
    "GJ_3090_b":  ("3:00:00", "30:00:00", "48G", "24G", 3, ""),
    "TOI_776_c":  ("4:00:00", "30:00:00", "48G", "24G", 3, ""),
    "TOI_836_b":  ("6:00:00", "30:00:00", "48G", "24G", 3, ""),
    # Walltimes raised 2026-07-30 after both TIMED OUT during Stages 1-2
    # at the original 4 h / 12 h. Measured: every other target finished
    # Stages 1-2 in 27 min - 2 h; these two are the outliers by volume.
    "K2_18_b":    ("8:00:00", "2-12:00:00", "48G", "24G", 4,
                   "cross-program 4-visit combine APPROVED (Wasi 2026-07-30); "
                   "TIMED OUT at 4 h on the first attempt"),
    # Memory raised 64G -> 128G. MEASURED Stage 3 MaxRSS on Wave 1 scales
    # ~linearly with integration count (0.65 G + ~4 MB/integration over
    # 1278-2726 ints: 5.8/7.0/7.7/11.4 G). Extrapolated to this target's
    # 21228 integrations that is ~86 G — past the old 64 G request, and an
    # OOM would land AFTER up to 24 h of Stages 1-2. The extrapolation is
    # 8x beyond the measured range and may flatten if exoTEDRF chunks by
    # segment, but headroom is far cheaper than a lost day.
    "TOI_561_b":  ("24:00:00", "15:00:00", "128G", "32G", 4,
                   "21228 integrations in one visit; TIMED OUT at 12 h on the "
                   "first attempt; mem extrapolated from Wave 1 MaxRSS"),
    # GO 4126 observed TOI-125 b AND c, one transit each (proposal
    # abstract, ADS 2023jwst.prop.4126F). Manifest overrides:
    #   --override jw04126101001="TOI-125 b" --override jw04126201001="TOI-125 c"
    # If the b/c <-> obs 101/201 assignment is swapped, the fit-time
    # transit-in-window guard refuses loudly — then swap the overrides.
    "TOI_125_b":  ("3:00:00", "15:00:00", "48G", "24G", 2,
                   "ID from GO 4126 abstract; guard catches a b/c swap"),
    "TOI_125_c":  ("3:00:00", "15:00:00", "48G", "24G", 2,
                   "ID from GO 4126 abstract; guard catches a b/c swap"),
}

EXPECTED_DEPTH_PPM = {
    "TOI_561_b": 231, "TOI_836_b": 552, "K2_18_b": 2889, "TOI_776_c": 1183,
    "TOI_836_01": 1274, "GJ_3090_b": 1436, "GJ_9827_d": 957, "GJ_357_b": 955,
    "TOI_776_b": 999, "TOI_1231_b": 4914, "L_98_59_d": 2116, "TOI_270_c": 3136,
    "TOI_270_b": 942, "LTT_3780_c": 3352, "TOI_1468_c": 2767, "GJ_1214_b": 13430,
    # TOI-125 b/c: approximate, from 2.73/2.76 Re on a ~0.85 Rsun host —
    # the per-fit depth_check against the archive (Rp/Rs)^2 is authoritative.
    "TOI_125_b": 870, "TOI_125_c": 890,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default="~/patchwork/manifests")
    ap.add_argument("--output-root", default="~/scratch/patchwork")
    ap.add_argument("--job-dir", default="~/patchwork/jobs")
    ap.add_argument("--mail-user", default="")
    args = ap.parse_args()

    manifest_dir = Path(os.path.expanduser(args.manifest_dir))
    job_dir = Path(os.path.expanduser(args.job_dir))
    output_root = os.path.expanduser(args.output_root)
    job_dir.mkdir(parents=True, exist_ok=True)

    manifests = sorted(glob.glob(str(manifest_dir / "*.json")))
    manifests = [m for m in manifests if "_unresolved" not in m]
    if not manifests:
        print(f"No manifests in {manifest_dir}. Run discover.py first "
              "(login node — it needs the archive).")
        return 1

    mail = (f" --mail-user={args.mail_user} --mail-type=BEGIN,END,FAIL"
            if args.mail_user else "")

    by_wave: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}
    missing_priors = []
    for mpath in manifests:
        with open(mpath) as fh:
            manifest = json.load(fh)
        slug = _slug(manifest["planet_name"])
        if not manifest.get("priors"):
            missing_priors.append(slug)
        sizing = SIZING.get(slug)
        if sizing is None:
            sizing = ("6:00:00", "30:00:00", "48G", "24G", 4,
                      "UNKNOWN TARGET — not in the sizing table, defaults used")
        red_t, fit_t, red_m, fit_m, wave, note = sizing

        script = write_fir_slurm_script(
            mpath, output_root,
            script_path=job_dir / f"run_{slug}.sbatch",
        )
        depth = EXPECTED_DEPTH_PPM.get(slug)
        block = [f"# --- {manifest['planet_name']}"
                 + (f"  (expect ~{depth} ppm)" if depth else "")]
        if note:
            block.append(f"#     {note}")
        # STEPS is passed via the submitting shell's environment +
        # --export=ALL: SLURM's --export=NAME=VALUE splits on the comma
        # inside 'inspect,reduce' no matter how it is quoted.
        block.append(
            f"STEPS=inspect,reduce sbatch --export=ALL --time={red_t} "
            f"--cpus-per-task=2 --mem={red_m}{mail} {script}"
        )
        block.append(
            f"# after reduce verifies (CAMPAIGN.md checklist), submit the fit:"
        )
        block.append(
            f"# STEPS=fit,combine FORCE_REFIT=1 sbatch --export=ALL "
            f"--time={fit_t} --cpus-per-task=4 --mem={fit_m}{mail} {script}"
        )
        by_wave[wave].append("\n".join(block))

    print(f"# Patchwork survey submission plan — {len(manifests)} targets")
    print(f"# Job scripts in {job_dir}; logs land in the directory you "
          "sbatch from.")
    print("# Submit from a shell with NO venv active; use absolute paths.")
    if missing_priors:
        print(f"#\n# NOTE: {len(missing_priors)} manifest(s) have no cached "
              f"'priors' block ({', '.join(missing_priors)}).")
        print("#   Regenerate manifests on a login node so compute nodes can "
              "fall back offline:")
        print("#   python -m aster_toolkit.data_reduction.discover "
              "--raw-root /project/def-ncowan/wasi/jwst_raw "
              "--manifest-dir ~/patchwork/manifests")
    titles = {1: "WAVE 1 — cheap single-visit validation (run these first)",
              2: "WAVE 2 — remaining single-visit targets",
              3: "WAVE 3 — multi-visit targets",
              4: "WAVE 4 — large / human-decision targets"}
    for wave in (1, 2, 3, 4):
        if by_wave[wave]:
            print(f"\n# ================ {titles[wave]} ================")
            print("\n\n".join(by_wave[wave]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
