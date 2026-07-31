# Patchwork campaign — agent operating brief

**READ THIS WHOLE FILE FIRST.** Your file reader returns ~100 lines per
call and this file is longer; call it again with an offset until you
reach the end. Do not act on a partial read or claim a full one.

**Do not ask permission to execute the PROTOCOL** — it is a standing
instruction from Wasi. Execute the next eligible action and report.
Ask only when a STOP condition below is met.

**Never invent a result.** Empty output means the tool failed, not that
the answer is nothing. Say "command returned no output" and stop.

You operate the Patchwork JWST NIRSpec/G395H sub-Neptune survey on DRAC
**Fir**: submit, monitor, verify, archive. The science is frozen; Wasi
adjudicates every flag you raise.

## HARD RULES — each has already cost real jobs

1. **Never reduce or fit in-process.** Do not call `RunPatchworkTarget`,
   `ReduceNirspecG395hTso`, `FitNirspecG395hWhiteLight`, or
   `FitNirspecG395hTransmissionSpectrum`. They would run hours of compute
   on a login node and be killed. Everything is a SLURM job.

2. **Submit only through this pattern.** The GUI runs inside a virtualenv;
   SLURM exports the submitting environment and `module purge` cannot undo
   an activation, which breaks exoTEDRF on the compute node:

   ```
   env -i HOME="$HOME" USER="$USER" bash -lc '<VARS> sbatch --export=ALL <FLAGS> <ABSOLUTE_SCRIPT>'
   ```

   **`env -i` is ONLY for `sbatch`.** It strips PATH, so `python` inside
   it has no aster-env and no numpy. The generator, `squeue`, `sacct`,
   `ls`, `rsync` all run as PLAIN commands — wrapping them breaks them.

3. **Absolute sbatch paths only.**

4. **Never two jobs on one target.** Check `squeue -u $USER` first.

5. **REDUCE alone first** (2 CPUs). **FIT only after CHECKLIST A passes.**

6. **`FORCE_REFIT=1` on any refit.** juliet reloads old posteriors and
   "completes" in minutes with stale numbers.

7. **Verify physics, not exit codes.** The two worst failures here both
   exited 0: a fit returning the Rp/Rs prior (~1800 ppm every target,
   errors exceeding the signal), and a cached-posterior reload.

## PATHS

`RunCommandTool` is unrestricted — **absolute paths**. `ReadFileTool` is
sandboxed to `workspace/` — use the **link** column.

| what | absolute | link |
|---|---|---|
| repo / briefs / generator | `/home/wasi/aster/maiea` (`scripts/patchwork`) | `patchwork_scripts/` |
| job scripts | `/home/wasi/patchwork/jobs/run_<SLUG>.sbatch` | `jobs/` |
| manifests | `/home/wasi/patchwork/manifests/<SLUG>.json` | `manifests/` |
| output root | `/home/wasi/scratch/patchwork` (PURGED) | `results/` |
| archive | `/project/def-ncowan/wasi/patchwork_reductions` | — |
| exoTEDRF engine | `/home/wasi/exoTEDRF` (`optimizer` branch) | — |

## PROTOCOL — run this every session

**Step 0 — assess.** Run each, report RAW output; if any returns nothing,
stop and say so:

```
squeue -u $USER
ls -l /home/wasi/patchwork/manifests/
ls -d /home/wasi/scratch/patchwork/*/ 2>/dev/null
```

Then a one-line-per-target state table, each target in one of: `not
started | reduce queued/running/done/verified | fit
queued/running/done/verified | archived | BLOCKED`.

**Step 1 — preflight** (once per session, before any submission). Run the
`VerifyPatchworkEnvironment` tool. It must print **READY** *and* name
`/home/wasi/exoTEDRF` as the source. If it names the installed release,
STOP and report — the optimizer branch is not the engine.

**Step 2 — get the sizing.** Never hand-write walltime or memory. Run
this as a PLAIN command — no `env -i` wrapper, it needs aster-env:

```
python /home/wasi/aster/maiea/scripts/patchwork/generate_survey_jobs.py --manifest-dir /home/wasi/patchwork/manifests --output-root /home/wasi/scratch/patchwork --job-dir /home/wasi/patchwork/jobs --mail-user naqviw802@gmail.com
```

It prints the wave order, each target's expected depth, and the exact
`--time/--cpus-per-task/--mem` per phase — use its numbers verbatim. Work
waves in order; within a wave submit all targets together.

**Step 3 — advance ONE step.** Exactly one of: submit the next wave's
reduces; verify finished reduces (CHECKLIST A); submit fits for verified
targets; verify finished fits (CHECKLIST B); archive verified targets
(CHECKLIST C). Never fit a target that has not passed CHECKLIST A; never
start a wave before the previous wave's fits verify.

**Step 4 — report and stop.** Table of what you did, what you found, the
next eligible action. Do not poll in a loop; jobs take hours and Wasi
will re-invoke you.

**Reporting rules.** Quote raw command output rather than paraphrasing
it. If a command returns nothing, say so — never describe a directory as
empty, or a queue as clear, on the basis of blank output. State job IDs
explicitly. When a check fails, quote the last 20 lines of the relevant
`reduction.log` rather than summarising the error.

## STOP AND ASK WASI

- Any CHECKLIST B flag: `depth_check.suspect` true, `beta_median > 1.2`,
  `ld_source: "uniform"`, `fit_version: "1.2-nopca"`, large
  `nrs1_nrs2_offset_ppm`.
- Any `exotedrf.path` or `crds_context` disagreement between targets —
  that breaks survey uniformity.
- A job OOM-killed or hitting walltime: report MaxRSS and elapsed,
  propose new sizing, wait.
- **GJ 357 b**: folder label says `d`, headers resolve to `b`. The reduce
  is unaffected (b/c/d share a host, identical stellar parameters) — but
  do NOT submit its fit until Wasi confirms identity from APT.
- **TOI-125 b vs c**: identity comes from the GO 4126 abstract. If a
  fit's transit-in-window guard refuses, the assignment is swapped —
  report it, do not re-map it yourself.
- Anything that would change `PATCHWORK_G395H_CONFIG`, the fit version,
  or the R=100 binning. These define the survey.

---
---

# CHECKLISTS (read when you reach Step 3)

## CHECKLIST A — after REDUCE, before submitting the fit

Read `results/<SLUG>/patchwork_summary.json` and
`results/<SLUG>/reductions/*/reduction_manifest.json` with `ReadFileTool`.

- [ ] `ls /home/wasi/scratch/patchwork/<SLUG>/reductions/*/nrs?/**/Stage3/*box_spectra_fullres.fits`
      returns **2 files per visit** (NRS1 + NRS2)
- [ ] `segments_complete: true` for every visit
- [ ] `exotedrf.path` is `/home/wasi/exoTEDRF` in **every** manifest —
      one tree for the whole survey
- [ ] `crds_context` identical across every target reduced so far
- [ ] `config_version` is `1.1` everywhere

Report `MaxRSS` from
`sacct -j <id> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode`
so later targets can be right-sized.

Any unchecked box: do not submit the fit. Report and wait.

## CHECKLIST B — after FIT

Read every `results/<SLUG>/fits/*/nrs*/white_fit_summary.json` plus the
target's `patchwork_summary.json`. One row per visit × detector, with a
FLAG column:

- [ ] transit coverage ~100% (log line, seconds into the fit)
- [ ] white depth within tens of ppm of the generator's expected value;
      `depth_check.suspect: false`
- [ ] `median_depth_err_ppm` << depth — tens of ppm, **not ~1500**
- [ ] `residual_rms_ppm` in the hundreds, not thousands
- [ ] `rednoise.beta_median` ~ 1 — record it; **> ~1.2 flags this target
      for error inflation** (COMPASS: real G395H errors run 5-12% above
      the photon prediction)
- [ ] `ld_source: "exotic-ld"` (not `"uniform"` — grids were missing)
- [ ] `priors_source: "archive"` (`"manifest-cache"` means the node was
      offline: acceptable, but note it)
- [ ] `fit_version: "1.2"` — `"1.2-nopca"` means the PCA regressors could
      not be built and that target is NOT uniform with the rest
- [ ] ~29 (NRS1) + ~24 (NRS2) channels at R=100
- [ ] `nrs1_nrs2_offset_ppm` small; `nrs1_nrs2_t0_offset_s` recorded

## CHECKLIST C — archive (scratch is purged; do this as each target passes B)

```
rsync -av --include='*/' --include='*box_spectra_fullres.fits' \
   --include='*centroids.csv' --include='*.json' --include='*.csv' \
   --include='*.pdf' --include='*.svg' --exclude='*' --prune-empty-dirs \
   /home/wasi/scratch/patchwork/<SLUG>/ \
   /project/def-ncowan/wasi/patchwork_reductions/<SLUG>/
```

**Never flatten the tree.** Stage 3 filenames repeat across visits
(`GJ9827_nrs1_box_spectra_fullres.fits` exists under both o010 and o091),
so the visit is encoded *only* in the directory name. This rsync is
additive — no `--delete`, ever.

## If a workspace link is missing

Links live under `workspace/`, which is gitignored, so a fresh checkout
will not have them:

```
ln -sfn /home/wasi/aster/maiea/scripts/patchwork /home/wasi/aster/maiea/workspace/patchwork_scripts
ln -sfn /home/wasi/patchwork/manifests /home/wasi/aster/maiea/workspace/manifests
ln -sfn /home/wasi/patchwork/jobs      /home/wasi/aster/maiea/workspace/jobs
ln -sfn /home/wasi/scratch/patchwork   /home/wasi/aster/maiea/workspace/results
```

END OF BRIEF — if you have read this line, you have the whole file.
