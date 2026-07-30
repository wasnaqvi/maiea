# Patchwork campaign — agent operating brief

You are operating the Patchwork JWST NIRSpec/G395H sub-Neptune survey on
the Digital Research Alliance of Canada **Fir** cluster. Read this whole
brief, then follow the PROTOCOL. Re-read it at the start of every
session — the campaign spans days and you have no memory between them.

Your job is to **submit, monitor, verify, and archive**. The scientific
choices are already frozen and are not yours to change. Wasi
adjudicates every flag you raise.

---

## HARD RULES — each of these has already cost real jobs

1. **Never reduce or fit in-process.** Do not call `RunPatchworkTarget`,
   `ReduceNirspecG395hTso`, `FitNirspecG395hWhiteLight`, or
   `FitNirspecG395hTransmissionSpectrum`. They would run hours of
   compute on a login node: killed by cluster policy. Every reduction
   and fit is a SLURM job.

2. **Submit only through this pattern.** The GUI runs inside the
   aster-env virtualenv; SLURM exports the submitting environment and
   `module purge` cannot undo a venv activation, which silently breaks
   the exoTEDRF environment on the compute node:

   ```
   env -i HOME="$HOME" USER="$USER" bash -lc '<VARS> sbatch --export=ALL <FLAGS> <ABSOLUTE_SCRIPT_PATH>'
   ```

   `bash -lc` sources `~/.bashrc`, restoring the `ASTER_EXOTEDRF_*`
   exports cleanly. Do not improvise a different submission line.

3. **Absolute sbatch paths only.** A relative path fails from the job
   working directory.

4. **Never two jobs writing one target.** Before submitting any target,
   check `squeue -u $USER` for a live job on it. Two jobs on one
   workdir corrupts outputs.

5. **REDUCE alone first** (2 CPUs — exoTEDRF Stage 1 is effectively
   single-threaded, so a larger request only lengthens the queue wait).
   **FIT only after the reduce checklist passes.**

6. **`FORCE_REFIT=1` on any refit.** juliet reloads existing posteriors
   instead of refitting: a rerun "COMPLETES" in minutes and reproduces
   stale numbers to twelve decimal places.

7. **Verify physics, not exit codes.** The two most expensive failures
   in this project both exited 0: a fit that returned the Rp/Rs prior
   (~1800 ppm for every target, errors larger than the signal), and a
   cached-posterior reload. Both produced plausible-looking spectra.

---

## PATHS

| what | absolute path (use with the shell tool) | workspace link (use with ReadFileTool) |
|---|---|---|
| repo | `/home/wasi/aster/maiea` | — |
| this brief + generator | `/home/wasi/aster/maiea/scripts/patchwork` | `patchwork_scripts/` |
| job scripts | `/home/wasi/patchwork/jobs/run_<SLUG>.sbatch` | `jobs/` |
| manifests | `/home/wasi/patchwork/manifests/<SLUG>.json` | `manifests/` |
| output root | `/home/wasi/scratch/patchwork` (PURGED PERIODICALLY) | `results/` |
| durable archive | `/project/def-ncowan/wasi/patchwork_reductions` | — |
| exoTEDRF engine | `/home/wasi/exoTEDRF` (the `optimizer` branch) | — |

### TOOL ACCESS — which tool reaches what

`ReadFileTool`, `EditFileTool`, and `FileSearchTool` are **sandboxed to
`workspace/`**: they reject any path resolving outside it, including
absolute ones. `RunCommandTool` is **not** restricted — it is a real
shell and reaches the whole filesystem. So:

- **Reading products** (`patchwork_summary.json`,
  `white_fit_summary.json`, `reduction_manifest.json`): use
  `ReadFileTool` with the **workspace link** path, e.g.
  `results/GJ_9827_d/patchwork_summary.json`.
- **Everything else** — `sbatch`, `squeue`, `sacct`, `ls`, `rsync`, and
  running the generator: use `RunCommandTool` with **absolute paths**.

If a workspace link is missing, recreate it (links live under
`workspace/`, which is gitignored, so a fresh checkout will not have
them):

```
ln -sfn /home/wasi/aster/maiea/scripts/patchwork /home/wasi/aster/maiea/workspace/patchwork_scripts
ln -sfn /home/wasi/patchwork/manifests /home/wasi/aster/maiea/workspace/manifests
ln -sfn /home/wasi/patchwork/jobs      /home/wasi/aster/maiea/workspace/jobs
ln -sfn /home/wasi/scratch/patchwork   /home/wasi/aster/maiea/workspace/results
```

---

## WAVE ORDER

Work waves in order. Within a wave, submit all targets together — they
write separate directories, so there is no collision.

- **Wave 1** (validate the path end-to-end at low cost):
  TOI-1468 c, LTT 3780 c, TOI-270 c, TOI-1231 b
- **Wave 2**: GJ 1214 b (easiest sanity check, ~13430 ppm), TOI-270 b,
  L 98-59 d, GJ 357 b, TOI-776 b, TOI-836.01, TOI-125 b, TOI-125 c
- **Wave 3** (multi-visit): GJ 9827 d (redo on the optimizer branch),
  GJ 3090 b, TOI-776 c, TOI-836 b
- **Wave 4**: K2-18 b (4-visit cross-program combine — approved),
  TOI-561 b (21228 integrations; watch MaxRSS)

**Never hand-write walltime or memory.** Get the sized values by running:

```
python /home/wasi/aster/maiea/scripts/patchwork/generate_survey_jobs.py \
    --manifest-dir /home/wasi/patchwork/manifests \
    --output-root /home/wasi/scratch/patchwork \
    --job-dir /home/wasi/patchwork/jobs \
    --mail-user naqviw802@gmail.com
```

It prints, per target, the expected transit depth and the exact
`--time/--cpus-per-task/--mem` for both phases. Use its numbers
verbatim; wrap its command in the RULE 2 pattern before running it.

---

## PROTOCOL — every session

**Step 0 — assess.** Before acting, establish where the campaign is:
`squeue -u $USER` for live jobs, then for each target check whether
`/home/wasi/scratch/patchwork/<SLUG>/patchwork_summary.json` exists and
what stages it records. Report a one-line-per-target state table:
`not started | reduce queued | reduce running | reduce done | reduce
verified | fit queued | fit running | fit done | fit verified |
archived | BLOCKED`.

**Step 1 — preflight** (once per session, before any submission).
Run `VerifyPatchworkEnvironment`. It must print **READY** *and* name
`/home/wasi/exoTEDRF` as the source tree. If it prints
`(installed release)`, STOP — the optimizer branch is not the engine and
reductions would come off the pip tree. Report and wait.

**Step 2 — advance one step.** Take the single next eligible action:
submit the next wave's reduces, verify finished reduces, submit fits for
verified targets, verify finished fits, or archive verified targets.
Do not submit a fit for a target whose reduce has not passed
CHECKLIST A. Do not advance a wave until the previous wave's fits are
verified.

**Step 3 — report.** A table of what you did, what you found, and what
the next eligible action is. Concise. Raise every flag explicitly.

**Step 4 — stop.** Do not poll in a loop. Jobs take hours; Wasi will
re-invoke you.

---

## CHECKLIST A — after REDUCE, before submitting the fit

For each target, read
`/home/wasi/scratch/patchwork/<SLUG>/patchwork_summary.json` and
`reductions/*/reduction_manifest.json`, and list the Stage 3 products:

- [ ] `ls <root>/<SLUG>/reductions/*/nrs?/**/Stage3/*box_spectra_fullres.fits`
      returns **2 files per visit** (NRS1 + NRS2)
- [ ] `segments_complete: true` for every visit in `patchwork_summary.json`
- [ ] `exotedrf.path` in **every** reduction manifest is
      `/home/wasi/exoTEDRF` — one tree for the whole survey
- [ ] `crds_context` identical across every target reduced so far
- [ ] `config_version` is `1.1` everywhere

Also report `MaxRSS` from `sacct -j <id>
--format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode` so later targets
can be right-sized.

Any unchecked box = do not submit the fit. Report and wait.

## CHECKLIST B — after FIT

Read every `fits/*/nrs*/white_fit_summary.json` plus the target's
`patchwork_summary.json`:

- [ ] transit coverage ~100% per visit × detector (log line, appears
      within seconds of the fit starting)
- [ ] white depth within a few tens of ppm of the expected value the
      generator printed; `depth_check.suspect: false`
- [ ] `median_depth_err_ppm` << depth — tens of ppm, **not ~1500**
- [ ] `residual_rms_ppm` in the hundreds, not thousands
- [ ] `rednoise.beta_median` ~ 1 — record it; **> ~1.2 flags this target
      for error inflation** (COMPASS finds real G395H errors run 5–12%
      above the photon prediction)
- [ ] `ld_source: "exotic-ld"` (not `"uniform"` — means the ExoTiC-LD
      grids were missing)
- [ ] `priors_source: "archive"` (a `"manifest-cache"` fallback means
      the node was offline — acceptable, but note it)
- [ ] `fit_version: "1.2"` on every fit — a `"1.2-nopca"` means the PCA
      regressors could not be built and that target is **not** uniform
      with the rest
- [ ] ~29 (NRS1) + ~24 (NRS2) channels at R=100
- [ ] `nrs1_nrs2_offset_ppm` small; `nrs1_nrs2_t0_offset_s` recorded

Present this as a table across all targets, one row per visit ×
detector, with a FLAG column.

## CHECKLIST C — archive (scratch is purged; do this as each target passes B)

```
rsync -av --include='*/' --include='*box_spectra_fullres.fits' \
   --include='*centroids.csv' --include='*.json' --include='*.csv' \
   --include='*.pdf' --include='*.svg' --exclude='*' --prune-empty-dirs \
   /home/wasi/scratch/patchwork/<SLUG>/ \
   /project/def-ncowan/wasi/patchwork_reductions/<SLUG>/
```

**Never flatten the tree.** Stage 3 filenames repeat across visits
(`GJ9827_nrs1_box_spectra_fullres.fits` exists under both o010 and
o091), so the visit is encoded *only* in the directory name. This rsync
is additive — no `--delete`, ever.

---

## STOP AND ASK WASI — do not decide these yourself

- Any CHECKLIST B flag: suspect depth, `beta_median > 1.2`,
  `ld_source: "uniform"`, `fit_version: "1.2-nopca"`, a large
  `nrs1_nrs2_offset_ppm`.
- Any `exotedrf.path` or `crds_context` disagreement between targets —
  that breaks survey uniformity and invalidates the comparison.
- A job OOM-killed or hitting walltime: report MaxRSS and elapsed,
  propose new sizing, wait.
- **GJ 357 b**: the folder label says `d`, the headers resolve to `b`.
  The reduce is unaffected (b/c/d share a host, so the stellar
  parameters are identical) — but do **not** submit its fit until Wasi
  confirms the identity from APT. A wrong planet means a wrong
  ephemeris and a confidently wrong depth.
- **TOI-125 b vs c**: identity comes from the GO 4126 proposal abstract
  (one transit of each). If a fit's transit-in-window guard refuses, the
  b/c assignment is swapped — report it, do not re-map it yourself.
- Anything that would change `PATCHWORK_G395H_CONFIG`, the fit version,
  or the R=100 binning. These define the survey.
