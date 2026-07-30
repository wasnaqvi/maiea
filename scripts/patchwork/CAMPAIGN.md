# Patchwork G395H survey campaign — runbook (2026-07-30)

Full uniform reduction of every G395H sub-Neptune on Fir (18 planets
with TOI-125 b & c, 27 usable visits), on the **exoTEDRF optimizer
branch** (`ASTER_EXOTEDRF_REPO=~/exoTEDRF`), with the frozen Patchwork
config v1.1 and **fit v1.2** (PCA detrending included). GJ 9827 d's
earlier PyPI-exotedrf reduction was a test run — it is REDONE here on
the optimizer branch for uniformity.

Survey decisions locked 2026-07-30 (Wasi):
- **K2-18 b**: combine all 4 visits across GO 2372 + GO 2722.
- **TOI-125**: include both planets. GO 4126 ("Comparative Atmospheric
  Chemistry Within One System") observed TOI-125 **b and c**, one
  transit each — override jw04126101001="TOI-125 b",
  jw04126201001="TOI-125 c". If the assignment is swapped the
  transit-in-window guard refuses at fit time; swap the overrides.
- **PCA detrending**: survey default, fit version 1.2.
- **Binning**: R=100 for every target, frozen (standard for sub-Neptune
  G395H spectra, e.g. arXiv:2501.18477; COMPASS publishes at R~200 —
  rebin theirs for overlays, never Patchwork's).

Companion to `session_summary_2026-07-29.txt` (the hard-won operational
detail lives there; this file is the campaign sequence).

## 0. What changed in the toolkit (this session)

QA fixes:
- `bin_at_resolution` clips the final bin edge to the wavelength cut
  (last channel's centre used to be reported outside the detector
  range). Stage 4 version bumped to 1.1. **Do not combine spectra
  binned before/after this change.**
- Per-channel ExoTiC-LD priors are computed in ONE batched call
  (`compute_ld_coeffs_batch`) instead of ~53 subprocess spawns per
  detector — the spectroscopic fit step starts minutes faster.
- `combine_visit_spectra` now tolerates a channel dropping out of one
  visit (bad column): channels are aligned by centre, combined from the
  visits that have them, with `n_visits_per_channel` recorded. Truly
  different binning schemes still refuse.
- Survey/optimizer sbatch logs go to the SUBMISSION directory
  (`%x-%j.out`); the old `--output` under output_root pointed at a
  directory that does not exist before the first run, which kills the
  job with no log at all.

New guards / diagnostics:
- **Expected-depth check**: every white-light fit compares its depth to
  the archive (Rp/Rs)^2 and stamps `depth_check.suspect` in
  `white_fit_summary.json` + a loud log line. Backstop for the
  prior-returning-fit failure class.
- **Red-noise beta** (Pont+2006, 5–30 min bins) recorded per white fit
  (`rednoise.beta_median`). COMPASS (arXiv:2511.18196) finds real G395H
  errors run ~5% (NRS1) / ~12% (NRS2) above photon predictions —
  beta >> 1 flags a target whose depth errors need inflating.
- **Priors caching**: discover.py embeds the archive ephemeris/stellar
  priors in each manifest; fits still query the archive at fit time but
  fall back to the cache offline (`priors_source` recorded).
- **NRS1–NRS2 white-light t0 offset** recorded per visit in the combine
  step (COMPASS sees significant offsets on some targets).
- **PCA detrending — SURVEY DEFAULT (fit v1.2, decision Wasi
  2026-07-30)**: COMPASS-style relative-pixel-flux PCA regressors
  (6 components) are part of the frozen fit definition. A fit where the
  components could not be built (no Stage 2 calints, or the as-is
  escape hatch) is stamped `1.2-nopca` and must not be mixed with
  survey fits.

## 1. One-time setup on Fir (before anything)

```bash
cd ~/aster/maiea && git pull
cp scripts/patchwork/exotedrf-python ~/bin/ && chmod +x ~/bin/exotedrf-python

# re-patch exoTEDRF (required after any pip install / git pull of ~/exoTEDRF)
source ~/envs/exotedrf/bin/activate
python ~/aster/maiea/scripts/patchwork/patch_exotedrf_env.py
deactivate

# preflight — must print READY
source ~/aster/aster-env/bin/activate
python -c "from aster_toolkit.data_reduction.exotedrf import *; \
           print(format_environment_report(verify_exotedrf_environment()))"
```

Also verify `ASTER_EXOTEDRF_REPO=~/exoTEDRF` is exported in `~/.bashrc`
(the optimizer-branch checkout is the survey's reduction engine) and
the ExoTiC-LD mps1 grids exist at `$ASTER_EXOTIC_LD_DATA`.

## 2. Regenerate manifests (login node — needs the archive)

The priors cache is new; regenerate so every manifest carries it:

```bash
python -m aster_toolkit.data_reduction.discover \
    --raw-root /project/def-ncowan/wasi/jwst_raw \
    --manifest-dir ~/patchwork/manifests
```

Include the TOI-125 overrides (GO 4126 observed b and c):

```bash
python -m aster_toolkit.data_reduction.discover \
    --raw-root /project/def-ncowan/wasi/jwst_raw \
    --manifest-dir ~/patchwork/manifests \
    --override jw04126101001="TOI-125 b" \
    --override jw04126201001="TOI-125 c"
```

Sanity check the assignment in the discovery report (predicted transit
offsets); a swapped b/c is also caught loudly by the fit-time
transit-in-window guard — then swap the overrides and refit.

## 3. Generate the job scripts + submission plan

```bash
python scripts/patchwork/generate_survey_jobs.py \
    --manifest-dir ~/patchwork/manifests \
    --output-root ~/scratch/patchwork \
    --job-dir ~/patchwork/jobs \
    --mail-user <you>
```

This prints every sbatch command, sized per target, in order:

- **Wave 1** (validate the path end-to-end at low cost):
  TOI-1468 c, LTT 3780 c, TOI-270 c, TOI-1231 b
- **Wave 2**: GJ 1214 b (easiest sanity check, ~13430 ppm), TOI-270 b,
  L 98-59 d, GJ 357 b, TOI-776 b, TOI-836.01, TOI-125 b, TOI-125 c
- **Wave 3** (multi-visit): GJ 9827 d (redo), GJ 3090 b, TOI-776 c,
  TOI-836 b
- **Wave 4**: K2-18 b (4-visit cross-program combine, approved),
  TOI-561 b (21k integrations; watch MaxRSS)

Rules that have already cost jobs: submit with NO venv active; absolute
sbatch paths; never two jobs on one workdir; REDUCE alone first
(2 CPUs), FIT only after the checklist below; `FORCE_REFIT=1` on any
refit.

## 4. Per-target verification (before submitting the fit / after it)

After REDUCE:
- [ ] `ls <root>/<SLUG>/reductions/*/nrs?/**/Stage3/*box_spectra_fullres.fits`
      -> 2 files per visit (NRS1+NRS2)
- [ ] `segments_complete: true` in `patchwork_summary.json`
- [ ] `exotedrf.path` in the reduction manifest = `~/exoTEDRF` for
      EVERY target (one tree for the whole survey)

After FIT:
- [ ] transit coverage ~100% (log line, seconds into the fit)
- [ ] white depth within tens of ppm of the expected value printed in
      the submission plan; `depth_check.suspect: false`
- [ ] `median_depth_err_ppm` << depth (tens of ppm, not ~1500)
- [ ] white rms hundreds of ppm; `rednoise.beta_median` ~ 1 (record it;
      > ~1.2 = flag for error inflation, per COMPASS)
- [ ] `ld_source: "exotic-ld"`, `priors_source: "archive"` (cache
      fallback means the node was offline — fine, but note it)
- [ ] ~29 (NRS1) + ~24 (NRS2) channels at R=100;
      `nrs1_nrs2_offset_ppm` small; `nrs1_nrs2_t0_offset_s` recorded

Then archive to `/project/def-ncowan/wasi/patchwork_reductions/<SLUG>/`
with the rsync in session summary 0.9 (NEVER flatten the tree).

## 5. Literature notes (2026-07-30 review)

- **COMPASS uniform reanalysis** (Ahrer et al. 2025, arXiv:2511.18196)
  is the closest published analogue to Patchwork: 7 G395H small-planet
  spectra, uniform pipeline. Their PCA systematics model is implemented
  here as the opt-in `pca_detrending`; their error-inflation finding
  motivates the beta diagnostic; systematics concentrate at 2.8–3.5 um
  (NRS1) — treat features there with suspicion.
- **Reproduction checks**: COMPASS has published G395H spectra of
  several Patchwork targets — TOI-836 b (arXiv:2404.00093), TOI-836 c
  (arXiv:2404.01264), GJ 357 b (arXiv:2507.07165), TOI-776 b & c
  (AJ 2025), L 98-59 (arXiv:2409.07552) — ready-made overlays.
- **GJ 9827 d caveat**: the published Piaulet-Ghorayeb GO 4098 result
  (arXiv:2410.03527) is **NIRISS/SOSS + HST**, not G395H. A published
  G395H spectrum to overlay may not exist yet; the reproduction check
  may need to compare in the small SOSS/G395H overlap or wait for the
  GO 4098 G395H paper. Wasi decides the comparison strategy.
- K2-18 b G395H reanalysis (arXiv:2501.18477) used exoTEDRF with
  scale-achromatic group-level 1/f — matches Patchwork config v1.1.
- ERS WASP-39 b G395H (arXiv:2211.10488): tilt-event handling by step
  function / trace detrending — matches the v1.1 tilt-step approach.
