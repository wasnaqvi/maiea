# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ASTER (Agentic Science Toolkit for Exoplanet Research) is a refactored agentic system for exoplanet atmospheric research. The system uses the `orchestral-ai` package to provide AI agents with tools for downloading exoplanet data, reducing raw JWST time-series observations, fitting transit light curves, running forward models, and performing atmospheric retrievals.

Two largely independent halves:

- **Atmospheric modeling** — TauREx forward models and retrievals, driven from an observed spectrum.
- **Data reduction** — the *Patchwork* pipeline: raw JWST NIRSpec/G395H uncals to a transmission spectrum, via exoTEDRF (Stages 1-3) and juliet (Stages 5-6). See [JWST Data Reduction](#jwst-data-reduction--patchwork-aster_toolkitdata_reduction). This half runs on DRAC's **Fir** cluster, where the data and the pinned environments live.

## Skills System

ASTER includes specialized skill files in `workspace/skills/` that contain detailed knowledge about specific tasks:

- **taurex_setup.md**: TauREx configuration, line list downloads, path setup, and troubleshooting
- **corner_plots.md**: How to create publication-quality corner plots from retrieval results
- **retrieval_best_practices.md**: Parameter bounds guidance, optimizer selection, and retrieval strategies for different use cases

**Important**: These skill files are NOT loaded into the system prompt. When you need information about these topics, use the ReadFileTool to read the relevant skill file from `workspace/skills/`.

## Environment Setup

The environment differs by machine — check which one you are on before
activating:

**On DRAC Fir** (and anywhere the repo ships with `aster-env/`), a
virtualenv at `aster-env/`:

```bash
source aster-env/bin/activate
pip install -r requirements.txt
```

**On the local Mac**, a conda environment named `exo` (there is no
`aster-env/` directory here):

```bash
conda activate exo
```

Both provide `orchestral-ai`; note the numpy versions differ (Fir's
aster-env is python 3.13, the Mac's `exo` is python 3.14 / numpy 2.x),
so anything numerically sensitive should be checked on both.

Key dependencies:
- `orchestral-ai` - Agent framework
- `taurex` - Atmospheric modeling engine
- `astropy`, `pandas`, `numpy` - Data handling
- `corner-2.2.3` - Posterior visualization

## Running the Application

```bash
# Start the ASTER web UI server
python run_aster.py
```

This starts a web server on `localhost:8000` with an agent that has access to TauREx modeling tools.

## Architecture

### Agent System

The agent system is built on `orchestral-ai` and configured in [run_aster.py](run_aster.py):
- Uses Claude or GPT as the LLM backend (configured via environment variables in `.env`)
- Provides a persistent command execution environment in the `workspace/` directory
- Includes safety hooks (`DangerousCommandHook`) to prevent destructive operations

### Tool Organization

Tools are organized in the `aster_toolkit/` package with clear separation of concerns:

**TauREx Tools** (`aster_toolkit/taurex/`):
- `forward_model.py` - `RunTaurexModelTool` for generating synthetic transmission spectra
- `retrieval.py` - `SimulateTaurexRetrieval` for atmospheric parameter fitting
- `set_paths.py` - `SetTaurexPaths` for configuring opacity/CIA data paths

**Data Acquisition** (`aster_toolkit/data_acquisition/`):
- `exoarchive.py` - `GetExoplanetParameters` for TAP queries to NASA Exoplanet Archive
- `exoarchive.py` - `DownloadDataset` for downloading and processing spectra from NASA archive
- `mast.py` - MAST search/download of raw JWST products, plus `archive_tap_query()` (used by the reduction tools for ephemerides)

**Data Reduction** (`aster_toolkit/data_reduction/`) — the Patchwork pipeline:
- `discover.py` - `DiscoverPatchworkVisits`: index a raw tree, identify planets, write manifests
- `exotedrf.py` - `ReduceNirspecG395hTso`, `InspectNirspecG395hUncalData`, `VerifyPatchworkEnvironment`, `MakeUncalTestSubset` (Stages 1-3)
- `lightcurves.py` - `DetectTiltEvents` (Stage 4; pure numpy/astropy. `DetectNirspecTiltEvents` is a back-compat alias)
- `juliet.py` - `FitNirspecG395hWhiteLight`, `FitNirspecG395hTransmissionSpectrum`, `CombineNirspecG395hVisits` (Stages 5-6)
- `contamination.py` - `DetectLightCurveAnomalies` (Stage 5.5), `ModelStellarContamination` (Stage 6.5), `VerifyContaminationBackends`
- `survey.py` - `RunPatchworkTarget`, `GeneratePatchworkFirJob` (end-to-end driver)
- `optimize.py` - `OptimizeNirspecG395hReduction`, `SummarizeG395hOptimization`, `GenerateFirOptimizerJobs`

### TauREx Path Configuration

TauREx requires opacity cross-sections and CIA (collision-induced absorption) files:

1. **Download line lists** (first time setup):
   ```bash
   python download_linelists.py
   ```
   This downloads molecular cross-sections (H2O, CO2, NH3, CH4, CO) to `workspace/linelists/xsec/` and CIA files (H2-H2, H2-He) to `workspace/linelists/cia/`

2. **Set paths before running models**:
   The agent must call `SetTaurexPaths` with absolute paths. The agent should:
   - First run `pwd` to get the current working directory
   - Then construct the full absolute paths to `linelists/xsec/` and `linelists/cia/`
   - Never hardcode or guess paths (e.g., `/app/linelists` is wrong)

   Example:
   ```python
   # If pwd returns /Users/username/project/workspace
   SetTaurexPaths(
       opacity_path='/Users/username/project/workspace/linelists/xsec',
       cia_path='/Users/username/project/workspace/linelists/cia'
   )
   ```

### Forward Modeling

`RunTaurexModelTool` generates synthetic transmission spectra given planet/star parameters:

Key parameters:
- Physical: `planet_radius` (RJup), `planet_mass` (MJup), `star_radius` (Rsun)
- Orbital: `orbital_period` (days), `semi_major_axis` (AU)
- Atmospheric: `planet_temp` (K), `atm_min_pressure`/`atm_max_pressure` (bar)
- Chemistry: `molecular_abundances` (optional dict, e.g., `{'H2O': 0.02, 'CH4': 0.001}`)
- Output: `filename` (saves as `{filename}_spectrum.png`)

The tool uses:
- Isothermal temperature profile
- TaurexChemistry with H2/He background (ratio 0.17)
- Default molecular abundances (if not specified): H2O (0.02), CH4 (0.001), CO2 (0.0001), CO (0.001), NH3 (0.0001)
- Custom abundances can be specified via `molecular_abundances` parameter
- Absorption, Rayleigh, and CIA contributions

Output files in `workspace/`:
- `{filename}_spectrum.png` - Plot
- `fm_wavelength.npy`, `fm_spectrum.npy` - Raw data

**Important**: The forward model outputs spectra at full line-list resolution (~100k points). For visualization purposes, these should be binned to observational resolution. Unbinned spectra are too noisy to display meaningfully. Use numpy to bin wavelength and spectrum arrays before custom plotting.

### Atmospheric Retrieval

`SimulateTaurexRetrieval` fits atmospheric parameters to observed spectra using nested sampling.

**Retrieval Modes**:
- `"reduced"` (default) - Fits mixing ratios of 5 predefined molecules (H2O, CH4, CO2, CO, NH3)
- `"equilibrium"` - Fits metallicity and C/O ratio using ACE thermochemical equilibrium
- `"full"` - Fits custom list of molecules specified by user

**Key Parameters**:
- `observation_path` - **REQUIRED**. Path to 3-4 column spectrum file (wavelength μm, depth, error, [bin width]). Use exact path from DownloadDataset output or user-provided file.
- `fit_params` - Parameters to fit (minimum: `['planet_radius', 'T']` + chemistry params). Can be passed as a list or string representation.
- `bounds` - Dict of `{param: [low, high]}` bounds. Can be passed as a dict or string representation. **Optional** - if not provided, reasonable defaults are auto-generated.
- `optimizer` - `"nestle"` (recommended, always works) or `"multinest"` (faster but requires difficult installation)

**Important Notes**:
- Pressure units in TauREx are **Pascals**, not bars (default range: 1e-3 to 1e5 Pa)
- Molecular abundance bounds should be `[1e-9, 1e-2]`
- Standard `nlayers=100` (only change if user requests)
- **String parameters**: `fit_params`, `bounds`, and `molecular_abundances` accept both native Python objects and string representations (e.g., `"['H2O', 'CH4']"` or `['H2O', 'CH4']`). The tool will parse strings automatically.
- **Auto-generated bounds**: If `bounds` is not provided, the tool generates sensible defaults: planet_radius [0.5, 2.5] RJup, T [500, 3000] K, molecules [1e-9, 1e-2], metallicity [0.1, 10.0], c_o_ratio [0.1, 2.0]

**Outputs** (saved to `output_path` with `output_basename` prefix):
- `*_fit.png` - Observed vs best-fit comparison
- `*_corner.png` - Posterior distributions
- `*_samples.npy`, `*_weights.npy` - Full posterior samples
- `*_wavelength.npy`, `*_spectrum.npy` - Best-fit spectrum

### Data Acquisition

The `exoarchive.py` module provides access to NASA Exoplanet Archive data:

**Tools**:
- `GetExoplanetParameters` - TAP queries for planet/star parameters from pscomppars table
  - Parameters: `planet_name`, `columns` (list of parameter names), `table` (default: "pscomppars")
  - Returns: Dictionary with requested parameters

- `DownloadDataset` - Download and process spectra from NASA archive
  - **Three input methods** (provide only ONE):
    1. `wgets_file_path` - Path to file containing wget commands (user created)
    2. `wget_text` - Raw wget commands pasted directly into chat
    3. `wget_url` - URL to Firefly wget page (tool scrapes commands automatically) ⭐ EASIEST
  - Parameters: `output_dir` (default: "spectra")
  - **File organization**:
    - Working files: `workspace/download_dataset_tool/query{NNN}/` (for debugging)
    - Final spectra: `workspace/spectra/PLANET_NAME_3/DATASET_ID/spectrum.dat`
    - Each download gets unique query ID (query001, query002, etc.)
    - **Tool output shows full spectrum file paths** for use in retrievals
  - Firefly interface: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/atmospheres/nph-firefly

**Key Functions** (for advanced use):
- `get_exoplanet_params_tap()` - Direct TAP query function
- `process_wgets_file()` - Download IPAC tables from URLs
- `process_downloads()` - Convert raw data to spectrum.dat format

## JWST Data Reduction — Patchwork (`aster_toolkit/data_reduction/`)

Uniform reduction of JWST NIRSpec/G395H BOTS time-series (raw uncals →
transmission spectrum) for the Patchwork sub-Neptune survey. Chain:

```
uncals → inspect → exoTEDRF Stages 1-3 → Stage 4 lightcurves
      → juliet white fit → Stage 5.5 occulted-spot scan → masked refit + channels
      → visit combination → Stage 6.5 unocculted-spot check
```

### Module map

- **`discover.py`** — indexes a raw tree by **FITS headers only**;
  directory names are untrustworthy (obsid-only folders, concatenated
  multi-target names, duplicates). Filters to the survey definition
  (G395H / F290LP / SUB2048 / NRS_BRIGHTOBJ), so target-acquisition
  exposures and other gratings drop out on their own. Identifies each
  visit's planet by archive cone search at the header pointing **plus a
  transit-window overlap test** — a host may have several planets, and
  the planet name drives the ephemeris, so a guess yields a confidently
  wrong depth. Writes per-planet manifests with cached archive priors.
- **`exotedrf.py`** — Stages 1-3 as a subprocess. Frozen
  `PATCHWORK_G395H_CONFIG`, environment preflight, CRDS pinning,
  per-visit baseline window.
- **`lightcurves.py`** — Stage 4, pure numpy/astropy (runs in the main
  ASTER env): MJD→BJD on load, out-of-transit normalization, constant-R
  binning, trace diagnostics, PCA regressors, red-noise beta, the
  transit-in-window guard, and tilt-event detection from the PSF
  diagnostics (see [Tilt events](#tilt-events--detect-in-the-psf-correct-with-a-heaviside)).
- **`juliet.py`** — Stages 5-6: white-light then per-channel fits with
  the orbit frozen to the white posterior; ExoTiC-LD priors; visit
  combination; NRS1-NRS2 offset; the same-band depth check.
- **`contamination.py`** — Stages 5.5 and 6.5, the two stellar
  contamination problems (pure numpy; `emcee` for the retrieval). See
  [Stellar contamination](#stellar-contamination--stages-55-and-65).
- **`survey.py`** — manifest-driven driver (`inspect → reduce → fit →
  combine → contamination`), restartable by step, plus the Fir sbatch
  generator. `fit` is compound: white pass 1 → Stage 5.5 scan → masked
  white refit → channels.
- **`optimize.py`** — wraps the exoTEDRF coordinate-descent optimizer as
  a frozen, replayable rule generator (SHA-256 `omega_hash` per run).

### exoTEDRF environment — the hard constraints

**Two Python versions, unavoidable.** `orchestral` needs 3.12+; exoTEDRF
pins `numpy<2` / `jwst==1.17.1` and lives on 3.11. They cannot be merged,
so Stages 1-3 run as a **subprocess** in the pinned environment.

**`ASTER_EXOTEDRF_PYTHON` must point at a WRAPPER, not an interpreter.**
On DRAC, calling `<venv>/bin/python` by path leaves site-packages off
`sys.path` on a compute node — virtualenvs must be *activated*. Use
`scripts/patchwork/exotedrf-python`, which module-loads its own
python/opencv, activates the venv, prepends the repo shadow, and execs.

**`ASTER_EXOTEDRF_REPO` selects which stage code runs.** When set it is
prepended to `PYTHONPATH` for every exoTEDRF subprocess, so a source
checkout shadows the installed release. Patchwork runs the **`optimizer`
branch**: reductions and optimizer sweeps must execute the *same* stage
code or tuned parameters do not transfer. Unset → installed release.
The tree actually used is recorded as `exotedrf.path` in every reduction
manifest — it must match across all targets.

Required environment on Fir:

```bash
export PYTHONPATH=~/aster/maiea${PYTHONPATH:+:$PYTHONPATH}
export ASTER_EXOTEDRF_PYTHON=~/bin/exotedrf-python
export ASTER_EXOTEDRF_REPO=~/exoTEDRF          # optimizer branch
export CRDS_PATH=~/scratch/crds_cache
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export CRDS_CONTEXT=jwst_1322.pmap             # pinned; part of the survey definition
export ASTER_EXOTIC_LD_DATA=~/scratch/exotic_ld_data
```

**A cleared `PYTHONPATH=` is for pip installs ONLY, never at runtime** —
at runtime it is how the `opencv` module delivers `cv2`, which
`stcal.jump` imports at module level (so JumpStep cannot run without it).

### Preflight — before committing any compute

```bash
python -c "from aster_toolkit.data_reduction.exotedrf import *; \
           print(format_environment_report(verify_exotedrf_environment()))"
```

Must print **READY** *and* name the intended source tree. Eight checks,
each traceable to a real failed job: `numpy<2`, pandas,
`jwst.__version_commit__`, `crds.jwst.locate`, `cv2`, stages 1-3,
`run_DMS.py`, BadPixStep resume guard. Also exposed as the
`VerifyPatchworkEnvironment` tool, and wired into `run_reduction` so it
raises before any compute.

### Environment patches

`scripts/patchwork/patch_exotedrf_env.py` applies import/resume fixes —
**none touch numerics**. Re-run after ANY `pip install` touching jwst /
stdatamodels / exotedrf, and after any `git pull` of the exoTEDRF
checkout: pip silently reverts patched files. Patch the tree that will
actually run — with `ASTER_EXOTEDRF_REPO` set, invoke it with
`PYTHONPATH=$ASTER_EXOTEDRF_REPO` so it targets the checkout rather than
site-packages.

Non-standard items, recorded so a stock reproduction knows why they
existed: numpy pinned 1.26.4; crds pinned 12.1.11; a
`stdatamodels.exceptions` shim; `__version_commit__ = ""` appended to
`jwst/__init__.py`; the BadPixStep resume guard; `PCAReconstructStep`
skipped (diagnostic-only, crashes with sklearn ≥1.3); OpenCV from the
`opencv/4.12.0` **module**, never pip (Alliance ships a deliberately
failing dummy wheel).

### Uniformity contract — frozen versions

Changing any of these changes the survey definition. Bump the version,
and never mix old and new products in one analysis.

| constant | value | module |
|---|---|---|
| `PATCHWORK_CONFIG_VERSION` | 1.1 | `exotedrf.py` |
| `PATCHWORK_STAGE4_VERSION` | 1.2 | `lightcurves.py` |
| `PATCHWORK_FIT_VERSION` | 1.3 | `juliet.py` |
| `PATCHWORK_CONTAM_VERSION` | 1.0 | `contamination.py` |
| `PATCHWORK_OPTIMIZER_VERSION` | 1.0 | `optimize.py` |
| `DEFAULT_RESOLUTION` | 100 | `lightcurves.py` |
| `CRDS_CONTEXT` | `jwst_1322.pmap` | `exotedrf.py` |

Fit v1.2 added COMPASS-style relative-pixel-flux PCA regressors (Ahrer et
al. 2025, arXiv:2511.18196); a fit that could not build them is stamped
`-nopca`. Fit **v1.3** adds two things, both from the 2026-08-03 referee
feedback: the Stage 5.5 occulted-spot scan (confirmed crossings masked,
lightcurve refit) and the rewritten tilt handling below. A fit where the
spot scan did not run is stamped `-noscan`. The stamps compose, so
`1.3-nopca-noscan` is possible and is **not** a survey fit. "Scan ran and
found nothing" and "scan never ran" are different states — do not treat
an unstamped and a `-noscan` fit as equivalent.

`-nopca` now carries a second meaning worth knowing: no PCA components
means no Stage 2 calints, which means no trace diagnostics, which means
the tilt search fell back to the out-of-transit flux and could not have
found an in-transit tilt. Treat `-nopca` as "this fit is blind to the
TOI-270 c failure mode", not merely as slightly worse detrending.

### Tilt events — detect in the PSF, correct with a Heaviside

A tilt event is a primary-mirror segment moving. Rare (~1 per day, so
~0.2 per visit — Loic Albert, 2026-08-26), and expensive when missed:
TOI-270 c's landed **inside** transit and left the depth unusable.

**Detection searches the trace diagnostics, not the flux.** A tilt
changes the PSF *shape* first and the aperture flux only in consequence.
Albert: *"the most direct effect that a tilt event has on the PSF is a
change in its FWHM, so PCA are good to catch that."* The KELT-7 b team
confirmed theirs as "a definite jump in the guide star **width**"
(arXiv:2509.12479); SOSSISSE catches tilts through its spatial-derivative
term for the same reason. `find_tilt_events` therefore step-searches
`trace_fwhm`, `trace_y`, `trace_x` and every PCA component, and requires
a coincident trigger in **≥2** of them — one series stepping alone is a
glitch in that series. Those series contain no transit, so the search
runs over the whole visit. The white flux is searched too but only out of
transit, since it is the one series the transit lives in.

This is the fix for the old `detect_tilt_events`, which searched the flux
alone and so had to mask the transit to stop ingress registering as a
step — making an in-transit tilt structurally undetectable. That function
is still there for single-series use, and its tests pin the old
behaviour, but it is not the survey path.

**Correction is a fitted Heaviside, not a mask or a split.** One step
regressor per event, amplitude free, break time fixed from the white fit
and reused per channel — the recipe all three KELT-7 b pipelines
(Eureka!, ExoTiC-JEDI, Tiberius) converged on: *"we also fixed the step
function's break point ... when fitting the wavelength-binned light
curves."* The amplitude must stay free **per channel and free in sign**:
the flux change is pixel-dependent and *"can be positive or negative and
varies between pixels"* (arXiv:2405.06737). Never reject a tilt for being
chromatic — unlike a spot crossing, chromaticity is expected.

Only `TILT_TRANSITION_MASK = 3` integrations are dropped at each
transition, because the integration straddling the tilt is a blend of two
PSF states that no Heaviside describes (same choice as the WASP-39 b
G395H analysis). Everything else is preserved: splitting the lightcurve
at the event would cost the joint constraint on the orbit, which is the
information the step is there to protect.

Cross-detector coincidence confirms an event (a segment tilt is a
telescope-level event, so it lands at the same instant in NRS1 and NRS2),
and the rate guard flags a visit with implausibly many detections rather
than fitting a step per false positive.

`extract_width` is fixed at 16 — never `'optimize'`, which silently
breaks per-target uniformity. The `overrides` argument to
`write_dms_config` / `run_reduction` exists for debugging single targets
only; production survey reductions must not use it.

### Stellar contamination — Stages 5.5 and 6.5

Two different problems, two different places in the chain. Added after
referee feedback on Patchwork 1 (2026-08-03).

**Stage 5.5 — occulted spots (`contamination.py`, inside the `fit` step).**
The planet crosses a spot; the lightcurve shows a positive bump *inside*
transit, which distorts the shape, inflates β, and biases the depth.
Visible only against a clean transit model, so the `fit` step is two
passes: white fit pass 1 (unmasked) → scan → masked white refit → the
spectroscopic channels, with the **same mask** on both. Detection is a
centred running mean over the residuals, long-window detrended, flagged
at ≥5 consecutive integrations beyond 3σ of the out-of-transit MAD.

Classification rules, and why each exists:
- **Both detectors, transient, in transit** → real crossing. Masked.
- **NRS2/NRS1 peak-amplitude ratio** is reported: a spot's contrast falls
  towards the infrared, so a ratio ≥ 0.9 is achromatic and argues
  instrumental despite the coincidence.
- **Persistent (level does not return)** → a tilt event, not stellar.
  Corrected with a step regressor, **never masked**.
- **One detector only** → detector systematics. Reported, never masked:
  masking it would drop integrations from one half of the spectrum only
  and corrupt the NRS1-NRS2 offset.

Masking (not modelling) is the survey default, on the referee's
recommendation — *"for a wholesale analysis like yours, masking
starspots seems a reasonable compromise between expediency and
accuracy."* `fit_spot_crossing_spotrod` is the per-target escape hatch
and its results are **not** uniform with the survey.

**Stage 6.5 — unocculted spots (the `contamination` step).** Spots
outside the transit chord never appear in the lightcurve but multiply
every depth by ε(λ) (Rackham+2018 transit light source effect). Fitted
as a flat intrinsic spectrum × ε(λ) — the null hypothesis, so the result
bounds how much structure the star alone could produce. Report
`delta_bic > 10` as a detection, otherwise quote the `f_het` upper limit.

Two limits, both load-bearing:
- `f_het` and `T_het` are **degenerate** over one G395H octave. Quote
  ε(λ) and the corrected spectrum, not `f_het` alone.
- Blackbody spot contrast at 3-5 µm is weak: a plausible M-dwarf spot
  moves the depth ~20 ppm across the band, below Patchwork's channel
  errors. Non-detection means *G395H cannot constrain this*, not that
  the star is quiet. Breaking either needs a bluer baseline (the ten
  targets with NIRISS SOSS overlap).

`spotrod`, `stctm` and `sage` are **optional** cross-check backends,
probed by `VerifyContaminationBackends` and never imported at module
scope. Nothing in the survey path requires them.

### VERIFY PHYSICS, NOT EXIT CODES

The two most expensive failures in this project both exited 0 and
produced plausible-looking spectra:

1. **The fit returns the prior.** ~1800 ppm for every target with tiny
   scatter, errors larger than the signal, `b` and `a/Rs` sitting
   mid-prior. Cause: the transit is not in the data — a stale ephemeris
   or an unconverted MJD time axis. *Guarded*: `build_lightcurves`
   refuses, reporting the offset and the data span.
2. **juliet reloads old posteriors instead of refitting.** "COMPLETES"
   in minutes and reproduces numbers to twelve decimal places.
   *Guarded*: the fit step refuses when posteriors exist; clear with
   `force_refit=True` (CLI `--force-refit`, sbatch `FORCE_REFIT=1`).

Every white-light fit also records `rednoise.beta_median` (Pont+2006;
> ~1.2 means that target's depth errors need inflating) and
`depth_check`. Check these, not the exit status.

`depth_check` compares against the best available reference and says
which it used:
- **`band: same-band`** — a published JWST/G395H depth from
  `PUBLISHED_G395H_DEPTHS` in `juliet.py`. Tight tolerance (10%); a
  disagreement is `status: suspect` and worth stopping for. Add entries
  as the literature grows — each needs a citation.
- **`band: cross-band`** — the archive optical (Rp/Rs)², i.e. TESS
  discovery photometry for these targets. Different bandpass: limb
  darkening differs and unocculted spots move the two bands by different
  amounts. Verified-healthy Patchwork 1 fits ran +0.3% to +14% from
  TESS, so this can only be `status: indicative` and **never**
  `suspect`. The old check flagged healthy targets survey-wide on
  exactly this.

An error bar larger than half the expected depth is `suspect` in either
band — that is the prior-returning fit, and it is bandpass-independent.

### Compute discipline

- **Never run a reduction or fit in-process on a login node.**
  `RunPatchworkTarget`, `ReduceNirspecG395hTso`, and the fit tools are
  for local subset smoke tests (`MakeUncalTestSubset`) only; real targets
  go through SLURM.
- Reduce with **2 CPUs** — exoTEDRF Stage 1 is effectively
  single-threaded, so a larger request only lengthens the queue wait.
- Submit from a shell with **no venv active**: SLURM exports the
  submitting environment and `module purge` cannot undo an activation.
  From inside the GUI, wrap submissions in
  `env -i HOME="$HOME" USER="$USER" bash -lc '...'`.
- **Never two jobs writing one output directory** — it corrupts outputs.
- Reduce is restartable (exoTEDRF skips completed stages); **the fit is
  not**.
- Measured rate: Stages 1+2+3 ≈ 20 min per visit × detector at 2753
  integrations. Fits are hours and not resumable.

### Campaign documents

- **`scripts/patchwork/CAMPAIGN.md`** — the survey runbook: one-time
  setup, wave order, per-target verification checklists, archiving.
- **`scripts/patchwork/AGENT_BRIEF.md`** — self-contained operating
  brief for an agent driving the campaign from the GUI (hard rules,
  per-session protocol, checklists, escalation list). Note the orchestral
  file tools are sandboxed to `workspace/`, so the brief documents which
  tool reaches what.
- **`scripts/patchwork/generate_survey_jobs.py`** — writes sized sbatch
  scripts and prints the submission plan. **Single source of truth for
  walltime and memory; never hand-write them.**
- `scripts/patchwork/patchwork_gj9827d.py` — a single-target, as-is
  baseline script. Do NOT generalize it to other targets; the toolkit is
  the target-agnostic path and has the guards.

### Testing

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/scripts/test_data_reduction.py -q
```

111 tests, no network and no exoTEDRF/juliet environment required. They
cover the transit-in-window and force-refit guards, MJD→BJD conversion,
constant-R binning, tilt detection (including that a real transit is not
flagged), planet resolution, staging deduplication, the frozen-config
assertions, and — for the contamination stages — injected spot and
facula crossings recovered across noise seeds, an **in-transit** step
still found (the TOI-270 c case, where a naive noise scale measures the
step height and the event hides itself), detrend lobes merged into one
event, two separated crossings kept separate, tilt steps and
single-detector excursions never masked, the chromatic amplitude ratio,
ε(λ) limits, and the same-band/cross-band depth-check logic.
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids an unrelated `pkg_resources`
breakage in the local pytest plugins.

## Common Workflows

### Running a Forward Model

1. Ensure line lists are downloaded (`download_linelists.py`)
2. Set TauREx paths using absolute paths
3. Call `RunTaurexModelTool` with planet/star parameters
4. Output saved to `workspace/{filename}_spectrum.png`

### Running a Retrieval

1. **Read the skill file**: Agent must use ReadFileTool to read `skills/retrieval_best_practices.md` first
2. Ensure line lists downloaded and TauREx paths set (with `ls`/`pwd` to get absolute paths)
3. Obtain observed spectrum (via `DownloadDataset` tool or user-provided)
4. Choose retrieval mode and configure fit parameters/bounds (or use auto-generated defaults)
5. Call `SimulateTaurexRetrieval` with `optimizer="nestle"` (default, always works)
6. Review outputs: fit plot, corner plot, and posterior samples

**Important**: The agent should read the skill file before running ANY retrieval to understand optimizer selection and parameter bounds.

### Reducing a JWST NIRSpec/G395H Target (Patchwork)

Full detail in `scripts/patchwork/CAMPAIGN.md`. The sequence:

1. **Preflight** — `VerifyPatchworkEnvironment` must print READY and name
   the intended exoTEDRF tree. Fix anything it flags with
   `patch_exotedrf_env.py` before spending compute.
2. **Discover** — `DiscoverPatchworkVisits` on the raw root (login node;
   it queries the archive) writes one manifest per planet. Supply
   `overrides` for any visit whose planet cannot be resolved; never guess.
3. **Generate jobs** — `generate_survey_jobs.py` writes sized sbatch
   scripts and prints the submission plan.
4. **Reduce** — submit `STEPS=inspect,reduce` alone, 2 CPUs.
5. **Verify the reduce** — 2 Stage 3 products per visit,
   `segments_complete: true`, identical `exotedrf.path` and
   `crds_context` across targets.
6. **Fit** — only after step 5 passes; `FORCE_REFIT=1` on any rerun.
   This one step runs white pass 1, the Stage 5.5 spot scan, the masked
   white refit, and the channels.
7. **Verify the science** — `depth_check.status` (`suspect` stops you;
   `indicative` is a cross-band note, not a problem),
   `rednoise.beta_median`, `ld_source`, `fit_version` (must not carry
   `-noscan` or `-nopca`), `tilt_handling` (events found, which
   diagnostics saw them, `rate_warning` empty),
   `spot_handling.n_masked`, channel counts,
   NRS1-NRS2 offset. Read `fits/<visit>/anomalies/anomaly_scan.pdf`:
   every masked span should be a bump the eye agrees with, and any
   `achromatic` or `single_detector` event should NOT have been masked.
8. **Contamination** — `STEPS=contamination` on the combined spectrum
   (seconds, no refit). Expect non-detections; see the Stage 6.5 limits.
9. **Archive** to durable storage (scratch is purged). Never flatten the
   directory tree: Stage 3 filenames repeat across visits, so the visit
   is encoded only in the directory name.

### Downloading Spectra

The `DownloadDataset` tool supports three input methods:

**Method 1: User provides URL (easiest)**
```
User: "Download spectra from https://exoplanetarchive.ipac.caltech.edu/staging/..."
Agent: DownloadDataset(wget_url="https://...")
```

**Method 2: User pastes wget text**
```
User: "Here are the wget commands: wget -O WASP_39_b.tbl '...'"
Agent: DownloadDataset(wget_text="wget -O WASP_39_b.tbl '...'")
```

**Method 3: User saves to file**
```
User: "I saved the wget commands to wgets.txt"
Agent: DownloadDataset(wgets_file_path="wgets.txt")
```

### Querying Exoplanet Data

Use `GetExoplanetParameters` tool for programmatic access to archive data:
```python
# Get planet parameters
GetExoplanetParameters(
    planet_name="WASP-39 b",
    columns=["pl_radj", "pl_bmassj", "st_rad", "st_teff"]
)
```

## Workspace Organization

```
workspace/
├── linelists/          # TauREx opacity/CIA data
│   ├── xsec/          # Molecular cross-sections (.h5 files)
│   └── cia/           # CIA files (.cia files)
├── tmp/               # Downloaded spectra and processed data
│   └── processed_data/PLANET_NAME_3/DATASET_ID/spectrum.dat
├── fm_*.npy           # Forward model outputs
└── *.png              # Plots
```

## Tool Usage Patterns

When working with the agent system:

1. **StateField vs RuntimeField**: Tools use `StateField` for agent-managed state (e.g., `base_directory`) and `RuntimeField` for user/LLM-provided inputs
2. **Streaming callbacks**: Retrieval functions support streaming output via `stream_callback` parameter for real-time progress
3. **Lazy imports**: The codebase uses lazy imports to speed startup time
4. **CamelCase naming**: All tool names follow Python class conventions (e.g., `RunTaurexModelTool`, not `run_taurex_model_tool`)

## Important Notes

**TauREx**
- Always use **absolute paths** for TauREx opacity/CIA configuration
- Pressure units in TauREx are **Pascals** (Pa), not bars
- For retrieval, use `"nestle"` optimizer by default (multinest requires complex installation)

**Data reduction**
- exoTEDRF writes the Time extension in **MJD**; ephemerides are **BJD**.
  `load_stage3_spectra` converts on load — anything reading a Stage 3
  product by another route must convert too. Getting this wrong costs
  hours of offset and yields a fit that returns the prior.
- Reduction pipeline paths are **absolute** everywhere; manifests are
  copied between machines.
- Preflight before compute; re-patch after any pip install or exoTEDRF
  `git pull`.
- Never edit the frozen config for a single target — that is a survey
  definition change requiring a version bump.
- Verify the depth against the archive expectation, not the exit code.

**General**
- The `.env` file contains API keys for LLM backends - never commit this file
- Planet names in archive queries use format like `"WASP-39 b"` (space, lowercase designation). Some targets keep a TOI candidate designation (e.g. `TOI-836.01`) — do not "correct" these, the archive returns nothing for the tidier name.
- The orchestral file tools (`ReadFileTool`, `EditFileTool`, `FileSearchTool`) are sandboxed to `base_directory` (`workspace/`) and reject absolute paths outside it; `RunCommandTool` is not restricted. Symlinks inside `workspace/` are followed out, since the check uses `abspath`, not `realpath`.
