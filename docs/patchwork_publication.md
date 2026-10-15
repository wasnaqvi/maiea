# Patchwork Publication Validation

The live checkout is `/Users/wasi/MAIA/aster`. Historical Fable documents
describe older architecture and paths; they are context, not deployment state.
This change set preserves preexisting identity guards and summary-merge work.

## Science-critical Repairs in This Pass

- TauREx forward models now receive the supplied Planet and BlackbodyStar,
  including planet mass/radius and stellar radius/temperature. Previously,
  silently using default objects could change gravity, scale height, and depth.
- Forward-model pressure inputs are documented as Pa, with positive/ordered
  physical parameter validation. The retrieval tool uses parsed fit-parameter
  names instead of iterating a JSON string character by character.
- DAT exporters now write FULL wavelength widths. Internal spectrum CSVs use
  half-widths; depths remain ppm in CSV and fractions in DAT.
- Rebin centers and half-widths describe the actual contributing support,
  not an error-weighted center paired with a displaced nominal interval.
- Survey beta gates preserve the worst visit; mixed grazing fit versions and
  missing diagnostics cannot be hidden by an aggregate. These flags are
  diagnostics, not calibrated acceptance rules.
- Review-page step markers map original integration indices through masks.
  Labels retain visit beta ranges. Scale-height panels no longer clip all
  targets at +/-4.5 H; historical heuristic mu plots require explicit opt-in.
- `publication_audit.py` measures fixed-time residual scatter, absolute and
  detector-centered spectra, visit-shape differences, and GJ 3090 diagnostics.
  Every input is hashed; raw files and fit posteriors are preserved.
- Exploratory band contrasts retain exact channel selections and coefficients.
  Constant offsets cancel; missing flanks are flagged, not filled or assigned
  a different window. Diagonal errors are not molecular-detection significances.
- The NASA Archive currently stores TOI-836 c as `TOI-836.01`. The scale-height
  lookup now queries the alias separately from the display name. Missing or
  nonpositive physical parameters are reported; absolute-depth plots retain
  those targets. A dated JSON records normalization parameters and omissions.

## Reproduce

```sh
cd /Users/wasi/MAIA/aster
/opt/anaconda3/envs/exo/bin/python -m pytest tests/scripts/test_publication.py tests/scripts/test_data_reduction.py -q
/opt/anaconda3/envs/exo/bin/python scripts/patchwork/publication_audit.py /Users/wasi/Patchwork_Data --output /Users/wasi/Patchwork_Data/publication_audit_2026-09-06
/opt/anaconda3/envs/exo/bin/python /Users/wasi/Desktop/Patchwork_Naqvi_Cowan/scripts/rebuild.py --repo /Users/wasi/MAIA/aster --data /Users/wasi/Patchwork_Data
```

These are regression checks, not atmospheric validation. No opacities or
cluster jobs are required for the audit. A successful mock TauREx test does
not demonstrate accurate molecular cross-sections or converged retrievals.
Do not promote the existing retrieval defaults to survey priors: define
target-specific mass/radius/temperature/abundance bounds and cloud models.

## Release Contract

Before inference, retain per-channel response/weights, covariance, shared
white-light posterior propagation, parameter sources, and exact model/config
versions. A current code version does not retroactively update old fits.
Before a common-mode correction, perform leave-channel-out injection/recovery
and account for noisy templates. Before an overlapping transit is admitted,
validate a multi-planet model and keep the single-planet guard effective.
Before a population claim, freeze selection and retain non-detections.

See the manuscript's `SCIENCE_PLAN.md` for ordered physical tests. Fable's
proposed archive rewrite was read but not blindly installed: unrelated HTTP,
security, and agent-architecture changes are outside this science repair.
