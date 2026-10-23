"""Patchwork uniform transit lightcurve fitting built on juliet.

Stages 5-6 of the Patchwork pipeline. Consumes exoTEDRF
``*_spectra_fullres.fits`` products (via ``lightcurves.py`` Stage 4) and
produces, per visit x detector:

1. White-light transit fit -> orbital parameters (t0, a/Rs, b, Rp/Rs).
2. Spectroscopic per-channel fits with the orbit frozen to the
   white-light posterior -> transmission spectrum (ppm CSV + TauREx dat).

and per detector: an inverse-variance combination of all visits, plus the
survey health metrics (NRS1-NRS2 offset near the detector gap, white and
spectroscopic residual rms).

Uniformity contract (PATCHWORK_FIT_VERSION)
-------------------------------------------
Fitting choices are frozen module-wide. v1.2 = v1.1 plus the
COMPASS-style relative-pixel-flux PCA regressors (6 components, Ahrer
et al. 2025, arXiv:2511.18196) as part of the frozen definition; fits
without them are stamped "-nopca". v1.1 (the "patched" fit of the
Patchwork plan, superseding the v1.0 baseline):

- Out-of-transit baseline normalization (ephemeris + literature duration).
- ExoTiC-LD Gaussian priors on Kipping (q1, q2) from the stellar
  parameters (truncated normal, sigma 0.1, mode JWST_NIRSpec_G395H);
  uniform fallback when ExoTiC-LD or its grids are unavailable — the
  fallback is recorded in the fit summary.
- Linear decorrelation regressors: time slope + trace x/y + FWHM (from
  Stage 2 calints, standardized), via juliet ``linear_regressors_lc``.
- Tilt-event step terms: step discontinuities detected on the white
  lightcurve (out-of-transit only) get one fitted 0/1 step regressor
  each, shared by the spectroscopic fits.
- dynesty nested sampling; ecc fixed 0, omega 90.

Set ``decorrelate=False, tilt_correction=False, ld_priors='uniform'`` to
reproduce the v1.0 baseline fit for comparison.

NOTE: this module is named ``juliet.py`` but the fitting engine is the
``juliet`` package. ``_import_juliet()`` strips this file's directory
from ``sys.path`` during import so site-packages always wins.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .lightcurves import (
    DEFAULT_RESOLUTION,
    G395H_WAVE_RANGES,
    build_lightcurves,
    build_regressor_matrix,
    detect_tilt_events,  # noqa: F401 — re-export; find_tilt_events superseded it
    find_stage2_calints,
    find_tilt_events,
    load_stage3_spectra,
    pca_regressors,
    propagate_t0,
    rednoise_beta,
    tilt_transition_keep_mask,
    trace_diagnostics,
)

try:
    try:  # shared base + field helpers; see aster_toolkit/tool_base.py
        from aster_toolkit.tool_base import (AsterBaseTool as BaseTool, OptionalRuntimeField, RuntimeField, StateField)
    except ModuleNotFoundError as _exc:  # loaded by file path, outside the package
        if _exc.name != "aster_toolkit":
            raise
        import sys as _sys, pathlib as _pl
        _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
        from aster_toolkit.tool_base import (AsterBaseTool as BaseTool, OptionalRuntimeField, RuntimeField, StateField)
except ModuleNotFoundError:
    class BaseTool:
        """Fallback that keeps plain fitting wrappers importable without Orchestral."""

    def RuntimeField(default=None, description=None):
        return default

    def StateField(default=None, description=None):
        return default

    def OptionalRuntimeField(description=None, **kwargs):
        return None


def _import_juliet():
    """Import the installed ``juliet`` package, never this module."""
    import importlib
    import sys

    cached = sys.modules.get("juliet")
    if cached is not None:
        if str(getattr(cached, "__file__", "")) == str(__file__):
            del sys.modules["juliet"]
        else:
            return cached

    this_dir = os.path.abspath(os.path.dirname(__file__))
    saved_path = list(sys.path)
    sys.path = [
        p for p in sys.path if os.path.abspath(p or os.getcwd()) != this_dir
    ]
    try:
        _repair_batman_namespace()
        return importlib.import_module("juliet")
    finally:
        sys.path = saved_path


def _repair_batman_namespace() -> None:
    """Re-export ``TransitParams``/``TransitModel`` at ``batman`` top level.

    juliet calls ``batman.TransitParams()``. A second PyPI distribution
    that also installs a ``batman`` package (a CLI toolkit) can overwrite
    batman-package's ``__init__.py`` with an empty one, leaving only the
    ``batman.transitmodel`` submodule (seen in the Mac ``exo`` env). The
    classes are the same objects either way; this touches no numerics.
    """
    try:
        import batman
        if hasattr(batman, "TransitParams"):
            return
        from batman import transitmodel
    except ImportError:
        return
    batman.TransitParams = transitmodel.TransitParams
    batman.TransitModel = transitmodel.TransitModel


# Frozen survey-wide fitting settings. Bump the version if any change.
# v1.2 (decision Wasi 2026-07-30): COMPASS-style relative-pixel-flux PCA
# regressors (Ahrer et al. 2025, arXiv:2511.18196) are part of the frozen
# fit definition. A fit where the PCA components could not be built
# (no Stage 2 calints) is stamped "1.2-nopca" so it is never silently
# mixed with the survey fits.
# v1.3 (referee feedback 2026-08-03): the Stage 5.5 occulted-spot scan
# (contamination.py) is part of the frozen fit — confirmed spot and
# facula crossings are masked and the lightcurve is refit. A fit where
# the scan did not run is stamped "1.3-noscan". The stamps compose:
# "1.3-nopca-noscan" is possible and is not a survey fit.
# v1.6 (decision Wasi 2026-10-20): the white light is ONE joint fit of
# NRS1 and NRS2, and the noise model gains a GP.
#   * Joint: P, t0, a/Rs and b are shared -- the two detectors record the
#     same photons at the same instant, so the geometry is one set of
#     numbers. Depth (p_p1_NRS1/NRS2), limb darkening, jitter, baseline
#     and every regressor stay per detector. Fitted separately, HD 15337 c
#     o019 put the two mid-transits 77 min apart (NRS2 caught ingress,
#     missed egress and faked the recovery with its time slope; b 0.89 vs
#     0.14). Sarkar et al. (2026) fit TOI-1231 b's two detectors as one
#     white lightcurve for the same reason.
#   * GP: celerite Matern-3/2 in time per detector (final white pass and
#     every channel), for the correlated noise behind betas of 3-14 that
#     linear regressors leave. The channels fit GP_sigma and keep GP_rho
#     fixed to the white-light posterior, the same freeze the orbit gets.
#   * Pass 1 (input to the Stage 5.5 scan) is joint but GP-free on
#     purpose: a GP would absorb the spot bumps the scan is looking for.
# Stamps: "-1det" when only one detector had Stage 3 products (no shared
# geometry to constrain it), "-nogp" when the GP was off. Either is not a
# survey fit.
PATCHWORK_FIT_VERSION = "1.6"
SAMPLER = "dynesty"
N_LIVE_WHITE = 500
N_LIVE_SPEC = 300
LD_LAW = "quadratic"          # Kipping q1/q2 parameterization in juliet
LD_PRIOR_SIGMA = 0.1          # Gaussian width on ExoTiC-LD (q1, q2)
# Above this impact parameter the per-channel limb darkening is FIXED to
# the ExoTiC-LD values instead of fitted.
#
# Measured on the survey's own lightcurves (2026-09-04), refitting at
# per-channel signal-to-noise with limb darkening free versus fixed:
#
#   TOI-1468 c  b=0.68   3047+/-37 -> 3038+/-38 ppm   1.03x
#   GJ 3090 b   b=0.66   1502+/-30 -> 1502+/-29 ppm   0.98x
#   TOI-836 b   b=0.64    652+/-20 ->  652+/-21 ppm   1.03x
#   L 98-59 d   b=0.94   2063+/-35 -> 1965+/-87 ppm   2.47x
#
# So freeing limb darkening costs nothing at moderate b and should stay
# free there -- fixing it survey-wide would impose theoretical
# coefficients for no gain. Near grazing it is different: the transit
# turns V-shaped, limb darkening becomes degenerate with depth, the
# error inflates 2.5x AND the depth itself moves 5%. The fitted q1
# values show the mechanism plainly, collapsing towards zero against an
# ExoTiC value near 0.09 (L 98-59 d 0.005, LTT 3780 c 0.010) as the fit
# trades limb darkening against radius.
LD_FIX_IMPACT_PARAM = 0.80
LD_MODE = "JWST_NIRSpec_G395H"
LD_MODEL = "mps1"
THETA_BOUNDS = [-0.1, 0.1]    # uniform prior on every linear regressor coeff
# v1.6 joint white light: parameters both detectors share.
JOINT_SHARED_PARAMS = ("P_p1", "t0_p1", "a_p1", "b_p1")
# v1.6 GP: celerite Matern-3/2 (juliet's GP_sigma + GP_rho pair) in time.
GP_KERNEL = "celerite-matern32"
GP_SIGMA_BOUNDS = [1e-6, 1e-2]        # amplitude, relative flux (1-10000 ppm)
# rho below a few minutes is indistinguishable from the white jitter that
# sigma_w already fits; above the visit span it is a baseline offset that
# mflux and the time slope already fit. Either end would only let the two
# trade against each other.
GP_RHO_MIN_DAYS = 2.0 / 1440.0


def gp_time_input(time: np.ndarray, origin: float) -> np.ndarray:
    """The GP's input coordinate: time in days from ``origin``.

    NEVER absolute BJD. celerite's Matern-3/2 term is a semiseparable
    approximation, and at t ~ 2.46e6 d it loses the precision to resolve
    a 2-20 s cadence: factorizing the covariance failed with
    ``celerite.solver.LinAlgError`` on 15 of the 22 first v1.6 fits
    (2026-10-20), and on 38 of 44 survey lightcurves when replayed
    offline, while the same data shifted to a near-zero origin
    factorized on all 44. The kernel is stationary, so the shift changes
    nothing in the model. Where it did not crash, the absolute-time
    likelihood could still fail at some parameter values and silently
    reject them, so fits made before this fix are not trusted either
    (they carry no ``gp_time_origin`` in ``noise_model``).
    """
    return np.asarray(time, dtype=float) - float(origin)

# Back-compat alias (pre-Stage4 name).
load_extracted_spectra = load_stage3_spectra


# -------------------- published same-band depths --------------------

# The automatic depth check used to compare every white-light depth
# against the archive (Rp/Rs)^2, which for these targets comes from TESS
# discovery photometry — a different bandpass. Limb darkening differs
# between the optical and 3-5 um, and unocculted spots change the two
# bands by different amounts, so a healthy G395H depth routinely sits
# 10-15% from the TESS value and the check fired on healthy targets
# survey-wide (referee feedback, 2026-08-03). A cross-band comparison
# can only ever be indicative.
#
# Where a published *same-band* (JWST G395H) depth exists, compare
# against that instead and keep the tolerance tight. Depths in ppm,
# errors as published; keys are archive planet names, with aliases
# resolved by ``published_depth_reference``.
PUBLISHED_G395H_DEPTHS: dict[str, dict[str, Any]] = {
    "LTT 3780 c": {
        "depth_ppm": 3434.0, "err_ppm": 73.0,
        "reference": "A JWST Transmission Spectrum of the Temperate "
                     "Sub-Neptune TOI-732 c, ApJL (2025), arXiv:2512.15844",
        "note": "GO 3557; NIRSpec G395H + MIRI LRS, 0.9-12 um.",
    },
    "TOI-1231 b": {
        "depth_ppm": 5571.0, "err_ppm": 21.0,
        "reference": "Sarkar et al. (2026), arXiv:2607.00973",
        "note": "GO 3557; NIRISS SOSS + NIRSpec G395H. They model the "
                "spot crossings with SPOTROD and fit one white lightcurve "
                "from NRS1+NRS2 combined; Patchwork masks the crossings "
                "and fits the detectors separately.",
    },
}

# Names the archive and the literature disagree on. Left side is any
# name a caller might pass; right side is the PUBLISHED_G395H_DEPTHS key.
_PLANET_ALIASES = {
    "TOI-732 c": "LTT 3780 c",
    "TOI-732.02": "LTT 3780 c",
}

# Fractional tolerance before a depth is called out. Same-band: a real
# 3-sigma disagreement with a published G395H depth is worth stopping
# for. Cross-band: only a gross failure (the prior-returning fit, which
# lands ~1800 ppm regardless of target) should trigger anything.
SAME_BAND_TOL_FRAC = 0.10
CROSS_BAND_TOL_FRAC = 0.35


def published_depth_reference(planet_name: str) -> dict[str, Any] | None:
    """Published same-band (JWST G395H) transit depth for a planet, or None.

    Add entries to ``PUBLISHED_G395H_DEPTHS`` as the literature grows;
    every entry needs a citation, because this value is what the survey
    validates itself against.
    """
    if not planet_name:
        return None
    key = planet_name.strip()
    key = _PLANET_ALIASES.get(key, key)
    entry = PUBLISHED_G395H_DEPTHS.get(key)
    return dict(entry, planet_name=key) if entry else None


# -------------------- archive priors --------------------


def fetch_transit_priors(planet_name: str) -> dict[str, Any]:
    """Literature ephemeris/orbit/star from the NASA Exoplanet Archive
    (pscomppars). Seeds the white-light priors and the OOT mask."""
    from aster_toolkit.data_acquisition.mast import archive_tap_query

    rows = archive_tap_query(
        [f"pl_name = '{planet_name}'"],
        columns=[
            "pl_name", "pl_orbper", "pl_orbpererr1", "pl_tranmid",
            "pl_tranmiderr1", "pl_ratdor", "pl_ratdorerr1", "pl_ratror",
            "pl_orbincl", "pl_imppar", "pl_impparerr1", "pl_orbeccen",
            "pl_trandur", "st_rad", "st_teff", "st_logg", "st_met",
        ],
    )
    if not rows:
        raise ValueError(
            f"No NASA Exoplanet Archive entry for '{planet_name}'. "
            "Check the name format (e.g. 'GJ 1214 b')."
        )
    row = rows[0]

    def _f(key: str) -> float | None:
        value = row.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return {
        "planet_name": planet_name,
        "period": _f("pl_orbper"),
        "period_err": _f("pl_orbpererr1"),
        "t0": _f("pl_tranmid"),
        "t0_err": _f("pl_tranmiderr1"),
        "a_rs": _f("pl_ratdor"),
        "a_rs_err": _f("pl_ratdorerr1"),
        "rp_rs": _f("pl_ratror"),
        "inclination": _f("pl_orbincl"),
        "impact_param": _f("pl_imppar"),
        "impact_param_err": _f("pl_impparerr1"),
        "eccentricity": _f("pl_orbeccen"),
        "duration_hr": _f("pl_trandur"),
        "st_rad": _f("st_rad"),
        "st_teff": _f("st_teff"),
        "st_logg": _f("st_logg"),
        "st_met": _f("st_met"),
    }


# Back-compat alias.
def predict_mid_transit(t0_ref: float, period: float, times: np.ndarray) -> float:
    return propagate_t0(t0_ref, period, times)


# -------------------- ExoTiC-LD priors --------------------


def _ld_data_path() -> str:
    return os.environ.get(
        "ASTER_EXOTIC_LD_DATA", str(Path.home() / "exotic_ld_data")
    )


def compute_ld_coeffs(
    st_teff: float, st_logg: float, st_met: float,
    wave_min_um: float, wave_max_um: float,
) -> dict[str, float] | None:
    """Quadratic limb-darkening (u1, u2) -> Kipping (q1, q2) from
    ExoTiC-LD stellar grids over one wavelength range.

    Tries an in-process import first, then falls back to running inside
    the pinned exoTEDRF environment (where exotic_ld is installed).
    Returns None (-> uniform LD priors) if both fail; callers must record
    that in the fit summary.
    """
    args = (st_teff, st_logg, st_met, wave_min_um, wave_max_um)
    if any(a is None for a in args):
        return None

    script = (
        "import json, sys\n"
        "from exotic_ld import StellarLimbDarkening\n"
        "teff, logg, met, wmin, wmax, path = (\n"
        "    float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]),\n"
        "    float(sys.argv[4]), float(sys.argv[5]), sys.argv[6])\n"
        f"sld = StellarLimbDarkening(M_H=met, Teff=teff, logg=logg,\n"
        f"                           ld_model='{LD_MODEL}', ld_data_path=path)\n"
        "u1, u2 = sld.compute_quadratic_ld_coeffs(\n"
        "    wavelength_range=[wmin * 1e4, wmax * 1e4],\n"
        f"    mode='{LD_MODE}')\n"
        "print(json.dumps([u1, u2]))\n"
    )

    def _in_process():
        from exotic_ld import StellarLimbDarkening

        sld = StellarLimbDarkening(
            M_H=st_met, Teff=st_teff, logg=st_logg,
            ld_model=LD_MODEL, ld_data_path=_ld_data_path(),
        )
        return sld.compute_quadratic_ld_coeffs(
            wavelength_range=[wave_min_um * 1e4, wave_max_um * 1e4],
            mode=LD_MODE,
        )

    def _subprocess():
        from .exotedrf import _exotedrf_python

        result = subprocess.run(
            [_exotedrf_python(), "-c", script,
             str(st_teff), str(st_logg), str(st_met),
             str(wave_min_um), str(wave_max_um), _ld_data_path()],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-500:])
        return json.loads(result.stdout.strip().splitlines()[-1])

    u1 = u2 = None
    for attempt in (_in_process, _subprocess):
        try:
            u1, u2 = attempt()
            break
        except Exception:
            continue
    if u1 is None:
        return None

    u1, u2 = float(u1), float(u2)
    s = u1 + u2
    q1 = s**2
    q2 = u1 / (2 * s) if s != 0 else 0.5
    return {"u1": u1, "u2": u2,
            "q1": float(np.clip(q1, 0, 1)), "q2": float(np.clip(q2, 0, 1))}


def compute_ld_coeffs_batch(
    st_teff: float, st_logg: float, st_met: float,
    ranges: list[tuple[float, float]],
) -> list[dict[str, float] | None]:
    """Quadratic -> Kipping LD coefficients for MANY wavelength ranges in
    one pass. The per-channel loop in ``fit_transmission_spectrum`` used
    to spawn one exoTEDRF-env subprocess per channel (~53 per detector,
    each reloading the stellar grids from disk); this builds the
    ``StellarLimbDarkening`` object once and evaluates every range.

    Returns one entry per range, None where the computation failed
    (-> uniform priors for that channel).
    """
    if any(a is None for a in (st_teff, st_logg, st_met)) or not ranges:
        return [None] * len(ranges)

    def _u_to_q(u1: float, u2: float) -> dict[str, float]:
        s = u1 + u2
        q1 = s**2
        q2 = u1 / (2 * s) if s != 0 else 0.5
        return {"u1": float(u1), "u2": float(u2),
                "q1": float(np.clip(q1, 0, 1)),
                "q2": float(np.clip(q2, 0, 1))}

    def _in_process():
        from exotic_ld import StellarLimbDarkening

        sld = StellarLimbDarkening(
            M_H=st_met, Teff=st_teff, logg=st_logg,
            ld_model=LD_MODEL, ld_data_path=_ld_data_path(),
        )
        out = []
        for w0, w1 in ranges:
            try:
                u1, u2 = sld.compute_quadratic_ld_coeffs(
                    wavelength_range=[w0 * 1e4, w1 * 1e4], mode=LD_MODE)
                out.append((float(u1), float(u2)))
            except Exception:
                out.append(None)
        return out

    def _subprocess():
        from .exotedrf import _exotedrf_python

        script = (
            "import json, sys\n"
            "from exotic_ld import StellarLimbDarkening\n"
            "teff, logg, met, path = (float(sys.argv[1]), float(sys.argv[2]),\n"
            "                         float(sys.argv[3]), sys.argv[4])\n"
            "ranges = json.loads(sys.argv[5])\n"
            f"sld = StellarLimbDarkening(M_H=met, Teff=teff, logg=logg,\n"
            f"                           ld_model='{LD_MODEL}', ld_data_path=path)\n"
            "out = []\n"
            "for w0, w1 in ranges:\n"
            "    try:\n"
            "        u1, u2 = sld.compute_quadratic_ld_coeffs(\n"
            f"            wavelength_range=[w0 * 1e4, w1 * 1e4], mode='{LD_MODE}')\n"
            "        out.append([float(u1), float(u2)])\n"
            "    except Exception:\n"
            "        out.append(None)\n"
            "print('LD_BATCH_JSON ' + json.dumps(out))\n"
        )
        result = subprocess.run(
            [_exotedrf_python(), "-c", script,
             str(st_teff), str(st_logg), str(st_met), _ld_data_path(),
             json.dumps([[float(a), float(b)] for a, b in ranges])],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-500:])
        for line in result.stdout.splitlines():
            if line.startswith("LD_BATCH_JSON "):
                return [tuple(v) if v else None
                        for v in json.loads(line[len("LD_BATCH_JSON "):])]
        raise RuntimeError("no LD_BATCH_JSON line in subprocess output")

    pairs = None
    for attempt in (_in_process, _subprocess):
        try:
            pairs = attempt()
            break
        except Exception:
            continue
    if pairs is None:
        return [None] * len(ranges)
    return [_u_to_q(*p) if p is not None else None for p in pairs]


def _ld_prior_entries(instrument: str, ld: dict[str, float] | None) -> dict[str, dict]:
    """Truncated-normal (q1, q2) priors when ExoTiC-LD coefficients are
    available, uniform otherwise."""
    if ld is None:
        return {
            f"q1_{instrument}": {"distribution": "uniform", "hyperparameters": [0.0, 1.0]},
            f"q2_{instrument}": {"distribution": "uniform", "hyperparameters": [0.0, 1.0]},
        }
    return {
        f"q1_{instrument}": {
            "distribution": "truncatednormal",
            "hyperparameters": [ld["q1"], LD_PRIOR_SIGMA, 0.0, 1.0],
        },
        f"q2_{instrument}": {
            "distribution": "truncatednormal",
            "hyperparameters": [ld["q2"], LD_PRIOR_SIGMA, 0.0, 1.0],
        },
    }


# -------------------- priors --------------------


def _common_instrument_priors(instrument: str, n_regressors: int) -> dict[str, dict]:
    p = {
        f"mdilution_{instrument}": {"distribution": "fixed", "hyperparameters": 1.0},
        f"mflux_{instrument}": {"distribution": "normal", "hyperparameters": [0.0, 0.1]},
        f"sigma_w_{instrument}": {"distribution": "loguniform",
                                  "hyperparameters": [0.1, 10000.0]},
    }
    for i in range(n_regressors):
        p[f"theta{i}_{instrument}"] = {
            "distribution": "uniform", "hyperparameters": list(THETA_BOUNDS),
        }
    return p


def _white_priors(
    priors: dict[str, Any],
    t0_obs: float,
    instrument: str,
    n_regressors: int,
    ld: dict[str, float] | None,
) -> dict[str, dict[str, Any]]:
    p = _orbit_priors(priors, t0_obs)
    p["p_p1"] = {"distribution": "uniform", "hyperparameters": [0.0, 0.3]}
    p.update(_ld_prior_entries(instrument, ld))
    p.update(_common_instrument_priors(instrument, n_regressors))
    return p


def _orbit_priors(priors: dict[str, Any], t0_obs: float) -> dict[str, dict[str, Any]]:
    """P, t0, b, a/Rs (+ circular orbit): the geometry of one transit."""
    period = priors["period"]
    if period is None:
        raise ValueError("No orbital period available from the archive.")

    p: dict[str, dict[str, Any]] = {
        "P_p1": {
            "distribution": "normal",
            "hyperparameters": [period, max(priors["period_err"] or 0, 1e-5) * 3],
        },
        # t0 propagated to this visit; 0.01 d Gaussian (validated run).
        "t0_p1": {"distribution": "normal", "hyperparameters": [t0_obs, 0.01]},
        "b_p1": {"distribution": "uniform", "hyperparameters": [0.0, 1.0]},
        "ecc_p1": {"distribution": "fixed", "hyperparameters": 0.0},
        "omega_p1": {"distribution": "fixed", "hyperparameters": 90.0},
    }
    a_rs = priors["a_rs"]
    if a_rs is not None:
        err = priors["a_rs_err"] or 0.1 * a_rs
        p["a_p1"] = {"distribution": "normal", "hyperparameters": [a_rs, 3 * err]}
    else:
        p["a_p1"] = {"distribution": "uniform", "hyperparameters": [1.0, 300.0]}
    return p


def _gp_priors(instrument: str, time: np.ndarray) -> dict[str, dict[str, Any]]:
    """Matern-3/2 GP in time for one detector (juliet: GP_sigma + GP_rho)."""
    span = float(np.ptp(np.asarray(time, dtype=float)))
    return {
        f"GP_sigma_{instrument}": {"distribution": "loguniform",
                                   "hyperparameters": list(GP_SIGMA_BOUNDS)},
        f"GP_rho_{instrument}": {"distribution": "loguniform",
                                 "hyperparameters": [GP_RHO_MIN_DAYS,
                                                     max(span, 2 * GP_RHO_MIN_DAYS)]},
    }


def _joint_white_priors(
    priors: dict[str, Any],
    t0_obs: float,
    detectors: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Priors for one white-light fit of every detector in ``detectors``.

    ``detectors`` maps a detector name to ``{"n_regressors", "ld",
    "gp_time"}`` (``gp_time`` None = no GP). The orbit (JOINT_SHARED_PARAMS)
    is one set of parameters; the depth is ``p_p1_<det>`` per detector.
    """
    p = _orbit_priors(priors, t0_obs)
    for det, spec in detectors.items():
        p[f"p_p1_{det}"] = {"distribution": "uniform", "hyperparameters": [0.0, 0.3]}
        p.update(_ld_prior_entries(det, spec.get("ld")))
        p.update(_common_instrument_priors(det, int(spec["n_regressors"])))
        if spec.get("gp_time") is not None:
            p.update(_gp_priors(det, spec["gp_time"]))
    return p


def _require_celerite() -> None:
    """The v1.6 GP needs celerite. Fail before sampling, never silently
    fall back to a white-noise fit stamped as a survey fit."""
    try:
        import celerite  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Patchwork fit v1.6 models red noise with a celerite GP, and "
            "celerite is not importable in this environment. Install it "
            "(pip install celerite) in the env that runs the fit."
        ) from exc


# -------------------- fitting --------------------


def evaluate_depth_check(
    measured_ppm: float,
    err_ppm: float,
    priors: dict[str, Any],
    *,
    planet_name: str = "",
) -> dict[str, Any]:
    """Compare a white-light depth against the best available reference.

    Reference selection, in order:

    1. A published JWST/G395H depth for this planet
       (``PUBLISHED_G395H_DEPTHS``) — the same bandpass, same physics.
       Tolerance ``SAME_BAND_TOL_FRAC``, and a disagreement is flagged.
    2. The archive (Rp/Rs)^2, which for the Patchwork targets is TESS
       discovery photometry. This is a *different band*: limb darkening
       differs, and unocculted spots on an M dwarf change the optical and
       infrared depths by different amounts. Measured G395H depths in
       Patchwork 1 ran +0.3% to +14% from the TESS value on targets whose
       fits were independently verified as healthy, so a cross-band
       comparison flags healthy targets and cannot be called "suspect".
       Tolerance ``CROSS_BAND_TOL_FRAC``, and the status stays
       ``indicative``.

    Either way an error bar larger than half the expected depth is a
    hard failure — that is the signature of a fit that returned its
    prior, and it does not depend on the bandpass.

    Returns the check dict stored in ``white_fit_summary.json``:
    ``status`` is one of ``ok``, ``suspect`` (same-band disagreement or a
    prior-returning fit), ``indicative`` (cross-band, informational), or
    ``unavailable``. ``suspect`` is kept as a boolean for back-compat
    and is True only for ``status == 'suspect'``.
    """
    published = published_depth_reference(planet_name)
    if published:
        expected = float(published["depth_ppm"])
        expected_err = float(published.get("err_ppm") or 0.0)
        band, source = "same-band", published["reference"]
        tol_frac = SAME_BAND_TOL_FRAC
    elif priors.get("rp_rs"):
        expected = float(priors["rp_rs"]) ** 2 * 1e6
        expected_err = 0.0
        band, source = "cross-band", "NASA Exoplanet Archive (Rp/Rs)^2, optical"
        tol_frac = CROSS_BAND_TOL_FRAC
    else:
        return {"status": "unavailable", "suspect": False,
                "reason": "No published G395H depth and no archive Rp/Rs."}

    # Combined scale for the disagreement: the fit error, the published
    # error, and the tolerance floor. Without the floor a target with a
    # 21 ppm published error and a 6 ppm fit error is "5 sigma off" at a
    # 0.5% difference that no one would question.
    scale = np.hypot(err_ppm, expected_err)
    sigma_off = abs(measured_ppm - expected) / scale if scale > 0 else float("inf")
    frac_off = abs(measured_ppm - expected) / expected if expected else float("inf")

    returned_prior = err_ppm > 0.5 * expected
    disagrees = frac_off > tol_frac and sigma_off > 3.0

    if returned_prior:
        status = "suspect"
    elif band == "same-band" and disagrees:
        status = "suspect"
    elif band == "cross-band" and disagrees:
        status = "indicative"
    else:
        status = "ok"

    return {
        "status": status,
        "suspect": status == "suspect",
        "band": band,
        "reference_source": source,
        "expected_ppm": expected,
        "expected_err_ppm": expected_err,
        "measured_ppm": float(measured_ppm),
        "err_ppm": float(err_ppm),
        "difference_ppm": float(measured_ppm - expected),
        "difference_frac": float(measured_ppm / expected - 1.0) if expected else None,
        "difference_sigma": float(sigma_off),
        "tolerance_frac": tol_frac,
        "returned_prior": bool(returned_prior),
        # Kept so nothing downstream that reads the old key breaks.
        "expected_ppm_from_archive": expected,
    }


def _apply_keep_mask(arrays: dict[str, np.ndarray], keep: np.ndarray | None):
    """Subset parallel per-integration arrays by a KEEP mask."""
    if keep is None:
        return arrays
    keep = np.asarray(keep, dtype=bool)
    return {k: (v[keep] if getattr(v, "shape", (0,))[:1] == keep.shape else v)
            for k, v in arrays.items()}


def fit_white_lightcurve(
    lc: dict[str, Any],
    priors: dict[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    instrument: str = "NRS1",
    regressors: np.ndarray | None = None,
    regressor_names: list[str] | None = None,
    ld: dict[str, float] | None = None,
    model_type: str = "transit",
    program: str = "",
    visit: str = "",
    n_live: int = N_LIVE_WHITE,
    keep_mask: np.ndarray | None = None,
    anomaly_scan: dict[str, Any] | None = None,
    tilt_report: dict[str, Any] | None = None,
    n_spot_masked: int = 0,
    n_tilt_masked: int = 0,
    gp: bool = True,
) -> dict[str, Any]:
    """Fit the white-light transit of ONE detector with juliet.

    The single-detector form of ``fit_white_lightcurve_joint``, kept for
    the per-detector tools and smoke tests. The survey fits both
    detectors jointly; a fit made here is stamped ``-1det`` because
    nothing outside this one detector constrains its geometry.
    """
    if model_type != "transit":
        raise NotImplementedError(
            "Only model_type='transit' is implemented. Phase-curve support "
            "is a planned extension."
        )
    out = fit_white_lightcurve_joint(
        {instrument: lc}, priors, {instrument: output_dir},
        joint_dir=output_dir,
        regressors={instrument: regressors},
        regressor_names={instrument: regressor_names},
        ld={instrument: ld},
        keep_masks={instrument: keep_mask},
        gp=gp, program=program, visit=visit, n_live=n_live,
        anomaly_scan=anomaly_scan,
        tilt_reports={instrument: tilt_report},
        n_spot_masked={instrument: n_spot_masked},
        n_tilt_masked={instrument: n_tilt_masked},
    )
    return out[instrument]


def _posterior_stat(samples: np.ndarray) -> dict[str, float]:
    med, lo, hi = np.percentile(samples, [50, 16, 84])
    return {"median": float(med), "minus": float(med - lo), "plus": float(hi - med)}


def fit_white_lightcurve_joint(
    lcs: dict[str, dict[str, Any]],
    priors: dict[str, Any],
    output_dirs: dict[str, str | os.PathLike[str]],
    *,
    joint_dir: str | os.PathLike[str],
    regressors: dict[str, np.ndarray | None] | None = None,
    regressor_names: dict[str, list[str] | None] | None = None,
    ld: dict[str, dict[str, float] | None] | None = None,
    keep_masks: dict[str, np.ndarray | None] | None = None,
    gp: bool = True,
    program: str = "",
    visit: str = "",
    n_live: int = N_LIVE_WHITE,
    anomaly_scan: dict[str, Any] | None = None,
    tilt_reports: dict[str, dict[str, Any] | None] | None = None,
    n_spot_masked: dict[str, int] | None = None,
    n_tilt_masked: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit one visit's white light on every detector at once (fit v1.6).

    ``lcs`` maps detector -> Stage 4 lightcurve. The orbit
    (JOINT_SHARED_PARAMS: P, t0, a/Rs, b) is shared; depth
    (``p_p1_<det>``), limb darkening, mflux, jitter, regressors and the GP
    are per detector. juliet writes its products to ``joint_dir``; each
    detector gets the usual per-detector summary, residual npz and figure
    in ``output_dirs[det]``, with its own depth reported as ``p_p1`` so
    every downstream reader (spectroscopic fits, plots, depth check)
    works unchanged.

    ``gp=True`` adds a Matern-3/2 GP in time per detector. Leave it off
    only for pass 1, whose residuals feed the Stage 5.5 scan.

    Returns ``{det: summary}``.
    """
    juliet = _import_juliet()
    if gp:
        _require_celerite()
    dets = list(lcs)
    if not dets:
        raise ValueError("fit_white_lightcurve_joint needs at least one detector.")
    regressors = dict(regressors or {})
    regressor_names = dict(regressor_names or {})
    ld = dict(ld or {})
    keep_masks = dict(keep_masks or {})
    tilt_reports = dict(tilt_reports or {})
    n_spot_masked = dict(n_spot_masked or {})
    n_tilt_masked = dict(n_tilt_masked or {})

    jdir = Path(joint_dir)
    jdir.mkdir(parents=True, exist_ok=True)

    times, keeps, regs, names = {}, {}, {}, {}
    for det in dets:
        lc = lcs[det]
        t = lc["time"]
        r = regressors.get(det)
        n = regressor_names.get(det)
        if r is None:
            r, n = build_regressor_matrix(t)
        keep = (np.ones(t.size, dtype=bool) if keep_masks.get(det) is None
                else np.asarray(keep_masks[det], dtype=bool))
        if keep.size != t.size:
            raise ValueError(
                f"{det}: keep_mask has {keep.size} entries but the lightcurve "
                f"has {t.size} integrations."
            )
        if keep.sum() < 50:
            raise ValueError(
                f"{det}: the anomaly mask leaves only {int(keep.sum())} of "
                f"{t.size} integrations. Masking that much of a visit is not a "
                "correction — inspect the Stage 5.5 scan; the event is probably "
                "instrumental or the whole visit is bad."
            )
        times[det], keeps[det], regs[det], names[det] = t, keep, r, list(n or [])

    # Both detectors are propagated from the same ephemeris, so their
    # t0_obs agree; the mean only guards against rounding.
    t0_obs = float(np.mean([lcs[d]["t0_obs"] for d in dets]))
    # The GP sees time from the predicted mid-transit, never absolute BJD
    # (see gp_time_input). One origin for both detectors.
    gp_origin = t0_obs
    prior_dict = _joint_white_priors(priors, t0_obs, {
        det: {"n_regressors": regs[det].shape[1], "ld": ld.get(det),
              "gp_time": times[det][keeps[det]] if gp else None}
        for det in dets
    })
    dataset = juliet.load(
        priors=prior_dict,
        t_lc={d: times[d][keeps[d]] for d in dets},
        y_lc={d: lcs[d]["wl_flux"][keeps[d]] for d in dets},
        yerr_lc={d: lcs[d]["wl_err"][keeps[d]] for d in dets},
        linear_regressors_lc={d: regs[d][keeps[d]] for d in dets},
        GP_regressors_lc=({d: gp_time_input(times[d][keeps[d]], gp_origin)
                           for d in dets} if gp else None),
        out_folder=str(jdir),
    )
    results = dataset.fit(sampler=SAMPLER, n_live_points=n_live, verbose=False)
    posterior = results.posteriors["posterior_samples"]

    joint_record = {
        "detectors": dets,
        "shared": list(JOINT_SHARED_PARAMS),
        "joint_dir": str(jdir),
    }
    summaries: dict[str, dict[str, Any]] = {}
    for det in dets:
        summaries[det] = _summarize_white_detector(
            results, posterior, det, lcs[det], priors,
            Path(output_dirs[det]),
            keep=keeps[det], regressor_names=names[det], ld=ld.get(det),
            gp=gp, gp_origin=gp_origin, n_detectors=len(dets),
            joint_record=joint_record,
            program=program, visit=visit,
            anomaly_scan=anomaly_scan, tilt_report=tilt_reports.get(det),
            n_spot_masked=int(n_spot_masked.get(det, 0)),
            n_tilt_masked=int(n_tilt_masked.get(det, 0)),
        )
    if len(dets) > 1:
        shared = {k: summaries[dets[0]][k] for k in JOINT_SHARED_PARAMS
                  if k in summaries[dets[0]]}
        with (jdir / "joint_white_fit_summary.json").open("w") as handle:
            json.dump({"fit_version": summaries[dets[0]]["fit_version"],
                       "detectors": dets, "shared": shared,
                       "depth_ppm": {d: summaries[d]["depth_ppm"] for d in dets}},
                      handle, indent=2)
    return summaries


def _summarize_white_detector(
    results,
    posterior: dict[str, np.ndarray],
    instrument: str,
    lc: dict[str, Any],
    priors: dict[str, Any],
    out: Path,
    *,
    keep: np.ndarray,
    regressor_names: list[str],
    ld: dict[str, float] | None,
    gp: bool,
    n_detectors: int,
    gp_origin: float | None = None,
    joint_record: dict[str, Any],
    program: str,
    visit: str,
    anomaly_scan: dict[str, Any] | None,
    tilt_report: dict[str, Any] | None,
    n_spot_masked: int,
    n_tilt_masked: int,
) -> dict[str, Any]:
    """One detector's white-fit summary, residuals and figure from a
    (possibly joint) juliet result. Same layout as every fit before v1.6,
    plus ``joint_fit`` and ``noise_model``."""
    out.mkdir(parents=True, exist_ok=True)
    times = lc["time"]
    n_reg = len(regressor_names)
    n_masked = int((~keep).sum())

    # PCA detrending is part of frozen v1.2. A fit without the PCA
    # columns (calints unavailable, or the as-is escape hatch) is
    # stamped so it can never be silently mixed with the survey fits.
    fit_version = PATCHWORK_FIT_VERSION
    if not any(str(n).startswith("pca_") for n in regressor_names):
        fit_version = f"{fit_version}-nopca"
    # v1.3: the Stage 5.5 occulted-spot scan is part of the frozen fit.
    # "scan ran and found nothing" and "scan never ran" are different
    # states and must not share a version stamp.
    if anomaly_scan is None:
        fit_version = f"{fit_version}-noscan"
    # v1.6: a lone detector has no partner to share the geometry with,
    # and a GP-free fit has a different noise model.
    if n_detectors < 2:
        fit_version = f"{fit_version}-1det"
    if not gp:
        fit_version = f"{fit_version}-nogp"
    summary: dict[str, Any] = {
        "instrument": instrument,
        "fit_version": fit_version,
        "regressor_names": list(regressor_names),
        "ld_source": "exotic-ld" if ld is not None else "uniform",
        "ld_coeffs": ld,
        "t0_obs": lc["t0_obs"],
        "planet_name": priors.get("planet_name", ""),
        "program": program,
        "visit": visit,
        "joint_fit": dict(joint_record, n_detectors=n_detectors),
        "noise_model": {"kind": GP_KERNEL if gp else "white"},
        "spot_handling": {
            "scan_ran": anomaly_scan is not None,
            "method": ("mask" if n_spot_masked else
                       "gaussian regressor" if any(
                           str(name).startswith("spot_gauss_")
                           for name in regressor_names) else
                       "none detected" if anomaly_scan is not None else None),
            "n_masked": int(n_spot_masked),
            "frac_masked": float(n_spot_masked / max(1, times.size)),
            "contam_version": (anomaly_scan or {}).get("contam_version"),
            "events": [
                {k: e.get(k) for k in
                 ("kind", "confirmed", "amplitude_ppm", "peak_sigma",
                  "hours_from_mid", "amplitude_ratio", "achromatic")}
                for e in (anomaly_scan or {}).get("events", [])
                if e.get("kind") != "step"
            ],
        },
        # Tilt events are CORRECTED, not masked: one Heaviside step
        # regressor each, amplitude free (and free per channel
        # downstream, since the flux change is pixel-dependent and can
        # go either way). Only the few integrations at the transition
        # are dropped.
        "tilt_handling": {
            "search_ran": tilt_report is not None,
            "method": ("heaviside step regressor + transition mask"
                       if tilt_report is not None else None),
            "n_events": len(((tilt_report or {}).get("events")) or []),
            "n_transition_masked": int(n_tilt_masked),
            "searched_diagnostics": bool(
                (tilt_report or {}).get("diagnostics_available")),
            "sources_searched": (tilt_report or {}).get("sources_searched"),
            "rate_warning": (tilt_report or {}).get("rate_warning"),
            "events": [
                {k: e.get(k) for k in
                 ("index", "width_ints", "amplitude_ppm", "hours_from_mid",
                  "in_transit", "n_sources", "sources", "source",
                  "fitted_shape")}
                for e in ((tilt_report or {}).get("events") or [])
            ],
        },
        "n_masked_total": n_masked,
        "n_used": int(keep.sum()),
    }
    # A joint fit names this detector's depth p_p1_<det>; a single-detector
    # fit may name it either way. Report it as p_p1 regardless, so readers
    # of white_fit_summary.json never need to know which kind of fit ran.
    p_key = f"p_p1_{instrument}" if f"p_p1_{instrument}" in posterior else "p_p1"
    keys = list(JOINT_SHARED_PARAMS) + [
        f"q1_{instrument}", f"q2_{instrument}", f"sigma_w_{instrument}",
        f"GP_sigma_{instrument}", f"GP_rho_{instrument}",
    ]
    keys += [f"theta{i}_{instrument}" for i in range(n_reg)]
    for key in keys:
        if key in posterior:
            summary[key] = _posterior_stat(posterior[key])
    summary["p_p1"] = _posterior_stat(posterior[p_key])
    d = np.percentile(posterior[p_key] ** 2, [50, 16, 84]) * 1e6
    summary["depth_ppm"] = {"median": float(d[0]),
                            "minus": float(d[0] - d[1]), "plus": float(d[2] - d[0])}
    if gp:
        summary["noise_model"].update({
            "GP_sigma_ppm": summary[f"GP_sigma_{instrument}"]["median"] * 1e6,
            "GP_rho_min": summary[f"GP_rho_{instrument}"]["median"] * 1440.0,
            # The channels must use the same GP input coordinate. Its
            # presence also marks a fit made after the absolute-time fix.
            "gp_time_origin": float(gp_origin),
        })

    flux = lc["wl_flux"][keep]
    model = results.lc.evaluate(instrument)
    residual = flux - model
    # juliet's full model is transit + linear model + GP. Without the GP
    # the residual still holds the correlated noise -- that is the series
    # the pre-v1.6 beta was measured on, so it is kept for comparison.
    if gp:
        _, components = results.lc.evaluate(instrument, return_components=True)
        deterministic = np.asarray(components["transit"], dtype=float)
        if "lm" in components:
            deterministic = deterministic + np.asarray(components["lm"], dtype=float)
        gp_component = model - deterministic
    else:
        gp_component = np.zeros_like(model)
    summary["residual_rms_ppm"] = float(np.nanstd(residual) * 1e6)

    # Stage 5.5 reads these back to scan for occulted-spot crossings
    # without refitting. Indices are the ORIGINAL integration numbers, so
    # a mask derived from a scan of these residuals lines up with the
    # full Stage 4 lightcurve even when this fit was already masked.
    np.savez(
        out / "white_lightcurve_residuals.npz",
        detector=np.asarray(instrument),
        index=np.flatnonzero(keep),
        n_total=np.asarray(times.size),
        time=times[keep],
        flux=flux,
        flux_err=lc["wl_err"][keep],
        model=model,
        residual=residual,
        gp=gp_component,
        oot_mask=lc["oot_mask"][keep],
        t0_obs=np.asarray(lc["t0_obs"]),
        n_integrations_full=np.asarray(times.size),
    )
    # Red-noise diagnostic (Pont+2006 beta over 5-30 min bins). COMPASS
    # finds real G395H errors run ~5-12% above the photon prediction;
    # beta_median >> 1 here means this target's depth errors need
    # inflating before population-level use.
    # times[keep], NOT times: the residual is computed on the kept
    # subset, so passing the full time axis mismatches the two arrays.
    # Latent until 2026-08-27 -- the tilt search had been finding zero
    # events on every target, so keep was all-True and the sizes agreed
    # by accident. The moment a step got masked (7 integrations per
    # event) it raised IndexError mid-fit, after the sampling was
    # already done.
    # rednoise is measured WITHOUT the GP term (data - transit - linear
    # model): how much correlated noise there was, the number every
    # version since v1.2 has reported. With a GP the depth error already
    # carries that noise, so beta here is a description, not a reason to
    # inflate anything.
    # rednoise_after_gp subtracts the GP's conditional mean as well. That
    # mean follows short-timescale noise, so the binned residual shrinks
    # and this beta sits BELOW 1 by construction (0.2-0.8 on the first
    # v1.6 fits). It is only informative on the high side: > ~1.5 means
    # structure even the GP could not describe.
    summary["rednoise"] = rednoise_beta(residual + gp_component, times[keep])
    if gp:
        summary["rednoise_after_gp"] = rednoise_beta(residual, times[keep])

    # Sanity check against the best available reference depth. Prefers a
    # published same-band (G395H) value; falls back to the archive
    # optical (Rp/Rs)^2, which can only ever be indicative. See
    # evaluate_depth_check.
    check = evaluate_depth_check(
        summary["depth_ppm"]["median"],
        max(summary["depth_ppm"]["plus"], summary["depth_ppm"]["minus"]),
        priors,
        planet_name=priors.get("planet_name", ""),
    )
    summary["depth_check"] = check
    if check["status"] == "suspect":
        print(f"[patchwork] WARNING ({instrument} {visit}): white-light "
              f"depth {check['measured_ppm']:.0f} +/- {check['err_ppm']:.0f} "
              f"ppm vs {check['band']} reference {check['expected_ppm']:.0f} "
              f"ppm ({check['difference_sigma']:.1f} sigma) — "
              f"{check['reference_source']}. Check the fit before using this "
              "spectrum (see depth_check in white_fit_summary.json).")
    elif check["status"] == "indicative" and check.get("difference_frac") is not None:
        print(f"[patchwork] note ({instrument} {visit}): white-light depth is "
              f"{check['difference_frac'] * 100:+.1f}% from the optical "
              "archive value. Different bandpass — informational only, not a "
              "problem in itself.")

    _plot_white_fit(
        results, lc, instrument, out, summary,
        title=figure_title(priors.get("planet_name", ""), instrument,
                           program=program, visit=visit,
                           suffix="white light"),
        keep=keep, model=model)

    with (out / "white_fit_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    summary["output_dir"] = str(out)
    return summary


# Patchwork house style for circulated figures: no grid (it competes with
# error bars at these amplitudes), muted data, one strong accent for the model.
_PLOT_STYLE = {
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "legend.frameon": False, "legend.fontsize": 10.5,
    "figure.dpi": 150,
}
DATA_COLOR = "#1A2F6B"      # deep navy
BINNED_COLOR = "#0B1B45"
RAW_COLOR = "#8E9AB3"       # slate -- unbinned integrations sit behind
EXCLUDED_COLOR = "#C2573A"  # rust -- integrations the fit dropped
MODEL_COLOR = "#E8A33D"     # amber -- reads strongly over dense navy points
NRS2_COLOR = "#1B7F79"      # teal
GAP_COLOR = "#B9C2CC"
MEAN_COLOR = "#8A94A6"


def figure_title(planet: str, instrument: str = "", *, program: str = "",
                 visit: str = "", suffix: str = "") -> str:
    """Standard Patchwork figure heading.

    "GJ 9827 d   ·   GO 4098   ·   JWST NIRSpec/G395H NRS1   ·   visit o010"
    Empty fields are dropped rather than left as blanks.
    """
    mode = "JWST NIRSpec/G395H"
    if instrument:
        mode += f" {instrument.upper()}"
    parts = [p for p in (planet, program, mode) if p]
    if visit:
        parts.append(f"visit {visit}")
    if suffix:
        parts.append(suffix)
    return "   ·   ".join(parts)


# Display bins for the white-light figure: ~10-15 min over a typical visit.
WHITE_PLOT_BINS = 45


def _bin_series(x, y, n_bins: int = 90):
    """Equal-count binning for a readable overlay on dense photometry."""
    idx = np.argsort(x)
    x, y = np.asarray(x)[idx], np.asarray(y)[idx]
    edges = np.linspace(0, x.size, n_bins + 1).astype(int)
    bx, by, be = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 3:
            continue
        bx.append(np.mean(x[a:b])); by.append(np.mean(y[a:b]))
        be.append(np.std(y[a:b], ddof=1) / np.sqrt(b - a))
    return map(np.asarray, (bx, by, be))


def _savefig(fig, out: Path, stem: str) -> str:
    # Patchwork figure rule: PDF + SVG, nothing else.
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.svg")
    return str(out / f"{stem}.pdf")


def transit_only_model(summary: dict[str, Any], time: np.ndarray,
                       instrument: str | None = None) -> np.ndarray:
    """The fitted transit alone, from the white-fit posterior medians.

    juliet's full model is the transit plus a linear model in every
    regressor (time, trace x/y/FWHM, PCA). Those regressors jitter from
    integration to integration, so the full model drawn over the data is
    a ragged band, not a transit -- which hides whether the TRANSIT was
    fitted at all. This is the batman light curve juliet evaluates
    inside that model (quadratic law, Kipping q1/q2, circular orbit),
    normalised to 1 out of transit.
    """
    from batman.transitmodel import TransitModel, TransitParams

    inst = instrument or summary.get("instrument", "")
    med = lambda key: float(summary[key]["median"])  # noqa: E731
    q1, q2 = med(f"q1_{inst}"), med(f"q2_{inst}")
    params = TransitParams()
    params.t0, params.per = med("t0_p1"), med("P_p1")
    params.rp, params.a = med("p_p1"), med("a_p1")
    params.inc = float(np.degrees(np.arccos(med("b_p1") / med("a_p1"))))
    params.ecc, params.w = 0.0, 90.0
    params.limb_dark = "quadratic"
    params.u = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)]
    time = np.asarray(time, dtype=float)
    return TransitModel(params, time).light_curve(params)


def plot_white_fit_series(
    out: Path,
    summary: dict[str, Any],
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    model: np.ndarray,
    *,
    title: str = "",
    excluded_time: np.ndarray | None = None,
    excluded_flux: np.ndarray | None = None,
    n_bins: int | None = None,
) -> str:
    """White-light fit figure: systematics removed, transit model alone.

    The data are drawn as ``flux - (model - transit)``, i.e. the fitted
    transit plus the residuals, with the transit as one smooth curve.
    The residual panel is unchanged by this (it is ``flux - model``
    either way), so nothing the fit got wrong is hidden. Time is centred
    on the FITTED mid-transit. When the ephemeris prediction sits more
    than 5 min away, it is marked, because a large offset is exactly the
    situation in which the t0 prior fights the data.

    The annotated depth is the posterior median of p^2, never
    ``1 - min(model)``: the latter contains the systematics too
    (TOI-270 c: 7249 vs 5230 ppm).

    ``excluded_*`` are integrations dropped at fitted step transitions.
    They are drawn as the same circles in ``EXCLUDED_COLOR``, corrected by
    the systematics model interpolated from their neighbours, so the
    figure does not silently hide them. They are not in the binned series.

    ``n_bins`` sets the display binning (default ``WHITE_PLOT_BINS``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.asarray(time, float)
    flux, flux_err = np.asarray(flux, float), np.asarray(flux_err, float)
    model = np.asarray(model, float)
    transit = transit_only_model(summary, time)
    systematics = model - transit
    corrected = flux - systematics
    residual = flux - model
    rms = float(np.nanstd(residual) * 1e6)
    depth_ppm = float(summary["depth_ppm"]["median"])
    beta = (summary.get("rednoise") or {}).get("beta_median")
    t0_fit = float(summary["t0_p1"]["median"])
    t_hr = (time - t0_fit) * 24
    order = np.argsort(time)
    grid = np.linspace(time.min(), time.max(), 3000)
    grid_hr = (grid - t0_fit) * 24
    smooth = transit_only_model(summary, grid)

    with plt.rc_context(_PLOT_STYLE):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.0, 6.4), sharex=True,
                                     height_ratios=[3, 1])
        n_bins = n_bins or WHITE_PLOT_BINS
        # Unbinned integrations as a slate backdrop with no error bars (the
        # residual rms states their scatter); the binned series is the one
        # to read. Thousands of navy points with error bars buried it.
        raw = dict(ms=2.0, mew=0, alpha=0.75, zorder=1, rasterized=True)
        binned = dict(fmt="o", mec="white", mew=0.8, color=BINNED_COLOR,
                      ecolor=BINNED_COLOR, elinewidth=1.3, capsize=0, zorder=5)
        a1.plot(t_hr, corrected, "o", color=RAW_COLOR, label="integrations",
                **raw)
        ex_hr = ex_corr = ex_res = None
        if excluded_time is not None and len(excluded_time):
            ex_t = np.asarray(excluded_time, float)
            ex_f = np.asarray(excluded_flux, float)
            ex_hr = (ex_t - t0_fit) * 24
            ex_corr = ex_f - np.interp(ex_t, time[order], systematics[order])
            ex_res = (ex_f - np.interp(ex_t, time[order], model[order])) * 1e6
            a1.plot(ex_hr, ex_corr, "o", color=EXCLUDED_COLOR,
                    label=f"excluded at tilt step ({ex_t.size})",
                    **{**raw, "zorder": 2, "alpha": 0.9, "ms": 2.6})
        bx, by, be = _bin_series(t_hr, corrected, n_bins=n_bins)
        bin_min = float(np.median(np.diff(bx))) * 60 if len(bx) > 1 else 0.0
        a1.errorbar(bx, by, yerr=be, ms=6.0,
                    label=f"binned ({bin_min:.0f} min)", **binned)
        a1.plot(grid_hr, smooth, color=MODEL_COLOR, lw=1.3, zorder=4,
                label="transit model")
        t0_obs = summary.get("t0_obs")
        if t0_obs is not None:
            offset_min = (float(t0_obs) - t0_fit) * 1440
            if abs(offset_min) > 5:
                for ax in (a1, a2):
                    ax.axvline(offset_min / 60, ls="--", lw=1.1,
                               color=MEAN_COLOR, zorder=0)
                a1.plot([], [], ls="--", lw=1.1, color=MEAN_COLOR,
                        label=f"ephemeris mid-transit ({offset_min:+.0f} min)")
        a1.set_ylabel("Relative flux, systematics removed")
        a1.set_title(title or figure_title(
            summary.get("planet_name", ""), summary.get("instrument", ""),
            program=summary.get("program", ""), visit=summary.get("visit", ""),
            suffix="white light"), pad=12)
        a1.legend(loc="lower left", ncol=2, columnspacing=1.4,
                  handletextpad=0.5, markerscale=1.5)
        note = f"depth {depth_ppm:.0f} ppm\nresidual rms {rms:.0f} ppm"
        if beta is not None and np.isfinite(beta):
            note += f"\nred-noise $\\beta$ = {beta:.2f}"
        n_reg = len(summary.get("regressor_names") or [])
        if n_reg:
            note += f"\n{n_reg} systematics regressors removed"
        a1.annotate(note, xy=(0.985, 0.06), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=10.5, color="#4E5866")

        a2.plot(t_hr, residual * 1e6, "o", color=RAW_COLOR, **raw)
        if ex_hr is not None:
            a2.plot(ex_hr, ex_res, "o", color=EXCLUDED_COLOR,
                    **{**raw, "zorder": 2, "alpha": 0.9, "ms": 2.6})
        rbx, rby, rbe = _bin_series(t_hr, residual * 1e6, n_bins=n_bins)
        a2.errorbar(rbx, rby, yerr=rbe, ms=5.0, **binned)
        a2.axhline(0, color=MODEL_COLOR, lw=1.0, zorder=2)
        a2.set_xlabel("Time from fitted mid-transit  [h]")
        a2.set_ylabel("Residual  [ppm]")
        a2.set_ylim(-4 * rms, 4 * rms)
        fig.tight_layout()
        path = _savefig(fig, out, "white_lightcurve_fit")
        plt.close(fig)
    return path


def _plot_white_fit(results, lc, instrument: str, out: Path,
                    summary: dict[str, Any], title: str = "",
                    keep: np.ndarray | None = None,
                    model: np.ndarray | None = None) -> None:
    """Draw the fit figure (see ``plot_white_fit_series``) and save the
    plotted series so figures can be restyled without refitting.

    ``keep`` is the fit mask; excluded integrations are drawn in
    ``EXCLUDED_COLOR``.
    ``model`` is the already-evaluated full model (evaluating a GP model
    again costs a posterior-sample loop).
    """
    keep = (np.ones(lc["time"].size, dtype=bool) if keep is None
            else np.asarray(keep, dtype=bool))
    flux = lc["wl_flux"][keep]
    if model is None:
        model = results.lc.evaluate(instrument)
    plot_white_fit_series(
        out, summary, lc["time"][keep], flux, lc["wl_err"][keep], model,
        title=title,
        excluded_time=lc["time"][~keep], excluded_flux=lc["wl_flux"][~keep])

    # Save the plotted series so figures can be restyled without refitting.
    np.savetxt(out / "white_lightcurve_series.csv",
               np.column_stack([lc["time"][keep], flux,
                                lc["wl_err"][keep], model]),
               delimiter=",", header="time_bjd,flux,flux_err,model",
               # %.10g on BJD ~2.46e6 rounds time to 86 s; keep it to ~1 ms.
               comments="", fmt=["%.8f", "%.10g", "%.10g", "%.10g"])


def load_white_fit_series(fit_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The plotted series of a finished white fit, for redrawing it.

    Reads ``white_lightcurve_residuals.npz`` (full-precision time). The
    series CSV is a fallback only: before 2026-10-23 it was written with
    ``%.10g``, which rounds BJD to 86 s and misplaces ingress/egress
    points. Integrations the fit dropped are recovered from
    ``pass1/white_lightcurve_residuals.npz`` -- pass 1 is unmasked, holds
    every index, and shares the raw flux -- as ``excluded_time`` /
    ``excluded_flux`` (None when there is no pass 1 or nothing dropped).
    """
    fit_dir = Path(fit_dir)
    npz_path = fit_dir / "white_lightcurve_residuals.npz"
    csv_path = fit_dir / "white_lightcurve_series.csv"
    out: dict[str, Any] = {"excluded_time": None, "excluded_flux": None}
    if npz_path.is_file():
        with np.load(npz_path) as d:
            out.update({k: np.asarray(d[k], float)
                        for k in ("time", "flux", "flux_err", "model")})
            index = np.asarray(d["index"])
        pass1 = fit_dir / "pass1" / "white_lightcurve_residuals.npz"
        if pass1.is_file():
            with np.load(pass1) as p:
                drop = ~np.isin(p["index"], index)
                if drop.any():
                    out["excluded_time"] = np.asarray(p["time"][drop], float)
                    out["excluded_flux"] = np.asarray(p["flux"][drop], float)
        out["source"] = npz_path.name
    elif csv_path.is_file():
        d = np.genfromtxt(csv_path, delimiter=",", names=True)
        out.update({"time": d["time_bjd"], "flux": d["flux"],
                    "flux_err": d["flux_err"], "model": d["model"]})
        out["source"] = csv_path.name
    else:
        return None
    return out


def replot_white_fit(fit_dir: str | os.PathLike[str],
                     out: str | os.PathLike[str] | None = None, *,
                     n_bins: int | None = None) -> str | None:
    """Redraw ``white_lightcurve_fit.pdf/.svg`` for one fit dir, no refit.

    Writes into ``out`` (default: in place). Returns the PDF path, or None
    when the dir has no summary or saved series.
    """
    fit_dir = Path(fit_dir)
    json_path = fit_dir / "white_fit_summary.json"
    series = load_white_fit_series(fit_dir)
    if series is None or not json_path.is_file():
        return None
    summary = json.loads(json_path.read_text())
    out = Path(out) if out is not None else fit_dir
    out.mkdir(parents=True, exist_ok=True)
    return plot_white_fit_series(
        out, summary, series["time"], series["flux"], series["flux_err"],
        series["model"], excluded_time=series["excluded_time"],
        excluded_flux=series["excluded_flux"], n_bins=n_bins)


def fit_transmission_spectrum(
    lc: dict[str, Any],
    white_summary: dict[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    instrument: str = "NRS1",
    regressors: np.ndarray | None = None,
    stellar: dict[str, float] | None = None,
    program: str = "",
    visit: str = "",
    n_live: int = N_LIVE_SPEC,
    keep_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit every spectroscopic channel with the orbit (P, t0, b, a/Rs)
    frozen to the white-light posterior medians. Free per channel: Rp/Rs,
    (q1, q2) with per-channel ExoTiC-LD priors when ``stellar`` is given,
    mflux, jitter, and the linear regressor coefficients.

    ``keep_mask`` must be the SAME Stage 5.5 mask used for the white fit
    that produced ``white_summary``. A spot crossing masked in the white
    fit but left in the channels would put the orbit and the depths on
    different data, and the crossing's chromatic signature would print
    straight onto the transmission spectrum — which is the shape a
    retrieval reads as an atmosphere.

    Writes ``transmission_spectrum.csv`` (ppm, with per-channel residual
    rms) and ``transmission_spectrum.dat`` (fractional depth, TauREx
    input format).
    """
    juliet = _import_juliet()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    times = lc["time"]
    if regressors is None:
        regressors, _ = build_regressor_matrix(times)
    n_reg = regressors.shape[1]

    fixed = {
        "P_p1": white_summary["P_p1"]["median"],
        "t0_p1": white_summary["t0_p1"]["median"],
        "b_p1": white_summary["b_p1"]["median"],
        "a_p1": white_summary["a_p1"]["median"],
    }

    # v1.6: a white fit with a GP passes it on. The timescale is a
    # property of the visit's systematics, not of the channel, so it is
    # frozen to the white posterior like the orbit; only the amplitude is
    # free per channel.
    gp_rho = gp_origin = None
    noise = white_summary.get("noise_model") or {}
    if noise.get("kind") == GP_KERNEL:
        _require_celerite()
        gp_rho = float(white_summary[f"GP_rho_{instrument}"]["median"])
        if noise.get("gp_time_origin") is None:
            raise ValueError(
                "This white-light fit used a GP on absolute BJD (made before the "
                "gp_time_input fix) — refit the white light; its GP posterior "
                "is not trusted and the channels cannot reuse it.")
        gp_origin = float(noise["gp_time_origin"])

    use_ld = stellar is not None and white_summary.get("ld_source") == "exotic-ld"
    # Near-grazing geometry: fix (q1, q2) per channel rather than fit them.
    b_white = (white_summary.get("b_p1") or {}).get("median")
    fix_ld = bool(use_ld and b_white is not None
                  and float(b_white) > LD_FIX_IMPACT_PARAM)

    # One LD computation for every channel up front (single grid load)
    # instead of one subprocess per channel.
    channel_ld: list[dict[str, float] | None] = [None] * lc["wave"].size
    if use_ld:
        ranges = [
            (float(lc["wave"][i] - lc["wave_err"][i]),
             float(lc["wave"][i] + lc["wave_err"][i]))
            for i in range(lc["wave"].size)
        ]
        channel_ld = compute_ld_coeffs_batch(
            stellar["st_teff"], stellar["st_logg"], stellar["st_met"], ranges)

    keep = (np.ones(times.size, dtype=bool) if keep_mask is None
            else np.asarray(keep_mask, dtype=bool))
    if keep.size != times.size:
        raise ValueError(
            f"keep_mask has {keep.size} entries but the lightcurve has "
            f"{times.size} integrations."
        )

    rows = []
    for i in range(lc["wave"].size):
        f, e = lc["sp_flux"][:, i], lc["sp_err"][:, i]
        good = np.isfinite(f) & np.isfinite(e) & keep
        if good.sum() < 50:
            continue

        ld = channel_ld[i]

        prior_dict: dict[str, dict[str, Any]] = {
            key: {"distribution": "fixed", "hyperparameters": value}
            for key, value in fixed.items()
        }
        prior_dict.update({
            "ecc_p1": {"distribution": "fixed", "hyperparameters": 0.0},
            "omega_p1": {"distribution": "fixed", "hyperparameters": 90.0},
            "p_p1": {"distribution": "uniform", "hyperparameters": [0.0, 0.3]},
        })
        if fix_ld and ld is not None:
            prior_dict.update({
                f"q1_{instrument}": {"distribution": "fixed",
                                     "hyperparameters": float(ld["q1"])},
                f"q2_{instrument}": {"distribution": "fixed",
                                     "hyperparameters": float(ld["q2"])},
            })
        else:
            prior_dict.update(_ld_prior_entries(instrument, ld))
        prior_dict.update(_common_instrument_priors(instrument, n_reg))
        if gp_rho is not None:
            prior_dict[f"GP_sigma_{instrument}"] = {
                "distribution": "loguniform",
                "hyperparameters": list(GP_SIGMA_BOUNDS)}
            prior_dict[f"GP_rho_{instrument}"] = {
                "distribution": "fixed", "hyperparameters": gp_rho}

        dataset = juliet.load(
            priors=prior_dict,
            t_lc={instrument: times[good]},
            y_lc={instrument: f[good]},
            yerr_lc={instrument: e[good]},
            linear_regressors_lc={instrument: regressors[good]},
            GP_regressors_lc=({instrument: gp_time_input(times[good], gp_origin)}
                              if gp_rho is not None else None),
            out_folder=str(out / f"bin{i:03d}"),
        )
        results = dataset.fit(sampler=SAMPLER, n_live_points=n_live, verbose=False)
        p_samples = results.posteriors["posterior_samples"]["p_p1"]
        depth = np.percentile(p_samples**2, [50, 16, 84])
        model = results.lc.evaluate(instrument)
        rows.append({
            "wave": float(lc["wave"][i]),
            "wave_err": float(lc["wave_err"][i]),
            "depth": float(depth[0]),
            "depth_err": float(0.5 * (depth[2] - depth[1])),
            "rms_ppm": float(np.nanstd(f[good] - model) * 1e6),
        })

    csv_path = out / "transmission_spectrum.csv"
    # Stamp the version the WHITE fit carries, not the module constant:
    # that one already records -nopca / -noscan, and the spectroscopic
    # fit inherits both conditions from it.
    fit_version = white_summary.get("fit_version", PATCHWORK_FIT_VERSION)
    spot = white_summary.get("spot_handling") or {}
    write_spectrum_csv(csv_path, rows, header=(
        f"Patchwork juliet fit v{fit_version}, "
        f"instrument={instrument}, ld={white_summary.get('ld_source')}, "
        f"regressors={','.join(white_summary.get('regressor_names', []))}, "
        f"masked_integrations={int((~keep).sum())}"
        + (f", spot_events={len(spot.get('events') or [])}" if spot else "")
        + (f", noise={GP_KERNEL} rho={gp_rho * 1440:.1f}min"
           if gp_rho is not None else ", noise=white")
    ))

    dat_path = out / "transmission_spectrum.dat"
    with dat_path.open("w") as handle:
        handle.write(
            "# wavelength_um  transit_depth  depth_err  bin_width_um\n"
            f"# Patchwork juliet fit v{fit_version}, instrument={instrument}, "
            f"masked_integrations={int((~keep).sum())}\n"
        )
        for r in rows:
            handle.write(
                f"{r['wave']:.6f}  {r['depth']:.8f}  "
                f"{r['depth_err']:.8f}  {2*r['wave_err']:.6f}\n"
            )

    _plot_spectrum(rows, out, instrument,
                   planet=white_summary.get("planet_name", ""),
                   program=program or white_summary.get("program", ""),
                   visit=visit or white_summary.get("visit", ""))
    return {"spectrum_csv": str(csv_path), "spectrum_path": str(dat_path),
            "n_bins": len(rows), "rows": rows,
            "ld_fixed_per_channel": fix_ld,
            "impact_parameter": (float(b_white) if b_white is not None else None),
            "fit_version": fit_version,
            "n_masked": int((~keep).sum()),
            "median_depth_err_ppm": float(np.nanmedian(
                [r["depth_err"] for r in rows]) * 1e6) if rows else None,
            "median_rms_ppm": float(np.nanmedian(
                [r["rms_ppm"] for r in rows])) if rows else None}


def write_spectrum_csv(path: str | os.PathLike[str],
                       rows: list[dict[str, float]], header: str = "") -> None:
    """ppm-unit spectrum CSV: wave_um, wave_err_um, depth_ppm,
    depth_err_ppm, resid_rms_ppm."""
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        if header:
            w.writerow([f"# {header}"])
        w.writerow(["wave_um", "wave_err_um", "depth_ppm",
                    "depth_err_ppm", "resid_rms_ppm"])
        for r in rows:
            w.writerow([
                f"{r['wave']:.6f}", f"{r['wave_err']:.6f}",
                f"{r['depth'] * 1e6:.2f}", f"{r['depth_err'] * 1e6:.2f}",
                f"{r.get('rms_ppm', float('nan')):.1f}",
            ])


def read_spectrum_csv(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Read a Patchwork spectrum CSV back into arrays (ppm depths)."""
    wave, wave_err, depth, depth_err, rms = [], [], [], [], []
    with Path(path).open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#") or row[0] == "wave_um":
                continue
            wave.append(float(row[0]))
            wave_err.append(float(row[1]))
            depth.append(float(row[2]))
            depth_err.append(float(row[3]))
            rms.append(float(row[4]) if len(row) > 4 and row[4] else np.nan)
    return {"wave": np.asarray(wave), "wave_err": np.asarray(wave_err),
            "depth_ppm": np.asarray(depth), "depth_err_ppm": np.asarray(depth_err),
            "rms_ppm": np.asarray(rms)}


def _plot_spectrum(rows: list[dict[str, float]], out: Path,
                   instrument: str, planet: str = "", program: str = "",
                   visit: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(_PLOT_STYLE):
        wave = np.array([r["wave"] for r in rows])
        depth = np.array([r["depth"] for r in rows]) * 1e6
        err = np.array([r["depth_err"] for r in rows]) * 1e6
        xerr = np.array([r["wave_err"] for r in rows])
        color = DATA_COLOR if instrument.upper() == "NRS1" else NRS2_COLOR

        fig, ax = plt.subplots(figsize=(10.5, 5.0))
        ax.errorbar(wave, depth, yerr=err, xerr=xerr, fmt="o", ms=4.5, mew=0,
                    color=color, ecolor=color, elinewidth=1.1, capsize=0,
                    alpha=0.85, zorder=3,
                    label=f"{instrument.upper()}  ({len(rows)} channels)")
        wmean = np.sum(depth / err ** 2) / np.sum(1 / err ** 2)
        ax.axhline(wmean, color=MEAN_COLOR, lw=1.2, ls="--", zorder=1,
                   label=f"weighted mean  {wmean:.0f} ppm")
        lo, hi = (depth - err).min(), (depth + err).max()
        pad = 0.18 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Wavelength  [μm]")
        ax.set_ylabel(r"Transit depth  $(R_{\rm p}/R_{\star})^{2}$  [ppm]")
        ax.set_title(figure_title(planet, instrument, program=program,
                                  visit=visit), pad=12)
        ax.legend(loc="upper left", ncol=2, columnspacing=1.4,
                  handletextpad=0.5)
        fig.tight_layout()
        _savefig(fig, out, "transmission_spectrum")
        plt.close(fig)


# -------------------- Stage 6: visit combination + metrics --------------


def combine_visit_spectra(csv_paths: list[str],
                          *, match_rtol: float = 1e-3) -> dict[str, np.ndarray]:
    """Inverse-variance combine per-visit spectra of one detector.

    Visits reduced with the same Patchwork binning scheme share their
    wavelength *edges*, but a channel can drop out of one visit and not
    another (a bad column makes ``bin_at_resolution`` skip that bin), so
    the per-visit grids are aligned channel-by-channel (centres matched
    to ``match_rtol`` relative tolerance) rather than required to be
    identical. Channels missing from a visit are combined from the
    visits that do have them; ``n_visits_per_channel`` records how many
    contributed to each. Truly incompatible grids (different binning
    scheme, so essentially no channels align) still raise.
    """
    specs = [read_spectrum_csv(p) for p in csv_paths]
    if len(specs) == 1:
        s = specs[0]
        return {"wave": s["wave"], "wave_err": s["wave_err"],
                "depth_ppm": s["depth_ppm"], "depth_err_ppm": s["depth_err_ppm"],
                "rms_ppm": s["rms_ppm"], "n_visits": 1,
                "n_visits_per_channel": np.ones(s["wave"].size, dtype=int)}

    # Union grid: cluster channel centres across visits.
    ref_waves: list[float] = []
    for s in specs:
        for w in s["wave"]:
            if not any(abs(w - r) <= match_rtol * r for r in ref_waves):
                ref_waves.append(float(w))
    ref = np.asarray(sorted(ref_waves))

    n_ch = ref.size
    D = np.full((len(specs), n_ch), np.nan)
    E = np.full((len(specs), n_ch), np.nan)
    R = np.full((len(specs), n_ch), np.nan)
    W = np.full(n_ch, np.nan)
    matched_per_visit = []
    for k, s in enumerate(specs):
        matched = 0
        for j, w in enumerate(s["wave"]):
            i = int(np.argmin(np.abs(ref - w)))
            if abs(ref[i] - w) <= match_rtol * w:
                D[k, i], E[k, i], R[k, i] = (s["depth_ppm"][j],
                                             s["depth_err_ppm"][j],
                                             s["rms_ppm"][j])
                if np.isnan(W[i]):
                    W[i] = s["wave_err"][j]
                matched += 1
        matched_per_visit.append(matched)

    # Same binning scheme -> visits overlap on nearly every channel. If
    # fewer than half the union channels have >= 2 visits contributing,
    # the grids do not actually align (different binning scheme).
    n_contrib = np.isfinite(E).sum(axis=0)
    if (n_contrib >= 2).sum() < 0.5 * n_ch:
        raise ValueError(
            "Visit spectra are on different wavelength grids (fewer than "
            "half the channels align across visits) — they must come from "
            "the same Patchwork binning scheme."
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / E**2
        w[~np.isfinite(w)] = 0.0
        wsum = w.sum(axis=0)
        depth = np.where(wsum > 0,
                         np.nansum(np.where(w > 0, D, 0.0) * w, axis=0)
                         / np.where(wsum > 0, wsum, 1.0), np.nan)
        err = np.where(wsum > 0, np.sqrt(1.0 / np.where(wsum > 0, wsum, 1.0)),
                       np.nan)
    return {"wave": ref, "wave_err": W,
            "depth_ppm": depth, "depth_err_ppm": err,
            "rms_ppm": np.nanmedian(R, axis=0), "n_visits": len(specs),
            "n_visits_per_channel": (w > 0).sum(axis=0)}


def detector_offset_ppm(nrs1: dict[str, np.ndarray],
                        nrs2: dict[str, np.ndarray],
                        n_edge: int = 5) -> float:
    """NRS1-NRS2 depth offset near the detector gap: median of the last
    ``n_edge`` finite NRS1 channels minus the first ``n_edge`` finite NRS2
    channels. A large jump flags an inter-detector systematic."""
    d1 = nrs1["depth_ppm"][np.isfinite(nrs1["depth_ppm"])]
    d2 = nrs2["depth_ppm"][np.isfinite(nrs2["depth_ppm"])]
    return float(np.median(d1[-n_edge:]) - np.median(d2[:n_edge]))


def _plot_combined(combined: dict[str, dict[str, np.ndarray]],
                   out: Path, title: str) -> None:
    """Combined per-detector transmission spectrum, formatted for circulation.

    The NIRSpec/G395H inter-detector gap is shaded and labelled so the eye
    does not read across a wavelength range that was never observed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"NRS1": DATA_COLOR, "NRS2": NRS2_COLOR}
    with plt.rc_context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10.5, 5.2))

        if {"NRS1", "NRS2"} <= set(combined):
            m1 = np.isfinite(combined["NRS1"]["depth_ppm"])
            m2 = np.isfinite(combined["NRS2"]["depth_ppm"])
            gap_lo = combined["NRS1"]["wave"][m1].max()
            gap_hi = combined["NRS2"]["wave"][m2].min()
            ax.axvspan(gap_lo, gap_hi, color=GAP_COLOR, alpha=0.30,
                       zorder=0, lw=0)
            ax.text(0.5 * (gap_lo + gap_hi), 0.035, "detector\ngap",
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=9.5, color="#4E5866",
                    linespacing=1.15)

        all_d, all_e = [], []
        for det in ("NRS1", "NRS2"):
            if det not in combined:
                continue
            S = combined[det]
            m = np.isfinite(S["depth_ppm"])
            all_d.append(S["depth_ppm"][m]); all_e.append(S["depth_err_ppm"][m])
            ax.errorbar(S["wave"][m], S["depth_ppm"][m], xerr=S["wave_err"][m],
                        yerr=S["depth_err_ppm"][m], fmt="o", ms=4.5, mew=0,
                        color=colors[det], ecolor=colors[det], elinewidth=1.1,
                        capsize=0, alpha=0.85, zorder=3,
                        label=f"{det}  ({int(m.sum())} channels)")

        if all_d:
            d = np.concatenate(all_d); e = np.concatenate(all_e)
            wmean = np.sum(d / e ** 2) / np.sum(1 / e ** 2)
            ax.axhline(wmean, color=MEAN_COLOR, lw=1.2, ls="--", zorder=1,
                       label=f"weighted mean  {wmean:.0f} ppm")
            lo, hi = (d - e).min(), (d + e).max()
            pad = 0.18 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)

        ax.set_xlabel("Wavelength  [μm]")
        ax.set_ylabel(r"Transit depth  $(R_{\rm p}/R_{\star})^{2}$  [ppm]")
        ax.set_title(title, pad=12)
        ax.legend(loc="upper left", ncol=3, columnspacing=1.4,
                  handletextpad=0.5)
        fig.tight_layout()
        _savefig(fig, out, "combined_transmission_spectrum")
        plt.close(fig)


# -------------------- shared tool plumbing --------------------


def _resolve(path: str, base: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base, path)


def prepare_visit_fit_inputs(
    spectra_path: str,
    planet_name: str,
    *,
    instrument: str,
    reduction_dir: str | None = None,
    decorrelate: bool = True,
    tilt_correction: bool = True,
    ld_priors: str = "exotic-ld",
    resolution: float = DEFAULT_RESOLUTION,
    n_bins: int | None = None,
    wave_min: float | None = None,
    wave_max: float | None = None,
    fallback_priors: dict[str, Any] | None = None,
    priors_override: dict[str, Any] | None = None,
    pca_detrending: bool = True,
    anomaly_report: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Shared Stage 4 -> Stage 5 preparation: archive priors, lightcurves,
    trace diagnostics, tilt events, regressor matrix, white-band LD.

    Priors come from the NASA Exoplanet Archive at fit time; when the
    query fails (offline compute node) the manifest-cached
    ``fallback_priors`` written by discover.py are used instead, and
    ``priors_source`` records which one actually ran.

    ``priors_override`` wins over BOTH. The archive is current, not
    correct: an ephemeris whose reference epoch is years old drifts by
    hours over hundreds of epochs, and for some targets the archive
    prediction misses a transit that is plainly present in the data
    (TOI-125 b visit o101, 2026-08-26: archive predicts mid-transit
    24.7 h after a window whose white lightcurve shows a full 870 ppm
    transit at its centre). The transit-in-window guard then correctly
    refuses to fit. The supported fix is to supply the measured value
    here — NOT to relax the guard, which is what stands between a stale
    ephemeris and a "spectrum" that is only the Rp/Rs prior.

    Keys map onto the ``fetch_transit_priors`` output (``t0``,
    ``period``, ``duration_hr``, ``a_rs``, ``rp_rs``, ``b``, ``st_teff``
    ...). Every overridden key is recorded in ``priors_override_used``
    and stamped into ``priors_source``, so a target fitted on a
    non-archive ephemeris can never be mistaken for a default one.

    The COMPASS-style relative-pixel-flux PCA regressors (Ahrer et al.
    2025) are part of the frozen v1.2 fit and on by default; a fit
    without them is version-stamped "-nopca" downstream. Disabling
    (``pca_detrending=False``) is the escape hatch for comparison fits.

    ``anomaly_report`` is the ``anomaly_report.json`` from a Stage 5.5
    scan (``contamination.DetectLightCurveAnomalies``). When given, the
    integrations it marks for this detector are dropped from both the
    white and the spectroscopic fits, and the scan payload is carried
    through so the fit summary records what was masked. Without it the
    fit is stamped "-noscan".
    """
    priors_source = "archive"
    try:
        priors = fetch_transit_priors(planet_name)
    except Exception:
        if not fallback_priors:
            raise
        priors = dict(fallback_priors)
        priors.setdefault("planet_name", planet_name)
        priors_source = "manifest-cache"

    # Applied last, so it beats the archive as well as the cache.
    override_used = sorted(k for k, v in (priors_override or {}).items()
                           if v is not None)
    if override_used:
        priors = dict(priors)
        priors.update({k: priors_override[k] for k in override_used})
        priors_source = f"{priors_source}+override({','.join(override_used)})"
    spectra = load_stage3_spectra(spectra_path)
    lc = build_lightcurves(
        spectra,
        detector=instrument,
        t0_ref=priors["t0"],
        period=priors["period"],
        duration_hr=priors["duration_hr"],
        resolution=resolution,
        n_bins=n_bins,
        wave_min=wave_min,
        wave_max=wave_max,
    )

    diagnostics = None
    pca = None
    if decorrelate and reduction_dir:
        calints = find_stage2_calints(reduction_dir, instrument)
        if calints:
            diagnostics = trace_diagnostics(calints)
            for key in list(diagnostics):
                if diagnostics[key].size != lc["time"].size:
                    diagnostics = None  # cube/spectra mismatch: skip, do not crash
                    break
            if pca_detrending and diagnostics is not None:
                try:
                    pca = pca_regressors(calints)
                    if pca.shape[0] != lc["time"].size:
                        pca = None
                except Exception:
                    pca = None

    # Tilt events are searched in the TRACE DIAGNOSTICS, not just the
    # flux: a mirror-segment tilt changes the PSF first, so the FWHM and
    # the PCA components carry it without any transit signal in the way,
    # and an event during transit stays visible. See find_tilt_events.
    tilt_report = None
    tilt_events: list[dict[str, Any]] = []
    if tilt_correction:
        tilt_report = find_tilt_events(
            lc["wl_flux"],
            diagnostics=diagnostics,
            pca_components=pca,
            oot_mask=lc["oot_mask"],
            time=lc["time"],
            t0_obs=lc["t0_obs"],
            detector=instrument,
        )
        tilt_events = tilt_report["events"]
        if tilt_report.get("rate_warning"):
            print(f"[patchwork] WARNING ({instrument}): "
                  f"{tilt_report['rate_warning']}")

    regressors, regressor_names = build_regressor_matrix(
        lc["time"], diagnostics, tilt_events, pca_components=pca
    )
    # The integration straddling a tilt is a blend of both PSF states and
    # no Heaviside describes it; drop those few points and step-correct
    # the rest. Everything else is preserved.
    tilt_keep = tilt_transition_keep_mask(lc["time"].size, tilt_events)

    ld = None
    if ld_priors == "exotic-ld":
        wlo, whi = G395H_WAVE_RANGES.get(instrument.upper(), (None, None))
        ld = compute_ld_coeffs(
            priors["st_teff"], priors["st_logg"], priors["st_met"],
            wave_min if wave_min is not None else wlo,
            wave_max if wave_max is not None else whi,
        )

    scan = None
    keep_mask = tilt_keep
    n_spot_masked = 0
    if anomaly_report:
        scan, spot_keep = load_anomaly_mask(
            anomaly_report, instrument, lc["time"].size)
        n_spot_masked = int((~spot_keep).sum())
        keep_mask = keep_mask & spot_keep

    return {"priors": priors, "priors_source": priors_source,
            "priors_override_used": override_used, "lc": lc,
            "regressors": regressors, "regressor_names": regressor_names,
            "tilt_events": tilt_events, "tilt_report": tilt_report,
            "tilt_keep_mask": tilt_keep,
            "n_tilt_masked": int((~tilt_keep).sum()),
            "ld": ld,
            "diagnostics_used": diagnostics is not None,
            "pca_used": pca is not None,
            "keep_mask": keep_mask, "anomaly_scan": scan,
            "n_spot_masked": n_spot_masked,
            "n_masked": int((~keep_mask).sum())}


def load_anomaly_mask(
    report_path: str | os.PathLike[str],
    instrument: str,
    n_integrations: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Read a Stage 5.5 ``anomaly_report.json`` into (payload, keep mask).

    The report stores explicit integration indices rather than a packed
    boolean array so it stays readable and diffable — the mask is a
    scientific decision, and someone will want to check it by eye.
    """
    with open(report_path) as handle:
        payload = json.load(handle)
    idx = (payload.get("mask_indices") or {}).get(instrument.upper(), [])
    keep = np.ones(int(n_integrations), dtype=bool)
    idx = [i for i in idx if 0 <= int(i) < int(n_integrations)]
    if idx:
        keep[np.asarray(idx, dtype=int)] = False
    return payload, keep


# -------------------- orchestral tools --------------------


class FitNirspecG395hWhiteLight(BaseTool):
    """
    Fit the white-light transit of one reduced JWST NIRSpec/G395H visit
    with juliet. One detector (NRS1 or NRS2) per call.

    Input is an exoTEDRF ``*_spectra_fullres.fits`` product (from
    ``ReduceNirspecG395hTso``). Ephemeris/orbit/stellar priors come from
    the NASA Exoplanet Archive via ``planet_name``; t0 is propagated to
    the observation epoch; the lightcurve is normalized by the
    out-of-transit baseline.

    Survey-frozen fit (Patchwork v1.1): Gaussian P and a/Rs, uniform
    Rp/Rs and b, ExoTiC-LD truncated-normal Kipping (q1, q2) [uniform
    fallback recorded in the summary], linear decorrelation against time
    + trace x/y + FWHM (pass ``reduction_dir`` so Stage 2 calints can be
    read), one fitted step per detected tilt event, dynesty sampling.
    Set decorrelate/tilt_correction False and ld_priors='uniform' for an
    baseline comparison fit.

    Outputs in ``output_dir``: juliet posteriors,
    ``white_lightcurve_fit.pdf/.svg``, ``white_fit_summary.json``
    (includes residual rms ppm and tilt events).

    Run BEFORE ``FitNirspecG395hTransmissionSpectrum`` — the spectroscopic
    fits freeze the orbit to this posterior.

    Example
    -------
        FitNirspecG395hWhiteLight(
            spectra_path="reductions/GJ_9827_d/o010/nrs1/.../..._box_spectra_fullres.fits",
            planet_name="GJ 9827 d",
            output_dir="fits/GJ_9827_d/o010/nrs1",
            instrument="NRS1",
            reduction_dir="reductions/GJ_9827_d/o010/nrs1",
        )
    """

    spectra_path: str = RuntimeField(
        description="Path to an exoTEDRF *_spectra_fullres.fits product."
    )
    planet_name: str = RuntimeField(
        description="Planet name for archive priors, e.g. 'GJ 9827 d'."
    )
    output_dir: str = RuntimeField(description="Directory for fit outputs.")
    instrument: str = RuntimeField(
        default="NRS1",
        description="Detector label for this dataset ('NRS1' or 'NRS2').",
    )
    reduction_dir: str | None = OptionalRuntimeField(
        description="Reduction dir containing Stage 2 calints — enables "
                    "trace x/y/FWHM decorrelation regressors.",
    )
    decorrelate: bool = RuntimeField(
        default=True,
        description="Include trace x/y/FWHM linear regressors (needs reduction_dir).",
    )
    tilt_correction: bool = RuntimeField(
        default=True,
        description="Detect tilt events and fit one step term per event.",
    )
    ld_priors: str = RuntimeField(
        default="exotic-ld",
        description="'exotic-ld' for Gaussian (q1,q2) priors, 'uniform' to disable.",
    )
    pca_detrending: bool = RuntimeField(
        default=True,
        description="COMPASS-style relative-pixel-flux PCA regressors "
                    "(Ahrer et al. 2025) — part of the frozen v1.2 survey "
                    "fit (needs reduction_dir). Disable only for comparison "
                    "fits; the fit version is then stamped '-nopca'.",
    )
    anomaly_report: str | None = OptionalRuntimeField(
        description="Path to a Stage 5.5 anomaly_report.json from "
                    "DetectLightCurveAnomalies. Masks the confirmed spot/"
                    "facula crossings it lists. Omit on the first pass (the "
                    "scan needs this fit's residuals); the fit is then "
                    "stamped '-noscan'.",
    )
    wave_min: float | None = OptionalRuntimeField(
        description="Optional lower wavelength cut in microns."
    )
    wave_max: float | None = OptionalRuntimeField(
        description="Optional upper wavelength cut in microns."
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        prep = prepare_visit_fit_inputs(
            _resolve(self.spectra_path, self.base_directory),
            self.planet_name,
            instrument=self.instrument,
            reduction_dir=(_resolve(self.reduction_dir, self.base_directory)
                           if self.reduction_dir else None),
            decorrelate=self.decorrelate,
            tilt_correction=self.tilt_correction,
            ld_priors=self.ld_priors,
            wave_min=self.wave_min,
            wave_max=self.wave_max,
            pca_detrending=self.pca_detrending,
            anomaly_report=(_resolve(self.anomaly_report, self.base_directory)
                            if self.anomaly_report else None),
        )
        summary = fit_white_lightcurve(
            prep["lc"],
            prep["priors"],
            _resolve(self.output_dir, self.base_directory),
            instrument=self.instrument,
            regressors=prep["regressors"],
            regressor_names=prep["regressor_names"],
            ld=prep["ld"],
            keep_mask=prep["keep_mask"],
            anomaly_scan=prep["anomaly_scan"],
            tilt_report=prep["tilt_report"],
            n_spot_masked=prep["n_spot_masked"],
            n_tilt_masked=prep["n_tilt_masked"],
        )
        lines = [
            f"White-light fit complete for {self.planet_name} ({self.instrument}).",
            f"Outputs: {summary['output_dir']}",
            f"  regressors: {', '.join(summary['regressor_names'])}"
            + ("" if prep["diagnostics_used"] else " (trace diagnostics unavailable)"),
            f"  LD priors: {summary['ld_source']}",
            f"  fit version: {summary['fit_version']}",
        ]
        th = summary["tilt_handling"]
        if th["n_events"]:
            lines.append(
                f"  tilt events: {th['n_events']} — corrected with a fitted "
                f"Heaviside step each; {th['n_transition_masked']} transition "
                "integration(s) dropped."
            )
            for e in th["events"]:
                where = (f"{e['hours_from_mid']:+.2f} h from mid-transit"
                         if e.get("hours_from_mid") is not None else "")
                lines.append(
                    f"    index {e['index']} ({where}"
                    + (", IN TRANSIT" if e.get("in_transit") else "")
                    + f"): {e['amplitude_ppm']:+.0f} ppm, seen in "
                    + ", ".join(e.get("sources") or [])
                )
        else:
            lines.append("  tilt events: none"
                         + ("" if th["searched_diagnostics"] else
                            " (NO trace diagnostics — flux-only search, an "
                            "in-transit tilt would be invisible; pass "
                            "reduction_dir)"))
        if th.get("rate_warning"):
            lines.append(f"  WARNING: {th['rate_warning']}")
        if prep["anomaly_scan"] is None:
            lines.append(
                "  Stage 5.5 spot scan: NOT RUN — run "
                "DetectLightCurveAnomalies on this output_dir, then refit "
                "with anomaly_report=<its anomaly_report.json> and "
                "force_refit. A '-noscan' fit is not uniform with the survey."
            )
        else:
            lines.append(
                f"  Stage 5.5 spot scan: {prep['n_spot_masked']} "
                "integration(s) masked."
            )
        for key in ["t0_p1", "p_p1", "b_p1", "a_p1"]:
            if key in summary:
                s = summary[key]
                lines.append(
                    f"  {key} = {s['median']:.6f} +{s['plus']:.6f} -{s['minus']:.6f}"
                )
        d = summary["depth_ppm"]
        lines.append(
            f"  white depth = {d['median']:.0f} +{d['plus']:.0f} -{d['minus']:.0f} ppm; "
            f"residual rms = {summary['residual_rms_ppm']:.0f} ppm"
        )
        rn = summary.get("rednoise") or {}
        if rn.get("beta_median") == rn.get("beta_median"):  # not NaN
            lines.append(
                f"  red noise: beta_median = {rn['beta_median']:.2f} "
                f"(beta > ~1.2 means depth errors need inflating)"
            )
        dc = summary.get("depth_check") or {}
        if dc.get("status") == "suspect":
            lines.append(
                f"  WARNING: depth {dc['measured_ppm']:.0f} ppm vs "
                f"{dc['band']} reference {dc['expected_ppm']:.0f} ppm "
                f"({dc['difference_sigma']:.1f} sigma; {dc['reference_source']})"
                " — verify the fit."
            )
        elif dc.get("status") == "indicative":
            lines.append(
                f"  depth check: {dc['difference_frac'] * 100:+.1f}% from the "
                "optical archive value (different bandpass — indicative only)."
            )
        elif dc.get("status") == "ok":
            lines.append(f"  depth check: consistent with the {dc['band']} "
                         f"reference ({dc['expected_ppm']:.0f} ppm).")
        lines.append(
            "Now run FitNirspecG395hTransmissionSpectrum with the same "
            "spectra_path and this output_dir as white_fit_dir."
        )
        return "\n".join(lines)


class FitNirspecG395hTransmissionSpectrum(BaseTool):
    """
    Fit the spectroscopic channels of one NIRSpec/G395H visit x detector
    and produce a transmission spectrum.

    Requires a completed ``FitNirspecG395hWhiteLight`` run: P, t0, b, a/Rs
    are frozen to the white-light posterior medians. Free per channel:
    Rp/Rs, (q1, q2) with per-channel ExoTiC-LD priors, normalization,
    jitter, and the same decorrelation + tilt-step regressors as the
    white fit. Channels are constant-resolution (default R=100).

    Outputs in ``output_dir``: per-channel posteriors,
    ``transmission_spectrum.csv`` (ppm + per-channel residual rms — the
    input to ``CombineNirspecG395hVisits``), ``transmission_spectrum.dat``
    (TauREx format, usable by SimulateTaurexRetrieval), and
    ``transmission_spectrum.pdf/.svg``.

    Compute: one nested-sampling fit per channel (~30-60), a few minutes
    each.

    Example
    -------
        FitNirspecG395hTransmissionSpectrum(
            spectra_path="reductions/GJ_9827_d/o010/nrs1/.../..._box_spectra_fullres.fits",
            planet_name="GJ 9827 d",
            white_fit_dir="fits/GJ_9827_d/o010/nrs1",
            output_dir="fits/GJ_9827_d/o010/nrs1/spectro",
            reduction_dir="reductions/GJ_9827_d/o010/nrs1",
        )
    """

    spectra_path: str = RuntimeField(
        description="Path to the same *_spectra_fullres.fits used for the white fit."
    )
    planet_name: str = RuntimeField(
        description="Planet name for archive priors, e.g. 'GJ 9827 d'."
    )
    white_fit_dir: str = RuntimeField(
        description="output_dir of the completed FitNirspecG395hWhiteLight run "
                    "(must contain white_fit_summary.json)."
    )
    output_dir: str = RuntimeField(description="Directory for spectroscopic outputs.")
    instrument: str = RuntimeField(
        default="NRS1",
        description="Detector label, must match the white-light fit.",
    )
    reduction_dir: str | None = OptionalRuntimeField(
        description="Reduction dir with Stage 2 calints for trace regressors.",
    )
    decorrelate: bool = RuntimeField(
        default=True, description="Include trace x/y/FWHM regressors."
    )
    tilt_correction: bool = RuntimeField(
        default=True, description="Include tilt-event step terms."
    )
    ld_priors: str = RuntimeField(
        default="exotic-ld",
        description="'exotic-ld' for per-channel Gaussian (q1,q2), 'uniform' to disable.",
    )
    resolution: float = RuntimeField(
        default=DEFAULT_RESOLUTION,
        description="Spectral binning resolution R.",
    )
    n_bins: int | None = OptionalRuntimeField(
        description="Optional fixed number of equal-width bins (overrides resolution).",
    )
    wave_min: float | None = OptionalRuntimeField(
        description="Optional lower wavelength cut in microns."
    )
    wave_max: float | None = OptionalRuntimeField(
        description="Optional upper wavelength cut in microns."
    )
    anomaly_report: str | None = OptionalRuntimeField(
        description="Stage 5.5 anomaly_report.json. MUST be the same report "
                    "used for the white fit — the orbit and the channel "
                    "depths have to come from the same integrations.",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        white_dir = _resolve(self.white_fit_dir, self.base_directory)
        summary_path = Path(white_dir) / "white_fit_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"{summary_path} not found — run FitNirspecG395hWhiteLight first."
            )
        with summary_path.open() as handle:
            white_summary = json.load(handle)

        prep = prepare_visit_fit_inputs(
            _resolve(self.spectra_path, self.base_directory),
            self.planet_name,
            instrument=self.instrument,
            reduction_dir=(_resolve(self.reduction_dir, self.base_directory)
                           if self.reduction_dir else None),
            decorrelate=self.decorrelate,
            tilt_correction=self.tilt_correction,
            ld_priors=self.ld_priors,
            resolution=self.resolution,
            n_bins=self.n_bins,
            wave_min=self.wave_min,
            wave_max=self.wave_max,
            anomaly_report=(_resolve(self.anomaly_report, self.base_directory)
                            if self.anomaly_report else None),
        )
        # The channels must see exactly the integrations the orbit was
        # fitted on. Catch the mismatch here rather than in a spectrum
        # nobody can explain six months later.
        white_masked = white_summary.get("n_masked_total")
        if white_masked is None:  # summary from a pre-v1.3 fit
            white_masked = (white_summary.get("spot_handling") or {}).get(
                "n_masked", 0)
        if int(white_masked) != int(prep["n_masked"]):
            raise ValueError(
                f"The white fit used {white_masked} masked integration(s) but "
                f"this run masks {prep['n_masked']}. Pass the same "
                "anomaly_report used for the white fit (or none, if the white "
                "fit used none), and the same tilt_correction setting — the "
                "frozen orbit and the channel depths must come from the same "
                "data."
            )
        stellar = {k: prep["priors"][k] for k in ("st_teff", "st_logg", "st_met")}
        result = fit_transmission_spectrum(
            prep["lc"],
            white_summary,
            _resolve(self.output_dir, self.base_directory),
            instrument=self.instrument,
            regressors=prep["regressors"],
            stellar=stellar if self.ld_priors == "exotic-ld" else None,
            keep_mask=prep["keep_mask"],
        )
        return (
            f"Transmission spectrum complete: {result['n_bins']} channels "
            f"(fit v{result['fit_version']}, {result['n_masked']} integration(s) "
            "masked).\n"
            f"  median depth error: {result['median_depth_err_ppm']:.0f} ppm; "
            f"median channel rms: {result['median_rms_ppm']:.0f} ppm\n"
            f"CSV (for visit combining): {result['spectrum_csv']}\n"
            f"TauREx .dat: {result['spectrum_path']}\n"
            f"Plot: transmission_spectrum.pdf/.svg in the same directory."
        )


class CombineNirspecG395hVisits(BaseTool):
    """
    Stage 6: inverse-variance combine the per-visit transmission spectra
    of one target into the final Patchwork spectrum, and report the
    survey health metrics.

    Input is a JSON mapping detector -> list of per-visit
    ``transmission_spectrum.csv`` paths (from
    ``FitNirspecG395hTransmissionSpectrum``). Visits of one detector must
    share the binning scheme (they do, if fit with defaults). A single
    visit per detector is fine — the "combination" is then a pass-through.

    Outputs in ``output_dir``: ``combined_{nrs1,nrs2}_transmission_spectrum.csv``,
    ``combined_transmission_spectrum.pdf/.svg`` (both detectors, ready to
    overlay a published spectrum on), and ``combined_summary.json`` with
    the NRS1-NRS2 offset near the detector gap and median rms values —
    the numbers the Patchwork log asks for.

    Example
    -------
        CombineNirspecG395hVisits(
            spectra_by_detector='{"NRS1": ["fits/GJ_9827_d/o010/nrs1/spectro/transmission_spectrum.csv", '
                                '"fits/GJ_9827_d/o091/nrs1/spectro/transmission_spectrum.csv"], '
                                '"NRS2": ["fits/GJ_9827_d/o010/nrs2/spectro/transmission_spectrum.csv", '
                                '"fits/GJ_9827_d/o091/nrs2/spectro/transmission_spectrum.csv"]}',
            output_dir="fits/GJ_9827_d/combined",
            title="GJ 9827 d — Patchwork G395H",
        )
    """

    spectra_by_detector: str = RuntimeField(
        description='JSON dict: {"NRS1": [csv, ...], "NRS2": [csv, ...]}. '
                    "One or both detectors."
    )
    output_dir: str = RuntimeField(description="Directory for combined outputs.")
    title: str = RuntimeField(
        default="Patchwork G395H transmission spectrum",
        description="Plot title, e.g. 'GJ 9827 d — Patchwork G395H'.",
    )
    base_directory: str = StateField()

    def _run(self) -> str:
        mapping = self.spectra_by_detector
        if isinstance(mapping, str):
            mapping = json.loads(mapping)

        out = Path(_resolve(self.output_dir, self.base_directory))
        out.mkdir(parents=True, exist_ok=True)

        combined: dict[str, dict[str, np.ndarray]] = {}
        summary: dict[str, Any] = {"fit_version": PATCHWORK_FIT_VERSION}
        lines = []
        for det, paths in mapping.items():
            det = det.upper()
            paths = [_resolve(p, self.base_directory) for p in paths]
            S = combine_visit_spectra(paths)
            combined[det] = S
            rows = [
                {"wave": w, "wave_err": we, "depth": d / 1e6,
                 "depth_err": de / 1e6, "rms_ppm": r}
                for w, we, d, de, r in zip(
                    S["wave"], S["wave_err"], S["depth_ppm"],
                    S["depth_err_ppm"], S["rms_ppm"])
            ]
            csv_path = out / f"combined_{det.lower()}_transmission_spectrum.csv"
            write_spectrum_csv(
                csv_path, rows,
                header=f"{self.title} — {det}, {S['n_visits']} visit(s) combined",
            )
            summary[det] = {
                "n_visits": S["n_visits"],
                "n_channels": int(np.isfinite(S["depth_ppm"]).sum()),
                "median_depth_err_ppm": float(np.nanmedian(S["depth_err_ppm"])),
                "median_channel_rms_ppm": float(np.nanmedian(S["rms_ppm"])),
                "csv": str(csv_path),
            }
            lines.append(
                f"  {det}: {S['n_visits']} visit(s), "
                f"{summary[det]['n_channels']} channels, median depth err "
                f"{summary[det]['median_depth_err_ppm']:.0f} ppm, median rms "
                f"{summary[det]['median_channel_rms_ppm']:.0f} ppm"
            )

        if "NRS1" in combined and "NRS2" in combined:
            offset = detector_offset_ppm(combined["NRS1"], combined["NRS2"])
            summary["nrs1_nrs2_offset_ppm"] = offset
            lines.append(f"  NRS1-NRS2 offset near detector gap: {offset:+.1f} ppm")

        _plot_combined(combined, out, self.title)
        with (out / "combined_summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)

        return "\n".join(
            [f"Combined transmission spectrum written to {out}."]
            + lines
            + ["Plot: combined_transmission_spectrum.pdf/.svg. Overlay your "
               "literature spectrum on the CSVs for the reproduction check."]
        )
