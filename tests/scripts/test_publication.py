"""Science-facing regressions for publication exports and model inputs."""
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "patchwork"))
from rebin_spectra import rebin, write_dat
from publication_audit import binned_noise, band_contrast
from plot_survey_scale_heights import _detector_diagnostics, amplitude_in_H, planet_params


def test_archive_alias_and_invalid_scale_height_parameters(monkeypatch):
    from aster_toolkit.data_acquisition import mast
    query = MagicMock(return_value=[
        {"pl_name": "TOI-836.01", "pl_rade": 2.587, "pl_bmasse": 9.6,
         "pl_eqt": 665, "st_rad": .665},
        {"pl_name": "Invalid b", "pl_rade": 2., "pl_bmasse": 0,
         "pl_eqt": 500, "st_rad": .5}])
    monkeypatch.setattr(mast, "archive_tap_query", query)
    params = planet_params(["TOI-836 c", "Invalid b"])
    assert "TOI-836.01" in query.call_args.args[0][0]
    assert params["TOI-836 c"]["archive_name"] == "TOI-836.01"
    assert params["TOI-836 c"]["delta_ppm"] > 0
    assert "Invalid b" not in params


def test_taurex_full_width_roundtrip(tmp_path):
    pytest.importorskip("taurex")
    from taurex.data.spectrum.observed import ObservedSpectrum
    rows = [{"wave": 3.1, "wave_err": .05, "depth": .002,
             "depth_err": .00002}]
    path = tmp_path / "spectrum.dat"
    write_dat(path, rows, "test")
    # TauREx needs a 2D input, so test two distinct bins.
    rows.append(dict(rows[0], wave=3.3))
    write_dat(path, rows, "test")
    spectrum = ObservedSpectrum(str(path))
    np.testing.assert_allclose(spectrum.rawData[:, 3], .1)
    np.testing.assert_allclose(spectrum.spectrum, .002)


def test_rebin_support_is_not_shifted_by_depth_precision():
    rows = rebin([3.01, 3.04, 3.07, 3.10], [.015]*4,
                 [100]*4, [1, 10, 1, 10], [1]*4, resolution=50)
    assert rows
    assert rows[0]["wave"] == pytest.approx(3.025)
    assert rows[0]["wave_err"] == pytest.approx(.03)


def test_noisy_visit_not_hidden_by_median(tmp_path):
    for visit, beta in (("o013", 3.37), ("o014", 2.06)):
        p = tmp_path / "GJ_3090_b" / "fits" / visit / "nrs2"
        p.mkdir(parents=True)
        (p / "white_fit_summary.json").write_text(json.dumps({
            "rednoise": {"beta_median": beta}, "fit_version": "1.4"}))
    diag = _detector_diagnostics(tmp_path, "GJ_3090_b", "NRS2")
    assert diag["rednoise_beta"] < 3
    s = {"wave": np.array([4.1, 4.2, 4.3]), "depth_ppm": np.array([100., 101., 99.]),
         "depth_err_ppm": np.ones(3)*10, **diag}
    assert "beta>3" in amplitude_in_H({"NRS2": s}, 100)["quality_flags"]


def test_time_binning_white_noise_scaling():
    rng = np.random.default_rng(721)
    t = np.arange(20000)/86400
    r = rng.normal(0, 100e-6, t.size)
    rms, count = binned_noise(t, r, 5)
    assert count > 50
    assert rms == pytest.approx(100/np.sqrt(300), rel=.25)


def test_band_contrast_offset_invariance_and_errors():
    spectrum = {"wave": np.array([3., 3.1, 3.25, 3.4, 3.6, 3.65]),
                "depth_ppm": np.array([100., 100., 130., 130., 100., 100.]),
                "depth_err_ppm": np.full(6, 10.)}
    windows = ((2.95, 3.15), (3.20, 3.50), (3.55, 3.70))
    result = band_contrast(spectrum, windows)
    assert result["contrast_ppm"] == pytest.approx(30)
    assert result["error_diagonal_ppm"] == pytest.approx(np.sqrt(75))
    assert sum(json.loads(result["coefficients"])) == pytest.approx(0)
    spectrum["depth_ppm"] += 2000
    assert band_contrast(spectrum, windows)["contrast_ppm"] == pytest.approx(30)
    with pytest.raises(ValueError, match="no valid channels"):
        band_contrast(spectrum, ((1., 2.), *windows[1:]))


def test_forward_model_uses_system_parameters(monkeypatch, tmp_path):
    pytest.importorskip("taurex")
    pytest.importorskip("orchestral")
    module = importlib.import_module("aster_toolkit.taurex.forward_model")
    model = MagicMock()
    model.model.return_value = (np.array([2000., 3000.]), np.array([.001, .0011]))
    constructor = MagicMock(return_value=model)
    monkeypatch.setattr(module, "TransmissionModel", constructor)
    module.generate_taurex_model(star_radius=.3, planet_radius=.2,
        planet_mass=.02, star_temp=3200, semi_major_axis=.07,
        orbital_period=7, molecular_abundances={}, base_directory=str(tmp_path))
    kwargs = constructor.call_args.kwargs
    from taurex.planet import Planet
    from taurex.stellar import BlackbodyStar
    expected = Planet(planet_radius=.2, planet_mass=.02)
    assert kwargs["planet"].fullRadius == pytest.approx(expected.fullRadius)
    assert kwargs["planet"].mass == pytest.approx(expected.mass)
    assert kwargs["star"].radius == pytest.approx(BlackbodyStar(radius=.3).radius)
    with pytest.raises(ValueError, match="positive"):
        module.generate_taurex_model(atm_min_pressure=0)
