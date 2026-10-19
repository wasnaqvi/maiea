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


def _v14_target(root, impacts, version="1.4"):
    """A minimal v1.4-shaped target: one visit per impact parameter."""
    from aster_toolkit.data_reduction.juliet import write_spectrum_csv
    target = root / "T_1_b"
    rows = [{"wave": 3.0, "wave_err": .015, "depth": .001, "depth_err": 2e-5,
             "rms_ppm": 500.}]
    summary = {"fit_version": version, "visits": {}}
    # v1.4 wrote the half bin width to the DAT, v1.5 the full width.
    width, scale = (("bin_half_width_um", 1) if version == "1.4"
                    else ("bin_width_um", 2))
    for i, b in enumerate(impacts):
        visit = f"o00{i + 1}"
        fit_dir = target / "fits" / visit / "nrs1"
        (fit_dir / "pass1").mkdir(parents=True)
        (fit_dir / "spectro").mkdir()
        white = {"fit_version": version, "b_p1": {"median": b}}
        (fit_dir / "white_fit_summary.json").write_text(json.dumps(white))
        (fit_dir / "pass1" / "white_fit_summary.json").write_text(
            json.dumps(dict(white, fit_version=f"{version}-noscan")))
        write_spectrum_csv(fit_dir / "spectro" / "transmission_spectrum.csv",
                           rows, header=f"Patchwork juliet fit v{version}, instrument=NRS1")
        (fit_dir / "spectro" / "transmission_spectrum.dat").write_text(
            f"# wavelength_um  transit_depth  depth_err  {width}\n"
            f"# Patchwork juliet fit v{version}, instrument=NRS1\n"
            f"3.000000  0.00100000  0.00002000  {.015 * scale:.6f}\n")
        summary["visits"][visit] = {"fits": {"NRS1": {"fit_version": version}}}
    (target / "combined").mkdir()
    write_spectrum_csv(target / "combined" / "combined_nrs1_transmission_spectrum.csv",
                       rows, header=f"T 1 b — NRS1, 1 visit(s), Patchwork v{version}")
    (target / "patchwork_summary.json").write_text(json.dumps(summary))
    return target


def test_restamp_v14_below_grazing_threshold(tmp_path):
    from restamp_fit_version import apply_changes, plan_target
    target = _v14_target(tmp_path, [0.3, 0.69])
    plan = plan_target(target)
    assert plan["status"] == "restamp" and plan["max_b"] == pytest.approx(0.69)
    apply_changes(plan["changes"])

    summary = json.loads((target / "patchwork_summary.json").read_text())
    assert summary["fit_version"] == "1.5"
    fit = summary["visits"]["o002"]["fits"]["NRS1"]
    assert fit["fit_version"] == "1.5" and fit["ld_fixed_per_channel"] is False
    assert fit["impact_parameter"] == pytest.approx(0.69)
    assert summary["resolution"] == 100
    assert summary["fit_restamps"][0]["from"] == "1.4"
    fit_dir = target / "fits" / "o001" / "nrs1"
    pass1 = json.loads((fit_dir / "pass1" / "white_fit_summary.json").read_text())
    assert pass1["fit_version"] == "1.5-noscan"

    # CSV: only the header changes, CRLF endings and data kept byte for byte.
    csv_path = fit_dir / "spectro" / "transmission_spectrum.csv"
    old = (csv_path.with_name(csv_path.name + ".pre-restamp")).read_bytes()
    new = csv_path.read_bytes()
    assert b"fit v1.5 (restamped from v1.4)" in new.split(b"\r\n")[0]
    assert old.split(b"\r\n")[1:] == new.split(b"\r\n")[1:]
    combined = (target / "combined" / "combined_nrs1_transmission_spectrum.csv").read_text()
    assert "Patchwork v1.5 (restamped from v1.4)" in combined

    # DAT: half width -> full width, as v1.5 writes it.
    dat = (fit_dir / "spectro" / "transmission_spectrum.dat").read_text().splitlines()
    assert "bin_width_um" in dat[0] and "half" not in dat[0]
    assert dat[2].split()[3] == "0.030000"

    # Idempotent: a second pass finds nothing to do.
    assert plan_target(target)["status"] == "current"


def test_restamp_refuses_grazing_target_whole(tmp_path):
    from restamp_fit_version import Refused, main, plan_target
    target = _v14_target(tmp_path, [0.3, 0.94])
    with pytest.raises(Refused, match="b = 0.94"):
        plan_target(target)
    main([str(tmp_path), "--write"])
    assert not list(target.rglob("*.pre-restamp"))
    assert json.loads((target / "patchwork_summary.json").read_text())["fit_version"] == "1.4"


def test_restamp_leaves_native_v15_grazing_fit_alone(tmp_path):
    from restamp_fit_version import plan_target
    target = _v14_target(tmp_path, [0.94], version="1.5")
    summary = json.loads((target / "patchwork_summary.json").read_text())
    summary["resolution"] = 100.0
    fit = summary["visits"]["o001"]["fits"]["NRS1"]
    fit.update(ld_fixed_per_channel=True, impact_parameter=0.94)
    (target / "patchwork_summary.json").write_text(json.dumps(summary))
    assert plan_target(target)["status"] == "current"


def _white_summary(**over):
    s = {"instrument": "NRS1", "t0_obs": 100.0 - 50 / 1440,
         "P_p1": {"median": 5.0}, "t0_p1": {"median": 100.0},
         "p_p1": {"median": 0.03}, "b_p1": {"median": 0.3},
         "a_p1": {"median": 20.0}, "q1_NRS1": {"median": 0.1},
         "q2_NRS1": {"median": 0.2}, "depth_ppm": {"median": 900.0},
         "rednoise": {"beta_median": 1.4}, "regressor_names": ["time"],
         "planet_name": "Test b", "visit": "o001"}
    s.update(over)
    return s


def test_transit_only_model_is_the_bare_transit():
    pytest.importorskip("batman.transitmodel")
    from aster_toolkit.data_reduction.juliet import transit_only_model
    t = np.linspace(99.8, 100.2, 2000)
    transit = transit_only_model(_white_summary(), t)
    assert transit[0] == pytest.approx(1.0) and transit[-1] == pytest.approx(1.0)
    # Limb darkening deepens the centre past p^2 = 900 ppm, not by much.
    assert 900 < (1 - transit.min()) * 1e6 < 1100
    assert t[np.argmin(transit)] == pytest.approx(100.0, abs=1e-3)


def test_white_fit_figure_removes_systematics(tmp_path, monkeypatch):
    pytest.importorskip("batman.transitmodel")
    pytest.importorskip("matplotlib")
    from aster_toolkit.data_reduction import juliet
    s = _white_summary()
    t = np.linspace(99.85, 100.15, 1500)
    transit = juliet.transit_only_model(s, t)
    rng = np.random.default_rng(1)
    jitter = 300e-6 * rng.standard_normal(t.size)   # a ragged regressor
    model = transit + 2e-3 * (t - 100.0) + jitter
    flux = model + 100e-6 * rng.standard_normal(t.size)
    drawn = []
    real_bin = juliet._bin_series

    def capture(x, y, **kwargs):  # the figure bins exactly what it draws
        drawn.append(np.asarray(y))
        return real_bin(x, y, **kwargs)

    monkeypatch.setattr(juliet, "_bin_series", capture)
    path = juliet.plot_white_fit_series(tmp_path, s, t, flux, np.full(t.size, 1e-4), model)
    assert Path(path).is_file() and (tmp_path / "white_lightcurve_fit.svg").is_file()
    corrected, residual = drawn[0], drawn[1]
    # Drawn data = transit + residual: no trend, no regressor jitter left.
    np.testing.assert_allclose(corrected, transit + (flux - model))
    np.testing.assert_allclose(residual, (flux - model) * 1e6)
    assert np.std(corrected - transit) < 150e-6
