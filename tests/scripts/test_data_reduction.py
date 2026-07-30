"""QA tests for aster_toolkit.data_reduction (Patchwork toolkit).

Pure-local: no network, no juliet/exoTEDRF environments. Everything that
can be exercised with numpy/astropy synthetics is exercised here —
especially the guards that protect against the two documented silent
failures (prior-returning fit, stale-posterior reload) and the physics
helpers whose bugs would not crash but would corrupt the survey.

Run:  python3 -m pytest tests/scripts/test_data_reduction.py -v
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from aster_toolkit.data_reduction.lightcurves import (  # noqa: E402
    MJD_TO_BJD_OFFSET,
    bin_at_resolution,
    build_lightcurves,
    build_regressor_matrix,
    detect_tilt_events,
    load_stage3_spectra,
    oot_mask_from_baseline,
    oot_mask_from_ephemeris,
    pca_regressors,
    propagate_t0,
    rednoise_beta,
    step_regressors,
)
from aster_toolkit.data_reduction.juliet import (  # noqa: E402
    combine_visit_spectra,
    detector_offset_ppm,
    read_spectrum_csv,
    write_spectrum_csv,
)
from aster_toolkit.data_reduction.discover import (  # noqa: E402
    group_visits,
    parse_uncal_name,
    resolve_planet_from_rows,
)
from aster_toolkit.data_reduction.exotedrf import (  # noqa: E402
    PATCHWORK_G395H_CONFIG,
    inspect_uncal_directory,
    write_dms_config,
)
from aster_toolkit.data_reduction.optimize import (  # noqa: E402
    G395H_SWEEP,
    best_params_to_overrides,
    omega_hash,
    parse_cost_table,
    summarize_sweep,
    write_optimize_config,
)
from aster_toolkit.data_reduction.survey import (  # noqa: E402
    existing_fit_products,
    load_manifest,
    run_patchwork_target,
    stage_visit_uncals,
    write_fir_slurm_script,
)


# -------------------- synthetic transit helpers --------------------

PERIOD = 6.2018300
T0_REF = 2460265.10196
DUR_HR = 1.28
DEPTH = 1000e-6


def synthetic_visit(n=2000, span_hr=5.0, t_center=None, depth=DEPTH,
                    noise=200e-6, seed=0):
    """White lightcurve with a box transit centred in the window."""
    rng = np.random.default_rng(seed)
    if t_center is None:
        t_center = propagate_t0(T0_REF, PERIOD, np.array([2460500.0]))
    time = t_center + np.linspace(-span_hr / 2, span_hr / 2, n) / 24.0
    flux = np.ones(n)
    in_tr = np.abs(time - t_center) < 0.5 * DUR_HR / 24.0
    flux[in_tr] -= depth
    flux += rng.normal(0, noise, n)
    return time, flux, in_tr


def synthetic_spectra(n_time=600, n_wave=500, depth=DEPTH, seed=1):
    """Fake Stage 3 dict for build_lightcurves (NRS1 wavelength range)."""
    rng = np.random.default_rng(seed)
    wave = np.linspace(2.7, 3.9, n_wave)
    t_center = propagate_t0(T0_REF, PERIOD, np.array([2460500.0]))
    time = t_center + np.linspace(-2.5, 2.5, n_time) / 24.0
    flux = np.ones((n_time, n_wave)) * 1e5
    in_tr = np.abs(time - t_center) < 0.5 * DUR_HR / 24.0
    flux[in_tr] *= (1 - depth)
    flux += rng.normal(0, 30, flux.shape)
    flux_err = np.full_like(flux, 30.0)
    return {"wave": wave, "wave_err": np.full(n_wave, 0.001),
            "flux": flux, "flux_err": flux_err, "time": time}


# -------------------- lightcurves --------------------


class TestPropagation:
    def test_propagate_t0_lands_near_data(self):
        times = np.linspace(2460500.0, 2460500.2, 100)
        t0 = propagate_t0(T0_REF, PERIOD, times)
        assert abs(t0 - np.median(times)) < PERIOD / 2

    def test_propagate_integral_epochs(self):
        t0 = propagate_t0(T0_REF, PERIOD, np.array([T0_REF + 33 * PERIOD]))
        assert t0 == pytest.approx(T0_REF + 33 * PERIOD, abs=1e-9)


class TestOotMasks:
    def test_baseline_mask(self):
        m = oot_mask_from_baseline(100, [20, -20])
        assert m[:20].all() and m[-20:].all() and not m[20:-20].any()
        assert m.sum() == 40

    def test_ephemeris_mask_pads_transit(self):
        t0 = 2460500.0
        times = t0 + np.linspace(-2, 2, 1000) / 24
        m = oot_mask_from_ephemeris(times, t0, DUR_HR)
        # every point within the padded half-duration is masked out
        half_pad_d = 0.5 * DUR_HR / 24 * 1.15
        assert not m[np.abs(times - t0) <= half_pad_d * 0.999].any()
        assert m[np.abs(times - t0) > half_pad_d * 1.001].all()


class TestBinning:
    def test_constant_r_spacing(self):
        wave = np.linspace(3.0, 4.0, 2000)
        flux = np.ones((10, 2000))
        err = np.full_like(flux, 0.01)
        b = bin_at_resolution(wave, flux, err, resolution=100)
        # adjacent edges follow w*(1+1/R): center ratio approximately constant
        ratios = b["wave"][1:] / b["wave"][:-1]
        assert np.allclose(ratios, ratios[0], rtol=1e-3)
        assert np.allclose(b["flux"], 1.0, atol=1e-12)

    def test_binned_error_shrinks(self):
        wave = np.linspace(3.0, 4.0, 2000)
        flux = np.ones((5, 2000))
        err = np.full_like(flux, 0.01)
        b = bin_at_resolution(wave, flux, err, resolution=100)
        assert (b["flux_err"] < 0.01).all()

    def test_nan_column_ignored(self):
        wave = np.linspace(3.0, 4.0, 200)
        flux = np.ones((5, 200))
        flux[:, 50] = np.nan
        err = np.full_like(flux, 0.01)
        b = bin_at_resolution(wave, flux, err, resolution=50)
        assert np.isfinite(b["flux"]).all()


class TestBuildLightcurves:
    def test_happy_path(self):
        lc = build_lightcurves(synthetic_spectra(), detector="NRS1",
                               t0_ref=T0_REF, period=PERIOD,
                               duration_hr=DUR_HR)
        assert lc["transit_coverage"] == pytest.approx(1.0)
        # normalized baseline sits at 1
        assert np.nanmedian(lc["wl_flux"][lc["oot_mask"]]) == pytest.approx(
            1.0, abs=1e-4)
        # transit depth is recovered in the white lightcurve
        in_tr = ~lc["oot_mask"]
        depth = 1 - np.nanmedian(lc["wl_flux"][in_tr])
        assert depth == pytest.approx(DEPTH, rel=0.3)
        # wavelength cuts applied
        assert lc["wave"].min() >= 2.87 and lc["wave"].max() <= 3.72

    def test_transit_in_window_guard_stale_ephemeris(self):
        spectra = synthetic_spectra()
        with pytest.raises(ValueError, match="NOT in\\s+this data"):
            build_lightcurves(spectra, detector="NRS1",
                              t0_ref=T0_REF + 0.25 * PERIOD, period=PERIOD,
                              duration_hr=DUR_HR)

    def test_transit_in_window_guard_mjd_axis(self):
        spectra = synthetic_spectra()
        # un-convert the axis: load_stage3_spectra would fix this, but if a
        # caller passes raw MJD the guard must still fire
        spectra["time"] = spectra["time"] - MJD_TO_BJD_OFFSET
        with pytest.raises(ValueError):
            build_lightcurves(spectra, detector="NRS1", t0_ref=T0_REF,
                              period=PERIOD, duration_hr=DUR_HR)

    def test_needs_mask_source(self):
        with pytest.raises(ValueError, match="duration_hr or baseline_ints"):
            build_lightcurves(synthetic_spectra(), detector="NRS1",
                              t0_ref=T0_REF, period=PERIOD)


class TestMjdConversion:
    def test_load_stage3_converts_mjd(self, tmp_path):
        from astropy.io import fits

        n_t, n_w = 20, 50
        hdus = fits.HDUList([fits.PrimaryHDU()])
        for name, data in [
            ("Wave", np.linspace(3, 4, n_w)),
            ("Wave Err", np.full(n_w, 0.01)),
            ("Flux", np.ones((n_t, n_w))),
            ("Flux Err", np.full((n_t, n_w), 0.01)),
            ("Time", np.linspace(60465.0, 60465.2, n_t)),  # MJD!
        ]:
            hdus.append(fits.ImageHDU(data=data, name=name))
        path = tmp_path / "fake_box_spectra_fullres.fits"
        hdus.writeto(path)

        out = load_stage3_spectra(path)
        assert np.nanmedian(out["time"]) > 2.4e6  # converted to BJD

    def test_load_stage3_leaves_bjd_alone(self, tmp_path):
        from astropy.io import fits

        n_t, n_w = 20, 50
        t_bjd = np.linspace(2460465.5, 2460465.7, n_t)
        hdus = fits.HDUList([fits.PrimaryHDU()])
        for name, data in [
            ("Wave", np.linspace(3, 4, n_w)),
            ("Wave Err", np.full(n_w, 0.01)),
            ("Flux", np.ones((n_t, n_w))),
            ("Flux Err", np.full((n_t, n_w), 0.01)),
            ("Time", t_bjd),
        ]:
            hdus.append(fits.ImageHDU(data=data, name=name))
        path = tmp_path / "fake_box_spectra_fullres.fits"
        hdus.writeto(path)
        out = load_stage3_spectra(path)
        assert np.allclose(out["time"], t_bjd)


class TestTiltEvents:
    def test_detects_synthetic_step(self):
        rng = np.random.default_rng(2)
        flux = 1 + rng.normal(0, 100e-6, 3000)
        flux[1800:] += 800e-6
        events = detect_tilt_events(flux)
        assert len(events) == 1
        assert abs(events[0]["index"] - 1800) <= 3
        assert events[0]["amplitude"] == pytest.approx(800e-6, rel=0.3)

    def test_no_false_positive_on_noise(self):
        rng = np.random.default_rng(3)
        flux = 1 + rng.normal(0, 100e-6, 3000)
        assert detect_tilt_events(flux) == []

    def test_transit_not_flagged(self):
        # a real transit is a step at ingress/egress; the exclude mask must
        # keep it from being flagged
        time, flux, in_tr = synthetic_visit(n=3000, noise=50e-6)
        events = detect_tilt_events(flux, exclude_mask=~in_tr)
        assert events == []

    def test_min_separation_merges(self):
        rng = np.random.default_rng(4)
        flux = 1 + rng.normal(0, 50e-6, 2000)
        flux[1000:] += 900e-6
        events = detect_tilt_events(flux, min_separation=30)
        idx = [e["index"] for e in events]
        assert all(abs(a - b) >= 30 for i, a in enumerate(idx)
                   for b in idx[i + 1:])


class TestRegressors:
    def test_step_regressors_shape(self):
        m = step_regressors(100, [{"index": 40, "amplitude": 1e-4}])
        assert m.shape == (100, 1)
        assert m[:40].sum() == 0 and m[40:].sum() == 60

    def test_empty_events(self):
        assert step_regressors(50, []).shape == (50, 0)

    def test_matrix_columns_and_standardization(self):
        time = np.linspace(0, 1, 200)
        diag = {"x": np.random.default_rng(0).normal(0, 1, 200),
                "y": np.random.default_rng(1).normal(0, 1, 200),
                "fwhm": np.random.default_rng(2).normal(0, 1, 200)}
        events = [{"index": 100, "amplitude": 1e-4}]
        M, names = build_regressor_matrix(time, diag, events)
        assert names == ["time", "trace_x", "trace_y", "trace_fwhm",
                         "tilt_step_0"]
        assert M.shape == (200, 5)
        # standardized: median ~0 for the continuous columns
        for j in range(4):
            assert abs(np.median(M[:, j])) < 0.2
        # step column stays 0/1
        assert set(np.unique(M[:, 4])) == {0.0, 1.0}

    def test_size_mismatch_diagnostics_skipped(self):
        time = np.linspace(0, 1, 200)
        diag = {"x": np.zeros(150)}  # wrong length
        M, names = build_regressor_matrix(time, diag, [])
        assert names == ["time"]


class TestRednoiseBeta:
    def test_white_noise_beta_near_one(self):
        rng = np.random.default_rng(5)
        t = 2460500.0 + np.arange(5000) * 20 / 86400  # 20 s cadence
        r = rng.normal(0, 300e-6, 5000)
        out = rednoise_beta(r, t)
        assert 0.8 < out["beta_median"] < 1.25
        assert out["rms_unbinned_ppm"] == pytest.approx(300, rel=0.1)

    def test_red_noise_beta_large(self):
        rng = np.random.default_rng(6)
        t = 2460500.0 + np.arange(5000) * 20 / 86400
        # slow sinusoid (~25 min period) buried in white noise
        red = 300e-6 * np.sin(2 * np.pi * (t - t[0]) * 86400 / 1500)
        r = rng.normal(0, 300e-6, 5000) + red
        out = rednoise_beta(r, t)
        assert out["beta_median"] > 1.5

    def test_short_series_nan(self):
        out = rednoise_beta(np.zeros(10), np.linspace(0, 1, 10))
        assert np.isnan(out["beta_median"])


class TestPcaRegressors:
    def _write_calints(self, path, nints=200, ny=8, nx=16, seed=7):
        from astropy.io import fits

        rng = np.random.default_rng(seed)
        cube = np.full((nints, ny, nx), 100.0)
        # a drifting gaussian trace: coherent structure for the PCA
        y0 = 4 + 0.5 * np.sin(np.linspace(0, 4 * np.pi, nints))
        yy = np.arange(ny)
        for i in range(nints):
            cube[i] += 1000 * np.exp(-0.5 * ((yy - y0[i]) / 1.2) ** 2)[:, None]
        cube += rng.normal(0, 5, cube.shape)
        hdu = fits.HDUList([fits.PrimaryHDU(),
                            fits.ImageHDU(data=cube, name="SCI")])
        hdu.writeto(path)

    def test_components_shape_and_standardization(self, tmp_path):
        p = tmp_path / "seg1_nrs1_calints.fits"
        self._write_calints(p)
        comps = pca_regressors([str(p)], n_components=6, max_pixels=100)
        assert comps.shape == (200, 6)
        assert np.isfinite(comps).all()
        # robust-standardized: median ~0
        assert np.all(np.abs(np.median(comps, axis=0)) < 0.5)
        # first component captures the injected trace drift (correlates
        # with the sinusoid)
        drift = np.sin(np.linspace(0, 4 * np.pi, 200))
        c = abs(np.corrcoef(comps[:, 0], drift)[0, 1])
        assert c > 0.7

    def test_build_matrix_with_pca(self):
        time = np.linspace(0, 1, 200)
        pca = np.random.default_rng(0).normal(0, 1, (200, 3))
        M, names = build_regressor_matrix(time, None, [], pca_components=pca)
        assert names == ["time", "pca_0", "pca_1", "pca_2"]
        assert M.shape == (200, 4)


# -------------------- juliet stage 6 --------------------


def _rows(depths, rng=None):
    wave = np.linspace(3.0, 3.7, len(depths))
    return [{"wave": float(w), "wave_err": 0.02, "depth": float(d),
             "depth_err": 50e-6, "rms_ppm": 400.0}
            for w, d in zip(wave, depths)]


class TestSpectrumCsvRoundtrip:
    def test_roundtrip(self, tmp_path):
        rows = _rows(np.full(10, 1000e-6))
        path = tmp_path / "spec.csv"
        write_spectrum_csv(path, rows, header="test")
        back = read_spectrum_csv(path)
        assert back["wave"].size == 10
        assert np.allclose(back["depth_ppm"], 1000.0, atol=0.01)
        assert np.allclose(back["depth_err_ppm"], 50.0, atol=0.01)


class TestCombine:
    def test_two_visits_shrink_error(self, tmp_path):
        p1, p2 = tmp_path / "v1.csv", tmp_path / "v2.csv"
        write_spectrum_csv(p1, _rows(np.full(10, 900e-6)))
        write_spectrum_csv(p2, _rows(np.full(10, 1100e-6)))
        S = combine_visit_spectra([str(p1), str(p2)])
        assert S["n_visits"] == 2
        assert np.allclose(S["depth_ppm"], 1000.0, atol=0.5)
        assert np.allclose(S["depth_err_ppm"], 50 / np.sqrt(2), rtol=1e-3)

    def test_dropped_channel_tolerated(self, tmp_path):
        # one visit lost a channel (bad column): the combination keeps the
        # union grid and records how many visits fed each channel
        rows_full = _rows(np.full(10, 1000e-6))
        rows_missing = [r for i, r in enumerate(_rows(np.full(10, 1000e-6)))
                        if i != 4]
        p1, p2 = tmp_path / "v1.csv", tmp_path / "v2.csv"
        write_spectrum_csv(p1, rows_full)
        write_spectrum_csv(p2, rows_missing)
        S = combine_visit_spectra([str(p1), str(p2)])
        assert S["wave"].size == 10
        n_per = S["n_visits_per_channel"]
        assert n_per[4] == 1 and (np.delete(n_per, 4) == 2).all()
        assert np.isfinite(S["depth_ppm"]).all()

    def test_incompatible_grids_refused(self, tmp_path):
        # a genuinely different binning scheme must still refuse
        rows_a = _rows(np.full(10, 1000e-6))
        rows_b = _rows(np.full(10, 1000e-6))
        for r in rows_b:
            r["wave"] += 0.03  # off-grid by more than the tolerance
        p1, p2 = tmp_path / "v1.csv", tmp_path / "v2.csv"
        write_spectrum_csv(p1, rows_a)
        write_spectrum_csv(p2, rows_b)
        with pytest.raises(ValueError, match="wavelength grid"):
            combine_visit_spectra([str(p1), str(p2)])

    def test_single_visit_passthrough(self, tmp_path):
        p1 = tmp_path / "v1.csv"
        write_spectrum_csv(p1, _rows(np.full(10, 1000e-6)))
        S = combine_visit_spectra([str(p1)])
        assert S["n_visits"] == 1
        assert np.allclose(S["depth_ppm"], 1000.0, atol=0.01)


class TestDetectorOffset:
    def test_offset_sign(self):
        nrs1 = {"depth_ppm": np.full(20, 1100.0)}
        nrs2 = {"depth_ppm": np.full(20, 1000.0)}
        assert detector_offset_ppm(nrs1, nrs2) == pytest.approx(100.0)


# -------------------- discover --------------------


class TestParseUncalName:
    def test_segmented(self):
        rec = parse_uncal_name(
            "jw04098010001_04102_00001-seg003_nrs1_uncal.fits")
        assert rec["visit_prefix"] == "jw04098010001"
        assert rec["segment"] == 3
        assert rec["detector"] == "NRS1"
        assert rec["exp_tag"] == "04102"

    def test_unsegmented_defaults_seg1(self):
        rec = parse_uncal_name("jw01185018001_04102_00001_nrs2_uncal.fits")
        assert rec["segment"] == 1

    def test_rejects_non_uncal(self):
        assert parse_uncal_name("jw04098010001_04102_00001_nrs1_rate.fits") is None
        assert parse_uncal_name("random.fits") is None


def _archive_row(name, period, tranmid, dur=1.3, rade=2.0):
    return {"pl_name": name, "pl_orbper": period, "pl_tranmid": tranmid,
            "pl_trandur": dur, "pl_rade": rade}


class TestResolvePlanet:
    # exposure window: 5 h centred on BJD 2460500.0
    START = 2460500.0 - 2.5 / 24 - MJD_TO_BJD_OFFSET
    END = 2460500.0 + 2.5 / 24 - MJD_TO_BJD_OFFSET

    def test_single_confident_match(self):
        rows = [_archive_row("X b", 3.0, 2460500.0 - 100 * 3.0),
                _archive_row("X c", 7.7, 2460497.0)]
        res = resolve_planet_from_rows(rows, self.START, self.END)
        assert res["confident"]
        assert res["matches"][0]["pl_name"] == "X b"
        assert res["matches"][0]["transit_coverage"] == pytest.approx(1.0)

    def test_neighbouring_epoch_not_matched(self):
        # transit 6 h from window centre on a 5 h window: must NOT match
        rows = [_archive_row("X b", 3.0, 2460500.0 + 6 / 24 - 100 * 3.0)]
        res = resolve_planet_from_rows(rows, self.START, self.END)
        assert res["matches"] == []

    def test_partial_overlap_matches(self):
        # mid-transit right at the window edge -> ingress-only coverage
        rows = [_archive_row("X b", 3.0, 2460500.0 + 2.4 / 24 - 50 * 3.0)]
        res = resolve_planet_from_rows(rows, self.START, self.END)
        assert len(res["matches"]) == 1
        cov = res["matches"][0]["transit_coverage"]
        assert 0.0 < cov <= 1.0

    def test_untestable_recorded_not_matched(self):
        rows = [_archive_row("X b", 3.0, 2460500.0 - 100 * 3.0),
                {"pl_name": "X d", "pl_orbper": None, "pl_tranmid": None,
                 "pl_trandur": None, "pl_rade": 1.5}]
        res = resolve_planet_from_rows(rows, self.START, self.END)
        assert res["untestable"] == ["X d"]
        assert res["confident"]  # exactly one *testable* match

    def test_empty_rows(self):
        res = resolve_planet_from_rows([], self.START, self.END)
        assert not res["confident"] and "No archive planet" in res["note"]


class TestGroupVisits:
    def _exp(self, prefix, tag, det, nints=1000, **hdr):
        header = {"GRATING": "G395H", "FILTER": "F290LP",
                  "SUBARRAY": "SUB2048", "EXP_TYPE": "NRS_BRIGHTOBJ",
                  "NINTS": nints, "TARGPROP": "HOST",
                  "TARG_RA": 10.0, "TARG_DEC": -10.0,
                  "EXPSTART": 60465.0, "EXPEND": 60465.2}
        header.update(hdr)
        return {"visit_prefix": prefix, "exp_tag": tag, "detector": det,
                "program": prefix[2:7], "observation": prefix[7:10],
                "visit": prefix[10:13],
                "files": [f"{prefix}_{tag}_{det}.fits"],
                "directories": [f"/raw/{prefix}"],
                "segments_found": [1], "segments_expected": 1,
                "missing_segments": [], "complete": True, "header": header}

    def test_ta_dropped_by_exptype(self):
        scan = {"exposures": [
            self._exp("jw01111001001", "04102", "NRS1"),
            self._exp("jw01111001001", "04102", "NRS2"),
            self._exp("jw01111001001", "02101", "NRS1", nints=3,
                      EXP_TYPE="NRS_TASLIT"),
        ]}
        visits = group_visits(scan)
        assert len(visits) == 1
        assert visits[0]["sci_tag"] == "04102"
        assert set(visits[0]["detectors"]) == {"NRS1", "NRS2"}
        assert visits[0]["complete"]

    def test_largest_nints_tag_wins(self):
        scan = {"exposures": [
            self._exp("jw01111001001", "04101", "NRS1", nints=10),
            self._exp("jw01111001001", "04102", "NRS1", nints=5000),
            self._exp("jw01111001001", "04102", "NRS2", nints=5000),
        ]}
        visits = group_visits(scan)
        assert visits[0]["sci_tag"] == "04102"

    def test_duplicate_download_flagged(self):
        e1 = self._exp("jw01111001001", "04102", "NRS1")
        e2 = self._exp("jw01111001001", "04102", "NRS2")
        e2["directories"] = ["/raw/other_dir"]
        visits = group_visits({"exposures": [e1, e2]})
        assert visits[0]["duplicate_download"]


# -------------------- exotedrf --------------------


class TestDmsConfig:
    def test_frozen_keys_present(self, tmp_path):
        path = write_dms_config(
            tmp_path / "run_DMS.yaml", input_dir="/data",
            filter_detector="NRS1", crds_cache_path="/crds",
            run_stages=[1, 2, 3], baseline_ints=[100, -100],
        )
        text = path.read_text()
        assert "extract_method : 'box'" in text
        assert "extract_width : 16" in text
        assert "oof_method : 'scale-achromatic'" in text
        assert "PCAReconstructStep : 'skip'" in text
        assert "baseline_ints : [100, -100]" in text
        # yaml-parsable by the pinned env's loader
        try:
            import yaml
            cfg = yaml.safe_load(path.read_text())
            assert cfg["extract_width"] == 16
        except ImportError:
            pass

    def test_config_version_frozen(self):
        # survey definition: these exact values, or the version must bump
        assert PATCHWORK_G395H_CONFIG["extract_method"] == "box"
        assert PATCHWORK_G395H_CONFIG["extract_width"] == 16
        assert PATCHWORK_G395H_CONFIG["oof_method"] == "scale-achromatic"
        assert PATCHWORK_G395H_CONFIG["OneOverFStep_int"] == "skip"


class TestInspectUncal:
    def _write_uncal(self, path, det, segnum, segtot):
        from astropy.io import fits

        h = fits.PrimaryHDU()
        h.header["DETECTOR"] = det
        h.header["EXSEGNUM"] = segnum
        h.header["EXSEGTOT"] = segtot
        h.header["NINTS"] = 100
        h.header["SUBARRAY"] = "SUB2048"
        fits.HDUList([h]).writeto(path)

    def test_complete_and_incomplete(self, tmp_path):
        self._write_uncal(tmp_path / "a_nrs1_uncal.fits", "NRS1", 1, 2)
        self._write_uncal(tmp_path / "b_nrs1_uncal.fits", "NRS1", 2, 2)
        self._write_uncal(tmp_path / "c_nrs2_uncal.fits", "NRS2", 1, 2)
        report = inspect_uncal_directory(tmp_path)
        assert report["detectors"]["NRS1"]["complete"]
        assert not report["detectors"]["NRS2"]["complete"]
        assert report["detectors"]["NRS2"]["missing_segments"] == [2]


# -------------------- optimize --------------------


class TestOptimizerRule:
    def test_omega_hash_deterministic(self, tmp_path):
        kw = dict(input_dir="/data", detector="NRS1",
                  baseline_ints=[100, -100], name_tag="t",
                  crds_cache_path="/crds")
        p1 = write_optimize_config(tmp_path / "a.yaml", **kw)
        p2 = write_optimize_config(tmp_path / "b.yaml", **kw)
        assert omega_hash(p1) == omega_hash(p2)

    def test_omega_hash_changes_with_inputs(self, tmp_path):
        p1 = write_optimize_config(
            tmp_path / "a.yaml", input_dir="/data", detector="NRS1",
            baseline_ints=[100, -100], name_tag="t", crds_cache_path="/crds")
        p2 = write_optimize_config(
            tmp_path / "b.yaml", input_dir="/data", detector="NRS2",
            baseline_ints=[100, -100], name_tag="t", crds_cache_path="/crds")
        assert omega_hash(p1) != omega_hash(p2)

    def test_sweep_grids_are_integers(self):
        for param, grid in G395H_SWEEP.items():
            assert all(isinstance(v, int) for v in grid), param

    def test_parse_cost_table(self, tmp_path):
        header = "nirspec_mask_width\textract_width\tduration_s\tcost"
        rows = ["16\t16\t1200\t0.00051", "14\t16\t1150\t0.00049",
                "14\t18\t1100\t0.00050"]
        path = tmp_path / "Cost_t.txt"
        path.write_text(header + "\n" + "\n".join(rows) + "\n")
        table = parse_cost_table(path)
        assert table["best_cost"] == pytest.approx(0.00049)
        assert table["best_params"] == {"nirspec_mask_width": 14,
                                        "extract_width": 16}
        sens = summarize_sweep(table)
        assert "nirspec_mask_width" in sens

    def test_best_params_to_overrides_nesting(self):
        ov = best_params_to_overrides({
            "nirspec_mask_width": 14, "time_window": 5,
            "box_size": 5, "window_size": 7, "extract_width": 18,
        })
        assert ov["nirspec_mask_width"] == 14
        assert ov["stage1_kwargs"] == {"JumpStep": {"time_window": 5}}
        assert ov["stage2_kwargs"] == {"BadPixStep": {"box_size": 5,
                                                      "window_size": 7}}
        assert ov["extract_width"] == 18


# -------------------- survey --------------------


class TestManifest:
    def test_load_requires_keys(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"planet_name": "X b"}))
        with pytest.raises(ValueError, match="visits"):
            load_manifest(p)


class TestStaging:
    def test_dedup_keeps_largest(self, tmp_path):
        raw = tmp_path / "raw"
        (raw / "named").mkdir(parents=True)
        (raw / "obsid").mkdir()
        name = "jw01111001001_04102_00001-seg001_nrs1_uncal.fits"
        (raw / "named" / name).write_bytes(b"x" * 100)   # truncated
        (raw / "obsid" / name).write_bytes(b"x" * 1000)  # complete
        staged = stage_visit_uncals(
            {"raw_root": str(raw), "visit_prefix": "jw01111001001"},
            tmp_path / "staged")
        files = os.listdir(staged)
        assert files == [name]
        target = os.readlink(os.path.join(staged, name))
        assert target.endswith("obsid/" + name)
        assert os.path.getsize(os.path.join(staged, name)) == 1000

    def test_ta_excluded_by_sci_tag(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        sci = "jw01111001001_04102_00001-seg001_nrs1_uncal.fits"
        ta = "jw01111001001_02101_00001_nrs1_uncal.fits"
        (raw / sci).write_bytes(b"x")
        (raw / ta).write_bytes(b"x")
        staged = stage_visit_uncals(
            {"raw_root": str(raw), "visit_prefix": "jw01111001001"},
            tmp_path / "staged")
        assert os.listdir(staged) == [sci]

    def test_plain_string_passthrough(self, tmp_path):
        assert stage_visit_uncals(str(tmp_path), tmp_path / "x") == str(tmp_path)


class TestForceRefitGuard:
    def _manifest(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir(exist_ok=True)
        return {
            "planet_name": "Test b",
            "planet_letter": "b",
            "visits": {"o001": str(raw)},
            "stellar": {"st_teff": 4000, "st_logg": 4.6, "st_met": 0.0},
        }

    def test_stale_posteriors_refused(self, tmp_path):
        out = tmp_path / "results"
        fits_dir = out / "Test_b" / "fits" / "o001" / "nrs1"
        fits_dir.mkdir(parents=True)
        (fits_dir / "_dynesty_NS_posteriors.pkl").write_bytes(b"stale")
        with pytest.raises(RuntimeError, match="force_refit"):
            run_patchwork_target(self._manifest(tmp_path), out,
                                 steps=("fit",), log=lambda *_: None)

    def test_force_refit_clears(self, tmp_path):
        out = tmp_path / "results"
        fits_dir = out / "Test_b" / "fits" / "o001" / "nrs1"
        fits_dir.mkdir(parents=True)
        (fits_dir / "_dynesty_NS_posteriors.pkl").write_bytes(b"stale")
        # no Stage 3 products exist, so the fit loop skips every detector;
        # what matters is that the stale posteriors were deleted, not refused
        run_patchwork_target(self._manifest(tmp_path), out,
                             steps=("fit",), force_refit=True,
                             log=lambda *_: None)
        assert existing_fit_products(out / "Test_b" / "fits") == []


class TestFirScript:
    def test_script_contents(self, tmp_path):
        m = tmp_path / "m.json"
        m.write_text(json.dumps({
            "planet_name": "Test b",
            "visits": {"o001": "/raw"},
        }))
        path = write_fir_slurm_script(m, "/scratch/patchwork")
        text = path.read_text()
        assert "ASTER_EXOTEDRF_REPO" in text
        assert "CRDS_CONTEXT" in text
        assert "verify_exotedrf_environment" in text
        assert "--manifest" in text
        assert "FORCE_REFIT" in text
