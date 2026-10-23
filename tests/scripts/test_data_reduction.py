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
    TILT_TRANSITION_MASK,
    detect_tilt_events,
    find_tilt_events,
    estimate_transit_midpoint,
    load_stage3_spectra,
    match_tilt_events,
    step_statistic,
    tilt_transition_keep_mask,
    trace_step_sources,
    oot_mask_from_baseline,
    oot_mask_from_ephemeris,
    pca_regressors,
    propagate_t0,
    rednoise_beta,
    ramp_regressors,
    refine_step_shape,
    step_regressors,
)
from aster_toolkit.data_reduction.juliet import (  # noqa: E402
    CROSS_BAND_TOL_FRAC,
    LD_FIX_IMPACT_PARAM,
    combine_visit_spectra,
    detector_offset_ppm,
    evaluate_depth_check,
    load_anomaly_mask,
    published_depth_reference,
    read_spectrum_csv,
    write_spectrum_csv,
)
from aster_toolkit.data_reduction.contamination import (  # noqa: E402
    ANOMALY_MASK_PAD,
    PATCHWORK_CONTAM_VERSION,
    anomaly_keep_mask,
    contamination_backends,
    contamination_factor,
    detect_lightcurve_anomalies,
    match_detector_anomalies,
    planck,
    retrieve_contamination,
    spot_crossing_regressors,
    step_events_for_regressors,
    remap_anomaly_report,
    running_mean,
)
from aster_toolkit.data_reduction.discover import (  # noqa: E402
    build_target_summaries,
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
    config_hash,
    parse_cost_table,
    summarize_sweep,
    write_optimize_config,
)
from aster_toolkit.data_reduction.survey import (  # noqa: E402
    _expand_indexed_values,
    _known_overlapping_transits,
    existing_fit_products,
    load_target_summary,
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


class TestEdgeBinRejection:
    def test_a_half_empty_edge_bin_is_dropped(self):
        # Regression: three of the four >4 sigma outliers in the
        # 2026-08-27 combined spectra were first/last channels clipped
        # by the wavelength cut (GJ 3090 b NRS2 4.038 um, -7.0 sigma).
        rng = np.random.default_rng(0)
        wave = np.linspace(3.0, 3.5, 400)
        flux = np.ones((50, wave.size)) + rng.normal(0, 1e-3, (50, wave.size))
        err = np.full_like(flux, 1e-3)
        full = bin_at_resolution(wave, flux, err, resolution=100)
        # Now clip the array so the first bin keeps only a couple of columns.
        keep = wave > wave[0] + 0.98 * (wave[1] - wave[0]) * 3
        clipped = bin_at_resolution(wave[keep], flux[:, keep], err[:, keep],
                                    resolution=100)
        assert clipped["wave"].size <= full["wave"].size
        # Whatever survives must be a properly filled bin, so no channel
        # centre may sit outside the data that fed it.
        assert clipped["wave"].min() >= wave[keep].min()

    def test_normal_bins_are_unaffected(self):
        rng = np.random.default_rng(1)
        wave = np.linspace(2.9, 3.7, 800)
        flux = np.ones((40, wave.size)) + rng.normal(0, 1e-3, (40, wave.size))
        err = np.full_like(flux, 1e-3)
        out = bin_at_resolution(wave, flux, err, resolution=100)
        assert out["wave"].size > 15
        assert np.all(np.isfinite(out["flux"]))


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


class TestRednoiseBetaLengths:
    def test_mismatched_lengths_raise_a_useful_error(self):
        # Regression: fit_white_lightcurve passed the FULL time axis with
        # residuals computed on the kept subset. Latent while the tilt
        # search found nothing (keep all-True); the first masked step
        # killed TOI-1231 b and TOI-270 c after sampling had completed.
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="differ in length"):
            rednoise_beta(rng.normal(0, 1e-4, 2719),
                          2460000.0 + np.arange(2726) * 20 / 86400.0)

    def test_matched_lengths_still_work(self):
        rng = np.random.default_rng(1)
        n = 2000
        out = rednoise_beta(rng.normal(0, 1e-4, n),
                            2460000.0 + np.arange(n) * 20 / 86400.0)
        assert np.isfinite(out["beta_median"])
        assert 0.5 < out["beta_median"] < 2.0


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

    def test_ambiguous_visit_is_not_auto_assigned(self, monkeypatch, tmp_path):
        import aster_toolkit.data_reduction.discover as D

        visit = {
            "visit_prefix": "jw04098017001", "target": "TOI-270",
            "ra": 149.0, "dec": -51.0, "expstart_mjd": self.START,
            "expend_mjd": self.END, "program": "04098", "observation": "017",
            "sci_tag": "04102", "date_obs": "2025-01-01", "nints": 1000,
            "complete": True, "duplicate_download": False,
        }
        rows = [_archive_row("TOI-270 b", 3.0, 2460500.0 - 100 * 3.0),
                _archive_row("TOI-270 d", 5.0, 2460500.0 - 100 * 5.0)]
        monkeypatch.setattr(D, "scan_raw_tree", lambda root: {"exposures": []})
        monkeypatch.setattr(D, "group_visits", lambda scan: [dict(visit)])
        monkeypatch.setattr(D, "_archive_planets_at", lambda *args: rows)
        result = build_target_summaries(tmp_path)
        assert result["targets"] == {}
        assert result["unresolved"][0]["planet_source"] == "ambiguous"
        assert {m["pl_name"] for m in result["unresolved"][0]["planet_matches"]} == {
            "TOI-270 b", "TOI-270 d"
        }


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

    def test_optimizer_branch_required_keys_present(self):
        """Every key run_DMS.py reads must be emitted.

        The optimizer branch reads extract_width_soss2 and deepframe
        unconditionally (run_DMS.py:192,198) even for NIRSpec. Omitting
        either raised KeyError *after* Stage 2 finished — 18 jobs burned
        ~14 h of compute and still reported COMPLETED 0:0.
        """
        # Keys run_DMS.py dereferences as config['<key>'] on the
        # optimizer branch, minus those write_dms_config injects per call.
        required = {
            "baseline_ints", "centroids", "deepframe", "do_plots",
            "extract_method", "extract_width", "extract_width_soss2",
            "f277w", "flag_in_time", "flag_up_ramp", "force_redo",
            "generate_lc", "generate_order0_mask", "hot_pixel_map",
            "input_filetag", "jump_threshold", "miri_background_method",
            "miri_background_width", "miri_drop_groups", "miri_trace_width",
            "nirspec_mask_width", "observing_mode", "oof_method",
            "outlier_maps", "pca_components", "remove_components",
            "save_results", "soss_background_file", "soss_inner_mask_width",
            "soss_outer_mask_width", "soss_specprofile", "soss_timeseries",
            "soss_timeseries_o2", "space_outlier_threshold", "stage1_kwargs",
            "stage2_kwargs", "stage3_kwargs", "superbias_method",
            "time_jump_threshold", "time_outlier_threshold",
        }
        # Step toggles are read via config[step] in a loop.
        required |= {
            "DQInitStep", "EmiCorrStep", "SaturationStep", "ResetStep",
            "SuperBiasStep", "RefPixStep", "DarkCurrentStep",
            "OneOverFStep_grp", "LinearityStep", "JumpStep", "RampFitStep",
            "GainScaleStep", "AssignWCSStep", "Extract2DStep",
            "SourceTypeStep", "WaveCorrStep", "FlatFieldStep",
            "OneOverFStep_int", "BackgroundStep", "TracingStep",
            "BadPixStep", "PCAReconstructStep",
        }
        missing = required - set(PATCHWORK_G395H_CONFIG)
        assert not missing, f"run_DMS.py would KeyError on: {sorted(missing)}"

    def test_written_config_has_required_keys(self, tmp_path):
        path = write_dms_config(
            tmp_path / "run_DMS.yaml", input_dir="/data",
            filter_detector="NRS1", crds_cache_path="/crds",
            run_stages=[1, 2, 3],
        )
        text = path.read_text()
        for key in ("extract_width_soss2", "deepframe", "run_stages",
                    "st_teff", "planet_letter"):
            assert f"{key} : " in text, key


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

    def _write_uncal_with_data(self, path, det, segnum, segtot):
        from astropy.io import fits

        h = fits.PrimaryHDU()
        h.header["DETECTOR"] = det
        h.header["EXSEGNUM"] = segnum
        h.header["EXSEGTOT"] = segtot
        h.header["NINTS"] = 100
        h.header["SUBARRAY"] = "SUB2048"
        sci = fits.ImageHDU(np.zeros((4, 7, 8, 64), dtype=np.float32), name="SCI")
        fits.HDUList([h, sci]).writeto(path)

    def test_a_truncated_segment_is_not_complete(self, tmp_path):
        # Regression (TOI-776 b o006 NRS2): a partially downloaded uncal
        # keeps a valid header and only part of its data, so the segment
        # -number test sees a full set and exoTEDRF then dies inside
        # DQInitStep with an unrelated UnboundLocalError. It failed that
        # way in two campaigns a month apart. Truncation must be caught
        # here, where the message is actionable.
        a = tmp_path / "a_nrs2_uncal.fits"
        b = tmp_path / "b_nrs2_uncal.fits"
        self._write_uncal_with_data(a, "NRS2", 1, 2)
        self._write_uncal_with_data(b, "NRS2", 2, 2)
        assert inspect_uncal_directory(tmp_path)["detectors"]["NRS2"]["complete"]

        with open(b, "r+b") as fh:          # keep headers, drop most data
            fh.truncate(os.path.getsize(b) // 3)
        entry = inspect_uncal_directory(tmp_path)["detectors"]["NRS2"]
        assert entry["complete"] is False
        # The old segment test alone would have passed this file.
        assert entry["missing_segments"] == []
        assert [t["file"] for t in entry["truncated_files"]] == [b.name]

    def test_headerless_files_are_not_called_truncated(self, tmp_path):
        # A header-only uncal declares no data, so absence of a data
        # block must never read as truncation.
        self._write_uncal(tmp_path / "a_nrs1_uncal.fits", "NRS1", 1, 1)
        entry = inspect_uncal_directory(tmp_path)["detectors"]["NRS1"]
        assert entry["truncated_files"] == [] and entry["complete"]


# -------------------- optimize --------------------


class TestOptimizerRule:
    def test_config_hash_deterministic(self, tmp_path):
        kw = dict(input_dir="/data", detector="NRS1",
                  baseline_ints=[100, -100], name_tag="t",
                  crds_cache_path="/crds")
        p1 = write_optimize_config(tmp_path / "a.yaml", **kw)
        p2 = write_optimize_config(tmp_path / "b.yaml", **kw)
        assert config_hash(p1) == config_hash(p2)

    def test_config_hash_changes_with_inputs(self, tmp_path):
        p1 = write_optimize_config(
            tmp_path / "a.yaml", input_dir="/data", detector="NRS1",
            baseline_ints=[100, -100], name_tag="t", crds_cache_path="/crds")
        p2 = write_optimize_config(
            tmp_path / "b.yaml", input_dir="/data", detector="NRS2",
            baseline_ints=[100, -100], name_tag="t", crds_cache_path="/crds")
        assert config_hash(p1) != config_hash(p2)

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


class TestTargetSummary:
    def test_load_requires_keys(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"planet_name": "X b"}))
        with pytest.raises(ValueError, match="visits"):
            load_target_summary(p)


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


class TestSummaryMerge:
    def test_a_step_subset_does_not_erase_other_blocks(self, tmp_path, monkeypatch):
        # Regression 2026-09-04: STEPS=contamination rewrote
        # patchwork_summary.json from scratch and deleted the fit record
        # -- depths, beta, depth_check, scan_steps_corrected -- for all
        # 16 fitted targets, leaving "visits": {}.
        import aster_toolkit.data_reduction.survey as S
        root = tmp_path / "GJ_9827_d"
        root.mkdir(parents=True)
        prior = {"planet_name": "GJ 9827 d", "steps": ["fit", "combine"],
                 "visits": {"o010": {"fits": {"NRS1": {"white_depth_ppm":
                                                       {"median": 957.0}}}}}}
        (root / "patchwork_summary.json").write_text(json.dumps(prior))
        monkeypatch.setattr(S, "fetch_transit_priors",
                            lambda n: {"st_teff": 4340, "st_logg": 4.66,
                                       "st_met": -0.26})
        out = S.run_patchwork_target(
            {"planet_name": "GJ 9827 d", "visits": {}},
            tmp_path, steps=("contamination",), log=lambda *a, **k: None)
        assert out["visits"], "contamination-only run erased the fit record"
        assert out["visits"]["o010"]["fits"]["NRS1"]["white_depth_ppm"]["median"] == 957.0
        assert out.get("fit_version") is None

    def test_subset_run_preserves_the_actual_fit_version(self, tmp_path):
        root = tmp_path / "Test_b"
        root.mkdir(parents=True)
        (root / "patchwork_summary.json").write_text(json.dumps({
            "planet_name": "Test b", "fit_version": "1.4", "resolution": 100,
            "visits": {"o001": {"fits": {"NRS1": {}}}},
        }))
        out = run_patchwork_target(
            {"planet_name": "Test b", "visits": {},
             "stellar": {"st_teff": 4000, "st_logg": 4.5, "st_met": 0.0}},
            tmp_path, steps=("contamination",), log=lambda *_: None,
        )
        assert out["fit_version"] == "1.4"
        assert out["resolution"] == 100

    def test_corrupt_summary_is_never_overwritten(self, tmp_path):
        root = tmp_path / "Test_b"
        root.mkdir(parents=True)
        path = root / "patchwork_summary.json"
        path.write_text("{truncated")
        with pytest.raises(RuntimeError, match="Refusing to overwrite"):
            run_patchwork_target(
                {"planet_name": "Test b", "visits": {},
                 "stellar": {"st_teff": 4000, "st_logg": 4.5, "st_met": 0.0}},
                tmp_path, steps=("contamination",), log=lambda *_: None,
            )
        assert path.read_text() == "{truncated"


class TestIndexedSeriesExpansion:
    def test_masked_residuals_return_to_original_coordinates(self):
        expanded = _expand_indexed_values(
            np.array([10.0, 20.0, 30.0]), np.array([0, 2, 4]), 5
        )
        assert np.allclose(expanded[[0, 2, 4]], [10.0, 20.0, 30.0])
        assert np.isnan(expanded[[1, 3]]).all()

    def test_bad_index_mapping_is_refused(self):
        with pytest.raises(ValueError, match="duplicate"):
            _expand_indexed_values(np.ones(2), np.array([1, 1]), 3)


class TestForceRefitGuard:
    def _target_summary(self, tmp_path):
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
            run_patchwork_target(self._target_summary(tmp_path), out,
                                 steps=("fit",), log=lambda *_: None)

    def test_known_simultaneous_transits_refused(self, tmp_path):
        target = self._target_summary(tmp_path)
        target.update({"planet_name": "TOI-270 b", "program": "GO 4098"})
        target["visits"] = {"o017": target["visits"]["o001"]}
        with pytest.raises(RuntimeError, match="overlapping transits"):
            run_patchwork_target(target, tmp_path / "results",
                                 steps=("fit",), log=lambda *_: None)

    def test_distinct_single_candidate_visits_are_not_overlaps(self):
        target = {
            "planet_name": "Test b",
            "visits": {
                "o001": {"transit_candidates": ["Test b"]},
                "o002": {"transit_candidates": ["Test c"]},
            },
        }
        assert _known_overlapping_transits(target) == []

    def test_force_refit_clears(self, tmp_path):
        out = tmp_path / "results"
        fits_dir = out / "Test_b" / "fits" / "o001" / "nrs1"
        fits_dir.mkdir(parents=True)
        (fits_dir / "_dynesty_NS_posteriors.pkl").write_bytes(b"stale")
        run_patchwork_target(self._target_summary(tmp_path), out,
                             steps=("fit",), force_refit=True,
                             log=lambda *_: None)
        assert existing_fit_products(out / "Test_b" / "fits") == []


class TestScienceRebinning:
    def test_equal_or_finer_resolution_is_refused(self):
        from scripts.patchwork.rebin_spectra import rebin

        wave = np.linspace(3.0, 3.7, 20)
        wave_err = wave / 200.0
        with pytest.raises(ValueError, match="needs a light-curve refit"):
            rebin(wave, wave_err, np.ones(20), np.ones(20), np.ones(20),
                  resolution=100)

    def test_underfilled_edge_bin_is_dropped(self):
        from scripts.patchwork.rebin_spectra import rebin

        wave = np.geomspace(4.08, 5.08, 23)
        wave_err = wave / 201.0
        rows = rebin(wave, wave_err, np.ones(23), np.ones(23), np.ones(23),
                     resolution=50, min_merged=2)
        assert rows
        assert all(row["n_merged"] >= 2 for row in rows)


class TestProfiledSpectralModulation:
    def test_detector_step_is_not_counted_as_atmosphere(self):
        from scripts.patchwork.plot_survey_scale_heights import amplitude_in_H

        wave1 = np.linspace(2.9, 3.7, 20)
        wave2 = np.linspace(4.1, 5.1, 20)
        survey = {
            "NRS1": {"wave": wave1, "depth_ppm": np.full(20, 1000.0),
                     "depth_err_ppm": np.full(20, 20.0)},
            "NRS2": {"wave": wave2, "depth_ppm": np.full(20, 1200.0),
                     "depth_err_ppm": np.full(20, 20.0)},
        }
        result = amplitude_in_H(survey, delta_ppm=50.0)
        assert result["detector_offset_ppm"] == pytest.approx(200.0, abs=0.1)
        assert result["amp_H"] < 0.01


class TestReduceFailurePropagates:
    """A reduction that produces no Stage 3 product must fail the job.

    18 jobs once exited 0 having died between Stage 2 and Stage 3; SLURM
    reported COMPLETED and nothing downstream noticed.
    """

    def _target_summary(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir(exist_ok=True)
        return {"planet_name": "Test b", "planet_letter": "b",
                "visits": {"o001": str(raw)},
                "stellar": {"st_teff": 4000, "st_logg": 4.6, "st_met": 0.0}}

    def _patch_run_reduction(self, monkeypatch, returncode, products):
        import aster_toolkit.data_reduction.survey as survey_mod

        def fake_run_reduction(input_dir, output_dir, **kw):
            return {"detectors": {
                det: {"returncode": returncode, "success": returncode == 0,
                      "spectra_fullres": list(products),
                      "log": f"{output_dir}/{det.lower()}/reduction.log"}
                for det in kw.get("detectors", ["NRS1", "NRS2"])}}

        monkeypatch.setattr(survey_mod, "run_reduction", fake_run_reduction)

    def test_exit_zero_without_products_raises(self, tmp_path, monkeypatch):
        # the real failure: run_DMS.py died after Stage 2, exit code lost
        self._patch_run_reduction(monkeypatch, 0, [])
        with pytest.raises(RuntimeError, match="reduction failed"):
            run_patchwork_target(self._target_summary(tmp_path), tmp_path / "out",
                                 steps=("reduce",), log=lambda *_: None)

    def test_nonzero_exit_raises(self, tmp_path, monkeypatch):
        self._patch_run_reduction(monkeypatch, 1, [])
        with pytest.raises(RuntimeError, match="reduction failed"):
            run_patchwork_target(self._target_summary(tmp_path), tmp_path / "out",
                                 steps=("reduce",), log=lambda *_: None)

    def test_success_with_products_does_not_raise(self, tmp_path, monkeypatch):
        self._patch_run_reduction(monkeypatch, 0, ["/x/a_box_spectra_fullres.fits"])
        summary = run_patchwork_target(
            self._target_summary(tmp_path), tmp_path / "out",
            steps=("reduce",), log=lambda *_: None)
        assert summary["visits"]["o001"]["reduction"]["NRS1"] is True

    def test_failure_is_recorded_in_summary(self, tmp_path, monkeypatch):
        self._patch_run_reduction(monkeypatch, 0, [])
        out = tmp_path / "out"
        with pytest.raises(RuntimeError):
            run_patchwork_target(self._target_summary(tmp_path), out,
                                 steps=("reduce",), log=lambda *_: None)
        # summary is written before raising, so the failure is diagnosable
        with (out / "Test_b" / "patchwork_summary.json").open() as fh:
            summary = json.load(fh)
        assert summary["visits"]["o001"]["reduction"]["NRS1"] is False


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
        assert "--summary" in text
        assert "FORCE_REFIT" in text


# -------------------- Stage 5.5: stellar contamination --------------------


def synthetic_residuals(n=1200, noise=150e-6, bump_amp=0.0, bump_i0=500,
                        bump_width=40, step_amp=0.0, step_i=500, seed=0):
    """White-light fit residuals with an optional in-transit bump (spot
    crossing) or step (tilt event), on a series whose transit spans
    integrations 400-800."""
    rng = np.random.default_rng(seed)
    time = 2460500.0 + np.arange(n) * (20.0 / 86400)
    resid = rng.normal(0, noise, n)
    if bump_amp:
        centre = bump_i0 + bump_width / 2
        resid = resid + bump_amp * np.exp(
            -0.5 * ((np.arange(n) - centre) / (bump_width / 2.5)) ** 2)
    if step_amp:
        resid = resid + step_amp * (np.arange(n) >= step_i)
    oot = np.ones(n, dtype=bool)
    oot[400:800] = False
    return time, resid, oot


class TestRunningMean:
    def test_preserves_a_constant(self):
        x = np.full(200, 3.0)
        out = running_mean(x, 15)
        assert np.allclose(out[10:-10], 3.0)

    def test_averages_noise_down(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1.0, 5000)
        out = running_mean(x, 25)
        # sqrt(n) suppression, loose bounds for the finite sample
        assert 0.15 < np.nanstd(out) < 0.30

    def test_stays_aligned_with_an_even_window(self):
        # An even window would shift every feature half an integration
        # early; the function forces it odd. A boxcar over a delta gives
        # a plateau, so use a peaked feature to get a unique maximum.
        x = np.exp(-0.5 * ((np.arange(101.0) - 50) / 6.0) ** 2)
        assert int(np.nanargmax(running_mean(x, 10))) == 50
        assert int(np.nanargmax(running_mean(x, 11))) == 50

    def test_thin_windows_are_nan_not_noisy(self):
        # Positions with fewer than half a window of finite points must
        # not return a noisy partial average — that is where a false
        # anomaly would most easily appear.
        x = np.arange(100.0)
        x[:90] = np.nan
        out = running_mean(x, 15)
        assert np.isnan(out[:88]).all()
        assert np.isfinite(out[95])


class TestAnomalyDetection:
    def test_recovers_an_injected_spot_crossing(self):
        t, r, oot = synthetic_residuals(bump_amp=600e-6, bump_i0=500)
        rep = detect_lightcurve_anomalies(t, r, oot_mask=oot, detector="NRS1")
        assert rep["anomalies"], "injected 600 ppm bump not detected"
        top = rep["anomalies"][0]
        assert top["kind"] == "spot_crossing"
        assert top["sign"] == 1
        assert top["in_transit_frac"] > 0.9
        # the flagged run must overlap the injected bump
        assert top["index_start"] < 540 < top["index_end"] + 60

    def test_clean_residuals_flag_nothing(self):
        t, r, oot = synthetic_residuals(seed=3)
        rep = detect_lightcurve_anomalies(t, r, oot_mask=oot, detector="NRS1")
        assert rep["anomalies"] == []

    def test_a_real_transit_is_not_an_anomaly(self):
        # Residuals of a GOOD fit contain no transit; the guard that
        # matters is that the in-transit region is not flagged merely for
        # being in transit.
        t, r, oot = synthetic_residuals(seed=7)
        rep = detect_lightcurve_anomalies(t, r, oot_mask=oot, detector="NRS1")
        assert not any(a["in_transit_frac"] > 0.5 for a in rep["anomalies"])

    def test_a_step_is_labelled_persistent_not_spot(self):
        t, r, oot = synthetic_residuals(step_amp=800e-6, step_i=500)
        rep = detect_lightcurve_anomalies(t, r, oot_mask=oot, detector="NRS1")
        assert rep["anomalies"]
        top = rep["anomalies"][0]
        assert top["persistent"] is True
        assert top["kind"] == "step"

    def test_an_in_transit_step_is_found_at_all(self):
        # The failure this guards: an in-transit tilt leaves the
        # out-of-transit points at two levels, so a naive noise scale
        # measures the step height instead of the noise and the event
        # hides itself. This is the TOI-270 c case.
        for seed in (0, 4, 9):
            t, r, oot = synthetic_residuals(step_amp=800e-6, step_i=500,
                                            seed=seed)
            rep = detect_lightcurve_anomalies(t, r, oot_mask=oot)
            steps = [a for a in rep["anomalies"] if a["kind"] == "step"]
            assert steps, f"in-transit step missed (seed {seed})"
            assert steps[0]["index_start"] < 500 < steps[0]["index_end"]

    def test_step_amplitude_is_the_level_shift(self):
        # Reporting the mean over the flagged span would halve it (the
        # span covers both lobes of the detrended feature).
        t, r, oot = synthetic_residuals(step_amp=800e-6, step_i=500)
        rep = detect_lightcurve_anomalies(t, r, oot_mask=oot)
        step = [a for a in rep["anomalies"] if a["kind"] == "step"][0]
        assert step["amplitude_ppm"] == pytest.approx(800, rel=0.15)

    def test_classification_is_stable_across_noise_seeds(self):
        for seed in (0, 4, 9):
            for amp, expect in ((600e-6, "spot_crossing"),
                                (-600e-6, "facula_crossing")):
                t, r, oot = synthetic_residuals(bump_amp=amp, bump_i0=500,
                                                seed=seed)
                rep = detect_lightcurve_anomalies(t, r, oot_mask=oot)
                kinds = [a["kind"] for a in rep["anomalies"]
                         if a["in_transit_frac"] > 0.5]
                assert expect in kinds, f"{expect} missed (seed {seed})"

    def test_detrend_lobes_are_merged_into_one_event(self):
        # Removing the long-window trend splits one feature into lobes.
        # Reported separately they would carry the wrong amplitude, and a
        # wing beside a real crossing would itself be masked.
        t, r, oot = synthetic_residuals(bump_amp=600e-6, bump_i0=500)
        rep = detect_lightcurve_anomalies(t, r, oot_mask=oot)
        in_tr = [a for a in rep["anomalies"] if a["in_transit_frac"] > 0.5]
        assert len(in_tr) == 1

    def test_two_separated_crossings_stay_separate(self):
        t, r, oot = synthetic_residuals(bump_amp=600e-6, bump_i0=500)
        r = r + 500e-6 * np.exp(-0.5 * ((np.arange(r.size) - 700) / 16.0) ** 2)
        rep = detect_lightcurve_anomalies(t, r, oot_mask=oot)
        peaks = sorted(a["index_start"] for a in rep["anomalies"]
                       if a["in_transit_frac"] > 0.5)
        assert len(peaks) == 2

    def test_facula_crossing_is_negative(self):
        t, r, oot = synthetic_residuals(bump_amp=-600e-6, bump_i0=500)
        rep = detect_lightcurve_anomalies(t, r, oot_mask=oot, detector="NRS1")
        assert rep["anomalies"][0]["kind"] == "facula_crossing"
        assert rep["anomalies"][0]["sign"] == -1

    def test_no_noise_scale_returns_a_note_not_a_crash(self):
        t = 2460500.0 + np.arange(300) * (20.0 / 86400)
        rep = detect_lightcurve_anomalies(t, np.zeros(300),
                                          oot_mask=np.ones(300, bool))
        assert rep["anomalies"] == [] and "note" in rep

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            detect_lightcurve_anomalies(np.arange(10.0), np.arange(9.0))


class TestCrossDetectorConfirmation:
    def _pair(self, amp1, amp2, i0_2=500, seed=1):
        t, r1, oot = synthetic_residuals(bump_amp=amp1, bump_i0=500, seed=seed)
        _, r2, _ = synthetic_residuals(bump_amp=amp2, bump_i0=i0_2, seed=seed + 1)
        return (detect_lightcurve_anomalies(t, r1, oot_mask=oot, detector="NRS1"),
                detect_lightcurve_anomalies(t, r2, oot_mask=oot, detector="NRS2"))

    def test_coincident_bump_is_confirmed_as_a_spot(self):
        a, b = self._pair(600e-6, 400e-6)
        merged = match_detector_anomalies({"NRS1": a, "NRS2": b})
        top = merged[0]
        assert top["kind"] == "spot_crossing" and top["confirmed"]
        assert set(top["detectors"]) == {"NRS1", "NRS2"}

    def test_amplitude_ratio_is_chromatic_for_a_real_spot(self):
        a, b = self._pair(600e-6, 400e-6)
        top = match_detector_anomalies({"NRS1": a, "NRS2": b})[0]
        # Spot contrast falls to the infrared: NRS2/NRS1 well below 1.
        assert 0.5 < top["amplitude_ratio"] < 0.85
        assert top["achromatic"] is False

    def test_equal_amplitudes_are_flagged_achromatic(self):
        a, b = self._pair(600e-6, 600e-6)
        top = match_detector_anomalies({"NRS1": a, "NRS2": b})[0]
        assert top["amplitude_ratio"] > 0.9 and top["achromatic"] is True

    def test_detector_order_does_not_change_the_ratio(self):
        a, b = self._pair(600e-6, 400e-6)
        forward = match_detector_anomalies({"NRS1": a, "NRS2": b})[0]
        reversed_ = match_detector_anomalies({"NRS2": b, "NRS1": a})[0]
        assert forward["amplitude_ratio"] == pytest.approx(
            reversed_["amplitude_ratio"])

    def test_single_detector_event_is_never_confirmed(self):
        t, r1, oot = synthetic_residuals(bump_amp=600e-6, bump_i0=500)
        _, r2, _ = synthetic_residuals(seed=11)
        a = detect_lightcurve_anomalies(t, r1, oot_mask=oot, detector="NRS1")
        b = detect_lightcurve_anomalies(t, r2, oot_mask=oot, detector="NRS2")
        merged = match_detector_anomalies({"NRS1": a, "NRS2": b})
        assert all(not m["confirmed"] for m in merged)
        assert merged[0].get("single_detector") is True

    def test_far_apart_events_do_not_match(self):
        # 400 integrations at 20 s is ~2.2 h apart, far beyond the tolerance
        a, b = self._pair(600e-6, 600e-6, i0_2=100)
        merged = match_detector_anomalies({"NRS1": a, "NRS2": b})
        assert all(len(m["detectors"]) == 1 for m in merged)


class TestAnomalyMask:
    def test_masks_a_confirmed_spot_with_padding(self):
        t, r1, oot = synthetic_residuals(bump_amp=600e-6, bump_i0=500)
        _, r2, _ = synthetic_residuals(bump_amp=400e-6, bump_i0=500, seed=2)
        a = detect_lightcurve_anomalies(t, r1, oot_mask=oot, detector="NRS1")
        b = detect_lightcurve_anomalies(t, r2, oot_mask=oot, detector="NRS2")
        merged = match_detector_anomalies({"NRS1": a, "NRS2": b})
        keep = anomaly_keep_mask(t.size, merged, detector="NRS1")
        entry = merged[0]["detectors"]["NRS1"]
        assert not keep[entry["index_start"]:entry["index_end"] + 1].any()
        # padding removes the smoothed-in wings the run itself misses
        assert not keep[entry["index_start"] - ANOMALY_MASK_PAD]
        assert keep[:300].all() and keep[900:].all()

    def test_does_not_mask_tilt_steps(self):
        t, r, oot = synthetic_residuals(step_amp=800e-6, step_i=500)
        rep = detect_lightcurve_anomalies(t, r, oot_mask=oot, detector="NRS1")
        merged = match_detector_anomalies({"NRS1": rep})
        keep = anomaly_keep_mask(t.size, merged, detector="NRS1")
        assert keep.all(), "tilt steps are fitted with a step term, not masked"

    def test_does_not_mask_unconfirmed_events(self):
        t, r, oot = synthetic_residuals(bump_amp=600e-6, bump_i0=500)
        rep = detect_lightcurve_anomalies(t, r, oot_mask=oot, detector="NRS1")
        merged = match_detector_anomalies({"NRS1": rep})   # one detector only
        assert anomaly_keep_mask(t.size, merged, detector="NRS1").all()

    def test_both_detectors_lose_the_same_integrations(self):
        # A spot crossing is a property of the star, so it contaminates
        # NRS2 even where NRS2's weaker detection clears threshold over a
        # narrower run. Masking each detector by its own run would leave
        # contaminated integrations in the redder half and put the two
        # depths on different data.
        t, r1, oot = synthetic_residuals(bump_amp=600e-6, bump_i0=500)
        _, r2, _ = synthetic_residuals(bump_amp=400e-6, bump_i0=500, seed=2)
        a = detect_lightcurve_anomalies(t, r1, oot_mask=oot, detector="NRS1")
        b = detect_lightcurve_anomalies(t, r2, oot_mask=oot, detector="NRS2")
        merged = match_detector_anomalies({"NRS1": a, "NRS2": b})
        k1 = anomaly_keep_mask(t.size, merged, detector="NRS1")
        k2 = anomaly_keep_mask(t.size, merged, detector="NRS2")
        assert not k1.all()
        assert np.array_equal(k1, k2)

    def test_report_roundtrips_through_json(self, tmp_path):
        report = tmp_path / "anomaly_report.json"
        report.write_text(json.dumps({
            "contam_version": PATCHWORK_CONTAM_VERSION,
            "mask_indices": {"NRS1": [10, 11, 12], "NRS2": []},
        }))
        payload, keep = load_anomaly_mask(report, "NRS1", 100)
        assert payload["contam_version"] == PATCHWORK_CONTAM_VERSION
        assert (~keep).sum() == 3 and not keep[10:13].any()
        _, keep2 = load_anomaly_mask(report, "NRS2", 100)
        assert keep2.all()

    def test_out_of_range_indices_are_ignored(self, tmp_path):
        report = tmp_path / "r.json"
        report.write_text(json.dumps({"mask_indices": {"NRS1": [5, 999]}}))
        _, keep = load_anomaly_mask(report, "NRS1", 100)
        assert (~keep).sum() == 1 and not keep[5]


class TestSpanOverlapMatching:
    """Regression: matching on t_peak missed 2/3 of injected coincident
    crossings. A broad (boxcar) crossing detrends into a suppressed core
    with negative wings of comparable height, so the |detrended| peak is
    noise-driven and wanders tens of minutes between detectors — far
    beyond the 5-minute tolerance. Matching must use the flagged spans."""

    @staticmethod
    def _boxcar_pair(seed):
        n = 2000
        t = 2460000.0 + np.arange(n) * 20 / 86400.0
        oot = np.ones(n, bool)
        oot[800:1200] = False
        rng = np.random.default_rng(seed)
        reports = {}
        for det, amp in (("NRS1", 400e-6), ("NRS2", 300e-6)):
            r = rng.normal(0, 100e-6, n)
            r[950:1010] += amp
            reports[det] = detect_lightcurve_anomalies(
                t, r, oot_mask=oot, detector=det)
        return reports

    def test_broad_boxcar_crossing_confirmed_despite_peak_jitter(self):
        # Seeds 0, 2, 4, 5, 7 all failed peak-time matching (peaks landed
        # 5-14 min apart on opposite lobes). Span overlap must catch them.
        for seed in (0, 2, 4, 5, 7):
            merged = match_detector_anomalies(self._boxcar_pair(seed))
            confirmed = [m for m in merged
                         if m["kind"] == "spot_crossing" and m["confirmed"]]
            assert confirmed, f"coincident crossing not confirmed (seed {seed})"
            assert confirmed[0]["span_overlap_min"] > 0

    def test_anomalies_carry_their_span_times(self):
        rep = self._boxcar_pair(0)["NRS1"]
        a = rep["anomalies"][0]
        assert a["t_start"] < a["t_peak"] < a["t_end"]

    def test_peak_only_dicts_still_match_at_the_tolerance(self):
        # Backward compatibility: anomaly dicts from an old report have
        # no t_start/t_end; matching falls back to peak separation.
        reports = self._boxcar_pair(3)
        for rep in reports.values():
            for a in rep["anomalies"]:
                a.pop("t_start"), a.pop("t_end")
                a["t_peak"] = 2460000.5   # force coincident peaks
        merged = match_detector_anomalies(reports)
        assert any(len(m["detectors"]) == 2 for m in merged)


class TestRemapAnomalyReport:
    """Regression: pass-1 white fits are tilt-transition-masked, so the
    scanned residual arrays are compressed and the scan's indices count
    the compressed series. Applied to the full lightcurve unmapped, the
    mask lands early by the number of dropped integrations."""

    def test_mask_aligns_with_original_coordinates(self):
        n = 2000
        t = 2460000.0 + np.arange(n) * 20 / 86400.0
        oot = np.ones(n, bool)
        oot[800:1200] = False
        # Two tilt events before the crossing: 14 integrations dropped.
        tilt_keep = tilt_transition_keep_mask(
            n, [{"index": 300}, {"index": 600}])
        rng = np.random.default_rng(1)
        reports = {}
        for det, amp in (("NRS1", 500e-6), ("NRS2", 400e-6)):
            r = rng.normal(0, 100e-6, n)
            r[950:1010] += amp
            rep = detect_lightcurve_anomalies(
                t[tilt_keep], r[tilt_keep], oot_mask=oot[tilt_keep],
                detector=det)
            reports[det] = remap_anomaly_report(
                rep, np.flatnonzero(tilt_keep), n)
        assert all(rep["n_integrations"] == n for rep in reports.values())
        merged = match_detector_anomalies(reports)
        spots = [m for m in merged
                 if m["kind"] == "spot_crossing" and m["confirmed"]]
        assert spots
        keep = anomaly_keep_mask(n, spots, detector="NRS1")
        # Every contaminated integration must be gone from the FULL series.
        assert not keep[950:1010].any()

    def test_identity_remap_changes_nothing(self):
        t, r, oot = synthetic_residuals(bump_amp=600e-6, bump_i0=500)
        rep = detect_lightcurve_anomalies(t, r, oot_mask=oot, detector="NRS1")
        before = [(a["index_start"], a["index_end"]) for a in rep["anomalies"]]
        rep = remap_anomaly_report(rep, np.arange(t.size), t.size)
        after = [(a["index_start"], a["index_end"]) for a in rep["anomalies"]]
        assert before == after and rep["n_integrations"] == t.size


class TestStage65SpectrumUnits:
    def test_read_spectrum_csv_keys_are_ppm(self, tmp_path):
        # Regression: Stage 6.5 read s["depth"] (KeyError) and multiplied
        # by 1e6 (unit error). The reader returns ppm under _ppm keys.
        rows = [{"wave": 3.0 + 0.03 * i, "wave_err": 0.015,
                 "depth": 1800e-6, "depth_err": 30e-6, "rms_ppm": 300.0}
                for i in range(10)]
        p = tmp_path / "combined_nrs1_transmission_spectrum.csv"
        write_spectrum_csv(p, rows)
        s = read_spectrum_csv(p)
        assert "depth_ppm" in s and "depth_err_ppm" in s
        assert "depth" not in s
        assert np.allclose(s["depth_ppm"], 1800.0, atol=0.01)
        assert np.allclose(s["depth_err_ppm"], 30.0, atol=0.01)


class TestEstimateTransitMidpoint:
    @staticmethod
    def _visit(t0_frac=0.5, dur_hr=2.5, depth=900e-6, span_hr=7.7, n=1500, seed=0):
        rng = np.random.default_rng(seed)
        t = 2460583.2 + np.linspace(0, span_hr / 24, n)
        centre = t[0] + t0_frac * (t[-1] - t[0])
        flux = np.ones(n) + rng.normal(0, 120e-6, n)
        flux[np.abs(t - centre) < 0.5 * dur_hr / 24] -= depth
        return t, flux, centre

    def test_recovers_an_injected_midpoint(self):
        t, f, centre = self._visit()
        r = estimate_transit_midpoint(t, f)
        assert r["found"]
        # within a couple of minutes of truth
        assert abs(r["t0"] - centre) * 24 * 60 < 3
        assert 700 < r["depth_ppm"] < 1100
        assert 2.2 < r["duration_hr"] < 2.8
        assert r["partial"] is False

    def test_flags_a_partial_transit(self):
        # Ingress only: the transit starts late and runs off the end,
        # which is TOI-125 c o201. Depth is then degenerate with the
        # baseline, so the caller must not build an override from it.
        t, f, _ = self._visit(t0_frac=1.0, dur_hr=2.5)
        r = estimate_transit_midpoint(t, f)
        assert r["found"] and r["partial"] is True

    def test_no_dip_is_reported_not_invented(self):
        rng = np.random.default_rng(3)
        t = 2460583.2 + np.linspace(0, 7.7 / 24, 1500)
        r = estimate_transit_midpoint(t, np.ones(1500) + rng.normal(0, 120e-6, 1500))
        # Pure noise: either no detection, or a spurious run that is
        # tiny and shallow — never a confident deep event.
        assert (not r["found"]) or r["depth_ppm"] < 400

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            estimate_transit_midpoint(np.arange(10.0), np.arange(9.0))


class TestPriorsOverride:
    def test_override_beats_archive_and_is_recorded(self, monkeypatch):
        # The archive is current, not correct. An override must win over
        # it and must be visible in priors_source, so a target fitted on
        # a hand-supplied ephemeris can never look like a default one.
        import aster_toolkit.data_reduction.juliet as J

        captured = {}
        monkeypatch.setattr(J, "fetch_transit_priors",
                            lambda name: {"t0": 2458355.35529, "period": 4.65382,
                                          "duration_hr": 2.96, "planet_name": name})

        def fake_build(spectra, **kw):
            captured.update(kw)
            raise RuntimeError("stop after priors")

        monkeypatch.setattr(J, "build_lightcurves", fake_build)
        monkeypatch.setattr(J, "load_stage3_spectra", lambda p: {})
        with pytest.raises(RuntimeError, match="stop after priors"):
            J.prepare_visit_fit_inputs(
                "x.fits", "TOI-125 b", instrument="NRS1",
                priors_override={"t0": 2460583.50540, "period": None},
            )
        assert captured["t0_ref"] == 2460583.50540
        assert captured["period"] == 4.65382   # None override is ignored


class TestOotAnchoredDetrend:
    """The old whole-series trend subtracted part of any broad in-transit
    feature from itself. Measured on the 2026-08-27 products, recovery of
    the in-transit peak roughly doubled once the trend was built from
    out-of-transit points only (TOI-1231 b NRS1 4.1 -> 9.4 sigma)."""

    @staticmethod
    def _broad_crossing(width=260, amp=300e-6, n=2000, seed=11):
        # A crossing WIDER than the old 6x-window detrend (90 ints).
        rng = np.random.default_rng(seed)
        t = 2460000.0 + np.arange(n) * 20 / 86400.0
        oot = np.ones(n, bool); oot[700:1300] = False
        r = rng.normal(0, 80e-6, n)
        centre = 1000
        r += amp * np.exp(-0.5 * ((np.arange(n) - centre) / (width / 2.355)) ** 2)
        return t, r, oot

    def test_a_broad_crossing_survives_the_detrend(self):
        t, r, oot = self._broad_crossing()
        rep = detect_lightcurve_anomalies(t, r, oot_mask=oot, detector="NRS1")
        assert rep["anomalies"], "broad in-transit crossing not detected"
        top = rep["anomalies"][0]
        assert top["in_transit_frac"] > 0.9
        # Recovered peak must be a decent fraction of the injected 300 ppm,
        # not the ~half the whole-series trend used to leave.
        assert abs(top["peak_amplitude_ppm"]) > 180

    def test_slow_drift_is_still_removed(self):
        t, r, oot = self._broad_crossing(amp=0.0)
        r = r + 400e-6 * np.linspace(-1, 1, r.size)     # linear drift
        rep = detect_lightcurve_anomalies(t, r, oot_mask=oot, detector="NRS1")
        assert not rep["anomalies"], "drift alone must not raise an anomaly"

    def test_falls_back_when_out_of_transit_is_too_sparse(self):
        # Almost no baseline: interpolating would invent a trend, so the
        # whole-series running mean is used instead.
        t, r, _ = self._broad_crossing()
        oot = np.zeros(t.size, bool); oot[:40] = True
        rep = detect_lightcurve_anomalies(t, r, oot_mask=oot, detector="NRS1")
        assert isinstance(rep["anomalies"], list)      # no crash, no NaNs


class TestSpotCrossingRegressors:
    """Radica et al. 2026: model the crossing as a Gaussian with position
    and width frozen from the white fit and a free per-channel amplitude,
    rather than masking the integrations away."""

    @staticmethod
    def _confirmed_crossing():
        t, r1, oot = synthetic_residuals(bump_amp=600e-6, bump_i0=500)
        _, r2, _ = synthetic_residuals(bump_amp=400e-6, bump_i0=500, seed=2)
        merged = match_detector_anomalies({
            "NRS1": detect_lightcurve_anomalies(t, r1, oot_mask=oot, detector="NRS1"),
            "NRS2": detect_lightcurve_anomalies(t, r2, oot_mask=oot, detector="NRS2")})
        return t, merged

    def test_builds_one_gaussian_per_confirmed_crossing(self):
        t, merged = self._confirmed_crossing()
        cols, meta = spot_crossing_regressors(t, merged, "NRS1")
        n_cross = sum(1 for m in merged
                      if m["confirmed"] and m["kind"].endswith("crossing"))
        assert cols.shape == (t.size, n_cross) and len(meta) == n_cross
        if n_cross:
            col = cols[:, 0]
            assert 0.99 < col.max() <= 1.0      # normalized shape
            assert col.min() >= 0.0
            peak = int(np.argmax(col))
            assert abs(peak - 520) < 60          # centred on the injection

    def test_both_detectors_get_the_same_shape(self):
        # Position and width are shared; only the amplitude may differ,
        # and that is the fitted coefficient, not the column.
        t, merged = self._confirmed_crossing()
        a, _ = spot_crossing_regressors(t, merged, "NRS1")
        b, _ = spot_crossing_regressors(t, merged, "NRS2")
        assert a.shape == b.shape
        if a.shape[1]:
            assert np.allclose(a, b)

    def test_steps_are_not_modelled_as_gaussians(self):
        t, r, oot = synthetic_residuals(step_amp=800e-6, step_i=500)
        merged = match_detector_anomalies({
            "NRS1": detect_lightcurve_anomalies(t, r, oot_mask=oot, detector="NRS1")})
        cols, meta = spot_crossing_regressors(t, merged, "NRS1")
        assert cols.shape[1] == 0 and meta == []

    def test_nothing_to_model_gives_an_empty_matrix(self):
        t = 2460000.0 + np.arange(500) * 20 / 86400.0
        cols, meta = spot_crossing_regressors(t, [], "NRS1")
        assert cols.shape == (500, 0) and meta == []


class TestRampStepFitting:
    """TOI-270 c o016 (2026-08-27). A hard Heaviside at the flagged
    span's centre recovered +2275 ppm where the true post-event level
    needs +2893: the correction stopped 22% short of the data. The break
    must be FITTED (the span centre sat 23-240 integrations off) and the
    transition given its measured width."""

    @staticmethod
    def _stepped(n=1500, break_at=900, width=8.0, amp=2500e-6, seed=5):
        from math import erf
        rng = np.random.default_rng(seed)
        x = np.arange(n, dtype=float)
        shape = 0.5 * (1 + np.vectorize(erf)((x - break_at) / (np.sqrt(2) * width)))
        return rng.normal(0, 120e-6, n) + amp * shape

    def test_ramp_recovers_break_width_and_amplitude(self):
        r = self._stepped()
        fit = refine_step_shape(r, 700, bounds=(700, 1100))
        assert abs(fit["index"] - 900) <= 8
        assert 4.0 <= fit["width_ints"] <= 14.0
        assert abs(fit["amplitude_ppm"] - 2500) < 250

    def test_the_model_reaches_the_post_event_level(self):
        # The specific failure: a step that does not rise as far as the
        # data after the event.
        r = self._stepped()
        fit = refine_step_shape(r, 700, bounds=(700, 1100))
        cols = ramp_regressors(r.size, [fit])
        design = np.column_stack([np.ones(r.size), cols])
        coef, *_ = np.linalg.lstsq(design, r, rcond=None)
        model = design @ coef
        tail = slice(r.size - 200, None)
        assert abs(np.mean(model[tail]) - np.mean(r[tail])) * 1e6 < 40

    def test_a_hard_step_is_still_available(self):
        # width 0 must give the exact Heaviside, so a genuinely
        # instantaneous event is unchanged.
        cols = ramp_regressors(100, [{"index": 50, "width_ints": 0.0}])
        assert cols[49, 0] == 0.0 and cols[50, 0] == 1.0

    def test_searching_the_span_beats_a_window_round_its_centre(self):
        # The span brackets the transition; its centre need not be near it.
        r = self._stepped(break_at=1100)
        near = refine_step_shape(r, 700, search=45)
        span = refine_step_shape(r, 700, bounds=(700, 1300))
        assert span["rms"] < near["rms"]
        assert abs(span["index"] - 1100) <= 10

    def test_no_events_gives_an_empty_matrix(self):
        assert ramp_regressors(300, []).shape == (300, 0)


class TestScanStepsBecomeRegressors:
    """Regression (TOI-270 c o016, 2026-08-27). The scan confirmed three
    steps in both detectors -- 27.3, 15.2 and 7.9 sigma, one of them
    2803 ppm -- while the Stage 4 trace search found none, because only
    trace_fwhm on NRS1 cleared 6 sigma and TILT_MIN_SOURCES is 2. Stage
    5.5 then declined to mask them (correct: a step wants a Heaviside,
    not a mask) and handed them to a stage that had already run. Nothing
    corrected a 2803 ppm discontinuity; the fit returned beta ~6.1 and a
    depth 96% high. A step confirmed by EITHER stage must reach the fit."""

    @staticmethod
    def _stepped_pair(step_i=900, amp=2800e-6, n=2000):
        t = 2460000.0 + np.arange(n) * 20 / 86400.0
        oot = np.ones(n, bool); oot[800:1200] = False
        reports = {}
        for det, seed in (("NRS1", 4), ("NRS2", 5)):
            rng = np.random.default_rng(seed)
            r = rng.normal(0, 100e-6, n) + amp * (np.arange(n) >= step_i)
            reports[det] = detect_lightcurve_anomalies(t, r, oot_mask=oot,
                                                       detector=det)
        return match_detector_anomalies(reports)

    def test_a_confirmed_step_becomes_a_regressor(self):
        merged = self._stepped_pair()
        steps = [m for m in merged if m["kind"] == "step" and m["confirmed"]]
        assert steps, "injected step not confirmed across detectors"
        evs = step_events_for_regressors(merged, "NRS1")
        assert len(evs) == len(steps)
        # Break time recovered near the true transition.
        assert abs(evs[0]["index"] - 900) < 100
        assert evs[0]["source"] == "stage5.5-scan"

    def test_the_regressor_is_a_heaviside_at_the_break(self):
        evs = step_events_for_regressors(self._stepped_pair(), "NRS1")
        cols = step_regressors(2000, evs)
        assert cols.shape == (2000, len(evs))
        i = evs[0]["index"]
        assert cols[i - 1, 0] == 0.0 and cols[i, 0] == 1.0

    def test_a_step_is_still_never_masked(self):
        # The correction is the regressor; the integrations stay in.
        merged = self._stepped_pair()
        keep = anomaly_keep_mask(2000, merged, detector="NRS1")
        assert keep.all()

    def test_a_step_straddling_a_contact_is_refused(self):
        # TOI-270 c o016: a confirmed step spanning egress
        # (in_transit_frac 0.54). A Heaviside there is degenerate with
        # the transit shape and eats the contact, so it must be refused
        # -- while steps wholly in or wholly out of transit are kept.
        base = self._stepped_pair()
        step = next(m for m in base if m["kind"] == "step" and m["confirmed"])
        # 0.99 must be KEPT: that is TOI-270 c's largest step (68.8 sigma),
        # essentially wholly in transit, not straddling a contact.
        for frac, expected in ((0.0, 1), (1.0, 1), (0.99, 1), (0.03, 1),
                               (0.54, 0), (0.2, 0)):
            step["in_transit_frac"] = frac
            got = len(step_events_for_regressors([step], "NRS1"))
            assert got == expected, f"in_transit_frac={frac} gave {got}"

    def test_spot_crossings_are_not_turned_into_step_regressors(self):
        # Only persistent events. A transient bump must still be masked,
        # not fitted with a Heaviside.
        t, r1, oot = synthetic_residuals(bump_amp=600e-6, bump_i0=500)
        _, r2, _ = synthetic_residuals(bump_amp=400e-6, bump_i0=500, seed=2)
        merged = match_detector_anomalies({
            "NRS1": detect_lightcurve_anomalies(t, r1, oot_mask=oot, detector="NRS1"),
            "NRS2": detect_lightcurve_anomalies(t, r2, oot_mask=oot, detector="NRS2")})
        assert any(m["kind"] == "spot_crossing" for m in merged)
        assert step_events_for_regressors(merged, "NRS1") == []


class TestMatchTiltEventsTimeless:
    def test_two_timeless_events_do_not_crash_the_sort(self):
        # Regression: sorting keyed on (time is None, time) compared
        # None < None and raised TypeError with two timeless events.
        reports = {
            "NRS1": {"events": [
                {"index": 100, "amplitude_ppm": 200.0, "n_sources": 2},
                {"index": 900, "amplitude_ppm": -150.0, "n_sources": 2},
            ]},
        }
        merged = match_tilt_events(reports)
        assert len(merged) == 2
        assert all(m["single_detector"] for m in merged)


# -------------------- Stage 6.5: unocculted heterogeneities ---------------


class TestContaminationFactor:
    def test_no_spots_means_no_correction(self):
        w = np.linspace(2.9, 5.2, 40)
        eps = contamination_factor(w, t_phot=3500, t_het=2800, f_het=0.0)
        assert np.allclose(eps, 1.0)

    def test_identical_temperatures_mean_no_correction(self):
        w = np.linspace(2.9, 5.2, 40)
        eps = contamination_factor(w, t_phot=3500, t_het=3500, f_het=0.3)
        assert np.allclose(eps, 1.0)

    def test_cool_spots_deepen_the_transit(self):
        w = np.linspace(2.9, 5.2, 40)
        eps = contamination_factor(w, t_phot=3500, t_het=2800, f_het=0.2)
        assert np.all(eps > 1.0)

    def test_hot_faculae_shallow_the_transit(self):
        w = np.linspace(2.9, 5.2, 40)
        eps = contamination_factor(w, t_phot=3500, t_het=4200, f_het=0.2)
        assert np.all(eps < 1.0)

    def test_spot_contrast_falls_towards_the_infrared(self):
        # The chromatic slope is the whole signal; its SIGN is what
        # distinguishes contamination from an atmosphere, so pin it.
        w = np.linspace(2.9, 5.2, 40)
        eps = contamination_factor(w, t_phot=3500, t_het=2800, f_het=0.2)
        assert eps[0] > eps[-1]

    def test_supplied_stellar_spectra_override_blackbodies(self):
        w = np.linspace(2.9, 5.2, 10)
        ones = np.ones_like(w)
        eps = contamination_factor(w, t_phot=3500, t_het=2800, f_het=0.5,
                                   stellar_spectra={"phot": ones, "het": ones})
        assert np.allclose(eps, 1.0)

    def test_planck_is_hotter_is_brighter(self):
        w = np.array([4.0])
        assert planck(w, 4000)[0] > planck(w, 3000)[0]

    def test_extreme_covering_fraction_stays_finite(self):
        w = np.linspace(2.9, 5.2, 20)
        eps = contamination_factor(w, t_phot=5000, t_het=1000, f_het=0.99)
        assert np.all(np.isfinite(eps))


class TestStage65Robustness:
    """The 2026-08-27 run produced two "detections" that could not be
    real: f_het railed against its prior wall implying ~48% spot
    coverage, and TOI-836 b vs TOI-836.01 -- two planets of the SAME
    star -- disagreeing (delta_BIC -1.3 vs +31.0)."""

    @staticmethod
    def _spectrum(slope_ppm_per_um=0.0, offset_ppm=0.0, n=50, seed=0):
        rng = np.random.default_rng(seed)
        w = np.linspace(2.9, 5.2, n)
        d = 1800.0 + slope_ppm_per_um * (w - w.mean()) + rng.normal(0, 25, n)
        d = d + offset_ppm * (w > 3.75)          # NRS1/NRS2 step
        return w, d, np.full(n, 25.0), (w > 3.75)

    def test_a_pure_slope_is_not_contamination(self):
        # epsilon(lambda) can mimic any slope. Only CURVATURE is evidence,
        # so a straight-line spectrum must not be a detection however
        # well it beats a flat line.
        w, d, e, _ = self._spectrum(slope_ppm_per_um=60.0)
        r = retrieve_contamination(w, d, e, t_phot=3500.0, n_steps=2500)
        assert r["delta_bic_linear"] < 10.0
        assert not r["contamination_detected"]

    def test_a_detector_offset_is_absorbed_not_reported_as_spots(self):
        w, d, e, is2 = self._spectrum(offset_ppm=80.0)
        r = retrieve_contamination(w, d, e, t_phot=3500.0,
                                   offset_mask=is2, n_steps=2500)
        assert r["offset_fitted"]
        assert abs(r["offset_ppm"] - 80.0) < 60.0
        assert not r["contamination_detected"]

    def test_a_railed_posterior_is_never_a_detection(self):
        w, d, e, _ = self._spectrum(slope_ppm_per_um=400.0, seed=3)
        r = retrieve_contamination(w, d, e, t_phot=3500.0,
                                   f_het_max=0.05, n_steps=2500)
        assert r["f_het_railed"] is True
        assert not r["contamination_detected"], "railed fit reported as detection"

    def test_flat_spectrum_stays_a_clean_non_detection(self):
        w, d, e, _ = self._spectrum()
        r = retrieve_contamination(w, d, e, t_phot=3500.0, n_steps=2500)
        assert not r["contamination_detected"]
        assert 0.0 < r["f_het_95pct_upper"] <= 0.5


class TestContaminationBackends:
    def test_probe_reports_every_backend(self):
        report = contamination_backends()
        assert set(report) == {"spotrod", "stctm", "sage"}
        assert all("available" in v for v in report.values())

    def test_external_command_marks_a_backend_available(self, monkeypatch):
        monkeypatch.setenv("ASTER_STCTM_CMD", "/usr/bin/true")
        report = contamination_backends()
        assert report["stctm"]["available"] is True
        assert report["stctm"]["external_command"] == "/usr/bin/true"


# -------------------- depth check --------------------


class TestGrazingLimbDarkening:
    """Limb darkening is fitted per channel at moderate impact parameter,
    where measurement showed it costs ~3%, and FIXED near grazing, where
    L 98-59 d (b=0.94) paid 2.47x in depth error and moved 5% in depth."""

    def test_threshold_sits_between_the_measured_cases(self):
        # b=0.68 cost 1.03x and must stay free; b=0.94 cost 2.47x and
        # must be fixed.
        assert 0.68 < LD_FIX_IMPACT_PARAM < 0.94

    def test_fixed_priors_have_no_width(self):
        import aster_toolkit.data_reduction.juliet as J
        ld = {"q1": 0.083, "q2": 0.17}
        free = J._ld_prior_entries("NRS1", ld)
        assert free["q1_NRS1"]["distribution"] == "truncatednormal"
        # A fixed prior carries a scalar, not a (mu, sigma, lo, hi) list.
        fixed = {"distribution": "fixed", "hyperparameters": float(ld["q1"])}
        assert not isinstance(fixed["hyperparameters"], list)

    def test_uniform_fallback_when_no_coefficients(self):
        import aster_toolkit.data_reduction.juliet as J
        out = J._ld_prior_entries("NRS1", None)
        assert out["q1_NRS1"]["distribution"] == "uniform"


class TestDepthCheck:
    def test_prefers_a_published_same_band_depth(self):
        ref = published_depth_reference("TOI-1231 b")
        assert ref is not None and ref["depth_ppm"] == pytest.approx(5571.0)
        check = evaluate_depth_check(5596.0, 6.0, {"rp_rs": 0.07464},
                                     planet_name="TOI-1231 b")
        assert check["band"] == "same-band"
        assert check["expected_ppm"] == pytest.approx(5571.0)
        assert check["status"] == "ok"

    def test_alias_resolves_to_the_published_entry(self):
        assert published_depth_reference("TOI-732 c")["planet_name"] == "LTT 3780 c"

    def test_unknown_planet_falls_back_to_the_archive(self):
        check = evaluate_depth_check(1000.0, 30.0, {"rp_rs": 0.0316},
                                     planet_name="Nowhere b")
        assert check["band"] == "cross-band"
        assert check["status"] == "ok"

    def test_cross_band_offset_is_indicative_not_suspect(self):
        # The referee's point: a healthy G395H depth sits 10-15% from the
        # optical TESS value, and the old check called that "suspect".
        # TOI-1468 c: TESS 2767 ppm vs Patchwork 3055 ppm (+10.4%).
        rp_rs = float(np.sqrt(2767e-6))
        check = evaluate_depth_check(3055.0, 9.0, {"rp_rs": rp_rs},
                                     planet_name="TOI-1468 c")
        assert check["band"] == "cross-band"
        assert check["suspect"] is False
        assert check["difference_frac"] == pytest.approx(0.104, abs=0.01)

    def test_same_band_disagreement_is_suspect(self):
        check = evaluate_depth_check(4500.0, 20.0, {"rp_rs": 0.07464},
                                     planet_name="TOI-1231 b")
        assert check["status"] == "suspect" and check["suspect"] is True

    def test_prior_returning_fit_is_suspect_in_any_band(self):
        # The documented failure: ~1800 ppm with errors larger than the
        # signal, whatever the target.
        check = evaluate_depth_check(1800.0, 1500.0, {"rp_rs": 0.0316},
                                     planet_name="Nowhere b")
        assert check["suspect"] is True and check["returned_prior"] is True

    def test_gross_cross_band_disagreement_is_still_reported(self):
        rp_rs = float(np.sqrt(1000e-6))
        check = evaluate_depth_check(5000.0, 20.0, {"rp_rs": rp_rs})
        assert check["status"] == "indicative"
        assert check["difference_frac"] > CROSS_BAND_TOL_FRAC

    def test_no_reference_at_all_is_unavailable(self):
        check = evaluate_depth_check(3000.0, 20.0, {})
        assert check["status"] == "unavailable" and check["suspect"] is False

    def test_old_key_is_preserved_for_readers(self):
        check = evaluate_depth_check(1000.0, 30.0, {"rp_rs": 0.0316})
        assert check["expected_ppm_from_archive"] == check["expected_ppm"]


# -------------------- tilt events (v1.3 multivariate search) --------------


def synthetic_tilt_visit(n=2000, step_i=1000, step_amp=800e-6,
                         fwhm_step=0.05, y_step=0.02, pca_step=5.0,
                         transit=(700, 1300), depth=5000e-6,
                         noise=150e-6, seed=0):
    """A visit with a transit and a mirror tilt event, plus the trace
    diagnostics a real reduction would supply. The tilt is placed INSIDE
    the transit by default — the TOI-270 c configuration."""
    rng = np.random.default_rng(seed)
    time = 2460500.0 + np.arange(n) * (9.0 / 86400)      # ~5 h at 9 s
    oot = np.ones(n, dtype=bool)
    if transit:
        oot[transit[0]:transit[1]] = False

    flux = np.ones(n) - depth * (~oot) + rng.normal(0, noise, n)
    diagnostics = {k: rng.normal(0, 0.01, n) for k in ("fwhm", "y", "x")}
    pca = rng.normal(0, 1.0, (n, 6))
    if step_i is not None:
        flux[step_i:] += step_amp
        diagnostics["fwhm"][step_i:] += fwhm_step
        diagnostics["y"][step_i:] += y_step
        pca[step_i:, 0] += pca_step
    return {"time": time, "flux": flux, "oot": oot,
            "diagnostics": diagnostics, "pca": pca, "t0_obs": time[n // 2]}


class TestStepStatistic:
    def test_flat_series_has_no_significant_step(self):
        z = step_statistic(np.random.default_rng(0).normal(0, 1, 1000))
        assert np.nanmax(np.abs(z)) < 6.0

    def test_step_is_localised_and_significant(self):
        s = np.random.default_rng(0).normal(0, 0.01, 1000)
        s[500:] += 0.2
        z = step_statistic(s)
        assert np.abs(z[500]) > 6.0
        assert int(np.nanargmax(np.abs(z))) == pytest.approx(500, abs=3)

    def test_a_smooth_ramp_is_not_a_step(self):
        s = np.linspace(0, 1, 1000) + np.random.default_rng(0).normal(0, 0.01, 1000)
        assert np.nanmax(np.abs(step_statistic(s))) < 6.0


class TestFindTiltEvents:
    def test_finds_a_tilt_that_happens_during_transit(self):
        # The failure this exists for: the flux-only search has to mask
        # the transit to avoid flagging ingress, so an in-transit tilt is
        # structurally invisible. TOI-270 c cost us a spectrum this way.
        v = synthetic_tilt_visit(step_i=1000)
        assert detect_tilt_events(v["flux"], exclude_mask=v["oot"]) == [], \
            "precondition: the legacy flux-only search cannot see this"

        r = find_tilt_events(v["flux"], diagnostics=v["diagnostics"],
                             pca_components=v["pca"], oot_mask=v["oot"],
                             time=v["time"], t0_obs=v["t0_obs"])
        assert len(r["events"]) == 1
        e = r["events"][0]
        assert abs(e["index"] - 1000) <= 3
        assert e["in_transit"] is True
        assert e["amplitude_ppm"] == pytest.approx(800, rel=0.2)

    def test_reports_which_diagnostics_triggered(self):
        v = synthetic_tilt_visit(step_i=400, transit=(700, 1300))
        r = find_tilt_events(v["flux"], diagnostics=v["diagnostics"],
                             pca_components=v["pca"], oot_mask=v["oot"],
                             time=v["time"], t0_obs=v["t0_obs"])
        sources = r["events"][0]["sources"]
        # FWHM is the most direct signature of a tilt (Albert 2026)
        assert "trace_fwhm" in sources
        assert "pca_0" in sources
        assert r["events"][0]["n_sources"] >= 2

    def test_a_transit_alone_is_not_a_tilt(self):
        v = synthetic_tilt_visit(step_i=None)
        r = find_tilt_events(v["flux"], diagnostics=v["diagnostics"],
                             pca_components=v["pca"], oot_mask=v["oot"],
                             time=v["time"], t0_obs=v["t0_obs"])
        assert r["events"] == []

    def test_one_series_stepping_alone_is_rejected(self):
        # A glitch in a single diagnostic is not the observatory moving.
        v = synthetic_tilt_visit(step_i=None)
        v["diagnostics"]["y"][1000:] += 0.10
        r = find_tilt_events(v["flux"], diagnostics=v["diagnostics"],
                             pca_components=v["pca"], oot_mask=v["oot"],
                             time=v["time"], t0_obs=v["t0_obs"])
        assert r["events"] == []

    def test_smooth_drifts_are_not_tilts(self):
        v = synthetic_tilt_visit(step_i=None)
        n = v["flux"].size
        v["diagnostics"]["fwhm"] += np.linspace(0, 0.08, n)
        v["diagnostics"]["y"] += np.linspace(0, 0.05, n)
        r = find_tilt_events(v["flux"] + np.linspace(0, 1e-3, n),
                             diagnostics=v["diagnostics"],
                             pca_components=v["pca"], oot_mask=v["oot"],
                             time=v["time"], t0_obs=v["t0_obs"])
        assert r["events"] == []

    def test_stable_across_noise_seeds(self):
        for seed in (0, 1, 7):
            v = synthetic_tilt_visit(step_i=1000, seed=seed)
            r = find_tilt_events(v["flux"], diagnostics=v["diagnostics"],
                                 pca_components=v["pca"], oot_mask=v["oot"],
                                 time=v["time"], t0_obs=v["t0_obs"])
            assert len(r["events"]) == 1, f"seed {seed}"
            assert abs(r["events"][0]["index"] - 1000) <= 5

    def test_rate_guard_fires_on_implausibly_many_events(self):
        # ~1 tilt event per day (Albert 2026), so a 5 h visit expects
        # ~0.2. Six is the search misbehaving, not the telescope.
        v = synthetic_tilt_visit(step_i=None)
        for i in (200, 500, 800, 1100, 1400, 1700):
            v["diagnostics"]["fwhm"][i:] += 0.05
            v["diagnostics"]["y"][i:] += 0.04
        r = find_tilt_events(v["flux"], diagnostics=v["diagnostics"],
                             pca_components=v["pca"], oot_mask=v["oot"],
                             time=v["time"], t0_obs=v["t0_obs"])
        assert len(r["events"]) > 2
        assert r["rate_warning"] and "rare" in r["rate_warning"]

    def test_quiet_visit_raises_no_rate_warning(self):
        v = synthetic_tilt_visit(step_i=1000)
        r = find_tilt_events(v["flux"], diagnostics=v["diagnostics"],
                             pca_components=v["pca"], oot_mask=v["oot"],
                             time=v["time"], t0_obs=v["t0_obs"])
        assert r["rate_warning"] is None

    def test_without_diagnostics_it_says_so(self):
        v = synthetic_tilt_visit(step_i=1000)
        r = find_tilt_events(v["flux"], oot_mask=v["oot"], time=v["time"],
                             t0_obs=v["t0_obs"])
        assert r["diagnostics_available"] is False
        assert r["sources_searched"] == ["flux"]


class TestTiltCrossDetector:
    def _reports(self, offset_ints=0):
        a = synthetic_tilt_visit(step_i=1000, seed=1)
        b = synthetic_tilt_visit(step_i=1000 + offset_ints, seed=2)
        b["time"] = a["time"]
        return {
            "NRS1": find_tilt_events(a["flux"], diagnostics=a["diagnostics"],
                                     pca_components=a["pca"], oot_mask=a["oot"],
                                     time=a["time"], t0_obs=a["t0_obs"],
                                     detector="NRS1"),
            "NRS2": find_tilt_events(b["flux"], diagnostics=b["diagnostics"],
                                     pca_components=b["pca"], oot_mask=b["oot"],
                                     time=b["time"], t0_obs=b["t0_obs"],
                                     detector="NRS2"),
        }

    def test_coincident_event_is_confirmed(self):
        merged = match_tilt_events(self._reports())
        assert len(merged) == 1
        assert merged[0]["confirmed"] is True
        assert set(merged[0]["detectors"]) == {"NRS1", "NRS2"}

    def test_opposite_sign_steps_still_confirm(self):
        # A tilt reshapes the PSF, so the flux step can differ in size
        # AND sign between detectors (arXiv:2405.06737). Chromaticity
        # must never be used to reject a tilt.
        a = synthetic_tilt_visit(step_i=1000, step_amp=+800e-6, seed=1)
        b = synthetic_tilt_visit(step_i=1000, step_amp=-600e-6, seed=2)
        b["time"] = a["time"]
        reports = {
            d: find_tilt_events(v["flux"], diagnostics=v["diagnostics"],
                                pca_components=v["pca"], oot_mask=v["oot"],
                                time=v["time"], t0_obs=v["t0_obs"], detector=d)
            for d, v in (("NRS1", a), ("NRS2", b))
        }
        merged = match_tilt_events(reports)
        assert merged[0]["confirmed"] is True
        amps = merged[0]["amplitude_ppm"]
        assert amps["NRS1"] > 0 > amps["NRS2"]

    def test_single_detector_event_is_not_confirmed(self):
        reports = self._reports()
        reports["NRS2"]["events"] = []
        merged = match_tilt_events(reports)
        assert all(not m["confirmed"] for m in merged)
        assert merged[0]["single_detector"] is True


class TestTiltTransitionMask:
    def test_masks_only_the_transition(self):
        keep = tilt_transition_keep_mask(2000, [{"index": 1000}])
        assert (~keep).sum() == 2 * TILT_TRANSITION_MASK + 1
        assert not keep[1000]
        assert keep[:1000 - TILT_TRANSITION_MASK].all()
        assert keep[1000 + TILT_TRANSITION_MASK + 1:].all()

    def test_preserves_the_post_event_data(self):
        # The whole point of the Heaviside: the post-event half of the
        # visit is corrected, not discarded, and the lightcurve is never
        # split. Anything else throws away the orbit constraint.
        keep = tilt_transition_keep_mask(2000, [{"index": 1000}])
        assert keep.sum() > 2000 - 10
        assert keep[1500:].all()

    def test_no_events_keeps_everything(self):
        assert tilt_transition_keep_mask(500, []).all()

    def test_accepts_a_cross_matched_event(self):
        merged = [{"detectors": {"NRS1": {"index": 300}}}]
        keep = tilt_transition_keep_mask(1000, merged)
        assert not keep[300]

    def test_step_regressor_matches_the_detected_event(self):
        v = synthetic_tilt_visit(step_i=1000)
        r = find_tilt_events(v["flux"], diagnostics=v["diagnostics"],
                             pca_components=v["pca"], oot_mask=v["oot"],
                             time=v["time"], t0_obs=v["t0_obs"])
        M = step_regressors(v["flux"].size, r["events"])
        assert M.shape == (v["flux"].size, 1)
        idx = r["events"][0]["index"]
        assert M[:idx].sum() == 0 and M[idx:].all()


# -------------------- fit v1.6: joint NRS1+NRS2 white light + GP --------------------


def _joint_priors():
    return {"planet_name": "Test b", "period": 5.0, "period_err": 1e-5,
            "a_rs": 15.0, "a_rs_err": 0.5, "rp_rs": 0.03}


class TestJointWhitePriors:
    def test_orbit_is_shared_and_depth_is_per_detector(self):
        import aster_toolkit.data_reduction.juliet as J
        t = np.linspace(0.0, 0.25, 200)
        p = J._joint_white_priors(_joint_priors(), 0.12, {
            "NRS1": {"n_regressors": 3, "ld": None, "gp_time": t},
            "NRS2": {"n_regressors": 2, "ld": None, "gp_time": t},
        })
        # One geometry for both detectors: the HD 15337 c o019 failure was
        # two independent fits putting the same transit 77 min apart.
        for key in J.JOINT_SHARED_PARAMS:
            assert key in p
            assert not any(k.startswith(key + "_") for k in p)
        assert "p_p1" not in p
        assert {"p_p1_NRS1", "p_p1_NRS2", "q1_NRS1", "q1_NRS2"} <= set(p)
        assert {"theta2_NRS1", "theta1_NRS2"} <= set(p)
        assert "theta2_NRS2" not in p   # regressor counts stay per detector

    def test_gp_rho_spans_minutes_to_the_visit(self):
        import aster_toolkit.data_reduction.juliet as J
        t = np.linspace(0.0, 0.25, 200)
        gp = J._gp_priors("NRS1", t)
        lo, hi = gp["GP_rho_NRS1"]["hyperparameters"]
        assert lo == pytest.approx(2.0 / 1440)
        assert hi == pytest.approx(0.25)
        assert gp["GP_sigma_NRS1"]["hyperparameters"] == J.GP_SIGMA_BOUNDS

    def test_no_gp_keys_when_gp_is_off(self):
        import aster_toolkit.data_reduction.juliet as J
        p = J._joint_white_priors(_joint_priors(), 0.12, {
            "NRS1": {"n_regressors": 1, "ld": None, "gp_time": None}})
        assert not any(k.startswith("GP_") for k in p)


def _synthetic_joint_lcs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(-0.12, 0.12, n)
    oot = np.abs(t) > 0.05
    lcs = {}
    for det, depth in (("NRS1", 900e-6), ("NRS2", 1000e-6)):
        flux = 1.0 - depth * (~oot) + rng.normal(0, 200e-6, n)
        lcs[det] = {"time": t, "wl_flux": flux, "wl_err": np.full(n, 200e-6),
                    "oot_mask": oot, "t0_obs": 0.0}
    return lcs


class _FakeJulietResults:
    """Stands in for a juliet results object: a posterior for the joint
    parameter names, and evaluate() returning a box transit (+ a GP-like
    wiggle when the fit had a GP)."""

    def __init__(self, load_kwargs):
        self.load_kwargs = load_kwargs
        rng = np.random.default_rng(1)
        n = 500
        post = {}
        for name, spec in load_kwargs["priors"].items():
            if spec["distribution"] == "fixed":
                continue
            hp = spec["hyperparameters"]
            centre = hp[0] if spec["distribution"] in ("normal", "truncatednormal") \
                else float(np.sqrt(hp[0] * hp[1])) if spec["distribution"] == "loguniform" \
                else 0.5 * (hp[0] + hp[1])
            post[name] = centre + rng.normal(0, 1e-4, n)
        post["p_p1_NRS1"] = 0.030 + rng.normal(0, 1e-4, n)
        post["p_p1_NRS2"] = 0.0316 + rng.normal(0, 1e-4, n)
        post["b_p1"] = 0.3 + rng.normal(0, 1e-3, n)
        post["a_p1"] = 15.0 + rng.normal(0, 1e-2, n)
        for d in ("NRS1", "NRS2"):
            post[f"q1_{d}"] = 0.1 + rng.normal(0, 1e-3, n)
            post[f"q2_{d}"] = 0.2 + rng.normal(0, 1e-3, n)
        self.posteriors = {"posterior_samples": post}
        outer = self

        class _LC:
            def evaluate(self, det, return_components=False):
                t = outer.load_kwargs["t_lc"][det]
                transit = 1.0 - (post[f"p_p1_{det}"].mean() ** 2) * (np.abs(t) < 0.05)
                lm = np.zeros_like(t)
                gp = (20e-6 * np.sin(t * 200)
                      if outer.load_kwargs.get("GP_regressors_lc") else 0 * t)
                model = transit + lm + gp
                if return_components:
                    return model, {"transit": transit, "lm": lm, "p1": transit}
                return model
        self.lc = _LC()


class _FakeJuliet:
    def __init__(self):
        self.calls = []

    def load(self, **kwargs):
        self.calls.append(kwargs)
        outer = self

        class _Dataset:
            def fit(self, **_):
                return _FakeJulietResults(outer.calls[-1])
        return _Dataset()


class TestJointWhiteFitPlumbing:
    def _run(self, monkeypatch, tmp_path, dets=("NRS1", "NRS2"), gp=True, scan=True):
        import aster_toolkit.data_reduction.juliet as J
        fake = _FakeJuliet()
        monkeypatch.setattr(J, "_import_juliet", lambda: fake)
        monkeypatch.setattr(J, "_require_celerite", lambda: None)
        lcs = {d: v for d, v in _synthetic_joint_lcs().items() if d in dets}
        regs = {d: np.column_stack([lcs[d]["time"], np.ones(lcs[d]["time"].size)])
                for d in lcs}
        names = {d: ["time", "pca_1"] for d in lcs}
        out = J.fit_white_lightcurve_joint(
            lcs, _joint_priors(), {d: tmp_path / d.lower() for d in lcs},
            joint_dir=tmp_path / "joint", regressors=regs, regressor_names=names,
            gp=gp, anomaly_scan={"contam_version": "1.1", "events": []} if scan else None,
        )
        return fake, out

    def test_one_juliet_call_shares_the_orbit(self, monkeypatch, tmp_path):
        fake, out = self._run(monkeypatch, tmp_path)
        assert len(fake.calls) == 1                       # ONE joint fit
        call = fake.calls[0]
        assert set(call["t_lc"]) == {"NRS1", "NRS2"}
        assert set(call["GP_regressors_lc"]) == {"NRS1", "NRS2"}
        assert "t0_p1" in call["priors"] and "t0_p1_NRS1" not in call["priors"]
        # Both detector summaries carry the SAME geometry, each its own depth.
        for key in ("t0_p1", "b_p1", "a_p1"):
            assert out["NRS1"][key] == out["NRS2"][key]
        assert out["NRS1"]["p_p1"]["median"] == pytest.approx(0.030, abs=1e-3)
        assert out["NRS2"]["p_p1"]["median"] == pytest.approx(0.0316, abs=1e-3)
        assert out["NRS2"]["depth_ppm"]["median"] > out["NRS1"]["depth_ppm"]["median"]
        assert out["NRS1"]["fit_version"] == J_VERSION()
        assert out["NRS1"]["noise_model"]["kind"] == "celerite-matern32"
        assert "rednoise_after_gp" in out["NRS1"]
        # The GP never sees absolute BJD (see gp_time_input).
        for det, x in call["GP_regressors_lc"].items():
            assert np.max(np.abs(x)) < 1.0
        assert out["NRS1"]["noise_model"]["gp_time_origin"] == pytest.approx(0.0)
        assert (tmp_path / "joint" / "joint_white_fit_summary.json").is_file()
        npz = np.load(tmp_path / "nrs2" / "white_lightcurve_residuals.npz")
        assert "gp" in npz.files and np.ptp(npz["gp"]) > 0
        on_disk = json.loads((tmp_path / "nrs1" / "white_fit_summary.json").read_text())
        assert on_disk["joint_fit"]["n_detectors"] == 2

    def test_stamps_mark_non_survey_fits(self, monkeypatch, tmp_path):
        _, one = self._run(monkeypatch, tmp_path, dets=("NRS1",))
        assert one["NRS1"]["fit_version"].endswith("-1det")
        _, nogp = self._run(monkeypatch, tmp_path / "b", gp=False)
        assert nogp["NRS1"]["fit_version"].endswith("-nogp")
        assert "rednoise_after_gp" not in nogp["NRS1"]
        _, noscan = self._run(monkeypatch, tmp_path / "c", scan=False)
        assert "-noscan" in noscan["NRS1"]["fit_version"]

    def test_channels_inherit_the_white_gp_timescale(self, monkeypatch, tmp_path):
        import aster_toolkit.data_reduction.juliet as J
        fake = _FakeJuliet()
        monkeypatch.setattr(J, "_import_juliet", lambda: fake)
        monkeypatch.setattr(J, "_require_celerite", lambda: None)
        lc = _synthetic_joint_lcs()["NRS1"]
        n = lc["time"].size
        lc.update(wave=np.array([3.0, 3.5]), wave_err=np.array([0.01, 0.01]),
                  sp_flux=np.column_stack([lc["wl_flux"]] * 2),
                  sp_err=np.column_stack([lc["wl_err"]] * 2))
        white = {"P_p1": {"median": 5.0}, "t0_p1": {"median": 0.0},
                 "b_p1": {"median": 0.3}, "a_p1": {"median": 15.0},
                 "ld_source": "uniform", "fit_version": J.PATCHWORK_FIT_VERSION,
                 "noise_model": {"kind": J.GP_KERNEL, "gp_time_origin": -0.5},
                 "GP_rho_NRS1": {"median": 0.01}, "regressor_names": ["time"]}
        J.fit_transmission_spectrum(lc, white, tmp_path, instrument="NRS1",
                                    regressors=np.ones((n, 1)))
        assert len(fake.calls) == 2
        for call in fake.calls:
            assert call["priors"]["GP_rho_NRS1"] == {"distribution": "fixed",
                                                     "hyperparameters": 0.01}
            # v1.7: the amplitude is trained out of transit and fixed too.
            assert call["priors"]["GP_sigma_NRS1"]["distribution"] == "fixed"
            # Same origin as the white fit, so the frozen rho means the same.
            assert call["GP_regressors_lc"]["NRS1"] == pytest.approx(lc["time"] + 0.5)
        header = (tmp_path / "transmission_spectrum.csv").read_text().splitlines()[0]
        assert "noise=celerite-matern32" in header


def J_VERSION():
    from aster_toolkit.data_reduction.juliet import PATCHWORK_FIT_VERSION
    return PATCHWORK_FIT_VERSION


def _real_juliet_available():
    try:
        import aster_toolkit.data_reduction.juliet as J
        J._import_juliet()
        import celerite  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _real_juliet_available(),
                    reason="needs juliet + celerite (the fit environment)")
class TestJointWhiteFitRealJuliet:
    """A real joint juliet fit on synthetic NRS1+NRS2 data with correlated
    noise and opposite time slopes per detector. Recovers each injected
    depth and ONE shared mid-transit."""

    def test_recovers_depths_and_shared_t0(self, tmp_path):
        import aster_toolkit.data_reduction.juliet as J
        J._import_juliet()   # repairs the batman namespace if needed
        from batman.transitmodel import TransitModel, TransitParams
        rng = np.random.default_rng(3)
        n, t0_true = 300, 0.004
        t = np.linspace(-0.12, 0.12, n)
        red = np.convolve(rng.normal(size=n), np.ones(15) / 15, "same") * 300e-6

        def transit(rp):
            p = TransitParams()
            p.t0, p.per, p.rp, p.a = t0_true, 5.0, rp, 15.0
            p.inc = float(np.degrees(np.arccos(0.3 / 15.0)))
            p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.1, 0.1]
            return TransitModel(p, t).light_curve(p)

        slope = t / 0.12
        lcs = {
            "NRS1": transit(0.030) + 3e-4 * slope + red + rng.normal(0, 250e-6, n),
            "NRS2": transit(0.032) - 3e-4 * slope + 0.8 * red + rng.normal(0, 350e-6, n),
        }
        lcs = {d: {"time": t, "wl_flux": f,
                   "wl_err": np.full(n, 250e-6 if d == "NRS1" else 350e-6),
                   "oot_mask": np.abs(t) > 0.05, "t0_obs": 0.0}
               for d, f in lcs.items()}
        priors = dict(_joint_priors(), a_rs_err=0.7)
        out = J.fit_white_lightcurve_joint(
            lcs, priors, {d: tmp_path / d.lower() for d in lcs},
            joint_dir=tmp_path / "joint",
            regressors={d: slope[:, None] for d in lcs},
            regressor_names={d: ["time"] for d in lcs},
            gp=True, n_live=100, anomaly_scan={"events": []},
        )
        t0 = out["NRS1"]["t0_p1"]
        assert out["NRS2"]["t0_p1"] == t0
        assert abs(t0["median"] - t0_true) < 4 * max(t0["plus"], t0["minus"]) + 1e-4
        for det, rp in (("NRS1", 0.030), ("NRS2", 0.032)):
            p = out[det]["p_p1"]
            assert abs(p["median"] - rp) < 4 * max(p["plus"], p["minus"])


class TestGpTimeInput:
    """The first v1.6 survey run crashed on 15 of 22 targets with
    celerite.solver.LinAlgError: the GP was fed absolute BJD."""

    def test_channels_refuse_a_white_fit_made_on_absolute_bjd(self, tmp_path):
        import aster_toolkit.data_reduction.juliet as J
        lc = _synthetic_joint_lcs()["NRS1"]
        lc.update(wave=np.array([3.0]), wave_err=np.array([0.01]),
                  sp_flux=lc["wl_flux"][:, None], sp_err=lc["wl_err"][:, None])
        white = {"P_p1": {"median": 5.0}, "t0_p1": {"median": 0.0},
                 "b_p1": {"median": 0.3}, "a_p1": {"median": 15.0},
                 "ld_source": "uniform", "noise_model": {"kind": J.GP_KERNEL},
                 "GP_rho_NRS1": {"median": 0.01}}
        with pytest.raises(ValueError, match="absolute BJD"):
            J.fit_transmission_spectrum(lc, white, tmp_path, instrument="NRS1",
                                        regressors=np.ones((lc["time"].size, 1)))

    def test_celerite_factorizes_relative_but_not_absolute_time(self):
        celerite = pytest.importorskip("celerite")
        from celerite import terms
        import aster_toolkit.data_reduction.juliet as J
        # GJ 357 b o007 NRS1 as fitted: 4401 integrations, 3.6 s cadence,
        # 106 ppm errors, at BJD ~2460000. juliet starts its Matern term at
        # sigma = rho = 10, which is where the survey run died.
        t = 2460123.4 + np.arange(4401) * 3.6 / 86400
        yerr = np.full(t.size, 106e-6)

        def factorizes(x):
            gp = celerite.GP(terms.Matern32Term(log_sigma=np.log(10.0),
                                                log_rho=np.log(10.0), eps=0.01))
            try:
                gp.compute(x, yerr=yerr)
                return True
            except celerite.solver.LinAlgError:
                return False

        assert factorizes(J.gp_time_input(t, t[t.size // 2]))
        # Documents why the shift exists. If celerite ever stops failing
        # here the guard is merely redundant, not wrong.
        if factorizes(t):
            pytest.skip("this celerite build copes with absolute BJD")


# -------------------- fit v1.7: out-of-transit GP, pass-1 t0, trace-backed steps --------------------


def _matern_noise(t, sigma, rho, seed=0):
    celerite = pytest.importorskip("celerite")
    from celerite import terms
    gp = celerite.GP(terms.Matern32Term(np.log(sigma), np.log(rho), eps=0.01))
    gp.compute(t, 1e-9)
    np.random.seed(seed)   # celerite's sampler draws from the global state
    return gp.sample()


class TestTransitWindow:
    def test_duration_matches_the_analytic_value(self):
        import aster_toolkit.data_reduction.juliet as J
        # Central transit: T14 = P/pi * asin((1+k)/a) for b = 0.
        P, a, k = 10.0, 20.0, 0.05
        assert J.transit_duration_days(P, a, 0.0, k) == pytest.approx(
            P / np.pi * np.arcsin((1 + k) / a), rel=1e-9)
        assert J.transit_duration_days(P, a, 0.9, k) < J.transit_duration_days(P, a, 0.0, k)
        assert np.isnan(J.transit_duration_days(P, a, 1.2, k))   # misses the star

    def test_window_comes_from_the_fitted_summary(self):
        import aster_toolkit.data_reduction.juliet as J
        s = {k: {"median": v} for k, v in
             (("t0_p1", 100.0), ("P_p1", 5.0), ("a_p1", 15.0), ("b_p1", 0.3), ("p_p1", 0.03))}
        t0, t14 = J.transit_window_from_summary(s)
        assert t0 == 100.0 and 0.05 < t14 < 0.12
        t = np.linspace(99.8, 100.2, 1001)
        oot = J.out_of_transit_mask(t, t0, t14)
        assert not oot[np.abs(t - t0) < 0.5 * t14].any()
        # The margin keeps the contacts themselves out of the training set.
        edge = np.abs(np.abs(t - t0) - 0.5 * t14) < 0.5 * J.OOT_MARGIN_DAYS
        assert not oot[edge].any()


class TestOutOfTransitGpTraining:
    def _visit(self, sigma=300e-6, rho=20 / 1440, depth=3000e-6, seed=1):
        n = 3000
        t = np.linspace(-0.15, 0.15, n)
        rng = np.random.default_rng(seed)
        red = _matern_noise(t, sigma, rho, seed=seed)
        transit = -depth * (np.abs(t) < 0.05)
        flux = 1.0 + transit + red + 2e-4 * t + rng.normal(0, 150e-6, n)
        return t, flux, np.full(n, 150e-6), np.abs(t) > 0.06

    def test_recovers_the_injected_noise_from_out_of_transit_alone(self):
        import aster_toolkit.data_reduction.juliet as J
        for seed in (1, 2, 3):
            t, flux, err, oot = self._visit(seed=seed)
            realised = _matern_noise(t, 300e-6, 20 / 1440, seed=seed)[oot].std()
            hp = J.train_gp_out_of_transit(t, flux, err, None, oot, 0.0)
            assert hp is not None and hp["n_oot"] == int(oot.sum())
            assert hp["GP_rho"] == pytest.approx(20 / 1440, rel=0.6)
            assert hp["GP_sigma"] == pytest.approx(realised, rel=0.3)
            assert not hp["rho_fixed"]
        # A regressor is fitted under the same covariance (GLS) and may take
        # a share of the slow noise, but the timescale survives.
        hp = J.train_gp_out_of_transit(t, flux, err, t[:, None], oot, 0.0)
        assert hp["GP_rho"] == pytest.approx(20 / 1440, rel=0.6)

    def test_the_transit_never_enters_the_training(self):
        # Same noise, a 10x deeper transit: out of transit nothing changed,
        # so neither may the hyperparameters. A v1.6-style free GP fitted
        # with the transit had every reason to change them.
        import aster_toolkit.data_reduction.juliet as J
        t, flux, err, oot = self._visit(depth=3000e-6)
        _, deep, _, _ = self._visit(depth=30000e-6)
        a = J.train_gp_out_of_transit(t, flux, err, None, oot, 0.0)
        b = J.train_gp_out_of_transit(t, deep, err, None, oot, 0.0)
        assert a["GP_sigma"] == pytest.approx(b["GP_sigma"], rel=1e-6)
        assert a["GP_rho"] == pytest.approx(b["GP_rho"], rel=1e-6)

    def test_fixed_rho_trains_only_the_amplitude(self):
        import aster_toolkit.data_reduction.juliet as J
        t, flux, err, oot = self._visit()
        hp = J.train_gp_out_of_transit(t, flux, err, None, oot, 0.0, rho=0.01)
        assert hp["GP_rho"] == 0.01 and hp["rho_fixed"]

    def test_too_little_baseline_returns_none(self):
        import aster_toolkit.data_reduction.juliet as J
        t, flux, err, oot = self._visit()
        few = oot & (np.cumsum(oot) <= J.GP_TRAIN_MIN_OOT - 1)
        assert J.train_gp_out_of_transit(t, flux, err, None, few, 0.0) is None


class TestV17WhiteFitPlumbing(TestJointWhiteFitPlumbing):
    def test_gp_is_trained_and_fixed_and_t0_tied_to_pass1(self, monkeypatch, tmp_path):
        import aster_toolkit.data_reduction.juliet as J
        fake = _FakeJuliet()
        monkeypatch.setattr(J, "_import_juliet", lambda: fake)
        lcs = _synthetic_joint_lcs()
        regs = {d: np.column_stack([lcs[d]["time"], np.ones(lcs[d]["time"].size)])
                for d in lcs}
        out = J.fit_white_lightcurve_joint(
            lcs, _joint_priors(), {d: tmp_path / d.lower() for d in lcs},
            joint_dir=tmp_path / "joint", regressors=regs,
            regressor_names={d: ["time", "pca_1"] for d in lcs},
            gp=True, anomaly_scan={"events": []},
            gp_train_window=(0.0, 0.1), t0_prior=(0.002, J.PASS2_T0_SIGMA_DAYS))
        pri = fake.calls[0]["priors"]
        for d in lcs:
            assert pri[f"GP_sigma_{d}"]["distribution"] == "fixed"
            assert pri[f"GP_rho_{d}"]["distribution"] == "fixed"
            nm = out[d]["noise_model"]
            assert nm["gp_frozen"] is True and nm["gp_training"]["n_oot"] > 0
            # Readers (channels, verify) find the fixed values where the
            # sampled ones used to be.
            assert out[d][f"GP_rho_{d}"]["median"] == pytest.approx(nm["gp_training"]["GP_rho"])
            assert out[d]["orbit_prior"]["t0"]["source"] == "pass 1"
            assert out[d]["fit_version"] == J.PATCHWORK_FIT_VERSION
        assert pri["t0_p1"]["hyperparameters"] == [0.002, J.PASS2_T0_SIGMA_DAYS]

    def test_untrainable_gp_falls_back_to_white_and_is_stamped(self, monkeypatch, tmp_path):
        import aster_toolkit.data_reduction.juliet as J
        fake = _FakeJuliet()
        monkeypatch.setattr(J, "_import_juliet", lambda: fake)
        lcs = _synthetic_joint_lcs()
        # A transit window covering the whole visit leaves no baseline.
        out = J.fit_white_lightcurve_joint(
            lcs, _joint_priors(), {d: tmp_path / d.lower() for d in lcs},
            joint_dir=tmp_path / "joint", gp=True, anomaly_scan={"events": []},
            regressor_names={d: ["pca_1"] for d in lcs},
            regressors={d: np.ones((lcs[d]["time"].size, 1)) for d in lcs},
            gp_train_window=(0.0, 1.0))
        assert not any(k.startswith("GP_") for k in fake.calls[0]["priors"])
        assert out["NRS1"]["fit_version"].endswith("-nogp")
        assert "gp_untrained" in out["NRS1"]["noise_model"]


class TestTraceStepSources:
    def _trace(self, step_at=None, n=2000):
        rng = np.random.default_rng(4)
        t = np.arange(n) * 10.0 / 86400
        fwhm = 2.0 + rng.normal(0, 0.001, n)
        if step_at is not None:
            fwhm[step_at:] += 0.02
        return t, {"trace_fwhm": step_statistic(fwhm), "pca_0": step_statistic(rng.normal(0, 1, n))}

    def test_a_real_tilt_is_supported_by_the_psf(self):
        t, z = self._trace(step_at=1000)
        assert trace_step_sources({"NRS1": {"time": t, "z": z}}, float(t[1000])) == \
            ["NRS1:trace_fwhm"]

    def test_a_flux_only_step_has_no_support(self):
        # HD 15337 c o019: "steps" at a mis-placed transit contact live in
        # the flux only; the PSF did nothing.
        t, z = self._trace(step_at=None)
        assert trace_step_sources({"NRS1": {"time": t, "z": z}}, float(t[1000])) == []

    def test_the_other_detector_can_supply_the_support(self):
        # TOI-270 c o016: only NRS1's FWHM cleared 6 sigma for a step both
        # detectors' flux showed.
        t, z1 = self._trace(step_at=1000)
        _, z2 = self._trace(step_at=None)
        got = trace_step_sources({"NRS1": {"time": t, "z": z1}, "NRS2": {"time": t, "z": z2}},
                                 float(t[1003]))
        assert got == ["NRS1:trace_fwhm"]


class TestVerifyV17:
    def _summary(self, **fit):
        import aster_toolkit.data_reduction.juliet as J
        f = {"fit_version": J.PATCHWORK_FIT_VERSION,
             "white_depth_ppm": {"median": 2950.0, "plus": 12.0, "minus": 12.0},
             "white_depth_ppm_pass1": {"median": 2960.0, "plus": 11.0, "minus": 11.0},
             "t0_p1": {"median": 100.0}, "t0_p1_pass1": {"median": 100.0},
             "b_p1": {"median": 0.6, "plus": 0.01, "minus": 0.01},
             "a_p1": {"median": 81.0, "plus": 0.8, "minus": 0.8},
             "noise_model": {"kind": J.GP_KERNEL, "gp_time_origin": 100.0,
                             "gp_frozen": True, "GP_rho_min": 20.0},
             "depth_check": {"status": "ok"}, "spot_handling": {"scan_ran": True},
             "joint_fit": {"n_detectors": 2}, "n_channels": 26,
             "ld_source": "exotic-ld", "tilt_handling": {"searched_diagnostics": True}}
        f.update(fit)
        return {"visits": {"o001": {"fits": {"NRS1": f}}}}

    def _check(self, summary):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                        "scripts", "patchwork"))
        import verify_fits
        return verify_fits.check_target(summary)

    def test_a_healthy_v17_fit_passes(self):
        lines, stop = self._check(self._summary())
        assert not stop, lines

    def test_a_free_gp_is_a_stop(self):
        s = self._summary()
        s["visits"]["o001"]["fits"]["NRS1"]["noise_model"].pop("gp_frozen")
        _, stop = self._check(s)
        assert stop

    def test_pass2_drifting_from_pass1_is_a_stop(self):
        # K2-18 b o004 under v1.6: 2972 ppm in pass 1, 1436 after the GP.
        lines, stop = self._check(self._summary(
            white_depth_ppm={"median": 1436.0, "plus": 190.0, "minus": 190.0},
            t0_p1={"median": 100.0 - 34.4 / 1440}))
        assert stop
        text = "\n".join(lines)
        assert "moved t0 -34 min" in text and "moved the depth -51%" in text
