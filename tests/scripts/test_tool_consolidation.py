"""Pins the consolidated ASTER tool surface (one tool per task, no overlap).

Covers the tools merged in the consolidation pass that are not exercised by
test_mast.py: QueryExoplanetArchive, GenerateFirJobs, the segment preflight
inside ReduceNirspecG395hTso, and the contamination-backend report inside
VerifyPatchworkEnvironment. Pure-local: no network, no exoTEDRF env.
"""
import ast
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import aster_toolkit  # noqa: E402

RETIRED = {
    "GetExoplanetParameters", "FindExoplanetsByCondition",
    "GetMastObservationProducts", "DownloadDemographicJwstProducts",
    "InspectNirspecG395hUncalData", "DetectNirspecTiltEvents",
    "VerifyContaminationBackends", "GeneratePatchworkFirJob",
    "GenerateFirOptimizerJobs",
}


def _construct(Tool, **kwargs):
    try:
        return Tool(**kwargs)
    except TypeError:
        tool = Tool()
        for k, v in kwargs.items():
            setattr(tool, k, v)
        return tool


# ---------------- registry ----------------

def test_exports_are_the_25_consolidated_tools():
    assert len(aster_toolkit.__all__) == 25
    assert len(set(aster_toolkit.__all__)) == 25
    assert not RETIRED & set(aster_toolkit.__all__)
    for name in RETIRED:
        with pytest.raises(AttributeError):
            getattr(aster_toolkit, name)


def test_run_aster_registers_exactly_the_exports():
    """run_aster.py's import list and tools=[...] list must match __all__."""
    tree = ast.parse((REPO / "run_aster.py").read_text())
    imported = set()
    registered = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "aster_toolkit":
            imported |= {a.name for a in node.names}
        if (isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "tools" for t in node.targets)
                and isinstance(node.value, ast.List)):
            for elt in node.value.elts:
                fn = elt.func if isinstance(elt, ast.Call) else elt
                if isinstance(fn, ast.Name):
                    registered.add(fn.id)
    assert imported == set(aster_toolkit.__all__)
    assert set(aster_toolkit.__all__) <= registered


def test_docs_do_not_route_to_retired_tools():
    for doc in ("aster_system_prompt.md", "CLAUDE.md",
                "scripts/patchwork/CAMPAIGN.md", "scripts/patchwork/AGENT_BRIEF.md"):
        text = (REPO / doc).read_text()
        stale = sorted(n for n in RETIRED if f"`{n}`" in text or f"{n}(" in text)
        assert not stale, f"{doc} still routes to {stale}"


# ---------------- QueryExoplanetArchive ----------------

from aster_toolkit.data_acquisition import exoarchive  # noqa: E402


def test_archive_planet_mode_formats_and_escapes_quotes():
    with patch.object(exoarchive, "get_exoplanet_params_tap",
                      return_value={"pl_radj": "1.27", "st_teff": "5400"}) as q:
        out = _construct(exoarchive.QueryExoplanetArchive,
                         planet_name="O'Test b", columns="['pl_radj','st_teff']")._run()
    assert q.call_args.kwargs["planet_name"] == "O''Test b"
    assert q.call_args.kwargs["columns"] == ["pl_radj", "st_teff"]
    assert "Planet Radius: 1.27 RJup" in out
    assert "Star Temp: 5400 K" in out


def test_archive_planet_mode_default_columns():
    with patch.object(exoarchive, "get_exoplanet_params_tap", return_value=None) as q:
        out = _construct(exoarchive.QueryExoplanetArchive, planet_name="X b")._run()
    assert q.call_args.kwargs["columns"] == ["pl_radj", "pl_bmassj", "pl_eqt"]
    assert "No data found" in out


def test_archive_conditions_mode():
    with patch.object(exoarchive, "find_exoplanets_by_condition",
                      return_value=[{"pl_name": "A b"}, {"pl_name": "B c"}]) as q:
        out = _construct(exoarchive.QueryExoplanetArchive,
                         conditions="['sy_pnum >= 2']", distinct="False",
                         limit="5")._run()
    kw = q.call_args.kwargs
    assert kw["conditions"] == ["sy_pnum >= 2"]
    assert kw["return_columns"] == ["pl_name"]
    assert kw["distinct"] is False and kw["limit"] == 5
    assert "(2 rows)" in out


@pytest.mark.parametrize("kwargs, msg", [
    ({}, "Supply planet_name"),
    ({"planet_name": "A b", "conditions": ["x > 1"]}, "not both"),
    ({"planet_name": "A b", "table": "foo.csv"}, "table must be"),
])
def test_archive_rejects_bad_calls(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        _construct(exoarchive.QueryExoplanetArchive, **kwargs)._run()


# ---------------- GenerateFirJobs ----------------

from aster_toolkit.data_reduction import survey  # noqa: E402


def _manifest(tmp_path):
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"planet_name": "Test b",
                             "visits": {"o001": "/raw/1", "o002": "/raw/2"}}))
    return m


def test_fir_jobs_patchwork_defaults(tmp_path):
    m = _manifest(tmp_path)
    out = _construct(survey.GenerateFirJobs, manifest_path=str(m),
                     output_root="/scratch/pw", base_directory=str(tmp_path))._run()
    assert "job=patchwork, 24:00:00, 8 CPUs, 40G" in out
    script = next(tmp_path.glob("*.sbatch")).read_text()
    assert "--time=24:00:00" in script and "data_reduction.survey" in script


def test_fir_jobs_optimizer_one_script_per_visit_detector(tmp_path):
    m = _manifest(tmp_path)
    out = _construct(survey.GenerateFirJobs, job="optimizer", manifest_path=str(m),
                     output_root="/scratch/pw", detectors="nrs1",
                     base_directory=str(tmp_path))._run()
    assert "Wrote 2 optimizer sbatch script(s)" in out
    assert "3-00:00:00, 16 CPUs, 64G" in out


def test_fir_jobs_overrides_and_bad_job(tmp_path):
    m = _manifest(tmp_path)
    out = _construct(survey.GenerateFirJobs, manifest_path=str(m), output_root="/s",
                     time="36:00:00", cpus="4", mem="",
                     base_directory=str(tmp_path))._run()
    assert "36:00:00, 4 CPUs, 40G" in out
    with pytest.raises(ValueError, match="job must be"):
        _construct(survey.GenerateFirJobs, job="reduce", manifest_path=str(m),
                   output_root="/s", base_directory=str(tmp_path))._run()


# ---------------- Reduce preflight / Verify ----------------

from aster_toolkit.data_reduction import exotedrf  # noqa: E402

_COMPLETE = {"input_dir": "x", "detectors": {"NRS1": {
    "complete": True, "missing_segments": [], "segments_expected": 2,
    "segments_found": [1, 2], "files": ["a", "b"], "target": "T",
    "date_obs": "d", "subarray": "SUB2048", "readpatt": "NRSRAPID",
    "nints_total": 10, "ngroups": 5}}}


def test_reduce_inspect_only_never_reduces(tmp_path):
    with patch.object(exotedrf, "inspect_uncal_directory", return_value=_COMPLETE), \
         patch.object(exotedrf, "run_reduction") as red:
        out = _construct(exotedrf.ReduceNirspecG395hTso, input_dir="raw",
                         inspect_only=True, base_directory=str(tmp_path))._run()
    red.assert_not_called()
    assert "Uncal inventory" in out and "COMPLETE" in out


def test_reduce_warns_on_incomplete_and_needs_output_dir(tmp_path):
    bad = json.loads(json.dumps(_COMPLETE))
    bad["detectors"]["NRS1"].update(complete=False, missing_segments=[2])
    with patch.object(exotedrf, "inspect_uncal_directory", return_value=bad), \
         patch.object(exotedrf, "run_reduction") as red:
        with pytest.raises(ValueError, match="output_dir is required"):
            _construct(exotedrf.ReduceNirspecG395hTso, input_dir="raw",
                       base_directory=str(tmp_path))._run()
        red.return_value = {"config_version": "1.1", "input_dir": "raw",
                            "output_dir": "o", "detectors": {}}
        out = _construct(exotedrf.ReduceNirspecG395hTso, input_dir="raw",
                         output_dir="o", base_directory=str(tmp_path))._run()
    red.assert_called_once()
    assert out.startswith("WARNING: incomplete")
    assert "exoTEDRF reduction finished" in out


def test_verify_reports_contamination_backends_even_without_exotedrf(tmp_path):
    with patch.object(exotedrf, "verify_exotedrf_environment",
                      side_effect=FileNotFoundError("no interpreter")):
        out = _construct(exotedrf.VerifyPatchworkEnvironment,
                         base_directory=str(tmp_path))._run()
    assert "Cannot locate the exoTEDRF interpreter" in out
    assert "Optional stellar-contamination backends:" in out
    for name in ("spotrod", "stctm", "sage"):
        assert name in out


def test_reduce_inspect_only_survives_orchestral_execute(tmp_path):
    """The real agent path: execute() enforces the schema's required list.
    Only input_dir may be required, or inspect_only calls die with
    'Missing Required Fields'."""
    pytest.importorskip("orchestral")
    Tool = exotedrf.ReduceNirspecG395hTso
    assert Tool._get_required_fields() == ["input_dir"]
    with patch.object(exotedrf, "inspect_uncal_directory", return_value=_COMPLETE):
        out = Tool(base_directory=str(tmp_path)).execute(
            input_dir="raw", inspect_only=True, st_teff="")
    assert "Missing Required Fields" not in out
    assert "Uncal inventory" in out


def test_reduce_normalizes_llm_numeric_strings(tmp_path):
    with patch.object(exotedrf, "inspect_uncal_directory", return_value=_COMPLETE), \
         patch.object(exotedrf, "run_reduction", return_value={
             "config_version": "1.1", "input_dir": "r", "output_dir": "o",
             "detectors": {}}) as red:
        _construct(exotedrf.ReduceNirspecG395hTso, input_dir="raw", output_dir="o",
                   stages="[3]", st_teff="3250", st_logg="", st_met=None,
                   base_directory=str(tmp_path))._run()
    kw = red.call_args.kwargs
    assert kw["stages"] == [3] and kw["st_teff"] == 3250.0
    assert kw["st_logg"] is None and kw["st_met"] is None


# ---------------- per-call argument reset (AsterBaseTool) ----------------

def test_every_aster_tool_uses_the_resetting_base():
    pytest.importorskip("orchestral")
    from aster_toolkit.tool_base import AsterBaseTool
    for name in aster_toolkit.__all__:
        if name == "SetTaurexPaths":      # @define_tool function, stateless
            continue
        assert issubclass(getattr(aster_toolkit, name), AsterBaseTool), name


def test_omitted_arguments_do_not_leak_from_the_previous_call(tmp_path):
    """orchestral builds a call from the instance's previous values; a long-
    lived GUI tool must not reuse an earlier call's optional arguments."""
    pytest.importorskip("orchestral")
    from aster_toolkit.data_acquisition import mast

    seen = []

    def fake_query(self, planet_name, proposal_id, ra, dec, radius_deg, filters, max_obs):
        seen.append(dict(planet_name=planet_name, proposal_id=proposal_id, filters=filters,
                         list_only=self.list_only, visits=self.visits))
        return []

    tool = mast.DownloadMastJwstProducts(base_directory=str(tmp_path))
    with patch.object(mast.DownloadMastJwstProducts, "_query_observations", fake_query):
        tool.execute(planet_name="WASP-39 b", proposal_id="1366", filters="['G395H']",
                     visits="[2]", list_only=True)
        tool.execute(proposal_id="1366", instruments="['NIRISS']")
    first, second = seen
    assert first["planet_name"] == "WASP-39 b" and first["filters"] == ["G395H"]
    assert second == dict(planet_name=None, proposal_id="1366", filters=None,
                          list_only=False, visits=None)


def test_state_fields_survive_the_reset(tmp_path):
    pytest.importorskip("orchestral")
    from aster_toolkit.data_acquisition import mast
    tool = mast.AggregateJwstObservations(base_directory=str(tmp_path))
    tool.execute(rows_path="missing.csv", group_by="['x']")
    assert tool.base_directory == str(tmp_path)


def test_failing_call_returns_error_text_instead_of_raising(tmp_path):
    """orchestral's rollback used to raise a ValidationError out of execute()
    when _run failed on a tool with required fields."""
    pytest.importorskip("orchestral")
    from aster_toolkit.data_acquisition import mast
    tool = mast.AggregateJwstObservations(base_directory=str(tmp_path))
    out = tool.execute(rows_path="missing.csv", group_by="['x']")
    assert out.startswith("Error: Execution Error")
    out = tool.execute(group_by="['x']")             # next call: still sane
    assert "Missing Required Fields" in out and "rows_path" in out
