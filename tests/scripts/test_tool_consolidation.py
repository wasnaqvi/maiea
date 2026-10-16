"""Pins the consolidated ASTER tool surface (one tool per task, no overlap).

Covers the tools merged in the consolidation pass that are not exercised by
test_mast.py: QueryExoplanetArchive, GenerateFirJobs, the segment preflight
inside ReduceNirspecG395hTso, and the contamination-backend report inside
VerifyPatchworkEnvironment. Pure-local: no network, no exoTEDRF env.
"""
import ast
import json
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
    # merged into SearchMastJwstObservations (population_preset / obs_types /
    # group_by / rows_path)
    "CrossmatchJwstToPlanets", "AggregateJwstObservations",
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

def test_exports_are_the_24_consolidated_tools():
    assert len(aster_toolkit.__all__) == 24
    assert len(set(aster_toolkit.__all__)) == 24
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
    tool = mast.SearchMastJwstObservations(base_directory=str(tmp_path))
    tool.execute(rows_path="missing.csv", group_by="['x']")
    assert tool.base_directory == str(tmp_path)


def test_failing_call_returns_error_text_instead_of_raising(tmp_path):
    """orchestral's rollback used to raise a ValidationError out of execute()
    when _run failed on a tool with required fields."""
    pytest.importorskip("orchestral")
    from aster_toolkit.data_acquisition import mast
    tool = mast.GetJwstProgramInfo(base_directory=str(tmp_path))
    out = tool.execute(proposal_ids="[]")
    assert out.startswith("Error: Execution Error")
    out = tool.execute()                             # next call: still sane
    assert "Missing Required Fields" in out and "proposal_ids" in out


# ---------------- schema contract: exactly these arguments are required ----------------

# What the MODEL is told it must supply. orchestral treats
# RuntimeField(default=None) as required, so an optional argument declared that
# way silently becomes mandatory (DownloadDataset used to require all three of
# its alternative inputs, so every normal call failed). Changing this table is
# a deliberate API change.
EXPECTED_REQUIRED = {
    "RunTaurexModelTool": [],
    "SimulateTaurexRetrieval": ["fit_params", "observation_path"],
    "PlotCornerPosteriors": ["labels", "retrieval_basename"],
    "QueryExoplanetArchive": [],
    "DownloadDataset": [],
    "SearchMastJwstObservations": [],
    "DownloadMastJwstProducts": [],
    "GetJwstProgramInfo": ["proposal_ids"],
    "AsterAlert": [],
    "DiscoverPatchworkVisits": ["raw_root"],
    "VerifyPatchworkEnvironment": [],
    "ReduceNirspecG395hTso": ["input_dir"],
    "MakeUncalTestSubset": ["output_path", "uncal_path"],
    "DetectTiltEvents": ["duration_hr", "period", "spectra_path", "t0_ref"],
    "FitNirspecG395hWhiteLight": ["output_dir", "planet_name", "spectra_path"],
    "FitNirspecG395hTransmissionSpectrum": ["output_dir", "planet_name", "spectra_path",
                                            "white_fit_dir"],
    "CombineNirspecG395hVisits": ["output_dir", "spectra_by_detector"],
    "DetectLightCurveAnomalies": ["output_dir", "white_fit_dirs"],
    "ModelStellarContamination": ["output_dir", "spectrum_csv"],
    "RunPatchworkTarget": ["manifest_path", "output_root"],
    "GenerateFirJobs": ["manifest_path", "output_root"],
    "OptimizeNirspecG395hReduction": ["detector", "input_dir", "name_tag", "output_dir"],
    "SummarizeG395hOptimization": ["cost_table"],
}


def test_required_arguments_match_the_contract():
    pytest.importorskip("orchestral")
    pytest.importorskip("taurex")
    actual = {n: sorted(getattr(aster_toolkit, n)._get_required_fields())
              for n in aster_toolkit.__all__ if n != "SetTaurexPaths"}
    assert actual == EXPECTED_REQUIRED


def test_schema_required_list_matches_execute_enforcement():
    """What the model is SHOWN as required (the JSON schema orchestral sends)
    must equal what execute() ENFORCES."""
    pytest.importorskip("orchestral")
    pytest.importorskip("taurex")
    from orchestral.llm.openai.tool_adapter import convert_tool_to_openai
    for name in EXPECTED_REQUIRED:
        cls = getattr(aster_toolkit, name)
        tool = cls(base_directory="/tmp") if "base_directory" in cls.model_fields else cls()
        schema = convert_tool_to_openai(tool.get_tool_spec())["function"]["parameters"]
        assert sorted(schema.get("required", [])) == EXPECTED_REQUIRED[name], name


def test_blank_strings_mean_not_provided(tmp_path):
    """Models send "" for arguments they mean to skip; for optional arguments
    that must fall back to the default, not reach the tool as a value."""
    pytest.importorskip("orchestral")
    from aster_toolkit.data_acquisition import mast
    seen = {}

    def fake_listing(groups, **kw):
        return "LIST"

    tool = mast.DownloadMastJwstProducts(base_directory=str(tmp_path))
    with patch.object(mast, "_format_product_listing", fake_listing), \
         patch.object(mast.DownloadMastJwstProducts, "_query_observations",
                      lambda self, *a: seen.update(label=self.label, filters=self.filters,
                                                   output_dir=self.output_dir) or []):
        tool.execute(proposal_id="1366", label="  ", filters="", output_dir="",
                     list_only=True)
    assert seen == {"label": None, "filters": None, "output_dir": None}  # all defaults


def test_blank_string_falls_back_to_a_non_none_default(tmp_path):
    pytest.importorskip("orchestral")
    from aster_toolkit.data_reduction import survey
    m = tmp_path / "m.json"
    m.write_text('{"planet_name": "T b", "visits": {"o001": "/r"}}')
    out = survey.GenerateFirJobs(base_directory=str(tmp_path)).execute(
        manifest_path=str(m), output_root="/s", job="  ", account="")
    assert "job=patchwork" in out                       # job "" -> "patchwork"
    assert "--account=def-ncowan" in next(tmp_path.glob("*.sbatch")).read_text()


def test_download_dataset_needs_only_one_of_its_inputs(tmp_path):
    pytest.importorskip("orchestral")
    from aster_toolkit.data_acquisition import exoarchive
    tool = exoarchive.DownloadDataset(base_directory=str(tmp_path))
    out = tool.execute(wget_url="https://example.invalid/wget.bat")
    assert "Missing Required Fields" not in out


def test_retrieval_reports_parameters_by_name_when_given_as_string(tmp_path):
    """fit_params usually arrives as "['T', 'H2O']"; the summary used to zip the
    raw string, pairing best-fit values with '[', "'", 'T', ..."""
    pytest.importorskip("orchestral")
    pytest.importorskip("taurex")
    from aster_toolkit.taurex import retrieval
    fake = {"best_parameters": [1234.0, -3.5], "best_value": 42.0, "outputs": {}}
    with patch.object(retrieval, "run_taurex_retrieval", return_value=fake) as run:
        out = retrieval.SimulateTaurexRetrieval(base_directory=str(tmp_path)).execute(
            observation_path="obs.dat", fit_params="['T', 'H2O']",
            free_chem_full_molecules="[['H2O', 0.02]]")
    assert "  - T: 1234.0" in out and "  - H2O: -3.5" in out
    assert run.call_args.kwargs["fit_params"] == ["T", "H2O"]
    assert run.call_args.kwargs["free_chem_full_molecules"] == [["H2O", 0.02]]
