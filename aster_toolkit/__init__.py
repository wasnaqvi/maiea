"""
ASTER Tools Package

All tools for the Agentic Science Toolkit for Exoplanet Research.
"""
_EXPORTS = {
    'RunTaurexModelTool': '.taurex.forward_model',
    'SetTaurexPaths': '.taurex.set_paths',
    'SimulateTaurexRetrieval': '.taurex.retrieval',
    'PlotCornerPosteriors': '.taurex.corner_plot',
    'QueryExoplanetArchive': '.data_acquisition.exoarchive',
    'DownloadDataset': '.data_acquisition.exoarchive',
    'SearchMastJwstObservations': '.data_acquisition.mast',
    'DownloadMastJwstProducts': '.data_acquisition.mast',
    'GetJwstProgramInfo': '.data_acquisition.mast',
    'AsterAlert': '.data_acquisition.alert',
    'DiscoverPatchworkVisits': '.data_reduction.discover',
    'VerifyPatchworkEnvironment': '.data_reduction.exotedrf',
    'ReduceNirspecG395hTso': '.data_reduction.exotedrf',
    'MakeUncalTestSubset': '.data_reduction.exotedrf',
    'DetectTiltEvents': '.data_reduction.lightcurves',
    'FitNirspecG395hWhiteLight': '.data_reduction.juliet',
    'FitNirspecG395hTransmissionSpectrum': '.data_reduction.juliet',
    'CombineNirspecG395hVisits': '.data_reduction.juliet',
    'DetectLightCurveAnomalies': '.data_reduction.contamination',
    'ModelStellarContamination': '.data_reduction.contamination',
    'RunPatchworkTarget': '.data_reduction.survey',
    'GenerateFirJobs': '.data_reduction.survey',
    'OptimizeNirspecG395hReduction': '.data_reduction.optimize',
    'SummarizeG395hOptimization': '.data_reduction.optimize',
}

__all__ = [
    'RunTaurexModelTool',
    'SetTaurexPaths',
    'SimulateTaurexRetrieval',
    'PlotCornerPosteriors',
    'QueryExoplanetArchive',
    'DownloadDataset',
    'SearchMastJwstObservations',
    'DownloadMastJwstProducts',
    'GetJwstProgramInfo',
    'AsterAlert',
    'DiscoverPatchworkVisits',
    'VerifyPatchworkEnvironment',
    'ReduceNirspecG395hTso',
    'MakeUncalTestSubset',
    'DetectTiltEvents',
    'FitNirspecG395hWhiteLight',
    'FitNirspecG395hTransmissionSpectrum',
    'CombineNirspecG395hVisits',
    'DetectLightCurveAnomalies',
    'ModelStellarContamination',
    'RunPatchworkTarget',
    'GenerateFirJobs',
    'OptimizeNirspecG395hReduction',
    'SummarizeG395hOptimization',
]


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
