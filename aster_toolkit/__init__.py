"""
ASTER Tools Package

All tools for the Agentic Science Toolkit for Exoplanet Research.
"""
_EXPORTS = {
    'RunTaurexModelTool': '.taurex.forward_model',
    'SetTaurexPaths': '.taurex.set_paths',
    'SimulateTaurexRetrieval': '.taurex.retrieval',
    'PlotCornerPosteriors': '.taurex.corner_plot',
    'GetExoplanetParameters': '.data_acquisition.exoarchive',
    'DownloadDataset': '.data_acquisition.exoarchive',
    'FindExoplanetsByCondition': '.data_acquisition.exoarchive',
    'SearchMastJwstObservations': '.data_acquisition.mast',
    'GetMastObservationProducts': '.data_acquisition.mast',
    'DownloadMastJwstProducts': '.data_acquisition.mast',
    'CrossmatchJwstToPlanets': '.data_acquisition.mast',
    'AggregateJwstObservations': '.data_acquisition.mast',
    'DownloadDemographicJwstProducts': '.data_acquisition.mast',
    'GetJwstProgramInfo': '.data_acquisition.mast',
    'DiscoverPatchworkVisits': '.data_reduction.discover',
    'InspectNirspecG395hUncalData': '.data_reduction.exotedrf',
    'ReduceNirspecG395hTso': '.data_reduction.exotedrf',
    'MakeUncalTestSubset': '.data_reduction.exotedrf',
    'DetectNirspecTiltEvents': '.data_reduction.lightcurves',
    'FitNirspecG395hWhiteLight': '.data_reduction.juliet',
    'FitNirspecG395hTransmissionSpectrum': '.data_reduction.juliet',
    'CombineNirspecG395hVisits': '.data_reduction.juliet',
    'RunPatchworkTarget': '.data_reduction.survey',
    'GeneratePatchworkFirJob': '.data_reduction.survey',
    'OptimizeNirspecG395hReduction': '.data_reduction.optimize',
    'SummarizeG395hOptimization': '.data_reduction.optimize',
    'GenerateFirOptimizerJobs': '.data_reduction.optimize',
}

__all__ = [
    'RunTaurexModelTool',
    'SetTaurexPaths',
    'SimulateTaurexRetrieval',
    'PlotCornerPosteriors',
    'GetExoplanetParameters',
    'DownloadDataset',
    'FindExoplanetsByCondition',
    'SearchMastJwstObservations',
    'GetMastObservationProducts',
    'DownloadMastJwstProducts',
    'CrossmatchJwstToPlanets',
    'AggregateJwstObservations',
    'DownloadDemographicJwstProducts',
    'GetJwstProgramInfo',
    'DiscoverPatchworkVisits',
    'InspectNirspecG395hUncalData',
    'ReduceNirspecG395hTso',
    'MakeUncalTestSubset',
    'DetectNirspecTiltEvents',
    'FitNirspecG395hWhiteLight',
    'FitNirspecG395hTransmissionSpectrum',
    'CombineNirspecG395hVisits',
    'RunPatchworkTarget',
    'GeneratePatchworkFirJob',
    'OptimizeNirspecG395hReduction',
    'SummarizeG395hOptimization',
    'GenerateFirOptimizerJobs',
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

# from .taurex_tools import (
#     SimulateTaurexSpectrum,
#     SimulateTaurexRetrieval,
#     CheckTaurexOpacityCiaPaths,
#     PlotCornerPosteriors,
# )
# from .exoplanet_tools import GetExoplanetParameters
# from .data_tools import DownloadDataset

# __all__ = [
#     "SimulateTaurexSpectrum",
#     "SimulateTaurexRetrieval",
#     "CheckTaurexOpacityCiaPaths",
#     "PlotCornerPosteriors",
#     "GetExoplanetParameters",
#     "DownloadDataset",
# ]
