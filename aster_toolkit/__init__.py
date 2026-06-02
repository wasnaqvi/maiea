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
