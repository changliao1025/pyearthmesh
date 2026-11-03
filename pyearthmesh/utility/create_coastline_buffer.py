import os
from typing import Optional

from hexwatershed_utility.preprocess.feature.coastline import (
    create_land_ocean_mask_from_naturalearth,
)
from osgeo import ogr


def create_coastline_buffer(
    dThreshold_area_island,
    dDistance_buffer_meter,
    sFilename_out,
    iFlag_antarctic_in=0,
    sResolution_nature_earth_coastline_in="110m",
    iFlag_simplify_in=1,
    iFlag_simplify_coastline_in=0,
    dTolerance_simplify_in=0.001,
    iFlag_verbose_in=0,
):
    """
    Create buffer zones around coastlines from Natural Earth data using hexwatershed_utility.

    Parameters
    ----------
    dThreshold_area_island : float
        Minimum island area in square meters to include
    dDistance_buffer_meter : float
        Buffer distance in meters
    sFilename_out : str
        Output file path
    iFlag_antarctic_in : int, optional
        Include Antarctic coastline (default: 0 = no)
    sResolution_nature_earth_coastline_in : str, optional
        Natural Earth resolution: "110m", "50m", or "10m" (default: "110m")
    iFlag_simplify_in : int, optional
        Buffer simplification mode (default: 1 = dissolve and split)
        - 0: No simplification (individual overlapping buffers)
        - 1: Dissolve overlapping buffers and split into individual polygons
    iFlag_simplify_coastline_in : int, optional
        Coastline vertex simplification flag (default: 0 = no simplification)
    dTolerance_simplify_in : float, optional
        Tolerance in degrees for coastline simplification (default: 0.001)
    iFlag_verbose_in : int, optional
        Verbosity level (default: 0)

    Returns
    -------
    int
        Number of buffer features created
    """
    sWorkspace_output = os.path.dirname(sFilename_out)
    if not sWorkspace_output:
        sWorkspace_output = os.getcwd()
    os.makedirs(sWorkspace_output, exist_ok=True)

    _, sFilename_vector = create_land_ocean_mask_from_naturalearth(
        sWorkspace_output=sWorkspace_output,
        sFilename_land_vector_in=None,
        sResolution_nature_earth=sResolution_nature_earth_coastline_in,
        iFlag_antarctic=iFlag_antarctic_in,
        dThreshold_area_island=dThreshold_area_island,
        iFlag_simplify_coastline=bool(iFlag_simplify_coastline_in),
        dTolerance_simplify=dTolerance_simplify_in,
        dDistance_buffer_meter=dDistance_buffer_meter,
        iFlag_simplify_buffer=iFlag_simplify_in,
        sFilename_vector_out=sFilename_out,
        iFlag_convert_raster=False,
        verbose=bool(iFlag_verbose_in),
    )

    n_buffered = 0
    pDs = ogr.Open(sFilename_vector, 0)
    if pDs is not None:
        pLayer = pDs.GetLayer()
        if pLayer is not None:
            n_buffered = pLayer.GetFeatureCount()
        pDs = None

    return n_buffered
