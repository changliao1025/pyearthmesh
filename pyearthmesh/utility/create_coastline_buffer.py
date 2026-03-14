import os
from pyearth.toolbox.data.ocean.define_land_ocean_mask import create_land_ocean_vector_mask_naturalearth
from pyearth.toolbox.management.vector.remove_small_polygon import remove_small_polygon
from pyearthbuffer.utility.create_gcs_buffer_zone import create_buffer_zone_polygon_file

def create_coastline_buffer(dThreshold_area_island, dDistance_buffer_meter, sFilename_out,
                            iFlag_antarctic_in=0, sResolution_nature_earth_coastline_in = "110m",
                            iFlag_simplify_in=1, iFlag_verbose_in=0):
    """
    Create buffer zones around coastlines from Natural Earth data.

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
    iFlag_verbose_in : int, optional
        Verbosity level (default: 0)
    """

    #find the output directory
    sWorkspace_coastline_output = os.path.dirname(sFilename_out)

    #get natual earth coastline data using the
    sFilename_naturalearth = os.path.join(sWorkspace_coastline_output, 'land_ocean_mask_naturalearth.parquet')
    create_land_ocean_vector_mask_naturalearth(sFilename_naturalearth, sResolution_coastal = sResolution_nature_earth_coastline_in)
    #remove small islands
    sFilename_wo_island = os.path.join(sWorkspace_coastline_output, 'land_ocean_mask_wo_island.parquet')
    remove_small_polygon(sFilename_naturalearth, sFilename_wo_island, dThreshold_area_island )

    #create buffer with simplification option
    create_buffer_zone_polygon_file(sFilename_wo_island, sFilename_out,
                                   dThreshold_area_island, dDistance_buffer_meter,
                                   iFlag_simplify=iFlag_simplify_in,
                                   verbose=iFlag_verbose_in)

    return

