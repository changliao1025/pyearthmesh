import os
from pyearth.toolbox.data.ocean.define_land_ocean_mask import create_land_ocean_vector_mask_naturalearth
from pyearth.toolbox.management.vector.remove_small_polygon import remove_small_polygon
from pyearthbuffer.utility.create_gcs_buffer_zone import create_buffer_zone_polygon_file

def create_coastline_buffer(dThreshold_area_island, dDistance_buffer_meter, sFilename_out,
                            iFlag_antarctic_in=0, sResolution_nature_earth_coastline_in = "110m",
                            iFlag_verbose_in=0):

    #find the output directory
    sWorkspace_coastline_output = os.path.dirname(sFilename_out)

    #get natual earth coastline data using the
    sFilename_naturalearth = os.path.join(sWorkspace_coastline_output, 'land_ocean_mask_naturalearth.parquet')
    create_land_ocean_vector_mask_naturalearth(sFilename_naturalearth, sResolution_coastal = sResolution_nature_earth_coastline_in)
    #removesmalle islands
    sFilename_wo_island = os.path.join(sWorkspace_coastline_output, 'land_ocean_mask_wo_island.parquet')
    remove_small_polygon(sFilename_naturalearth, sFilename_wo_island, dThreshold_area_island )

    #create buffer
    create_buffer_zone_polygon_file(sFilename_wo_island, sFilename_out,
    dThreshold_area_island, dDistance_buffer_meter,
      verbose=iFlag_verbose_in)

    return

