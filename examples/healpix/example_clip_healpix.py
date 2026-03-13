
import os

from pyearthmesh.meshes.structured.dggs.healpix.create_rhealpix_mesh import create_rhealpix_mesh
from pyearth.toolbox.management.vector.remove_small_polygon import remove_small_polygon
from pyearth.toolbox.management.vector.merge_features import merge_features

from pyearth.toolbox.conversion.convert_vector_to_global_raster import convert_vector_to_global_raster
from pyearth.toolbox.data.ocean.define_land_ocean_mask import create_land_ocean_vector_mask_naturalearth
from pyearthmesh.utility.extract_land_meshcells import extract_land_meshcells, extract_land_meshcells_multi_tile
from pyearthmesh.utility.mesh_utility import check_mesh_quality
from pyearth.toolbox.conversion.convert_vector_format import convert_vector_format
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_driver_from_format, get_vector_format_from_extension
from pyearthmesh.utility.create_coastline_buffer import create_coastline_buffer


dResolution_meter = 1000  #1 km
sResoluton = str(int(dResolution_meter/ 1000)) + "km"

dThreshold_area_island= 1000 * 1000 * 10 #50 km2
sResolution_nature_earth_coastline = "10m"
dDistance_buffer_meter= 5000 #5 km
sResolution_buffer = str(int(dDistance_buffer_meter/ 1000)) + "km"
case_index = 2
iFlag_use_tiles = 1
sCase_index = str(case_index).zfill(2)

sFilename_mesh = "healpix_mesh_fixed.parquet"
#get extension using the filename only
sExtension_mesh = os.path.splitext(sFilename_mesh)[1].lower()
#now use the extension to determine the gdal format
if sExtension_mesh == ".nc":
    sFormat_gdal = "NetCDF"
else:
    sFormat_gdal = get_vector_format_from_extension(sExtension_mesh)

sFormat_gdal_safe = sFormat_gdal.replace(' ', '_')

sWorkspace_output_base = '/compyfs/liao313/04model/pyearthmesh/'
sWorkspace_output = os.path.join(sWorkspace_output_base, 'healpix', sResoluton, sFormat_gdal_safe, sCase_index)
if not os.path.exists(sWorkspace_output):
    os.makedirs(sWorkspace_output)

sWorkspace_coastline_output = os.path.join(sWorkspace_output, 'coastline')
if not os.path.exists(sWorkspace_coastline_output):
    os.makedirs(sWorkspace_coastline_output)


if iFlag_use_tiles == 1:
    sWorkspace_input = os.path.join(sWorkspace_output, 'fixed')
    sWorkspace_output = sWorkspace_coastline_output
    #sFilename_coastline= '/compyfs/liao313/04model/pyearthmesh/healpix/coastline_buffer.parquet'
    extract_land_meshcells_multi_tile(sWorkspace_input, sWorkspace_output,
                                      sExtension_mesh,
                                      sFilename_coastline,
                                      sExtension_out='.geojson',
                                      iFlag_parallel=0,
                                      nWorker_in=4
                                      )

else:
    sFilename_mesh_new = os.path.join(sWorkspace_output, "healpix_mesh_fixed.geojson")
    sFilename_mesh_parquet = os.path.join(sWorkspace_output, "healpix_mesh_fixed.parquet")
    #convert_vector_format(sFilename_mesh_new, sFilename_mesh_parquet, use_ogr2ogr = True)
    sFilename_mesh_land_out = os.path.join(sWorkspace_output, "healpix_mesh_land.parquet")
    sFilename_mesh_touch_out = os.path.join(sWorkspace_output, "healpix_mesh_touch.parquet")


    sFilename_land_ocean_mask = os.path.join(sWorkspace_coastline_output, 'land_ocean_mask_wo_island_w_buffer.geojson')
    create_coastline_buffer(dThreshold_area_island,
                        dDistance_buffer_meter,
                        sFilename_land_ocean_mask,
                        iFlag_antarctic_in=0,
                        iFlag_verbose_in=1 )


    extract_land_meshcells(sFilename_mesh_parquet, sFilename_land_ocean_mask, sFilename_mesh_land_out, sFilename_mesh_touch_out = sFilename_mesh_touch_out)