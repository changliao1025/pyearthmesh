
import os

from pyearthmesh.meshes.structured.dggs.healpix.create_rhealpix_mesh import create_rhealpix_mesh
from pyearth.toolbox.management.vector.remove_small_polygon import remove_small_polygon
from pyearth.toolbox.management.vector.merge_features import merge_features
from pyearth.toolbox.geometry.create_gcs_buffer_zone import create_buffer_zone_polygon_file
from pyearth.toolbox.conversion.convert_vector_to_global_raster import convert_vector_to_global_raster
from pyearth.toolbox.data.ocean.define_land_ocean_mask import create_land_ocean_vector_mask_naturalearth
from pyearthmesh.utility.extract_land_meshcells import extract_land_meshcells
from pyearthmesh.utility.mesh_utility import check_mesh_quality
from pyearth.toolbox.conversion.convert_vector_format import convert_vector_format


dResolution_meter = 10000  #10 km
sResoluton = str(int(dResolution_meter/ 1000)) + "km"

sWorkspace_output_base = '/compyfs/liao313/04model/pyearthmesh/'
sWorkspace_output = os.path.join(sWorkspace_output_base, 'rhealpix', sResoluton)
if not os.path.exists(sWorkspace_output):
    os.makedirs(sWorkspace_output)

sWorkspace_coastline_output = os.path.join(sWorkspace_output, 'coastline')
if not os.path.exists(sWorkspace_coastline_output):
    os.makedirs(sWorkspace_coastline_output)

sFilename_mesh_new = os.path.join(sWorkspace_output, "rhealpix_mesh_fixed.geoparquet")
sFilename_mesh_land_out = os.path.join(sWorkspace_output, "rhealpix_mesh_land.geoparquet")
sFilename_mesh_touch_out = os.path.join(sWorkspace_output, "rhealpix_mesh_touch.geoparquet")

#prepare coastline for cutting rhealpix mesh
sFilename_naturalearth = os.path.join(sWorkspace_coastline_output, 'land_ocean_mask_naturalearth.geojson')
create_land_ocean_vector_mask_naturalearth(sFilename_naturalearth)
dThreshold_area_island = dResolution_meter * dResolution_meter
sFilename_wo_island = os.path.join(sWorkspace_coastline_output, 'land_ocean_mask_wo_island.geojson')
remove_small_polygon(sFilename_naturalearth, sFilename_wo_island, dThreshold_area_island )
#extract the mesh using the coastline polygon
extract_land_meshcells(sFilename_mesh_new, sFilename_wo_island, sFilename_mesh_land_out, sFilename_mesh_touch_out = sFilename_mesh_touch_out)