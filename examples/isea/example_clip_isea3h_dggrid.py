
import os

from pyearth.toolbox.management.vector.remove_small_polygon import remove_small_polygon

from pyearth.toolbox.data.ocean.define_land_ocean_mask import create_land_ocean_vector_mask_naturalearth
from pyearthmesh.utility.extract_land_meshcells import extract_land_meshcells

from pyearth.toolbox.conversion.convert_vector_format import convert_vector_format
from pyearthmesh.utility.mesh_utility import check_mesh_quality

dResolution_meter = 1000  #10 km
sResoluton = str(int(dResolution_meter/ 1000)) + "km"
sWorkspace_output_base = '/compyfs/liao313/04model/pyearthmesh/'
sWorkspace_output = os.path.join(sWorkspace_output_base, 'isea3h', sResoluton)
if not os.path.exists(sWorkspace_output):
    os.makedirs(sWorkspace_output)

sWorkspace_coastline_output = os.path.join(sWorkspace_output, 'coastline')
if not os.path.exists(sWorkspace_coastline_output):
    os.makedirs(sWorkspace_coastline_output)

sFilename_mesh_parquet = os.path.join(sWorkspace_output, 'dggrid', "isea3h_mesh.parquet")
sFilename_mesh_new = os.path.join(sWorkspace_output, "isea3h_mesh_fixed.parquet")
sFilename_mesh_land_out = os.path.join(sWorkspace_output, "isea3h_mesh_land.parquet")
sFilename_mesh_touch_out = os.path.join(sWorkspace_output, "isea3h_mesh_touch.parquet")
sFilename_mesh_new = check_mesh_quality( sFilename_mesh_parquet )

#prepare coastline for cutting isea3h mesh
sFilename_naturalearth = os.path.join(sWorkspace_coastline_output, 'land_ocean_mask_naturalearth.geojson')
create_land_ocean_vector_mask_naturalearth(sFilename_naturalearth)
dThreshold_area_island = dResolution_meter * dResolution_meter
sFilename_wo_island = os.path.join(sWorkspace_coastline_output, 'land_ocean_mask_wo_island.geojson')
remove_small_polygon(sFilename_naturalearth, sFilename_wo_island, dThreshold_area_island )
#extract the mesh using the coastline polygon
extract_land_meshcells(sFilename_mesh_new, sFilename_wo_island, sFilename_mesh_land_out, sFilename_mesh_touch_out = sFilename_mesh_touch_out)


