
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


dResolution_meter = 1000  #10 km

sWorkspace_output_base = '/compyfs/liao313/04model/pyearthmesh/'
sWorkspace_output = os.path.join(sWorkspace_output_base, 'rhealpix')
if not os.path.exists(sWorkspace_output):
    os.makedirs(sWorkspace_output)



sFilename_mesh = os.path.join(sWorkspace_output, "rhealpix_mesh.geojson")
#convert to geoparquet for faster read/write
sFilename_mesh_parquet = os.path.join(sWorkspace_output, "rhealpix_mesh.geoparquet")
print(sFilename_mesh_parquet)

create_rhealpix_mesh(dResolution_meter,
        sFilename_mesh,
        sWorkspace_output,)

convert_vector_format(sFilename_mesh, sFilename_mesh_parquet, use_ogr2ogr = True)

sFilename_mesh_new = check_mesh_quality( sFilename_mesh_parquet )
#sFilename_mesh_new = os.path.join(sWorkspace_output, "rhealpix_mesh_fixed.geoparquet")
