
import os
from pyearthmesh.meshes.structured.dggs.isea.create_isea_mesh import create_isea_mesh
from pyearthmesh.utility.mesh_utility import check_mesh_quality
from pyearth.toolbox.conversion.convert_vector_format import convert_vector_format

iFlag_create_mesh = 1

dResolution_meter = 1000  #10 km
sResoluton = str(int(dResolution_meter/ 1000)) + "km"
sWorkspace_output_base = '/compyfs/liao313/04model/pyearthmesh/'
sWorkspace_output = os.path.join(sWorkspace_output_base, 'isea3h', sResoluton)
if not os.path.exists(sWorkspace_output):
    os.makedirs(sWorkspace_output)

sFilename_mesh = os.path.join(sWorkspace_output, "isea3h_mesh.geojson")
#convert to parquet for faster read/write
sFilename_mesh_parquet = os.path.join(sWorkspace_output, "isea3h_mesh.parquet")
print(sFilename_mesh_parquet)

create_isea_mesh(dResolution_meter,
        'isea3h',
        sFilename_mesh,
        sWorkspace_output,)

convert_vector_format(sFilename_mesh, sFilename_mesh_parquet, use_ogr2ogr = True)
sFilename_mesh_new = os.path.join(sWorkspace_output, "isea3h_mesh_fixed.parquet")
sFilename_mesh_new = check_mesh_quality( sFilename_mesh_parquet )

print("Finished creating isea3h mesh at resolution " + sResoluton)