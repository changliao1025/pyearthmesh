
import os
from pyearthmesh.meshes.structured.dggs.dggrid.create_dggrid_mesh import create_dggrid_mesh,  dggrid_find_index_by_resolution, dggrid_find_resolution_by_index, copy_dggrid_binaries_to_output
from pyearthmesh.utility.mesh_utility import check_mesh_quality
from pyearth.toolbox.conversion.convert_vector_format import convert_vector_format

iFlag_create_mesh = 1

dResolution_meter = 1000  #10 km
sResoluton = str(int(dResolution_meter/ 1000)) + "km"
sWorkspace_output_base = '/compyfs/liao313/04model/pyearthmesh/'
sWorkspace_output = os.path.join(sWorkspace_output_base, 'isea3h', sResoluton)
if not os.path.exists(sWorkspace_output):
    os.makedirs(sWorkspace_output)

sWorkspace_output_dggrid = os.path.join(sWorkspace_output, 'dggrid')
if not os.path.exists(sWorkspace_output_dggrid):
    os.makedirs(sWorkspace_output_dggrid)

sFilename_mesh_base = "isea3h_mesh.geoparquet"
sFilename_mesh = os.path.join(sWorkspace_output, sFilename_mesh_base)

#convert to geoparquet for faster read/write
sFilename_mesh_parquet = os.path.join(sWorkspace_output, "isea3h_mesh.geoparquet")
print(sFilename_mesh_parquet)
iFlag_global = 1
iFlag_save_mesh= 1

sDggrid_type = 'ISEA3H'
iResolution_index_in = dggrid_find_index_by_resolution(sDggrid_type, dResolution_meter)
dResolution_meter = dggrid_find_resolution_by_index(sDggrid_type, iResolution_index_in)
print("Using resolution " + str(dResolution_meter) + " meters, index " + str(iResolution_index_in) )
#copy dggrid binary files to output folder
copy_dggrid_binaries_to_output(sWorkspace_output_dggrid)

create_dggrid_mesh(iFlag_global,
    iFlag_save_mesh,
    sFilename_mesh,
    sWorkspace_output_dggrid,  # for dggrid
    iResolution_index_in=iResolution_index_in,
    sDggrid_type_in='ISEA3H')

#convert_vector_format(sFilename_mesh, sFilename_mesh_parquet, use_ogr2ogr = True)
sFilename_mesh_new = os.path.join(sWorkspace_output, "isea3h_mesh_fixed.geoparquet")
sFilename_mesh_new = check_mesh_quality( sFilename_mesh_parquet )

print("Finished creating isea3h mesh at resolution " + sResoluton)