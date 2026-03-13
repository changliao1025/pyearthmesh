
import os

from pyearthmesh.meshes.structured.dggs.healpix.create_rhealpix_mesh import create_rhealpix_mesh
from pyearthmesh.utility.mesh_utility import check_mesh_quality
from pyearth.toolbox.conversion.convert_vector_format import convert_vector_format
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_driver_from_format, get_vector_format_from_extension
from pyearthmesh.meshes.structured.dggs.convert_vector_format_multi_tile import convert_multi_tile_vector_format
dResolution_meter = 10000  #10 km
iFlag_use_tiles = 1
sFilename_mesh = "rhealpix_mesh.geojson"
#get extension using the filename only
sExtension_mesh = os.path.splitext(sFilename_mesh)[1].lower()
#now use the extension to determine the gdal format
if sExtension_mesh == ".nc":
    sFormat_gdal = "NetCDF"
else:
    sFormat_gdal = get_vector_format_from_extension(sExtension_mesh)

sFormat_gdal_safe = sFormat_gdal.replace(' ', '_')

sWorkspace_output_base = '/compyfs/liao313/04model/pyearthmesh/'
sResoluton = str(int(dResolution_meter/ 1000)) + "km"
#sWorkspace_output = os.path.join(sWorkspace_output_base, 'healpix')
sWorkspace_output = os.path.join(sWorkspace_output_base, 'rhealpix', sResoluton, sFormat_gdal_safe)
if not os.path.exists(sWorkspace_output):
    os.makedirs(sWorkspace_output)



sFilename_mesh = os.path.join(sWorkspace_output, "rhealpix_mesh.geojson")
#convert to parquet for faster read/write
sFilename_mesh_parquet = os.path.join(sWorkspace_output, "rhealpix_mesh.parquet")
print(sFilename_mesh_parquet)
create_rhealpix_mesh(dResolution_meter,
        sFilename_mesh,
        sWorkspace_output,
        tile_mode= 'base',
        iFlag_use_tiles=1)

if iFlag_use_tiles == 1:
    sWorkspace_in = sWorkspace_output
    sWorkspace_out = os.path.join(sWorkspace_output, 'convert')
    if not os.path.exists(sWorkspace_out):
        os.makedirs(sWorkspace_out)
    sExtension_in= sExtension_mesh
    sExtension_out = '.parquet'
    convert_multi_tile_vector_format(sWorkspace_in, sWorkspace_out, sExtension_in, sExtension_out)

else:
    #convert to parquet for faster read/write
    sFilename_mesh = os.path.join(sWorkspace_output, sFilename_mesh)
    sFilename_mesh_parquet = os.path.join(sWorkspace_output, "healpix_mesh.parquet")
    print(sFilename_mesh_parquet)

    convert_vector_format(sFilename_mesh, sFilename_mesh_parquet, use_ogr2ogr = True)
