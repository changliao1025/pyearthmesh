
import os
from pyearth.toolbox.conversion.convert_vector_format import convert_vector_format
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_driver_from_format, get_vector_format_from_extension
from pyearthmesh.meshes.structured.dggs.check_mesh_quality_multi_tile import check_mesh_quality_multi_tile
from pyearthmesh.meshes.structured.dggs.healpix.create_healpix_mesh import create_healpix_mesh
from pyearthmesh.utility.mesh_utility import check_mesh_quality
from pyearthmesh.meshes.structured.dggs.convert_vector_format_multi_tile import convert_vector_format_multi_tile

dResolution_meter = 1000  #1 km
iFlag_create_mesh = 1
iFlag_use_tiles = 1
case_index = 4
sFilename_mesh = "healpix_mesh.parquet"
#get extension using the filename only
sExtension_mesh = os.path.splitext(sFilename_mesh)[1].lower()
#now use the extension to determine the gdal format
if sExtension_mesh == ".nc":
    sFormat_gdal = "NetCDF"
else:
    sFormat_gdal = get_vector_format_from_extension(sExtension_mesh)

sFormat_gdal_safe = sFormat_gdal.replace(' ', '_')

sCase_index = str(case_index).zfill(2)

sWorkspace_output_base = '/compyfs/liao313/04model/pyearthmesh/'
sResoluton = str(int(dResolution_meter/ 1000)) + "km"
#sWorkspace_output = os.path.join(sWorkspace_output_base, 'healpix')
sWorkspace_output = os.path.join(sWorkspace_output_base, 'healpix', sResoluton, sFormat_gdal_safe, sCase_index)
if not os.path.exists(sWorkspace_output):
    os.makedirs(sWorkspace_output)

if iFlag_create_mesh == 1:
    create_healpix_mesh(dResolution_meter,
        sFilename_mesh,
        sWorkspace_output,
        iFlag_use_tiles=iFlag_use_tiles,
        tile_mode='base')

if iFlag_use_tiles == 1:
    sWorkspace_in = sWorkspace_output
    sWorkspace_out = os.path.join(sWorkspace_output, 'fixed')
    if not os.path.exists(sWorkspace_out):
        os.makedirs(sWorkspace_out)
    sExtension_in= sExtension_mesh
    sExtension_out = '.parquet'
    check_mesh_quality_multi_tile(sWorkspace_in, sWorkspace_out, sExtension_in,
                                   sExtension_out, iParallel_in=1,nWorker_in=4)

else:
    #convert to parquet for faster read/write
    sFilename_mesh = os.path.join(sWorkspace_output, sFilename_mesh)
    if sExtension_mesh != '.parquet' and sExtension_mesh != '.parquet' and sExtension_mesh != '.nc':
        print("converting to parquet for faster read/write")
        sFilename_mesh_parquet = os.path.join(sWorkspace_output, "healpix_mesh.parquet")
        print(sFilename_mesh_parquet)
        convert_vector_format(sFilename_mesh, sFilename_mesh_parquet, use_ogr2ogr = True)
    else:
        sFilename_mesh_parquet = sFilename_mesh

    if sExtension_mesh != '.nc':
        sFilename_mesh_new = check_mesh_quality( sFilename_mesh_parquet , iFlag_verbose_in=1)
        #sFilename_mesh_new = os.path.join(sWorkspace_output, "healpix_mesh_fixed.parquet")
        print(sFilename_mesh_new)
