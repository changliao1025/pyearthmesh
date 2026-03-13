
import os
import shutil
from pyearth.toolbox.conversion.convert_vector_format import convert_vector_format
from pyearth.gis.gdal.gdal_vector_format_support import get_extension_from_vector_format
from pyearthmesh.meshes.structured.dggs.dggrid.create_dggrid_mesh import create_dggrid_mesh,  dggrid_find_index_by_resolution, dggrid_find_resolution_by_index, copy_dggrid_binaries_to_output, dggrid_get_total_number_of_cells
from pyearthmesh.meshes.structured.dggs.convert_vector_format_multi_tile import convert_multi_tile_vector_format
from pyearthmesh.utility.mesh_utility import check_mesh_quality


iFlag_create_mesh = 1
iFlag_use_tiles = 1
dResolution_meter = 1000  #10 km
sFormat_gdal = 'GeoJSON'  # support any gdal format, e.g., 'GeoJSON', 'GPKG', 'Parquet', etc. Note that some formats (e.g., shapefile) have limitations on the number of features per file, so for high resolution meshes, it's recommended to use formats that support larger files (e.g., GeoJSON, GPKG, Parquet). Parquet is particularly efficient for large datasets and is recommended for high-resolution meshes.
sFormat_gdal = 'ESRI Shapefile'  # for testing shapefile output, which has limitations on file size and number of features, so it will be split into multiple files if the mesh is too large. Note that shapefile format does not support more than 2^31-1 bytes per file, which is roughly around 10 million cells depending on the attributes. For very high resolution meshes with hundreds of millions or billions of cells, you may want to use a more efficient format like GeoJSON, GPKG, or Parquet instead of shapefile to avoid issues with file size and memory usage when writing the files.
#if the output format is shapefile, we dont need to use gdal, we can use DGGRID built in shapefile

sExtension_mesh = get_extension_from_vector_format(sFormat_gdal)

sResoluton = str(int(dResolution_meter/ 1000)) + "km"
sWorkspace_output_base = '/compyfs/liao313/04model/pyearthmesh/'
# Replace spaces with underscores in format name for directory path to avoid path issues
sFormat_gdal_safe = sFormat_gdal.replace(' ', '_')
sWorkspace_output = os.path.join(sWorkspace_output_base, 'isea3h', sResoluton, sFormat_gdal_safe)
if not os.path.exists(sWorkspace_output):
    os.makedirs(sWorkspace_output)
else:
    # remove directory and its contents if present (os.rmdir fails if not empty)
    shutil.rmtree(sWorkspace_output)
    os.makedirs(sWorkspace_output)

sWorkspace_output_dggrid = os.path.join(sWorkspace_output, 'dggrid')
if not os.path.exists(sWorkspace_output_dggrid):
    os.makedirs(sWorkspace_output_dggrid)

sFilename_mesh_base = "isea3h_mesh" + sExtension_mesh  # support any gdal format
sFilename_mesh = os.path.join(sWorkspace_output_dggrid, sFilename_mesh_base)

#convert to parquet for faster read/write
sFilename_mesh_parquet = os.path.join(sWorkspace_output, "isea3h_mesh.parquet")
print(sFilename_mesh_parquet)
if os.path.exists(sFilename_mesh_parquet):
    os.remove(sFilename_mesh_parquet)

iFlag_global = 1
iFlag_save_mesh= 1

sDggrid_type = 'ISEA3H'
iResolution_index_in = dggrid_find_index_by_resolution(sDggrid_type, dResolution_meter)
dResolution_meter = dggrid_find_resolution_by_index(sDggrid_type, iResolution_index_in)
print("Using resolution " + str(dResolution_meter) + " meters, index " + str(iResolution_index_in) )
#copy dggrid binary files to output folder
copy_dggrid_binaries_to_output(sWorkspace_output_dggrid, sFilename_executable_in = 'dggrid')

#find out the total number of cells for the given resolution, and determine how many cells to write to each output file
nCell_total = dggrid_get_total_number_of_cells(sDggrid_type, iResolution_index_in)
lMax_cells_per_output_file =  20000000  # set to 2 million cells per file by default, can adjust based on the total number of cells and the output format limitations.
#For example, shapefile has a limit of 2^31-1 bytes per file, which is roughly around 10 million cells depending on the attributes. For very high resolution meshes with hundreds of millions or billions of cells, you may want to set this to a lower value (e.g., 1 million) to avoid issues with file size and memory usage when writing the files. For lower resolution meshes with fewer cells, you can set this to a higher value to reduce the number of output files.
create_dggrid_mesh(iFlag_global,
    iFlag_save_mesh,
    sFilename_mesh,
    sWorkspace_output_dggrid,  # for dggrid
    iResolution_index_in=iResolution_index_in,
    lMax_cells_per_output_file = lMax_cells_per_output_file,
    sDggrid_type_in='ISEA3H')

if iFlag_use_tiles ==1:
    #convert one by one
    sWorkspace_in = sWorkspace_output_dggrid
    sWorkspace_out = os.path.join(sWorkspace_output, 'dggrid_convert')
    if not os.path.exists(sWorkspace_out):
        os.makedirs(sWorkspace_out)
    sExtension_in= sExtension_mesh
    sExtension_out = '.parquet'
    convert_multi_tile_vector_format(sWorkspace_in, sWorkspace_out, sExtension_in, sExtension_out)

else:
    #convert_vector_format(sFilename_mesh, sFilename_mesh_parquet, use_ogr2ogr = True)
    sFilename_mesh_new = os.path.join(sWorkspace_output, "isea3h_mesh_fixed.parquet")
    #sFilename_mesh_new = check_mesh_quality( sFilename_mesh_parquet )

print("Finished creating isea3h mesh at resolution " + sResoluton)