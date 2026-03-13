import os, sys
from pyearth.gis.gdal.gdal_vector_format_support import  get_vector_format_from_extension
from pyearth.toolbox.management.vector.merge_vector_files import merge_vector_files

aFilename_in = list()

dResolution_meter = 1000  #1 km
sResoluton = str(int(dResolution_meter/ 1000)) + "km"

dThreshold_area_island= 1000 * 1000 * 10 #50 km2
dDistance_buffer_meter= 5000 #5 km

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
sWorkspace_in = os.path.join(sWorkspace_output, 'coastline')
for root, dirs, files in os.walk(sWorkspace_in):
    # Prevent os.walk from descending into the output workspace when it's
    # inside the input workspace by removing matching dirs in-place
    # Skip processing if current root is the output workspace (safety)
    # Sort files to ensure consistent processing order
    files.sort()
aFilename_in = files
sFilename_out = '/compyfs/liao313/04model/pyearthmesh/healpix/land_ocean_mask_w_buffer.parquet'
copy_attributes = False
add_id_field = True

merge_vector_files(aFilename_in,    sFilename_out,    copy_attributes=copy_attributes,    add_id_field=add_id_field)