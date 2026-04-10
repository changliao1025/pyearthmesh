import os
from pyearth.toolbox.management.vector.merge_vector_files import merge_vector_files
from pyearthmesh.utility.extract_land_meshcells_vectorized import extract_land_meshcells_vectorized
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_format_from_extension

dResolution_meter = 1000  #1 km
sResoluton = str(int(dResolution_meter/ 1000)) + "km"

dThreshold_area_island= 1000 * 1000 * 10 #50 km2
dDistance_buffer_meter= 5000 #5 km

sResolution_nature_earth_coastline = "10m"
sResolution_buffer = str(int(dDistance_buffer_meter/ 1000)) + "km"

case_index = 4
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


sWorkspace_coastline_output = os.path.join(sWorkspace_output, 'coastline')
sWorkspace_coastline_output = os.path.abspath(sWorkspace_coastline_output)
sWorkspace_out_abs = os.path.join(sWorkspace_output, 'landoceanmask')
sWorkspace_out_abs = os.path.abspath(sWorkspace_out_abs)
if not os.path.exists(sWorkspace_out_abs):
    os.makedirs(sWorkspace_out_abs)
aFilename = list()
for root, dirs, files in os.walk(sWorkspace_coastline_output):
    # Prevent os.walk from descending into the output workspace when it's
    # inside the input workspace by removing matching dirs in-place
    if os.path.abspath(root) != sWorkspace_coastline_output:
        continue

    dirs[:] = []

    if os.path.abspath(root).startswith(sWorkspace_out_abs):
        continue

    # Sort files to ensure consistent processing order
    files.sort()
    # Exclude files that have '_land' in the name to avoid re-processing
    for file in files:
        if file.endswith(sExtension_mesh):
            sFile_in = os.path.join(root, file)
            aFilename.append(sFile_in)


sFilename_out = os.path.join(sWorkspace_out_abs, "merged_land_cells" + sExtension_mesh)

merge_vector_files(aFilename, sFilename_out, copy_attributes = True, add_id_field = False)