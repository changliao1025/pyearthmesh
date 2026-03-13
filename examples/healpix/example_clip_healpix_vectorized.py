
import os

from pyearthmesh.utility.extract_land_meshcells_vectorized import extract_land_meshcells_vectorized
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_format_from_extension

dResolution_meter = 10000  #1 km
sResoluton = str(int(dResolution_meter/ 1000)) + "km"

dThreshold_area_island= 1000 * 1000 * 10 #50 km2
dDistance_buffer_meter= 5000 #5 km

sResolution_nature_earth_coastline = "110m"
sResolution_buffer = str(int(dDistance_buffer_meter/ 1000)) + "km"

case_index = 1
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
if not os.path.exists(sWorkspace_output):
    os.makedirs(sWorkspace_output)

sWorkspace_coastline_output = os.path.join(sWorkspace_output, 'coastline')
if not os.path.exists(sWorkspace_coastline_output):
    os.makedirs(sWorkspace_coastline_output)

sWorkspace_coastline = '/compyfs/liao313/04model/pyearthmesh/healpix/'

if iFlag_use_tiles == 1:
    sWorkspace_input = os.path.join(sWorkspace_output, 'fixed')
    sWorkspace_output = sWorkspace_coastline_output
    sFilename_coastline= os.path.join(sWorkspace_coastline, 'coastline_buffer_' \
        + sResolution_nature_earth_coastline \
        + '_' + sResolution_buffer + '.parquet')
    extract_land_meshcells_vectorized(sWorkspace_input,
                                      sFilename_coastline,
                                       sWorkspace_output,
                                       sExtension_in= sExtension_mesh,
                                      sExtension_out= sExtension_mesh,
                        iFlag_multi_tile=1,
                        iParallel_in=1,
                        nWorker_in=4)