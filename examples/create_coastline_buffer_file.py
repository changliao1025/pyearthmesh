import os

from pyearthmesh.utility.create_coastline_buffer import create_coastline_buffer

dThreshold_area_island= 1000 * 1000 * 10 #50 km2
dDistance_buffer_meter= 10000 #5 km

sResolution_buffer = str(int(dDistance_buffer_meter/ 1000)) + "km"

sResolution_nature_earth_coastline = "10m"
sWorkspace_out = '/compyfs/liao313/04model/pyearthmesh/healpix/'
if not os.path.exists(sWorkspace_out):
    os.makedirs(sWorkspace_out)

sFilename_out= os.path.join(sWorkspace_out, 'coastline_buffer_' + sResolution_nature_earth_coastline + '_' + sResolution_buffer + '.parquet')

create_coastline_buffer(dThreshold_area_island,
                        dDistance_buffer_meter,
                        sFilename_out,
                        iFlag_antarctic_in=0,
                        iFlag_verbose_in=1,
                        sResolution_nature_earth_coastline_in = sResolution_nature_earth_coastline )