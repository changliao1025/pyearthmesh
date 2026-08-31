import sys, os
sPath_project = '/qfs/people/liao313/workspace/python/hrm'
#add the project path of the pythonpath

sys.path.append(sPath_project)


import numpy as np
from osgeo import ogr, osr, gdal
import cartopy.crs as ccrs

from pyearth.gis.gdal.read.vector.gdal_get_vector_boundary import gdal_get_vector_boundary

from pyearthviz.map.vector.map_vector_polygon_file import map_vector_polygon_file
from pyearthviz.map import RasterTileServer

from pyearthviz.color.pick_colormap import pick_colormap_hydrology


dLongitude_left_in=99.5
dLatitude_bot_in=36.5

dLongitude_right_in=101
dLatitude_top_in=37.5


sColormap = pick_colormap_hydrology('nse')
iFiletype_in=1
sFilename_in='C:\\Users\\chang\\workspace\\python\\pyearthmesh\\data\\qinghaihu\\lake_latlon.geojson'
sFilename_boundary = 'C:\\Users\\chang\\data\\modeldata\\hexwatershed\\qinghaihu\\vector\\contributing_polygons_qinghaihu_merged.geojson'



pBoundary_wkt, aExtent = gdal_get_vector_boundary(sFilename_boundary)

image_size = [1000, 1000]
dpi = 150
scale_denominator = RasterTileServer.calculate_scale_denominator(aExtent, image_size)
pSrc = osr.SpatialReference()
pSrc.ImportFromEPSG(3857) # mercator
pProjection = pSrc.ExportToWkt()
iFlag_openstreetmap_level = RasterTileServer.calculate_zoom_level(scale_denominator, pProjection, dpi=dpi)
print(iFlag_openstreetmap_level)

pProjection_map = ccrs.Orthographic(central_longitude=0, central_latitude=0, globe=None)

sBasemap_provider = 'Tianditu.Vector'
sFilename_output_in = 'C:\\Users\\chang\\workspace\\python\\pyearthmesh\\data\\qinghaihu\\lake_latlon_map' 
sFilename_output_in = sFilename_output_in + sBasemap_provider+ '.png'

map_vector_polygon_file( sFilename_in,                                       
                          sFilename_output_in= sFilename_output_in,                                                                 
                       aBasemap_provider_in=[sBasemap_provider],
                          iFlag_zebra_in = 1,
                          iFlag_fill_in = 0,
                          iDPI_in=None,
                          dMissing_value_in=None,
                          #aLegend_in=['(b)'],
                          aExtent_in = aExtent,
                           pProjection_map_in=None)