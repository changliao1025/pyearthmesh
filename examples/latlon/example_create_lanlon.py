from pyearthmesh.meshes.structured.latlon.create_latlon_mesh import create_latlon_mesh


#set an example for the qinghai lake

dLongitude_left_in=99.
dLatitude_bot_in=36.

dLongitude_right_in=101
dLatitude_top_in=37.5

dResolution_degree_in=0.50
ncolumn_in = 4
nrow_in = 4
sFilename_output_in= r'C:\\Users\\chang\\workspace\\python\\pyearthmesh\\data\\qinghaihu\\lake_latlon.geojson'
pBoundary_in = None
create_latlon_mesh(dLongitude_left_in, dLatitude_bot_in, 
dResolution_degree_in, ncolumn_in, nrow_in, sFilename_output_in,
 pBoundary_in)
  