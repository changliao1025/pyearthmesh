from pyearthmesh.meshes.structured.latlon.create_latlon_mesh import create_latlon_mesh


#set an example for the qinghai lake

dLongitude_left_in=98.
dLatitude_bot_in=36.

dLongitude_right_in=101
dLatitude_top_in=37.5

dResolution_degree_in= 30/3600.0 * 10
ncolumn_in = int(3 / dResolution_degree_in)
nrow_in = int(3 / dResolution_degree_in)
sFilename_output_in= r'C:\\Users\\chang_rnrigc1\\workspace\\python\\pyearthmesh\\data\\qinghaihu\\lake_latlon_10km.geojson'
pBoundary_in = None
create_latlon_mesh(dLongitude_left_in, dLatitude_bot_in,
dResolution_degree_in, ncolumn_in, nrow_in, sFilename_output_in,
 pBoundary_in)
