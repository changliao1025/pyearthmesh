from pyearthmesh.meshes.structured.dggs.convert_vector_format_multi_tile import convert_multi_tile_vector_format

sWorkspace_in='/compyfs/liao313/04model/pyearthmesh/isea3h/1km/ESRI Shapefile/dggrid'
sWorkspace_out='/compyfs/liao313/04model/pyearthmesh/isea3h/1km/ESRI Shapefile/convert'
sExtension_in='.shp'
sExtension_out='.parquet'
convert_multi_tile_vector_format(sWorkspace_in, sWorkspace_out, sExtension_in, sExtension_out)