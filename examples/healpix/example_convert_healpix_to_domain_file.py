
from pyearthmesh.utility.convert_vector_to_domain_file import convert_vector_to_domain_file

#sFilename_vector_in = "/compyfs/liao313/04model/pyearthmesh/healpix/1km/Parquet/02/landoceanmask/merged_land_cells.parquet"
#sFilename_netcdf_out = "/compyfs/liao313/04model/pyearthmesh/healpix/1km/Parquet/02/landoceanmask/land_domain_file_healpix_1km.nc"

sFilename_vector_in = "/compyfs/liao313/04model/pyearthmesh/healpix/10km/Parquet/02/landoceanmask/merged_land_cells.parquet"
sFilename_netcdf_out = "/compyfs/liao313/04model/pyearthmesh/healpix/10km/Parquet/02/landoceanmask/land_domain_file_healpix_10km.nc"
sFilename_scripgrid_out = "/compyfs/liao313/04model/pyearthmesh/healpix/10km/Parquet/02/landoceanmask/scrip_grid_healpix_10km.nc"

convert_vector_to_domain_file(sFilename_vector_in, sFilename_netcdf_out, sFilename_scripgrid_out=sFilename_scripgrid_out)