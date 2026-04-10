import numpy as np
import netCDF4 as nc
import os, sys
import getpass
from datetime import datetime
from pyearth.system.define_global_variables import earth_radius
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_driver_from_filename
from pyearth.gis.location.get_geometry_coordinates import get_geometry_coordinates
def convert_vector_to_domain_file(sFilename_vector_in, sFilename_netcdf_out, sFilename_scripgrid_out=None):

    #check input
    if not os.path.isfile(sFilename_vector_in):
        print('Error: file not exist: '+sFilename_vector_in)
        return

    #check if output file exist, if so, delete it
    if os.path.isfile(sFilename_netcdf_out):
        print('Warning: file exist, delete it: '+sFilename_netcdf_out)
        os.remove(sFilename_netcdf_out)

    #read vector file using gdal/ogr directly
    pDriver_vector = get_vector_driver_from_filename(sFilename_vector_in)
    pDataset_vector = pDriver_vector.Open(sFilename_vector_in, 0)
    if pDataset_vector is None:
        print('Error: cannot open file: '+sFilename_vector_in)
        return

    pLayer_vector = pDataset_vector.GetLayer()
    nFeature_count = pLayer_vector.GetFeatureCount()
    print('Number of features: '+str(nFeature_count))

    # Pre-allocate arrays for batch processing (max size = all features)
    cell_id_array = np.zeros(nFeature_count, dtype=np.int32)
    lon_array = np.zeros(nFeature_count, dtype=np.float32)
    lat_array = np.zeros(nFeature_count, dtype=np.float32)
    xv_array = np.zeros((nFeature_count, 4), dtype=np.float32)
    yv_array = np.zeros((nFeature_count, 4), dtype=np.float32)
    area_array = np.zeros(nFeature_count, dtype=np.float32)

    # Process all features and keep only valid cells
    n_valid_cells = 0
    pLayer_vector.ResetReading()
    for i in range(nFeature_count):
        pFeature = pLayer_vector.GetNextFeature()
        if pFeature is None:
            continue

        pGeometry = pFeature.GetGeometryRef()
        if pGeometry is None:
            print('Warning: geometry is None, skip feature index: '+str(i))
            continue

        sGeometry_type = pGeometry.GetGeometryName()
        if sGeometry_type != 'POLYGON':
            print('Error: geometry type is not polygon: '+sGeometry_type)
            pDataset_vector = None
            return

        #check the number of vertices for each polygon, if less than 4, skip this cell
        aCoord = get_geometry_coordinates(pGeometry)
        if aCoord is None or aCoord.ndim != 2 or aCoord.shape[1] != 2:
            print('Warning: invalid polygon coordinates, skip feature index: '+str(i))
            continue

        nVertex_count = aCoord.shape[0]
        if nVertex_count != 5:
            print('Warning: polygon has less than 4 vertices: '+str(nVertex_count))
            print('Polygon will be skipped: feature index: '+str(i))
            continue

        #read the attribute of the feature, including cell id and center lon lat
        #if not pFeature.GetFieldIndex('cellid') >= 0:
        #    print('Warning: cellid field not found, skip feature index: '+str(i))
        #    continue
        #if not pFeature.GetFieldIndex('lon') >= 0:
        #    print('Warning: lon field not found, skip feature index: '+str(i))
        #    continue
        #if not pFeature.GetFieldIndex('lat') >= 0:
        #    print('Warning: lat field not found, skip feature index: '+str(i))
        #    continue
        #if not pFeature.GetFieldIndex('area') >= 0:
        #    print('Warning: area field not found, skip feature index: '+str(i))
        #    continue
        cell_id = pFeature.GetField('cellid')
        dLon_center = pFeature.GetField('lon')
        dLat_center = pFeature.GetField('lat')
        area = pFeature.GetField('area')

        #remove the closed vertex if exist, which is the same as the first vertex
        if np.array_equal(aCoord[0, :], aCoord[-1, :]):
            aCoord = aCoord[:-1, :]
        else:
            print('Warning: polygon is not closed, number of vertices: '+str(nVertex_count))
            print('Polygon will be skipped: feature index: '+str(i))
            continue

        #get the center of the cell, which is the average of the vertices
        cell_id_array[n_valid_cells] = cell_id
        lon_array[n_valid_cells] = dLon_center
        lat_array[n_valid_cells] = dLat_center
        area_array[n_valid_cells] = area
        xv_array[n_valid_cells, :] = aCoord[:, 0]
        yv_array[n_valid_cells, :] = aCoord[:, 1]

        n_valid_cells += 1

        # Optional: print progress
        if (i % 50000 == 0) or (i == nFeature_count - 1):
            print(f"Processed {i+1}/{nFeature_count} features ({100*(i+1)/nFeature_count:.1f}%), valid cells: {n_valid_cells}")
            #flush the output to ensure progress is visible
            sys.stdout.flush()

    if n_valid_cells == 0:
        print('Error: no valid polygon cells found in input file')
        pDataset_vector = None
        return

    # Trim arrays to valid cell count and generate contiguous 1-based cell IDs
    n_cells = n_valid_cells
    cellid_array = cell_id_array[:n_cells]
    lon_array = lon_array[:n_cells]
    lat_array = lat_array[:n_cells]
    xv_array = xv_array[:n_cells, :]
    yv_array = yv_array[:n_cells, :]
    area_array = area_array[:n_cells]
    # Close the vector dataset
    pDataset_vector = None

    # Create netcdf file and write all data at once
    print('Writing to NetCDF file...')
    pDataset_out = nc.Dataset(sFilename_netcdf_out, 'w', format='NETCDF4')
    if pDataset_out is None:
        print('Error: cannot create file: '+sFilename_netcdf_out)
        return

    # Create dimensions
    pDataset_out.createDimension('cell', n_cells)
    pDataset_out.createDimension('nv', 4)

    # Create variables
    var_cellid = pDataset_out.createVariable('cellid', 'i4', ('cell',))
    var_lon = pDataset_out.createVariable('lon', 'f4', ('cell',))
    var_lat = pDataset_out.createVariable('lat', 'f4', ('cell',))
    var_xv = pDataset_out.createVariable('xv', 'f4', ('cell', 'nv'))
    var_yv = pDataset_out.createVariable('yv', 'f4', ('cell', 'nv'))
    var_area = pDataset_out.createVariable('area', 'f4', ('cell',))

    # Add attributes for cellid
    var_cellid.long_name = 'global cell id'
    var_cellid.units = '1'

    # Add attributes for lon
    var_lon.long_name = 'longitude of cell center'
    var_lon.standard_name = 'longitude'
    var_lon.units = 'degrees_east'

    # Add attributes for lat
    var_lat.long_name = 'latitude of cell center'
    var_lat.standard_name = 'latitude'
    var_lat.units = 'degrees_north'

    # Add attributes for xv
    var_xv.long_name = 'longitude of cell vertices'
    var_xv.units = 'degrees_east'

    # Add attributes for yv
    var_yv.long_name = 'latitude of cell vertices'
    var_yv.units = 'degrees_north'

    # Add attributes for area
    var_area.long_name = 'area of cell'
    var_area.standard_name = 'area'
    var_area.units = 'm2'

    # Write all data at once (batch write)
    var_cellid[:] = cellid_array
    var_lon[:] = lon_array
    var_lat[:] = lat_array
    var_xv[:, :] = xv_array
    var_yv[:, :] = yv_array
    var_area[:] = area_array

    pDataset_out.close()
    print('Successfully converted vector to netcdf: '+sFilename_netcdf_out)

    # Optionally write SCRIP grid file
    if sFilename_scripgrid_out is not None:
        print('Writing SCRIP grid file...')

        # Check if output file exists, if so, delete it
        if os.path.isfile(sFilename_scripgrid_out):
            print('Warning: file exist, delete it: '+sFilename_scripgrid_out)
            os.remove(sFilename_scripgrid_out)

        # Convert area from m2 to radians^2 (divide by Earth radius squared)
        dEarth_radius_m = earth_radius
        area_rad2 = area_array.astype(np.float64) / (dEarth_radius_m ** 2)

        # Build mask array (all 1 = unmasked)
        grid_imask = np.ones(n_cells, dtype=np.int32)

        nvertex = 4
        grid_rank = 1

        pDataset_scrip = nc.Dataset(sFilename_scripgrid_out, mode='w', format='NETCDF3_CLASSIC')

        pDataset_scrip.createDimension('grid_size', n_cells)
        pDataset_scrip.createDimension('grid_corners', nvertex)
        pDataset_scrip.createDimension('grid_rank', grid_rank)

        aDimension_tuple1 = ('grid_size',)
        aDimension_tuple2 = ('grid_size', 'grid_corners')
        aDimension_tuple3 = ('grid_rank',)

        var_grid_dims = pDataset_scrip.createVariable('grid_dims', 'i4', aDimension_tuple3, fill_value=-9999)
        var_grid_dims[:] = n_cells

        var_center_lon = pDataset_scrip.createVariable('grid_center_lon', float, aDimension_tuple1, fill_value=-9999)
        var_center_lon.setncatts({'units': 'degrees'})
        var_center_lon[:] = lon_array.astype(np.float64)

        var_center_lat = pDataset_scrip.createVariable('grid_center_lat', float, aDimension_tuple1, fill_value=-9999)
        var_center_lat.setncatts({'units': 'degrees'})
        var_center_lat[:] = lat_array.astype(np.float64)

        var_imask = pDataset_scrip.createVariable('grid_imask', 'i4', aDimension_tuple1, fill_value=-9999)
        var_imask.setncatts({'units': 'unitless'})
        var_imask[:] = grid_imask

        var_corner_lon = pDataset_scrip.createVariable('grid_corner_lon', float, aDimension_tuple2, fill_value=-9999)
        var_corner_lon.setncatts({'units': 'degrees'})
        var_corner_lon[:, :] = xv_array.astype(np.float64)

        var_corner_lat = pDataset_scrip.createVariable('grid_corner_lat', float, aDimension_tuple2, fill_value=-9999)
        var_corner_lat.setncatts({'units': 'degrees'})
        var_corner_lat[:, :] = yv_array.astype(np.float64)

        var_grid_area = pDataset_scrip.createVariable('grid_area', float, aDimension_tuple1, fill_value=-9999)
        var_grid_area.setncatts({'units': 'radians^2'})
        var_grid_area[:] = area_rad2

        user_name = getpass.getuser()
        setattr(pDataset_scrip, 'Created_by', user_name)
        setattr(pDataset_scrip, 'Created_on', datetime.now().strftime('%c'))

        pDataset_scrip.close()
        print('Successfully converted vector to SCRIP grid: '+sFilename_scripgrid_out)

    return


