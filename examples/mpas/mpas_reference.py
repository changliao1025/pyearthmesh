#refer to the e3sm confluence page for more details of the workflow

#each step may require different input datasets
#however, all the output will be saved in the same output directory for easy access

#you can change this to your preferred output directory
import os, platform
import glob
import logging
from datetime import datetime
from shutil import copy2

#using standalone dependency for this workflow, which will reduce the dependency to pyearth and pyflowline.
from mpas_land_mesh.utilities.change_json_key_value import change_json_key_value
from mpas_land_mesh.utilities.vector import get_field_and_value, merge_features, add_field_to_vector_file
from mpas_land_mesh.utilities.raster import convert_vector_to_global_raster
from mpas_land_mesh.utilities.constants import KM2_TO_M2, ISLAND_AREA_MULTIPLIER, DRAINAGE_AREA_MULTIPLIER

#from mpas_land_mesh.preprocessing.river_network import simplify_hydrorivers_network
#from mpas_land_mesh.preprocessing.coastline import create_land_ocean_mask_from_naturalearth, fix_naturalearth_hydrosheds_incompatibility

from hexwatershed_utility.preprocess.features.rivers import simplify_hydrorivers_network
from hexwatershed_utility.preprocess.features.coastline.simplify_hydrorivers_network import create_land_ocean_mask_from_naturalearth, fix_naturalearth_hydrosheds_incompatibility

from mpas_land_mesh.utilities.config_manager import create_jigsaw_template_configuration_file, read_jigsaw_configuration_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
sDate_today = datetime.now().strftime('%Y%m%d')


platform = platform.system()

#things that may need to be changed for different runs
sDate_today = '20260801'  #use a fixed date for easy repeatability
sMesh_type = 'mpas'  #
#index for different runs
iCase_index = 3
sModel = 'jigsaw'

#resolution settings
#for rivers and watershed
dResolution_ocean = 100
dResolution_land = 12.5  #unit in km
dResolution_river_network = 5
dResolution_coastline = 10  #unit in km
dResolution_lake_boundary = 3
dResolution_watershed_boundary =3

#for coastline
dThreshold_area_island = dResolution_ocean * dResolution_ocean * ISLAND_AREA_MULTIPLIER * KM2_TO_M2  #unit m2, this one may need to be adjusted based on the resolution
dResolution_coastline_buffer = dResolution_coastline * 1.0E3  #buffer zone for coastline line
#small island removal threshold
dDrainage_area_threshold = dResolution_land * dResolution_land * DRAINAGE_AREA_MULTIPLIER * KM2_TO_M2  #at least 100 grid cells of drainage area, this may be adjusted as well

#setup flags for debugging
iFlag_simplify_hydrosheds_river_network = 0
iFlag_process_coastline = 0

iFlag_endorheic_lake = 1

#number of largest outlet to be processed
nOutlet_largest = 10

#for jigsaw resolution control
dResolution_x_in = 30.0/3600 * dResolution_coastline
dResolution_y_in = dResolution_x_in
nrow = int(180 / dResolution_y_in)
ncolumn = int(360 / dResolution_x_in)

#string format for file names
#river first
dDistance_tolerance = dResolution_river_network * 1.0E3 #how far away two river need to be for mesh generation

sDistance_tolerance = "{:.2E}".format(dDistance_tolerance)
sDrainage_area_threshold = "{:.2E}".format(dDrainage_area_threshold) # m2

#coastline second
sCoastline_buffer = "{:.1E}".format(dResolution_coastline_buffer  ) # to m
sThreshold_area_island = "{:.1E}".format(dThreshold_area_island ) # to m2

#thing may not need to be changed
if platform == 'Linux':
    sWorkspace_input = '/qfs/people/liao313/workspace/python/unified_land_river_mesh/data/global/input'
    sWorkspace_output = '/data2/share/liaochang/04model/jigsaw/qinghaihu'
    #define global output directory
    sWorkspace_river_network_output = '/data2/share/liaochang/04model/jigsaw/qinghaihu/river_network'
    sWorkspace_coastline_output = '/data2/share/liaochang/04model/jigsaw/global/coastline'
    sWorkspace_data = '/public/home/liaochang/data/hexwatershed/qinghaihu/'
    sFilename_flowline_hydrosheds_in = '/data2/share/liaochang/data/raw/hydrology/hydrosheds/hydroriver/asian/HydroRIVERS_v10_as_shp/HydroRIVERS_v10_as.shp'
    sFilename_flowline_hydroshed_tmp = 'HydroRIVERS_v10_simplified_' + sDistance_tolerance + '_' + sDrainage_area_threshold + '.geojson'
    sFilename_geojson_geometery_feature = '/public/home/liaochang/data/hexwatershed/global/vector/region.geojson'
    sWorkspace_watershed_boundary_in = '/data2/share/liaochang/data/raw/hydrology/hydrosheds/hydrobasin'
else:
    if platform == 'Windows':
        sWorkspace_input = '/qfs/people/liao313/workspace/python/unified_land_river_mesh/data/global/input'
        sWorkspace_output = 'D:\\scratch\\04model\\jigsaw\\qinghaihu'
        sWorkspace_river_network_output = 'D:\\scratch\\04model\\jigsaw\\qinghaihu\\river_network'
        sWorkspace_coastline_output = 'D:\\scratch\\04model\\jigsaw\\global\\coastline'
        sWorkspace_data = 'D:\\data\\modeldata\\hexwatershed\\qinghaihu'
        sFilename_flowline_hydroshed_tmp = 'HydroRIVERS_v10_simplified_' + sDistance_tolerance + '_' + sDrainage_area_threshold + '.geojson'
        sFilename_geojson_geometery_feature = 'D:\\data\\modeldata\\hexwatershed\\global\\vector\\region.geojson'
        sWorkspace_watershed_boundary_in = '/data2/share/liaochang/data/raw/hydrology/hydrosheds/hydrobasin'

if os.path.exists(sWorkspace_river_network_output) is False:
    os.makedirs(sWorkspace_river_network_output)
if os.path.exists(sWorkspace_coastline_output) is False:
    os.makedirs(sWorkspace_coastline_output)

#add the threshol into the output folder
sWorkspace_river_network_output = os.path.join(sWorkspace_river_network_output,  sDistance_tolerance + '_' + sDrainage_area_threshold)
if os.path.exists(sWorkspace_river_network_output) is False:
    os.makedirs(sWorkspace_river_network_output)

sWorkspace_coastline_output = os.path.join(sWorkspace_coastline_output,  sCoastline_buffer + '_' + sThreshold_area_island )
if os.path.exists(sWorkspace_coastline_output) is False:
    os.makedirs(sWorkspace_coastline_output)


#Step 1
#prepare the river network and coastline line dataset

sFilename_flowline_hydrosheds_out = os.path.join(sWorkspace_river_network_output, sFilename_flowline_hydroshed_tmp)

#step 1: record attribute from the MPAS tools
aField, aValue = get_field_and_value(sFilename_geojson_geometery_feature)

sFilename_river_network_raster = os.path.join(sWorkspace_river_network_output, 'river_network_raster.tif')

#new
sFilename_river_network_raster = os.path.join(sWorkspace_data, 'raster' ,'river_networks.tif')
sFilename_watershed_boudnary_raster = os.path.join(sWorkspace_data, 'raster' ,'basin_boundary_mask.tif')

sFilename_lake_boudnary_vector = os.path.join(sWorkspace_data, 'vector' ,'lake_boundary_hydrobasins_qinghaihu.geojson')
sFilename_lake_boudnary_raster = os.path.join(sWorkspace_data, 'raster' ,'lake_boundary_mask.tif')

if iFlag_simplify_hydrosheds_river_network == 1:
    simplify_hydrorivers_network(sFilename_flowline_hydrosheds_in,
                       sFilename_flowline_hydrosheds_out,
                       dDistance_tolerance,
                        dDrainage_area_threshold,
                        nOutlet_largest=nOutlet_largest)
    convert_vector_to_global_raster(sFilename_flowline_hydrosheds_out, sFilename_river_network_raster,
                                         dResolution_x_in, dResolution_y_in )
else:
    #reuse the existing simplified river network for debug purpose, you can also change this to the original hydrosheds river network for testing
    sFilename_flowline_hydrosheds_out = '/public/home/liaochang/data/hexwatershed/qinghaihu/vector/flowline_hydroshed_simplified_clipped_clean.geojson'
    pass



sFilename_vector_coastline_merged = os.path.join(sWorkspace_coastline_output, 'land_ocean_mask_wo_island_merged.geojson')
sFilename_tif_wo_island = os.path.join(sWorkspace_coastline_output, 'land_ocean_mask_wo_island.tif')
if iFlag_process_coastline == 1:
    sFilename_tif_wo_island, sFilename_vector_coastline = create_land_ocean_mask_from_naturalearth(sWorkspace_coastline_output,
                                                                             dResolution_x_in, dResolution_y_in,
                                                                             dThreshold_area_island,
                                                                             dResolution_coastline_buffer,
                                                                             iRaster_buffer_pixel = 2)

    if iFlag_endorheic_lake == 1:
        #do we need to consider compatibility issue?
        pass
    else:
        ##we need to fix the incompatibilty between hydrosheds and naturalearth
        aFilename_flowline = list()
        for i in range(1, nOutlet_largest+1):
            sBasin_id = '{:04d}'.format(i)
            sFilename_flowline_simplified_basin = os.path.join(sWorkspace_river_network_output, 'HydroRIVERS_v10_simplified_' + sDistance_tolerance + '_' + sDrainage_area_threshold +'_'+ sBasin_id + '.geojson')
            aFilename_flowline.append(sFilename_flowline_simplified_basin)

        sFilename_vector_coastline_updated = os.path.join(sWorkspace_coastline_output, 'land_ocean_mask_wo_island_fixed.geojson')
        fix_naturalearth_hydrosheds_incompatibility(aFilename_flowline, sFilename_vector_coastline, sFilename_vector_coastline_updated )
        #should be merged into one single function
        merge_features(sFilename_vector_coastline_updated, sFilename_vector_coastline_merged, iFlag_force= True)
        add_field_to_vector_file(sFilename_vector_coastline_merged, aField, aValue)
else:
    #reuse
    sFilename_tif_wo_island = os.path.join(sWorkspace_coastline_output, 'land_ocean_mask_wo_island_buffered_fixed.tif')
    pass



#Step 2 - 4
#run the hexwatershed model, this step include three steps merged together.
#for debug purpose, you can also run then one by one, using the iFlag_debug flag to control
#sFilename_mpas_mesh_netcdf = '/compyfs/liao313/04model/pyhexwatershed/global/pyflowline20251122001/jigsaw/out/invert_mesh.nc'

iFlag_debug = 1  #0 for normal run, 1 for debug
sFilename_jigsaw_configuration_json = os.path.join(sWorkspace_river_network_output, 'jigsaw_configuration.json')

if iFlag_debug == 1:

    create_jigsaw_template_configuration_file(sFilename_jigsaw_configuration_json)
    change_json_key_value(sFilename_jigsaw_configuration_json, "sWorkspace_output", sWorkspace_output)
    oJigsaw = read_jigsaw_configuration_file(sFilename_jigsaw_configuration_json, \
    iCase_index_in=iCase_index, sDate_in=sDate_today, iFlag_create_directory_in=1)

    sWorkspace_output_case = oJigsaw.sWorkspace_output

    sFilename_jigsaw_configuration_copy = os.path.join( sWorkspace_output_case, 'jigsaw_configuration_copy.json' )
    copy2(sFilename_jigsaw_configuration_json, sFilename_jigsaw_configuration_copy)

    #update the jigsaw configuration file below
    change_json_key_value(sFilename_jigsaw_configuration_copy, "iFlag_geom", "true") # enable geometry control
    change_json_key_value(sFilename_jigsaw_configuration_copy, "iFlag_geom_river_network", "true") #set the resolution
    change_json_key_value(sFilename_jigsaw_configuration_copy, "iFlag_geom_lake_boundary", "true") #s


    change_json_key_value(sFilename_jigsaw_configuration_copy, "iFlag_spac", "true") #enable resolution control
    change_json_key_value(sFilename_jigsaw_configuration_copy, "iFlag_spac_ocean", "true")
    change_json_key_value(sFilename_jigsaw_configuration_copy, "iFlag_spac_river_network", "true")
    change_json_key_value(sFilename_jigsaw_configuration_copy, "iFlag_spac_lake_boundary", "true")
    change_json_key_value(sFilename_jigsaw_configuration_copy, "iFlag_spac_watershed_boundary", "true")

    #change_json_key_value(sFilename_jigsaw_configuration_copy, "iFlag_RRS18to6_ocean", "true")
    change_json_key_value(sFilename_jigsaw_configuration_copy, "iFlag_spac_land", "true")
    change_json_key_value(sFilename_jigsaw_configuration_copy, "iFlag_spac_coastline", "true") #set the resolution for coastline line

    change_json_key_value(sFilename_jigsaw_configuration_copy, "dResolution_ocean", dResolution_ocean) #set the resolution for ocean
    change_json_key_value(sFilename_jigsaw_configuration_copy, "dResolution_coastline", dResolution_coastline) #set the resolution for coastline line
    change_json_key_value(sFilename_jigsaw_configuration_copy, "dResolution_land", dResolution_land) #set the resolution for land
    change_json_key_value(sFilename_jigsaw_configuration_copy, "dResolution_river_network", dResolution_river_network) #set the resolution for river netwokr
    change_json_key_value(sFilename_jigsaw_configuration_copy, "dResolution_lake_boundary", dResolution_lake_boundary) #set the resolution for river network
    change_json_key_value(sFilename_jigsaw_configuration_copy, "dResolution_watershed_boundary", dResolution_watershed_boundary) #set the resolution for river network


    change_json_key_value(sFilename_jigsaw_configuration_copy, "ncolumn_space", ncolumn) #set the resolution for x direction
    change_json_key_value(sFilename_jigsaw_configuration_copy, "nrow_space", nrow) #set the resolution for y direction
    #change_json_key_value(sFilename_jigsaw_configuration_copy, "sFilename_dam_vector", sFilename_dam) #set the dam file
    change_json_key_value(sFilename_jigsaw_configuration_copy, "sFilename_river_network_vector", sFilename_flowline_hydrosheds_out) #set the resolution for x direction
    change_json_key_value(sFilename_jigsaw_configuration_copy, "sFilename_river_network_raster", sFilename_river_network_raster) #set the small island removal threshold

    change_json_key_value(sFilename_jigsaw_configuration_copy, "sFilename_watershed_boundary_raster", sFilename_watershed_boudnary_raster) #set the small island removal threshold

    change_json_key_value(sFilename_jigsaw_configuration_copy, "sFilename_lake_boundary_vector", sFilename_lake_boudnary_vector)
    change_json_key_value(sFilename_jigsaw_configuration_copy, "sFilename_lake_boundary_raster", sFilename_lake_boudnary_raster) #set the small island removal threshold

    change_json_key_value(sFilename_jigsaw_configuration_copy, "sFilename_coastline_raster", sFilename_tif_wo_island) #set the resolution for y direction

    #now we can set up the actual pyflowline to create the mesh
    oJigsaw = read_jigsaw_configuration_file(sFilename_jigsaw_configuration_copy,
                    iCase_index_in=iCase_index,
                    sDate_in= sDate_today)

    aCommand = list()
    aCommand.append('module load apps/miniconda3/2023.10')
    aCommand.append('source /public/software/apps/miniconda3/2023.10/etc/profile.d/conda.sh')
    oJigsaw._jigsaw_create_hpc_job(sSlurm_in = 'cpu_single', hours_in = 3, aCommand_list= aCommand )
    #now you should manually submit the job

else:
    pass

logger.info('='*80)
logger.info('Workflow completed successfully!')
logger.info('='*80)
