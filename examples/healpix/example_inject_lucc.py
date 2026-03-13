import os, sys, platform

sPlatform_os = platform.system()
from pyearth.toolbox.conversion.convert_vector_format import convert_vector_format
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_driver_from_format, get_vector_format_from_extension
# Get the directory of the current script


from uraster.classes.uraster import uraster
# Download input data using Pooch (downloads to system cache)

dResolution_meter = 50000  #10 km
sResoluton = str(int(dResolution_meter/ 1000)) + "km"
case_index = 1
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

sFilename_source_mesh = os.path.join(sWorkspace_output, "healpix_mesh_land.parquet")
sFilename_target_mesh =  os.path.join(sWorkspace_output, "uraster_lucc.parquet")
sFilename_mesh_png = os.path.join(sWorkspace_output, "mesh.png")
sFilename_variable_png = os.path.join(sWorkspace_output, "lucc.png")
sFilename_variable_animation = os.path.join(sWorkspace_output, "lucc.mp4")

aDict_discrete_labels = {
    11: 'Open Water',
    12: 'Perennial Ice/Snow',
    21: 'Developed, Open Space',
    22: 'Developed, Low Intensity',
    23: 'Developed, Medium Intensity',
    24: 'Developed, High Intensity',
    31: 'Barren Land (Rock/Sand/Clay)',
    41: 'Deciduous Forest',
    42: 'Evergreen Forest',
    43: 'Mixed Forest',
    51: 'Dwarf Scrub',
    52: 'Shrub/Scrub',
    71: 'Grassland/Herbaceous',
    72: 'Sedge/Herbaceous',
    81: 'Pasture/Hay',
    82: 'Cultivated Crops',
    90: 'Woody Wetlands',
    95: 'Emergent Herbaceous Wetlands',
    0: 'No Data'
}

aColor_discrete = {
    11: '#466B9F',  # Open Water (NLCD RGB: 70, 107, 159)
    12: '#D1DEF8',  # Perennial Ice/Snow (NLCD RGB: 209, 222, 248)
    21: '#DEE4E7',  # Developed, Open Space (NLCD RGB: 222, 228, 231)
    22: '#E4CEAD',  # Developed, Low Intensity (NLCD RGB: 228, 206, 173)
    23: '#EA9E64',  # Developed, Medium Intensity (NLCD RGB: 234, 158, 100)
    24: '#C25D42',  # Developed, High Intensity (NLCD RGB: 194, 93, 66)
    31: '#EADCDC',  # Barren Land (Rock/Sand/Clay) (NLCD RGB: 234, 220, 220)
    41: '#A8D88E',  # Deciduous Forest (NLCD RGB: 168, 216, 142)
    42: '#38814E',  # Evergreen Forest (NLCD RGB: 56, 129, 78)
    43: '#DCCA8F',  # Mixed Forest (NLCD RGB: 220, 202, 143)
    51: '#F4EBBD',  # Dwarf Scrub (NLCD RGB: 244, 235, 189)
    52: '#B8D9E3',  # Shrub/Scrub (NLCD RGB: 184, 217, 227)
    71: '#AAACCA',  # Grassland/Herbaceous (NLCD RGB: 170, 172, 202)
    72: '#C9DD93',  # Sedge/Herbaceous (NLCD RGB: 201, 221, 147)
    73: '#B6D1B6',  # Lichens/Mosses (NLCD RGB: 182, 209, 182)
    74: '#C6DCD7',  # Emergent Herbaceous Wetlands (NLCD RGB: 198, 220, 215)
    81: '#76A8B4',  # Pasture/Hay (NLCD RGB: 118, 168, 180)
    82: '#C79D8E',  # Cultivated Crops (NLCD RGB: 199, 157, 142)
    90: '#7E967D',  # Woody Wetlands (NLCD RGB: 126, 150, 125)
    95: '#A2AD9C',  # Emergent Herbaceous Wetlands (NLCD RGB: 162, 173, 156) - Note: NLCD has two "Emergent Herbaceous Wetlands" categories (74 and 95) with slightly different colors.
    0:  '#FFFFFF'   # No Data / Background (White)
}

def main():
    aConfig = dict()
    aConfig["sFilename_source_mesh"] = (
        sFilename_source_mesh  # use the L10-100 test mesh
    )
    aFilename_source_raster = []
    sFilename_dem0 = '/compyfs/liao313/00raw/nlcd/Annual_NLCD_LndCov_2023_CU_C1V0_wgs84.tif'
    aFilename_source_raster.append(sFilename_dem0)  # global dem from gebco
    aConfig["aFilename_source_raster"] = aFilename_source_raster
    aConfig["sFilename_target_mesh"] = sFilename_target_mesh
    aConfig["iFlag_discrete"] = 1
    pRaster = uraster(aConfig)

    if not os.path.exists(pRaster.sFilename_target_mesh):
        print(f"Target mesh file does not exist: {pRaster.sFilename_target_mesh}")

    pRaster.setup(iFlag_verbose_in=True)
    # pRaster.report_inputs()
    dLongitude_focus_in = -98.5
    dLatitude_focus_in = 39.8
    #pRaster.visualize_source_mesh(
    #    sFilename_out=sFilename_mesh_png,
    #    dLongitude_focus_in=dLongitude_focus_in,
    #    dLatitude_focus_in=dLatitude_focus_in,
    #    dImage_scale_in=10,
    #    iFlag_show_graticule=False,
    #    iFlag_wireframe_only=True,
    #)
    #pRaster.run_remap(iFlag_verbose_in=True)

    # pRaster.report_outputs()
 


    pRaster.visualize_target_mesh(
        sFilename_out=sFilename_variable_png,
        dLongitude_focus_in=dLongitude_focus_in,
        dLatitude_focus_in=dLatitude_focus_in,
        dImage_scale_in=5,
        iFlag_show_graticule=False,
        sUnit_in="NLCD Class",
        aDict_discrete_labels_in = aDict_discrete_labels,
        aDict_value_color_in = aColor_discrete,
    )

    # pRaster.visualize_target_mesh(
    #    sFilename_out=sFilename_variable_animation,
    #    sColormap=sColormap,
    #    dLongitude_focus_in=dLongitude_focus_in,
    #    dLatitude_focus_in=dLatitude_focus_in,
    #    iFlag_create_animation=True,
    #    iAnimation_frames=360,       # 1° longitude per frame
    #    sAnimation_format='mp4')

    pRaster.cleanup()

    print("done")


if __name__ == "__main__":
    main()
