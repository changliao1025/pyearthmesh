import os, stat, platform
import numpy as np
from pathlib import Path
import subprocess
import datetime
from shutil import copy2
from osgeo import osr, ogr, gdal

from pyearth.system.define_global_variables import earth_radius
from pyearth.gis.location.get_geometry_coordinates import get_geometry_coordinates
from pyearthmesh.utility.convert_coordinates import convert_gcs_coordinates_to_meshcell


pDate = datetime.datetime.today()
sDate_default = (
    "{:04d}".format(pDate.year)
    + "{:02d}".format(pDate.month)
    + "{:02d}".format(pDate.day)
)

slash = os.sep

#this module use the dggal tool:
#https://github.com/ecere/dggal

def generate_dggal_bash_script(sWorkspace_output, sCommand_in):
    os.chdir(sWorkspace_output)
    # detemine the system platform
    # Determine the appropriate executable name for the platform
    system = platform.system()

    if system == "Windows":
        # execute binary on Windows
        iFlag_unix = 0
    elif system == "Linux":
        # execute binary on Linux
        iFlag_unix = 1
    elif system == "Darwin":
        # execute binary on macOS
        iFlag_unix = 1
    else:
        # unsupported operating system
        print("Unsupported operating system: " + system)
        print("Please reach out to the developers for assistance.")
        # generate the bash/batch script
    if iFlag_unix == 1:
        sFilename_bash = os.path.join(str(Path(sWorkspace_output)), "run_dggal.sh")
        ofs = open(sFilename_bash, "w")
        sLine = "#!/bin/bash\n"
        ofs.write(sLine)
        sLine = "cd " + sWorkspace_output + "\n"
        ofs.write(sLine)
        sLine = sCommand_in + "\n"
        ofs.write(sLine)
        ofs.close()
        os.chmod(sFilename_bash, stat.S_IRWXU)
    else:
        sFilename_bash = os.path.join(str(Path(sWorkspace_output)), "run_dggal.bat")
        ofs = open(sFilename_bash, "w")
        sLine = "cd " + sWorkspace_output + "\n"
        ofs.write(sLine)
        sLine = sCommand_in + "\n"
        ofs.write(sLine)
        ofs.close()
        os.chmod(sFilename_bash, stat.S_IRWXU)

    return sFilename_bash

def healpix_find_level_by_resolution(sHealpix_type, dResolution_meter):
    """
    Find the appropriate rHEALPix level for a given target resolution.

    For rHEALPix with aperture 9:
    - Level 0: ~4,759 km resolution
    - Each level divides cells by 3 (aperture 9 = 3²)
    - Resolution at level n ≈ R₀ / 3ⁿ

    Args:
        sHealpix_type (str): Type of HEALPix ('rHEALPix', 'HEALPix')
        dResolution_meter (float): Target resolution in meters

    Returns:
        int: The appropriate refinement level
    """
    # Earth's radius in meters
    dRadius_earth = earth_radius

    # rHEALPix aperture 9 (each level divides by 3)
    if sHealpix_type.lower() == 'rhealpix':
        # Base resolution at level 0 (approximate)
        # For rHEALPix, level 0 has 12 base cells on a sphere
        # Average cell edge length ≈ sqrt(4πR²/12) ≈ 4,759 km
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 12)

        # Aperture factor: each level divides by 3
        aperture_factor = 3.0

        # Find level where resolution_level0 / (aperture_factor^level) ≈ target_resolution
        # level = log(resolution_level0 / target_resolution) / log(aperture_factor)
        level = np.log(dResolution_level0 / dResolution_meter) / np.log(aperture_factor)

        # Round to nearest integer level
        iLevel = int(np.round(level))

        # Ensure non-negative level
        if iLevel < 0:
            iLevel = 0

    elif sHealpix_type.lower() == 'healpix':
        # HEALPix aperture 4 (each level divides by 2)
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 12)
        aperture_factor = 2.0

        level = np.log(dResolution_level0 / dResolution_meter) / np.log(aperture_factor)
        iLevel = int(np.round(level))

        if iLevel < 0:
            iLevel = 0
    else:
        raise ValueError(f"Unknown HEALPix type: {sHealpix_type}")

    return iLevel


def healpix_find_resolution_by_level(sHealpix_type, iLevel):
    """
    Find the resolution for a given rHEALPix level.

    For rHEALPix with aperture 9:
    - Level 0: ~4,759 km resolution
    - Each level divides cells by 3 (aperture 9 = 3²)
    - Resolution at level n ≈ R₀ / 3ⁿ

    Args:
        sHealpix_type (str): Type of HEALPix ('rHEALPix', 'HEALPix')
        iLevel (int): Refinement level

    Returns:
        float: The resolution in meters
    """
    # Earth's radius in meters
    dRadius_earth = earth_radius

    if sHealpix_type.lower() == 'rhealpix':
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 12)
        aperture_factor = 3.0

        dResolution = dResolution_level0 / (aperture_factor ** iLevel)

    elif sHealpix_type.lower() == 'healpix':
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 12)
        aperture_factor = 2.0

        dResolution = dResolution_level0 / (aperture_factor ** iLevel)

    else:
        raise ValueError(f"Unknown HEALPix type: {sHealpix_type}")

    return dResolution

def create_rhealpix_mesh(dResolution_meter_in,
    sFilename_mesh,
    sWorkspace_output,
    pBoundary_in=None,):
    """
    Create a spherical mesh based on the HEALPix tessellation.

    Parameters
    ----------
    dResolution_meter_in : float
        The resolution parameter for the HEALPix mesh.
    radius : float, optional
        The radius of the sphere, by default 1.0.

    Returns
    -------
    Mesh
        A spherical mesh object based on the HEALPix tessellation.
    """



    #find the level based on the resolution
    iLevel = healpix_find_level_by_resolution('rHEALPix', dResolution_meter_in)
    print("Determined rHEALPix level: " + str(iLevel))
    dResolution_actual = healpix_find_resolution_by_level('rHEALPix', iLevel)
    print("Actual resolution at this level: " + str(dResolution_actual) + " meters )")

    #now generate the mesh using dggal
    # example command: dgg isea3h -crs ico grid 3 > isea3h-level3-isea.geojson

    #create a temporary output file if a boudary is given

    sFilename_mesh_temp = os.path.join(sWorkspace_output, "temp_rhealpix_mesh.geojson")
    if os.path.exists(sFilename_mesh_temp):
        os.remove(sFilename_mesh_temp)

    sCommand_in = "dgg rhealpix " + " -crs EPSG:4326 " + " grid " + str(iLevel) + " > " + sFilename_mesh_temp

    sFilename_bash = generate_dggal_bash_script(sWorkspace_output, sCommand_in)

    #execute the bash script
    os.chdir(sWorkspace_output)
    if platform.system() == "Windows":
        subprocess.run([sFilename_bash], shell=True)
    else:
        subprocess.run(["bash", sFilename_bash])


    #now we need to convert the mesh
    if pBoundary_in is not None:
        pass
    else:
        #only need to deal with antimeridian mesh cell issue
        pass

    return sFilename_mesh


