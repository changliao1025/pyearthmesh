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

def isea_find_level_by_resolution(sISEA_type, dResolution_meter):
    """
    Find the appropriate ISEA level for a given target resolution.

    For ISEA3H (aperture 3):
    - Level 0: ~4,759 km resolution
    - Each level increases cell count by 3 (aperture 3)
    - Cell area decreases by factor of 3, so edge length decreases by √3
    - Resolution at level n ≈ R₀ / √3ⁿ

    For ISEA4H (aperture 4):
    - Each level increases cell count by 4 (aperture 4)
    - Cell area decreases by factor of 4, so edge length decreases by 2
    - Resolution at level n ≈ R₀ / 2ⁿ

    Args:
        sISEA_type (str): Type of ISEA ('ISEA3H', 'ISEA4H')
        dResolution_meter (float): Target resolution in meters

    Returns:
        int: The appropriate refinement level
    """
    # Earth's radius in meters
    dRadius_earth = earth_radius

    if sISEA_type.lower() == 'isea3h':
        # Base resolution at level 0 (approximate)
        # For ISEA3H with 20 base triangles on icosahedron
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 20)
        # Aperture 3: edge length factor is sqrt(3)
        edge_factor = np.sqrt(3.0)
        level = np.log(dResolution_level0 / dResolution_meter) / np.log(edge_factor)
        iLevel = int(np.round(level))
        # Ensure non-negative level
        if iLevel < 0:
            iLevel = 0

    elif sISEA_type.lower() == 'isea4h':
        # Base resolution at level 0 (approximate)
        # For ISEA4H with 20 base triangles on icosahedron
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 20)
        # Aperture 4: edge length factor is 2
        edge_factor = 2.0
        level = np.log(dResolution_level0 / dResolution_meter) / np.log(edge_factor)
        iLevel = int(np.round(level))
        # Ensure non-negative level
        if iLevel < 0:
            iLevel = 0
    else:
        raise ValueError(f"Unknown ISEA type: {sISEA_type}")

    return iLevel


def isea_find_resolution_by_level(sISEA_type, iLevel):
    """
    Find the resolution for a given ISEA level.

    For ISEA3H (aperture 3):
    - Level 0: ~4,759 km resolution
    - Each level increases cell count by 3 (aperture 3)
    - Cell area decreases by factor of 3, so edge length decreases by √3
    - Resolution at level n ≈ R₀ / √3ⁿ

    For ISEA4H (aperture 4):
    - Each level increases cell count by 4 (aperture 4)
    - Cell area decreases by factor of 4, so edge length decreases by 2
    - Resolution at level n ≈ R₀ / 2ⁿ

    Args:
        sISEA_type (str): Type of ISEA ('ISEA3H', 'ISEA4H')
        iLevel (int): Refinement level

    Returns:
        float: The resolution in meters
    """
    # Earth's radius in meters
    dRadius_earth = earth_radius

    if sISEA_type.lower() == 'isea3h':
        # Base resolution at level 0 (approximate)
        # For ISEA3H with 20 base triangles on icosahedron
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 20)
        # Aperture 3: edge length factor is sqrt(3)
        edge_factor = np.sqrt(3.0)
        dResolution = dResolution_level0 / (edge_factor ** iLevel)
    elif sISEA_type.lower() == 'isea4h':
        # Base resolution at level 0 (approximate)
        # For ISEA4H with 20 base triangles on icosahedron
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 20)
        # Aperture 4: edge length factor is 2
        edge_factor = 2.0
        dResolution = dResolution_level0 / (edge_factor ** iLevel)
    else:
        raise ValueError(f"Unknown ISEA type: {sISEA_type}")
    return dResolution

def create_isea_mesh(dResolution_meter_in,
                     sISEA_type,
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
    iLevel = isea_find_level_by_resolution(sISEA_type, dResolution_meter_in)
    print("Determined ISEA level: " + str(iLevel))
    dResolution_actual = isea_find_resolution_by_level(sISEA_type, iLevel)
    print("Actual resolution at this level: " + str(dResolution_actual) + " meters )")

    #now generate the mesh using dggal
    # example command: dgg isea3h -crs ico grid 3 > isea3h-level3-isea.geojson

    #create a temporary output file if a boudary is given

    sCommand_in = "dgg " + sISEA_type.lower() + " " + " -crs EPSG:4326 " + " grid " + str(iLevel) + " > " + sFilename_mesh

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


