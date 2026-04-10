import os
import stat
import json
import datetime
from pathlib import Path
from pyearth.system.python import get_python_environment
#create a simple mesh run case class
pDate = datetime.datetime.today()
sDate_default = (
    "{:04d}".format(pDate.year)
    + "{:02d}".format(pDate.month)
    + "{:02d}".format(pDate.day)
)

class meshcase:
    """Class to represent a mesh run case with configuration and file paths"""
    iFlag_save_mesh = 1
    sFilename_land_ocean_mask = None
    pBoundary_wkt = None
    aConfig_mesh = None
    def __init__(self, aConfig_in,
        iFlag_standalone_in=None,
        iFlag_create_directory_in=1,
        sModel_in=None,
        sDate_in=None,
        sWorkspace_output_in=None):
        # flags
        if iFlag_standalone_in is not None:
            self.iFlag_standalone = iFlag_standalone_in
        else:
            if "iFlag_standalone" in aConfig_in:
                self.iFlag_standalone = int(aConfig_in["iFlag_standalone"])
            else:
                self.iFlag_standalone = 1

        if "iFlag_global" in aConfig_in:
            self.iFlag_global = int(aConfig_in["iFlag_global"])
        else:
            self.iFlag_global = 1

        if "iCase_index" in aConfig_in:
            iCase_index = int(aConfig_in["iCase_index"])
        else:
            iCase_index = 1

        sCase_index = "{:03d}".format(iCase_index)
        self.iCase_index = iCase_index

        if sWorkspace_output_in is not None:
            self.sWorkspace_output = sWorkspace_output_in
        else:
            if "sWorkspace_output" in aConfig_in:
                self.sWorkspace_output = aConfig_in["sWorkspace_output"]


        if "sFilename_land_ocean_mask" in aConfig_in:
            self.sFilename_land_ocean_mask = aConfig_in["sFilename_land_ocean_mask"]

        if "pBoundary_wkt" in aConfig_in:
            self.pBoundary_wkt = aConfig_in["pBoundary_wkt"]

        sDate = aConfig_in["sDate"]
        if sDate is not None:
            self.sDate = sDate
        else:
            self.sDate = sDate_default

        sCase = self.sModel + self.sDate + sCase_index
        self.sCase = sCase

        # the model can be run as part of hexwatershed or standalone
        if self.iFlag_standalone == 1:
            # in standalone case, will add case information and update output path
            sPath = str(Path(self.sWorkspace_output) / sCase)
            self.sWorkspace_output = sPath
        else:
            # use specified output path, also do not add output or input tag
            sPath = self.sWorkspace_output

        if iFlag_create_directory_in == 1:
            Path(sPath).mkdir(parents=True, exist_ok=True)

        if "sFilename_model_configuration" in aConfig_in:
            self.sFilename_model_configuration = aConfig_in["sFilename_model_configuration"]
        else:
            self.sFilename_model_configuration = None

        self.sFilename_mesh = os.path.join(
            str(Path(self.sWorkspace_output)),  "mesh.geojson"
        )

        if "sMesh_type" in aConfig_in:
            self.sMesh_type = aConfig_in["sMesh_type"]
        else:
            self.sMesh_type = 'mpas'




        return