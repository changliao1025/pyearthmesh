import datetime
import json
import os
import stat
from pathlib import Path

from pyearth.system.python.get_python_environment import get_python_environment

from pyearthmesh.classes.base_case import BaseCase

sDate_default = datetime.datetime.today().strftime("%Y%m%d")


class mpascase(BaseCase):
    iFlag_standalone = 1
    iFlag_create_directory = 1
    iFlag_save_config = 1
    """Class to represent a mesh run case with configuration and file paths"""
    iFlag_save_mesh = 1
    sFilename_land_ocean_mask = None
    sFilename_mesh_boundary = None
    sFilename_boundary = None
    pBoundary_wkt = None
    aConfig_mesh = None

    def __init__(
        self,
        aConfig_in,
        iFlag_create_directory_in=1,
        sDate_in=None,
        sWorkspace_output_in=None,
    ):
        super().__init__(aConfig_in)

        if "iCase_index" in aConfig_in:
            iCase_index = int(aConfig_in["iCase_index"])
        else:
            iCase_index = 1
        sCase_index = f"{iCase_index:03d}"
        self.iCase_index = iCase_index

        if sDate_in is not None:
            self.sDate = sDate_in
        elif "sDate" in aConfig_in:
            self.sDate = aConfig_in["sDate"]
        else:
            self.sDate = sDate_default

        if sWorkspace_output_in is not None:
            self.sWorkspace_output = sWorkspace_output_in
        elif "sWorkspace_output" in aConfig_in:
            self.sWorkspace_output = aConfig_in["sWorkspace_output"]

        if "sFilename_mpas_mesh_netcdf" in aConfig_in:
            self.sFilename_mpas_mesh_netcdf = aConfig_in["sFilename_mpas_mesh_netcdf"]
        else:
            self.sFilename_mpas_mesh_netcdf = None

        if "sFilename_jigsaw_mesh_netcdf" in aConfig_in:
            self.sFilename_jigsaw_mesh_netcdf = aConfig_in[
                "sFilename_jigsaw_mesh_netcdf"
            ]
        else:
            self.sFilename_jigsaw_mesh_netcdf = None

        if "sFilename_land_ocean_mask" in aConfig_in:
            self.sFilename_land_ocean_mask = aConfig_in["sFilename_land_ocean_mask"]

        if "sFilename_mesh_boundary" in aConfig_in:
            self.sFilename_mesh_boundary = aConfig_in["sFilename_mesh_boundary"]
        elif "sFilename_boundary" in aConfig_in:
            self.sFilename_mesh_boundary = aConfig_in["sFilename_boundary"]
        elif "sFilename_boundary_vector" in aConfig_in:
            self.sFilename_mesh_boundary = aConfig_in["sFilename_boundary_vector"]
        else:
            self.sFilename_mesh_boundary = None
        self.sFilename_boundary = self.sFilename_mesh_boundary

        if "pBoundary_wkt" in aConfig_in:
            self.pBoundary_wkt = aConfig_in["pBoundary_wkt"]

        # Polar region flags
        if "iFlag_antarctic" in aConfig_in:
            self.iFlag_antarctic = int(aConfig_in["iFlag_antarctic"])
        else:
            self.iFlag_antarctic = 0

        if "iFlag_arctic" in aConfig_in:
            self.iFlag_arctic = int(aConfig_in["iFlag_arctic"])
        else:
            self.iFlag_arctic = 0

        # Domain extent and resolution parameters
        if "dLatitude_top" in aConfig_in:
            self.dLatitude_top = float(aConfig_in["dLatitude_top"])
        else:
            self.dLatitude_top = 90.0

        if "dLatitude_bot" in aConfig_in:
            self.dLatitude_bot = float(aConfig_in["dLatitude_bot"])
        else:
            self.dLatitude_bot = -90.0

        if "dLongitude_left" in aConfig_in:
            self.dLongitude_left = float(aConfig_in["dLongitude_left"])
        else:
            self.dLongitude_left = -180.0

        if "dLongitude_right" in aConfig_in:
            self.dLongitude_right = float(aConfig_in["dLongitude_right"])
        else:
            self.dLongitude_right = 180.0

        if "dLatitude_mean" in aConfig_in and aConfig_in["dLatitude_mean"] is not None:
            self.dLatitude_mean = float(aConfig_in["dLatitude_mean"])
        else:
            self.dLatitude_mean = 0.5 * (self.dLatitude_bot + self.dLatitude_top)

        if (
            "dLongitude_mean" in aConfig_in
            and aConfig_in["dLongitude_mean"] is not None
        ):
            self.dLongitude_mean = float(aConfig_in["dLongitude_mean"])
        else:
            self.dLongitude_mean = 0.5 * (self.dLongitude_left + self.dLongitude_right)

        if (
            "dResolution_meter" in aConfig_in
            and aConfig_in["dResolution_meter"] is not None
        ):
            self.dResolution_meter = float(aConfig_in["dResolution_meter"])
        else:
            self.dResolution_meter = None

        if (
            "dResolution_degree" in aConfig_in
            and aConfig_in["dResolution_degree"] is not None
        ):
            self.dResolution_degree = float(aConfig_in["dResolution_degree"])
        else:
            self.dResolution_degree = None

        # Convert between meter and degree if one is missing and mean latitude is available
        if self.dResolution_degree is not None and self.dResolution_meter is None:
            try:
                from pyearth.gis.spatialref.convert_between_degree_and_meter import (
                    degree_to_meter,
                )

                self.dResolution_meter = degree_to_meter(
                    self.dResolution_degree, self.dLatitude_mean
                )
            except Exception:
                pass
        elif self.dResolution_meter is not None and self.dResolution_degree is None:
            try:
                from pyearth.gis.spatialref.convert_between_degree_and_meter import (
                    meter_to_degree,
                )

                self.dResolution_degree = meter_to_degree(
                    self.dResolution_meter, self.dLatitude_mean
                )
            except Exception:
                pass

        if "iFlag_use_mesh_dem" in aConfig_in:
            self.iFlag_use_mesh_dem = int(aConfig_in["iFlag_use_mesh_dem"])
        else:
            self.iFlag_use_mesh_dem = 0

        if "iFlag_mesh_boundary" in aConfig_in:
            self.iFlag_mesh_boundary = int(aConfig_in["iFlag_mesh_boundary"])
        else:
            self.iFlag_mesh_boundary = 1 if self.sFilename_mesh_boundary else 0

        if "iFlag_land_ocean_mask" in aConfig_in:
            self.iFlag_land_ocean_mask = int(aConfig_in["iFlag_land_ocean_mask"])
        else:
            self.iFlag_land_ocean_mask = 1 if self.sFilename_land_ocean_mask else 0

        self.sModel = "mpas"

        if "iFlag_global" in aConfig_in:
            self.iFlag_global = int(aConfig_in["iFlag_global"])
        else:
            self.iFlag_global = 1

        if "sFilename_model_configuration" in aConfig_in:
            self.sFilename_model_configuration = aConfig_in[
                "sFilename_model_configuration"
            ]
        else:
            self.sFilename_model_configuration = None

        if "iFlag_run_jigsaw" in aConfig_in:
            self.iFlag_run_jigsaw = int(aConfig_in["iFlag_run_jigsaw"])
        else:
            self.iFlag_run_jigsaw = 1

        if self.iFlag_run_jigsaw == 1:
            self.aConfig_jigsaw = aConfig_in

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

        self.sFilename_mesh = os.path.join(
            str(Path(self.sWorkspace_output)), "mpas.geojson"
        )
        self.config["sWorkspace_output"] = self.sWorkspace_output
        self.config["sFilename_mesh"] = self.sFilename_mesh

        if iFlag_create_directory_in == 1:
            Path(sPath).mkdir(parents=True, exist_ok=True)

        return

    def setup(self):
        """Prepare the MPAS workspace directories.

        Creates the required output directory and, when JIGSAW mesh generation
        is enabled, the standard JIGSAW subdirectories (tmp, out) inside
        self.sWorkspace_output so that run_jigsaw can write its intermediate
        and output files.
        """
        # Ensure the main output directory exists
        Path(self.sWorkspace_output).mkdir(parents=True, exist_ok=True)

        if self.iFlag_run_jigsaw == 1:
            # Create standard JIGSAW subdirectories
            for subdir in ("tmp", "out"):
                Path(os.path.join(self.sWorkspace_output, subdir)).mkdir(
                    parents=True, exist_ok=True
                )
            print("MPAS/JIGSAW setup complete in: " + self.sWorkspace_output)
        else:
            print("MPAS setup complete in: " + self.sWorkspace_output)

        return

    def get_boundary_wkt(self) -> str | None:
        """Resolve the domain boundary as a WKT polygon string.

        Resolution order:
        1. Explicit ``pBoundary_wkt`` in configuration or attribute.
        2. Vector boundary file (``sFilename_mesh_boundary`` / ``sFilename_boundary``)
           when ``iFlag_mesh_boundary`` is enabled or vector file is present.
        3. Regional bounding box (``dLongitude_left``, ``dLongitude_right``,
           ``dLatitude_bot``, ``dLatitude_top``) when not a global domain.
        4. None (for global mesh).
        """
        if self.pBoundary_wkt:
            return self.pBoundary_wkt

        from pyearthmesh.meshes.generate_mesh import determine_mesh_boundary

        boundary_file = self.sFilename_mesh_boundary or self.sFilename_boundary
        wkt, extent = determine_mesh_boundary(
            pBoundary_wkt=self.pBoundary_wkt,
            sFilename_mesh_boundary=boundary_file,
            iFlag_mesh_boundary=getattr(self, "iFlag_mesh_boundary", 0),
            dLongitude_left=self.dLongitude_left,
            dLongitude_right=self.dLongitude_right,
            dLatitude_bot=self.dLatitude_bot,
            dLatitude_top=self.dLatitude_top,
            iFlag_global=getattr(self, "iFlag_global", 1),
        )
        if wkt:
            self.pBoundary_wkt = wkt
        if extent:
            self.dLongitude_left = extent[0]
            self.dLongitude_right = extent[1]
            self.dLatitude_bot = extent[2]
            self.dLatitude_top = extent[3]
            self.dLongitude_mean = 0.5 * (extent[0] + extent[1])
            self.dLatitude_mean = 0.5 * (extent[2] + extent[3])
        return self.pBoundary_wkt

    def run(self):
        """Run the MPAS mesh generation workflow.

        Follows unified boundary-selection logic:
        * If ``iFlag_antarctic == 1`` **or** ``iFlag_arctic == 1``, the polar
          domain is handled internally by ``create_mpas_mesh`` — no
          ``pBoundary_wkt`` is passed.
        * Otherwise the boundary WKT resolved by ``get_boundary_wkt()`` (vector
          file, explicit WKT, or bounding box) is forwarded to ``create_mpas_mesh``.
        """
        from pyearthmesh.meshes.unstructured.mpas.create_mpas_mesh import (
            create_mpas_mesh,
        )

        sWorkspace_jigsaw = (
            self.sWorkspace_output if self.iFlag_run_jigsaw == 1 else None
        )
        aConfig_jigsaw = (
            getattr(self, "aConfig_jigsaw", None)
            if self.iFlag_run_jigsaw == 1
            else None
        )

        if self.iFlag_antarctic == 1 or self.iFlag_arctic == 1:
            pBoundary_wkt = None
        else:
            pBoundary_wkt = self.get_boundary_wkt()

        sFilename_land_ocean_mask = (
            self.sFilename_land_ocean_mask
            if getattr(self, "iFlag_land_ocean_mask", 0) == 1
            else None
        )

        self.aMpas = create_mpas_mesh(
            self.sFilename_mesh,
            iFlag_global_in=self.iFlag_global,
            iFlag_use_mesh_dem_in=self.iFlag_use_mesh_dem,
            iFlag_save_mesh_in=self.iFlag_save_mesh,
            iFlag_run_jigsaw_in=self.iFlag_run_jigsaw,
            iFlag_antarctic_in=self.iFlag_antarctic,
            iFlag_arctic_in=self.iFlag_arctic,
            pBoundary_in=pBoundary_wkt,
            sFilename_mesh_boundary_in=self.sFilename_mesh_boundary,
            sWorkspace_jigsaw_in=sWorkspace_jigsaw,
            sFilename_mpas_mesh_netcdf_in=self.sFilename_mpas_mesh_netcdf,
            sFilename_jigsaw_mesh_netcdf_in=self.sFilename_jigsaw_mesh_netcdf,
            sFilename_land_ocean_mask_in=sFilename_land_ocean_mask,
            aConfig_jigsaw_in=aConfig_jigsaw,
        )
        return self.aMpas

    def create_hpc_job(self, slurm=None, hours=10, **kwargs):
        """Create an HPC job for this MPAS mesh-generation simulation.

        Generates two files in self.sWorkspace_output:
          - run_mpas.py  : Python driver script that reads the configuration,
                           calls setup() and run() on the mpascase object.
          - submit.job   : SLURM batch script that activates the conda
                           environment and executes run_mpas.py.

        Args:
            slurm (str, optional): SLURM partition name. Defaults to 'slurm'.
            hours (int, optional): Wall-clock time limit in hours. Defaults to 10.
        """
        if slurm is None:
            slurm = kwargs.get("sSlurm_in", "slurm")
        if "hours_in" in kwargs:
            hours = kwargs["hours_in"]
        sSlurm_in = slurm
        hours_in = hours
        sConda_env_path, sConda_env_name, env_type = get_python_environment()

        # ------------------------------------------------------------------
        # Part 1 – Python driver script (run_mpas.py)
        # ------------------------------------------------------------------
        sFilename_mpas = os.path.join(str(Path(self.sWorkspace_output)), "run_mpas.py")
        ofs_mpas = open(sFilename_mpas, "w")

        sLine = (
            "#!/qfs/people/liao313/.conda/envs/"
            + sConda_env_name
            + "/bin/"
            + "python3"
            + "\n"
        )
        ofs_mpas.write(sLine)
        sLine = "import os" + "\n"
        ofs_mpas.write(sLine)
        sLine = (
            "os.environ['PROJ_LIB']="
            + '"'
            + sConda_env_path
            + "/share/proj"
            + '"'
            + "\n"
        )
        ofs_mpas.write(sLine)
        sLine = (
            "os.environ['LD_LIBRARY_PATH']="
            + '"'
            + sConda_env_path
            + '/lib:${LD_LIBRARY_PATH}"'
            + "\n"
        )
        ofs_mpas.write(sLine)
        sFilename_workflow_in = self.sFilename_model_configuration
        if not sFilename_workflow_in:
            try:
                from pyearthmesh.config.config_manager import dump_workflow_config

                sFilename_workflow_in = os.path.join(
                    str(Path(self.sWorkspace_output)), "workflow_config.yaml"
                )
                dump_workflow_config(self.config, sFilename_workflow_in, fmt="yaml")
            except Exception:
                sFilename_workflow_in = os.path.join(
                    str(Path(self.sWorkspace_output)), "workflow_config.json"
                )
                with open(sFilename_workflow_in, "w") as f_cfg:
                    json.dump(self.config, f_cfg, indent=2)

        sLine = "from pyearthmesh.config.config_manager import create_workflow" + "\n"
        ofs_mpas.write(sLine)
        sLine = "sFilename_workflow_in = " + '"' + str(sFilename_workflow_in) + '"\n'
        ofs_mpas.write(sLine)
        sLine = "oWorkflow = create_workflow(sFilename_workflow_in)" + "\n"
        ofs_mpas.write(sLine)
        sLine = "oWorkflow.setup()" + "\n"
        ofs_mpas.write(sLine)
        sLine = "oWorkflow.run()" + "\n"
        ofs_mpas.write(sLine)
        sLine = "print('Finished')" + "\n"
        ofs_mpas.write(sLine)
        ofs_mpas.close()
        os.chmod(sFilename_mpas, stat.S_IREAD | stat.S_IWRITE | stat.S_IXUSR)

        # ------------------------------------------------------------------
        # Part 2 – SLURM batch script (submit.job)
        # ------------------------------------------------------------------
        sFilename_job = os.path.join(str(Path(self.sWorkspace_output)), "submit.job")
        ofs = open(sFilename_job, "w")
        sLine = "#!/bin/bash\n"
        ofs.write(sLine)
        sLine = "#SBATCH -A E3SM\n"
        ofs.write(sLine)
        sLine = "#SBATCH --job-name=" + self.sModel + "\n"
        ofs.write(sLine)
        sHour = f"{hours_in:02d}"
        sLine = "#SBATCH -t " + sHour + ":00:00" + "\n"
        ofs.write(sLine)
        sLine = "#SBATCH --nodes=1" + "\n"
        ofs.write(sLine)
        sLine = "#SBATCH --ntasks-per-node=1" + "\n"
        ofs.write(sLine)
        if sSlurm_in is not None:
            sSlurm = sSlurm_in
        else:
            sSlurm = "slurm"
        sLine = "#SBATCH --partition=" + sSlurm + "\n"
        ofs.write(sLine)
        sLine = "#SBATCH -o stdout.out\n"
        ofs.write(sLine)
        sLine = "#SBATCH -e stderr.err\n"
        ofs.write(sLine)
        sLine = "module purge\n"
        ofs.write(sLine)
        sLine = "module load gcc/8.1.0" + "\n"
        ofs.write(sLine)
        sLine = "module load python/miniconda2024May29 " + "\n"
        ofs.write(sLine)
        sLine = (
            "source /share/apps/python/miniconda2024May29/etc/profile.d/conda.sh" + "\n"
        )
        ofs.write(sLine)
        sLine = "conda activate " + sConda_env_name + "\n"
        ofs.write(sLine)
        sLine = "cd $SLURM_SUBMIT_DIR\n"
        ofs.write(sLine)
        sLine = "JOB_DIRECTORY=" + self.sWorkspace_output + "\n"
        ofs.write(sLine)
        sLine = "cd $JOB_DIRECTORY" + "\n"
        ofs.write(sLine)
        sLine = "python3 run_mpas.py" + "\n"
        ofs.write(sLine)
        sLine = "conda deactivate" + "\n"
        ofs.write(sLine)
        ofs.close()
        return

    _create_hpc_job = create_hpc_job
    _mpas_create_hpc_job = create_hpc_job


# Backward-compatible and uniform adapter naming alias
MPASCase = mpascase
