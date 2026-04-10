import os
import stat
import json
import datetime
from pathlib import Path
from pyearth.system.python.get_python_environment import get_python_environment

sDate_default = datetime.datetime.today().strftime("%Y%m%d")

class mpascase():
    iFlag_standalone = 1
    iFlag_create_directory = 1
    iFlag_save_config = 1
    """Class to represent a mesh run case with configuration and file paths"""
    iFlag_save_mesh = 1
    sFilename_land_ocean_mask = None
    pBoundary_wkt = None
    aConfig_mesh = None

    def __init__(self, aConfig_in,
                         iFlag_create_directory_in=1,
                         sDate_in=None,
                 sWorkspace_output_in=None):

        if "iCase_index" in aConfig_in:
            iCase_index = int(aConfig_in["iCase_index"])
        else:
            iCase_index = 1
        sCase_index = "{:03d}".format(iCase_index)
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
            self.sFilename_jigsaw_mesh_netcdf = aConfig_in["sFilename_jigsaw_mesh_netcdf"]
        else:
            self.sFilename_jigsaw_mesh_netcdf = None

        if "sFilename_land_ocean_mask" in aConfig_in:
            self.sFilename_land_ocean_mask = aConfig_in["sFilename_land_ocean_mask"]

        if "pBoundary_wkt" in aConfig_in:
            self.pBoundary_wkt = aConfig_in["pBoundary_wkt"]

        self.sModel = 'mpas'

        if "iFlag_global" in aConfig_in:
            self.iFlag_global = int(aConfig_in["iFlag_global"])
        else:
            self.iFlag_global = 1

        if "sFilename_model_configuration" in aConfig_in:
            self.sFilename_model_configuration = aConfig_in["sFilename_model_configuration"]
        else:
            self.sFilename_model_configuration = None

        self.sFilename_mesh = os.path.join(
            str(Path(self.sWorkspace_output)), "mpas.geojson"
        )

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

    def run(self):
        """Run the MPAS mesh generation workflow."""
        from pyearthmesh.meshes.unstructured.mpas.create_mpas_mesh import create_mpas_mesh

        sWorkspace_jigsaw = self.sWorkspace_output if self.iFlag_run_jigsaw == 1 else None
        aConfig_jigsaw = getattr(self, 'aConfig_jigsaw', None) if self.iFlag_run_jigsaw == 1 else None

        self.aMpas = create_mpas_mesh(
            self.sFilename_mesh,
            iFlag_global_in=self.iFlag_global,
            iFlag_save_mesh_in=self.iFlag_save_mesh,
            iFlag_run_jigsaw_in=self.iFlag_run_jigsaw,
            pBoundary_in=self.pBoundary_wkt,
            sWorkspace_jigsaw_in=sWorkspace_jigsaw,
            sFilename_mpas_mesh_netcdf_in=self.sFilename_mpas_mesh_netcdf,
            sFilename_jigsaw_mesh_netcdf_in=self.sFilename_jigsaw_mesh_netcdf,
            sFilename_land_ocean_mask_in=self.sFilename_land_ocean_mask,
            aConfig_jigsaw_in=aConfig_jigsaw,
        )
        return self.aMpas

    def _mpas_create_hpc_job(self, sSlurm_in=None, hours_in=10):
        """Create an HPC job for this MPAS mesh-generation simulation.

        Generates two files in self.sWorkspace_output:
          - run_mpas.py  : Python driver script that reads the configuration,
                           calls setup() and run() on the mpascase object.
          - submit.job   : SLURM batch script that activates the conda
                           environment and executes run_mpas.py.

        Args:
            sSlurm_in (str, optional): SLURM partition name. Defaults to 'slurm'.
            hours_in  (int, optional): Wall-clock time limit in hours. Defaults to 10.
        """
        os.chdir(self.sWorkspace_output)
        sConda_env_path, sConda_env_name, env_type = get_python_environment()

        # ------------------------------------------------------------------
        # Part 1 – Python driver script (run_mpas.py)
        # ------------------------------------------------------------------
        sFilename_mpas = os.path.join(
            str(Path(self.sWorkspace_output)), "run_mpas.py"
        )
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
            "os.environ['PROJ_LIB']=" + '"' + sConda_env_path + "/share/proj" + '"' + "\n"
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
        sLine = (
            "from pyearthmesh.config.config_manager import read_configuration_file"
            + "\n"
        )
        ofs_mpas.write(sLine)
        sLine = (
            "sFilename_configuration_in = "
            + '"'
            + self.sFilename_model_configuration
            + '"\n'
        )
        ofs_mpas.write(sLine)
        sLine = (
            "oMPAS = read_configuration_file(sFilename_configuration_in,"
            + " iFlag_create_directory_in=1)"
            + "\n"
        )
        ofs_mpas.write(sLine)
        sLine = "oMPAS.setup()" + "\n"
        ofs_mpas.write(sLine)
        sLine = "oMPAS.run()" + "\n"
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
        sHour = "{:02d}".format(hours_in)
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
        sLine = "source /share/apps/python/miniconda2024May29/etc/profile.d/conda.sh" + "\n"
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