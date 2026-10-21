###########################
Frequently Asked Questions
###########################

Installation & Dependencies
===========================

1. Why does importing GDAL or JIGSAW fail?
------------------------------------------

   Mesh packages rely on compiled C/C++ libraries. Installing via ``conda-forge`` ensures all shared libraries (GDAL, GEOS, PROJ, JIGSAW) are binary-compatible:

   .. code-block:: bash

      conda install -c conda-forge gdal jigsaw jigsawpy mpas-tools netcdf4

2. PROJ library path issues (``PROJ_LIB``)
------------------------------------------

   If GDAL cannot find projection definitions, set the ``PROJ_LIB`` environment variable:

   .. code-block:: bash

      export PROJ_LIB=$CONDA_PREFIX/share/proj

Mesh Generation & Workflow
==========================

3. Can I run mesh generation without creating a JSON file?
----------------------------------------------------------

   Yes! You can pass a standard Python dictionary directly into ``create_workflow``:

   .. code-block:: python

      from pyearthmesh.config.config_manager import create_workflow

      spec = {
          "workflow": {"mesh_type": "mpas"},
          "resolutions": {"ocean_km": 100, "land_km": 40.0},
          "paths": {"output_workspace": "./output"},
      }
      workflow = create_workflow(spec)
      workflow.setup()
      workflow.run()

4. How do I generate SLURM job scripts for HPC?
-----------------------------------------------

   Set ``"run_on_hpc": true`` under the ``workflow`` section of your configuration and call ``workflow.create_hpc_job()``:

   .. code-block:: python

      workflow = create_workflow("workflow_config.json")
      workflow.create_hpc_job(slurm="slurm_partition", hours=8)

   This creates ``run_mpas.py`` and ``submit.job`` in your output workspace.
