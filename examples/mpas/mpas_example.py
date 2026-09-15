#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPAS mesh example (config-driven workflow).

A single sectioned JSON file (``workflow_config.json``) is the one and only source
of truth.  It is loaded by ``load_workflow_config`` and handed to
``create_workflow``, which selects the per-mesh-type workflow class
(``MPASWorkflow``) and builds the underlying ``mpascase`` in memory -- there is no
intermediate flat config file any more.

The example then either runs mesh generation directly or generates HPC (SLURM) job
scripts, depending on ``workflow.run_on_hpc``.
"""

import os

from pyearthmesh.config.config_manager import (
    load_workflow_config,
    create_workflow,
)

# ── Step 1: Load the single workflow configuration ────────────────────────────
sFilename_workflow = os.path.join(os.path.dirname(__file__), "workflow_config.json")
print("Step 1: Loading workflow configuration...")
print("   * " + sFilename_workflow)
aWorkflow = load_workflow_config(sFilename_workflow)

# ── Step 2: Build the workflow (selects MPASWorkflow, builds mpascase) ────────
print("\nStep 2: Creating workflow object...")
oWorkflow = create_workflow(aWorkflow)
oMPAS = oWorkflow.case
print("   * Mesh type:         " + aWorkflow["workflow"]["mesh_type"])
print("   * Output workspace:  " + oMPAS.sWorkspace_output)
print("   * Mesh output file:  " + oMPAS.sFilename_mesh)

# ── Step 3: Execute (mode + HPC settings come from the config) ────────────────
aJigsaw = aWorkflow["jigsaw"]

if aWorkflow["workflow"].get("run_on_hpc"):
    print("\nStep 3 (HPC): Creating SLURM job scripts...")
    oWorkflow.create_hpc_job(
        slurm=aJigsaw["slurm"],   # SLURM partition name
        hours=aJigsaw["hours"],   # wall-clock time limit in hours
    )
    print("   * run_mpas.py  written to: " + oMPAS.sWorkspace_output)
    print("   * submit.job   written to: " + oMPAS.sWorkspace_output)
    print("   * Submit with: sbatch " + os.path.join(oMPAS.sWorkspace_output, "submit.job"))
else:
    print("\nStep 3 (local): Setting up MPAS workspace...")
    oWorkflow.setup()

    print("\nStep 4 (local): Running MPAS mesh generation...")
    aMpas = oWorkflow.run()

    if aMpas is not None:
        print("   * MPAS mesh created: {:d} cells".format(len(aMpas)))
        print("   * GeoJSON saved to: " + oMPAS.sFilename_mesh)
    else:
        print("   * Mesh generation did not produce output.")

print("\nDone.")
