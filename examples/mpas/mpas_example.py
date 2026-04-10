#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPAS mesh example
This demonstrates how to create an MPAS mesh using pyearthmesh,
either by running directly or by submitting an HPC (SLURM) job.
"""

import os
from pathlib import Path

from pyearthmesh.config.config_manager import (
    create_template_configuration_file,
    read_configuration_file,
)
from pyearthmesh.utility.change_json_key_value import change_json_key_value
from shutil import copy2

iCase_index = 1
sDate = '20260401'

sMesh_type = 'mpas'

# ── Output workspace ──────────────────────────────────────────────────────────
sWorkspace_output = os.path.join('/compyfs/liao313/04model/pyearthmesh/', sMesh_type)    #'./output/mpas_example'
#covert to absolute path
sWorkspace_output = os.path.abspath(sWorkspace_output)
Path(sWorkspace_output).mkdir(parents=True, exist_ok=True)

# ── Step 1: Create MPAS configuration template ────────────────────────────────
print("Step 1: Creating MPAS configuration template...")
sFilename_mpas_configuration_json = os.path.join(
    sWorkspace_output, 'mpas_configuration.json'
)

config = create_template_configuration_file(
    sFilename_mpas_configuration_json,
    mesh_type=sMesh_type,
    sWorkspace_output=sWorkspace_output,
    iFlag_global=1,
    iFlag_run_jigsaw=1,
)
print("   * Configuration saved to: " + sFilename_mpas_configuration_json)


change_json_key_value(sFilename_mpas_configuration_json, "sWorkspace_output", sWorkspace_output)


# ── Step 2: Read configuration and create mpascase object ─────────────────────
print("\nStep 2: Reading configuration and creating mpascase object...")
oMPAS = read_configuration_file(
    sFilename_mpas_configuration_json,
    iCase_index_in= iCase_index,
    sDate_in= sDate,
    iFlag_create_directory_in=1,
)
print("   * mpascase created")
print("   * Output workspace: " + oMPAS.sWorkspace_output)
print("   * Mesh output file: " + oMPAS.sFilename_mesh)

sWorkspace_output_case = oMPAS.sWorkspace_output

sFilename_mpas_configuration_copy = os.path.join( sWorkspace_output_case, 'mpas_configuration_copy.json' )
copy2(sFilename_mpas_configuration_json, sFilename_mpas_configuration_copy)

oMPAS = read_configuration_file(
    sFilename_mpas_configuration_copy,
    iCase_index_in= iCase_index,
    sDate_in= sDate,
    iFlag_create_directory_in=1,
)

# ── Choose execution mode ─────────────────────────────────────────────────────
# Set iFlag_hpc=1 to generate SLURM job scripts instead of running directly.
iFlag_hpc = 1

if iFlag_hpc == 1:

    # ── Step 4 (HPC): Create SLURM job scripts ────────────────────────────────
    print("\nStep 4: Creating HPC job scripts...")
    oMPAS._mpas_create_hpc_job(
        sSlurm_in='slurm',   # SLURM partition name
        hours_in=10,          # wall-clock time limit in hours
    )
    print("   * run_mpas.py  written to: " + oMPAS.sWorkspace_output)
    print("   * submit.job   written to: " + oMPAS.sWorkspace_output)
    print("   * Submit with: sbatch " + os.path.join(oMPAS.sWorkspace_output, "submit.job"))
else:
    # ── Step 3: Setup workspace directories ───────────────────────────────────────
    print("\nStep 3: Setting up MPAS workspace...")
    oMPAS.setup()
    # ── Step 4 (local): Run MPAS mesh generation directly ─────────────────────
    print("\nStep 4: Running MPAS mesh generation...")
    aMpas = oMPAS.run()

    if aMpas is not None:
        print("   * MPAS mesh created: {:d} cells".format(len(aMpas)))
        print("   * GeoJSON saved to: " + oMPAS.sFilename_mesh)
    else:
        print("   * Mesh generation did not produce output.")

print("\nDone.")
