#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPAS mesh example (config-driven workflow).

A single sectioned JSON file (``workflow_config.json``) is the one and only source
of truth. It is loaded by ``create_workflow``, which selects the per-mesh-type
workflow class (``MPASWorkflow``) and builds the underlying ``mpascase`` in memory.

The example then either runs mesh generation directly or generates HPC (SLURM) job
scripts, depending on ``workflow.run_on_hpc``.
"""

import os
from pathlib import Path

from pyearthmesh.config.config_manager import create_workflow


def main() -> None:
    # ── Step 1 & 2: Build the workflow from configuration ─────────────────────
    sFilename_workflow = Path(__file__).with_name("workflow_config.json")
    print("Step 1: Loading workflow configuration...")
    print(f"   * {sFilename_workflow}")

    print("\nStep 2: Creating workflow object...")
    oWorkflow = create_workflow(sFilename_workflow)
    print(f"   * Mesh type:         {oWorkflow.workflow.get('mesh_type', 'mpas')}")
    print(f"   * Output workspace:  {oWorkflow.output_workspace}")
    print(f"   * Mesh output file:  {oWorkflow.mesh_file}")

    # ── Step 3: Execute (mode + HPC settings come from the config) ────────────
    if oWorkflow.workflow.get("run_on_hpc"):
        print("\nStep 3 (HPC): Creating SLURM job scripts...")
        oWorkflow.create_hpc_job()
        print(f"   * run_mpas.py  written to: {oWorkflow.output_workspace}")
        print(f"   * submit.job   written to: {oWorkflow.output_workspace}")
        print(f"   * Submit with: sbatch {os.path.join(oWorkflow.output_workspace, 'submit.job')}")
    else:
        print("\nStep 3 (local): Setting up MPAS workspace...")
        oWorkflow.setup()

        print("\nStep 4 (local): Running MPAS mesh generation...")
        aMpas = oWorkflow.run()

        if aMpas is not None:
            print(f"   * MPAS mesh created: {len(aMpas):d} cells")
            print(f"   * GeoJSON saved to: {oWorkflow.mesh_file}")
        else:
            print("   * Mesh generation did not produce output.")

    print("\nDone.")


if __name__ == "__main__":
    main()
