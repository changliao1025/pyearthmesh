Getting Started
===============

PyEarthMesh is a lightweight mesh generation and spatial modeling toolkit for Earth science applications.
It supports structured grids (Lat-Lon, Hexagon, Square), spherical DGGS (HEALPix, rHEALPix, ISEA), and unstructured meshes (MPAS, TIN, JIGSAW).

Installation
============

PyEarthMesh depends on spatial and scientific libraries including GDAL, JIGSAW, and MPAS-Tools.
We recommend using **conda** for easiest dependency resolution.

Using Conda (Recommended)
-------------------------

.. code-block:: bash

    # Create and activate conda environment
    conda create -n earthmesh python=3.12 -c conda-forge
    conda activate earthmesh

    # Install spatial and mesh dependencies
    conda install -c conda-forge gdal jigsaw jigsawpy mpas-tools netcdf4

    # Install PyEarth and PyEarthMesh
    conda install -c conda-forge pyearthmesh

Using pip
---------

If your system already has GDAL, JIGSAW, and C/Fortran libraries installed:

.. code-block:: bash

    pip install pyearthmesh

Building from Source
--------------------

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/changliao1025/pyearthmesh.git
    cd pyearthmesh

    # Install dependencies
    conda install -c conda-forge gdal jigsaw jigsawpy mpas-tools netcdf4

    # Install in editable mode
    pip install -e .

Quickstart: Two-Tier Workflow System
====================================

PyEarthMesh uses a **two-tier architecture**:
1. You describe the domain intent (region, resolution in km, features) in a single JSON or Python dict.
2. The compiler translates your intent into backend-native generator configurations.

Basic Workflow Example
----------------------

.. code-block:: python

    from pathlib import Path
    from pyearthmesh.config.config_manager import create_workflow

    # 1. Load configuration and build workflow
    workflow = create_workflow("workflow_config.json")

    # 2. Prepare workspace
    workflow.setup()

    # 3. Execute mesh generation
    mesh = workflow.run()

    print(f"Generated mesh with {len(mesh):,} cells!")
    print(f"Mesh file saved to: {workflow.mesh_file}")

Next Steps
==========

- For a complete, step-by-step guide on creating variable-resolution MPAS meshes, see :doc:`mpas_mesh_generation`.
- For details on the configuration hierarchy and supported parameters, see :doc:`config_manager_architecture`.
