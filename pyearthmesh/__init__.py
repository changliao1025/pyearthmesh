"""
PyEarthMesh: A lightweight mesh generation and manipulation package for Earth science applications.

This package provides tools for creating and manipulating various types of meshes:
- Structured meshes (square, latlon, hexagon, triangular, DGGRID, Healpix)
- Unstructured meshes (TIN, MPAS)

Part of the PyEarthSuite ecosystem.
"""

__version__ = "0.1.0"
__author__ = "Chang Liao"
__email__ = "changliao.climate@gmail.com"

# Import main classes
from pyearthmesh.classes.vertex import pyvertex
from pyearthmesh.classes.edge import pyedge
from pyearthmesh.classes.meshcell import pymeshcell
from pyearthmesh.classes.hexagon import pyhexagon
from pyearthmesh.classes.square import pysquare
from pyearthmesh.classes.latlon import pylatlon
from pyearthmesh.classes.link import pycelllink

__all__ = [
    "pyvertex",
    "pyedge",
    "pymeshcell",
    "pyhexagon",
    "pysquare",
    "pylatlon",
    "pycelllink",
]
