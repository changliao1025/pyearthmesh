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
from pyearthmesh.classes.vertex import Vertex
from pyearthmesh.classes.edge import Edge
from pyearthmesh.classes.meshcell import MeshCell
from pyearthmesh.classes.hexagon import Hexagon
from pyearthmesh.classes.square import Square
from pyearthmesh.classes.latlon import LatLon
from pyearthmesh.classes.link import Link

__all__ = [
    "Vertex",
    "Edge",
    "MeshCell",
    "Hexagon",
    "Square",
    "LatLon",
    "Link",
]
