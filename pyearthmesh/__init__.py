"""
PyEarthMesh: A lightweight mesh generation and manipulation package for Earth science applications.

This package provides tools for creating and manipulating various types of meshes:
- Structured meshes (square, latlon, hexagon, triangular, DGGRID, Healpix)
- Unstructured meshes (TIN, MPAS)

Part of the PyEarthSuite ecosystem.
"""

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.1"
__author__ = "Chang Liao"
__email__ = "changliao.climate@gmail.com"

if TYPE_CHECKING:
    from pyearthmesh.classes.dggrid import pydggrid
    from pyearthmesh.classes.edge import pyedge
    from pyearthmesh.classes.hexagon import pyhexagon
    from pyearthmesh.classes.latlon import pylatlon
    from pyearthmesh.classes.link import pycelllink
    from pyearthmesh.classes.meshcell import pymeshcell
    from pyearthmesh.classes.mpas import pympas
    from pyearthmesh.classes.square import pysquare
    from pyearthmesh.classes.tin import pytin
    from pyearthmesh.classes.vertex import pyvertex

from pyearthmesh.config import (
    create_model_configuration,
    create_workflow,
    load_workflow_config,
)

_CLASS_MAP = {
    "pydggrid": ("pyearthmesh.classes.dggrid", "pydggrid"),
    "pyedge": ("pyearthmesh.classes.edge", "pyedge"),
    "pyhexagon": ("pyearthmesh.classes.hexagon", "pyhexagon"),
    "pylatlon": ("pyearthmesh.classes.latlon", "pylatlon"),
    "pycelllink": ("pyearthmesh.classes.link", "pycelllink"),
    "pymeshcell": ("pyearthmesh.classes.meshcell", "pymeshcell"),
    "pympas": ("pyearthmesh.classes.mpas", "pympas"),
    "pysquare": ("pyearthmesh.classes.square", "pysquare"),
    "pytin": ("pyearthmesh.classes.tin", "pytin"),
    "pyvertex": ("pyearthmesh.classes.vertex", "pyvertex"),
}

__all__ = [
    "pyvertex",
    "pyedge",
    "pymeshcell",
    "pyhexagon",
    "pysquare",
    "pylatlon",
    "pycelllink",
    "pympas",
    "pytin",
    "pydggrid",
    "create_model_configuration",
    "create_workflow",
    "load_workflow_config",
]


def __getattr__(name: str):
    if name in _CLASS_MAP:
        module_path, attr = _CLASS_MAP[name]
        module = importlib.import_module(module_path)
        cls = getattr(module, attr)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
