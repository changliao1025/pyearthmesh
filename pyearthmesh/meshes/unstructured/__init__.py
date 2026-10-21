"""PyEarthMesh unstructured meshes module."""

__all__ = []

try:
    from pyearthmesh.meshes.unstructured.mpas.create_mpas_mesh import create_mpas_mesh

    __all__.append("create_mpas_mesh")
except ImportError:
    pass

try:
    from pyearthmesh.meshes.unstructured.tin.create_tin_mesh import create_tin_mesh

    __all__.append("create_tin_mesh")
except ImportError:
    pass
