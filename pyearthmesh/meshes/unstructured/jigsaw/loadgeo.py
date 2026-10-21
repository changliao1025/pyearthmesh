import numpy as np
from jigsawpy import jigsaw_msh_t
from osgeo import ogr

ogr.DontUseExceptions()


def pointgeo(point, nset, eset, nobj, last):
    """
    POINTGEO: read a geoJSON point into a jigsaw msh_t object.

    """
    # Authors: Darren Engwirda

    # ----------------------------------------- read POINT dataset
    temp = jigsaw_msh_t()
    temp.vert2 = np.zeros((1), dtype=temp.VERT2_t)
    temp.vert2["coord"] = point[0:2]
    nset.append(temp.vert2)

    nobj = 1

    return nobj, last


def linegeo(line, nset, eset, nobj, last):
    """
    LINEGEO: read a geoJSON line into a jigsaw msh_t object.

    """
    # Authors: Darren Engwirda

    npts = len(line) - 0

    if npts > 0:
        # ----------------------------------------- read LINE dataset
        temp = jigsaw_msh_t()
        temp.vert2 = np.zeros((npts + 0), dtype=temp.VERT2_t)
        temp.edge2 = np.zeros((npts - 1), dtype=temp.EDGE2_t)

        temp.edge2["IDtag"][:] = nobj

        nobj = nobj + 1

        index = np.arange(0, npts - 1) + last

        last = last + npts
        line_dummy = np.array(line)
        temp.vert2["coord"] = line_dummy[:, 0:2]  # line[+0::]

        temp.edge2["index"][:, 0] = index + 0
        temp.edge2["index"][:, 1] = index + 1

        nset.append(temp.vert2)
        eset.append(temp.edge2)

    return nobj, last


def polygeo(loop, nset, eset, nobj, last):
    """
    POLYGEO: read a geoJSON poly into a jigsaw msh_t object.

    """
    # Authors: Darren Engwirda

    npts = len(loop) - 1

    if npts > +0:
        # ----------------------------------------- read LOOP dataset
        temp = jigsaw_msh_t()
        temp.vert2 = np.zeros((npts + 0), dtype=temp.VERT2_t)
        temp.edge2 = np.zeros((npts - 0), dtype=temp.EDGE2_t)

        temp.edge2["IDtag"][:] = nobj

        nobj = nobj + 1

        index = np.arange(0, npts - 1) + last

        itop = last + npts - 1

        idx1 = np.full(npts, itop, dtype=np.int32)
        idx1[:-1] = index + 0

        idx2 = np.full(npts, last, dtype=np.int32)
        idx2[:-1] = index + 1

        last = last + npts

        # modified by Chang Liao
        # Check if z coordinates are present
        loop_array = np.array(loop)
        if len(loop[0]) == 3:
            temp.vert2["coord"] = loop_array[:-1, 0:2]
        else:
            temp.vert2["coord"] = loop_array[:-1, 0:2]

        temp.edge2["index"][:, 0] = idx1
        temp.edge2["index"][:, 1] = idx2

        nset.append(temp.vert2)
        eset.append(temp.edge2)

    return nobj, last


def readgeo(geom, nset, eset, nobj, last):
    """
    READGEO: read a geoJSON part into a jigsaw msh_t object.

    """
    # Authors: Darren Engwirda and Chang Liao

    geom_type = ogr.GT_Flatten(geom.GetGeometryType())

    if geom_type == ogr.wkbPoint:

        point = (geom.GetX(), geom.GetY())

        nobj, last = pointgeo(point, nset, eset, nobj, last)

    elif geom_type == ogr.wkbLineString:

        line = geom.GetPoints()

        nobj, last = linegeo(line, nset, eset, nobj, last)

    elif geom_type == ogr.wkbMultiLineString:

        for index in range(geom.GetGeometryCount()):
            nobj, last = readgeo(geom.GetGeometryRef(index), nset, eset, nobj, last)

    elif geom_type == ogr.wkbPolygon:

        for index in range(geom.GetGeometryCount()):
            loop = geom.GetGeometryRef(index).GetPoints()
            nobj, last = polygeo(loop, nset, eset, nobj, last)

    elif geom_type == ogr.wkbMultiPolygon:

        for index in range(geom.GetGeometryCount()):
            nobj, last = readgeo(geom.GetGeometryRef(index), nset, eset, nobj, last)

    elif geom_type == ogr.wkbGeometryCollection:

        for index in range(geom.GetGeometryCount()):
            nobj, last = readgeo(geom.GetGeometryRef(index), nset, eset, nobj, last)

    return nobj, last


def loadgeo(name, mesh):
    """
    LOADGEO: load a geoJSON file into a jigsaw msh_t object.

    """
    # Authors: Darren Engwirda

    if not isinstance(name, str):
        raise TypeError("name must be a string")

    if not isinstance(mesh, jigsaw_msh_t):
        raise TypeError("mesh must be a jigsaw_msh_t instance")

    nobj = +0
    last = +0
    nset = []
    eset = []

    datasource = ogr.Open(name, 0)
    if datasource is None:
        raise FileNotFoundError(f"Could not open {name}")

    layer = datasource.GetLayer()
    if layer is None:
        raise ValueError(f"Could not read a layer from {name}")

    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is not None:
            nobj, last = readgeo(geom, nset, eset, nobj, last)

    datasource = None

    if len(nset) == 0:
        raise ValueError(f"No supported geometry found in {name}")

    mesh.vert2 = np.concatenate(nset, axis=0)
    if len(eset) > 0:
        mesh.edge2 = np.concatenate(eset, axis=0)
    else:
        mesh.edge2 = None

    return
