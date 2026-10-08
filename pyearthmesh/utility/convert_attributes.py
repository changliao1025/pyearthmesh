import numpy as np

from pyearthmesh.classes.vertex import pyvertex
from pyearthmesh.classes.edge import pyedge
from pyearthmesh.classes.meshcell import pymeshcell, celltype
from pyearthmesh.classes.mpas import pympas
from pyearthmesh.classes.hexagon import pyhexagon
from pyearthmesh.classes.square import pysquare
from pyearthmesh.classes.latlon import pylatlon
from pyearthmesh.classes.tin import pytin


def convert_gcs_attributes_to_meshcell(
    iMesh_type_in,
    dLongitude_center_in=None,
    dLatitude_center_in=None,
    aCoordinates_gcs_in=None,
    aVertexID_in=None,
    aEdgeID_in=None,
    aVertexIndexOnEdge_in=None,
):
    """Convert GCS coordinates with topology attributes to a mesh cell object.

    Supports both the 7-parameter signature:
        (iMesh_type, lon_center, lat_center, coords_gcs, vertex_ids, edge_ids, vertex_index_on_edge)
    and the legacy 5-parameter signature:
        (lon_center, lat_center, coords_gcs, vertex_ids, edge_ids)

    Args:
        iMesh_type_in (int | str | celltype): Mesh type (1=hexagon, 2=square, 3=latlon, 4=mpas, 5=tin).
        dLongitude_center_in (float): Longitude of the cell center in degrees.
        dLatitude_center_in (float): Latitude of the cell center in degrees.
        aCoordinates_gcs_in (array-like): List/array of (lon, lat) coordinates for each vertex.
        aVertexID_in (array-like): Global vertex IDs corresponding to coordinates.
        aEdgeID_in (array-like): Global edge IDs.
        aVertexIndexOnEdge_in (array-like, optional): (N, 2) matrix of [start_vertex_id, end_vertex_id] for each edge.

    Returns:
        pymeshcell | pympas | pyhexagon | pysquare | pylatlon | pytin: Constructed cell object.
    """
    # Handle legacy 5-parameter call where iMesh_type_in was omitted:
    # convert_gcs_attributes_to_meshcell(dLon, dLat, coords, vertex_ids, edge_ids)
    if isinstance(dLatitude_center_in, (list, tuple, np.ndarray)):
        aEdgeID_in = aVertexID_in
        aVertexID_in = aCoordinates_gcs_in
        aCoordinates_gcs_in = dLatitude_center_in
        dLatitude_center_in = float(dLongitude_center_in)
        dLongitude_center_in = float(iMesh_type_in)
        iMesh_type_in = 4  # Default to mpas

    # Normalize mesh type
    if isinstance(iMesh_type_in, celltype):
        mesh_type_id = iMesh_type_in.value
    elif isinstance(iMesh_type_in, str):
        type_str = iMesh_type_in.strip().lower()
        mapping = {"hexagon": 1, "square": 2, "latlon": 3, "mpas": 4, "tin": 5, "dggrid": 6}
        mesh_type_id = mapping.get(type_str, 4)
    else:
        mesh_type_id = int(iMesh_type_in)

    aCoordinates_gcs_in = np.asarray(aCoordinates_gcs_in)
    aVertexID_in = np.asarray(aVertexID_in)
    aEdgeID_in = np.asarray(aEdgeID_in)
    npoint = len(aVertexID_in)

    if mesh_type_id == 4:  # MPAS
        aVertex = []
        for i in range(npoint):
            lon = float(aCoordinates_gcs_in[i, 0])
            lat = float(aCoordinates_gcs_in[i, 1])
            pVertex = pyvertex({"dLongitude_degree": lon, "dLatitude_degree": lat})
            pVertex.lVertexID = int(aVertexID_in[i])
            aVertex.append(pVertex)

        aEdge = []
        for j in range(npoint):
            if aVertexIndexOnEdge_in is not None:
                aVertexID_dummy = aVertexIndexOnEdge_in[j, :]
                dummy_index_0 = np.where(aVertexID_in == aVertexID_dummy[0])[0]
                if len(dummy_index_0) == 0:
                    print("Vertex ID not found:", aVertexID_dummy[0])
                    return None
                start_idx = dummy_index_0[0]

                dummy_index_1 = np.where(aVertexID_in == aVertexID_dummy[1])[0]
                if len(dummy_index_1) == 0:
                    print("Vertex ID not found:", aVertexID_dummy[1])
                    return None
                end_idx = dummy_index_1[0]

                pVertex_start = pyvertex({
                    "dLongitude_degree": float(aCoordinates_gcs_in[start_idx, 0]),
                    "dLatitude_degree": float(aCoordinates_gcs_in[start_idx, 1]),
                })
                pVertex_start.lVertexID = int(aVertexID_dummy[0])

                pVertex_end = pyvertex({
                    "dLongitude_degree": float(aCoordinates_gcs_in[end_idx, 0]),
                    "dLatitude_degree": float(aCoordinates_gcs_in[end_idx, 1]),
                })
                pVertex_end.lVertexID = int(aVertexID_dummy[1])
            else:
                pVertex_start = aVertex[j]
                pVertex_end = aVertex[(j + 1) % npoint]

            pEdge = pyedge(pVertex_start, pVertex_end)
            pEdge.lEdgeID = int(aEdgeID_in[j])
            aEdge.append(pEdge)

        pMpas = pympas(dLongitude_center_in, dLatitude_center_in, aEdge, aVertex)
        return pMpas

    # For standard structured/regular meshes (hexagon, square, latlon, tin)
    aVertex = []
    for i in range(npoint):
        lon = float(aCoordinates_gcs_in[i, 0])
        lat = float(aCoordinates_gcs_in[i, 1])
        pVertex = pyvertex({"dLongitude_degree": lon, "dLatitude_degree": lat})
        pVertex.lVertexID = int(aVertexID_in[i])
        aVertex.append(pVertex)

    aEdge = [
        pyedge(aVertex[i], aVertex[(i + 1) % npoint])
        for i in range(npoint)
    ]
    for i, edge in enumerate(aEdge):
        if i < len(aEdgeID_in):
            edge.lEdgeID = int(aEdgeID_in[i])

    if mesh_type_id == 1:
        return pyhexagon(dLongitude_center_in, dLatitude_center_in, aEdge, aVertex)
    elif mesh_type_id == 2:
        return pysquare(dLongitude_center_in, dLatitude_center_in, aEdge, aVertex)
    elif mesh_type_id == 3:
        return pylatlon(dLongitude_center_in, dLatitude_center_in, aEdge, aVertex)
    elif mesh_type_id == 5:
        return pytin(dLongitude_center_in, dLatitude_center_in, aEdge, aVertex)

    return pymeshcell(dLongitude_center_in, dLatitude_center_in, aEdge, aVertex)


def convert_pcs_attributes_to_meshcell(
    iMesh_type_in,
    aCoordinates_pcs_in,
    aVertexID_in,
    aEdgeID_in,
    aVertexIndexOnEdge_in=None,
):
    """Convert projected coordinate system attributes to a mesh cell."""
    aCoordinates_pcs_in = np.asarray(aCoordinates_pcs_in)
    aVertexID_in = np.asarray(aVertexID_in)
    aEdgeID_in = np.asarray(aEdgeID_in)
    npoint = len(aVertexID_in)

    aVertex = []
    for i in range(npoint):
        pVertex = pyvertex({
            "dLongitude_degree": float(aCoordinates_pcs_in[i, 0]),
            "dLatitude_degree": float(aCoordinates_pcs_in[i, 1]),
        })
        pVertex.lVertexID = int(aVertexID_in[i])
        aVertex.append(pVertex)

    aEdge = []
    for j in range(npoint):
        if aVertexIndexOnEdge_in is not None:
            aVertexID_dummy = aVertexIndexOnEdge_in[j, :]
            dummy_index_0 = np.where(aVertexID_in == aVertexID_dummy[0])[0]
            dummy_index_1 = np.where(aVertexID_in == aVertexID_dummy[1])[0]
            if len(dummy_index_0) == 0 or len(dummy_index_1) == 0:
                return None
            start_idx = dummy_index_0[0]
            end_idx = dummy_index_1[0]
            pVertex_start = pyvertex({
                "dLongitude_degree": float(aCoordinates_pcs_in[start_idx, 0]),
                "dLatitude_degree": float(aCoordinates_pcs_in[start_idx, 1]),
            })
            pVertex_start.lVertexID = int(aVertexID_dummy[0])
            pVertex_end = pyvertex({
                "dLongitude_degree": float(aCoordinates_pcs_in[end_idx, 0]),
                "dLatitude_degree": float(aCoordinates_pcs_in[end_idx, 1]),
            })
            pVertex_end.lVertexID = int(aVertexID_dummy[1])
        else:
            pVertex_start = aVertex[j]
            pVertex_end = aVertex[(j + 1) % npoint]

        pEdge = pyedge(pVertex_start, pVertex_end)
        pEdge.lEdgeID = int(aEdgeID_in[j])
        aEdge.append(pEdge)

    mesh_type_id = int(iMesh_type_in) if not isinstance(iMesh_type_in, celltype) else iMesh_type_in.value
    if mesh_type_id == 4:
        return pympas(aEdge, aVertex)
    elif mesh_type_id == 1:
        return pyhexagon(aEdge, aVertex)
    elif mesh_type_id == 2:
        return pysquare(aEdge, aVertex)
    elif mesh_type_id == 3:
        return pylatlon(aEdge, aVertex)
    elif mesh_type_id == 5:
        return pytin(aEdge, aVertex)
    return pymeshcell(aEdge, aVertex)


# Aliases for pyflowline naming compatibility
convert_gcs_attributes_to_cell = convert_gcs_attributes_to_meshcell
convert_pcs_attributes_to_cell = convert_pcs_attributes_to_meshcell
