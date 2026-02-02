import numpy as np

from pyearthmesh.classes.vertex import pyvertex
from pyearthmesh.classes.edge import pyedge
from pyearthmesh.classes.meshcell import pymeshcell

def convert_gcs_attributes_to_meshcell(
    dLongitude_center_in,
    dLatitude_center_in,
    aCoordinates_gcs_in,
    aVertexID_in,
    aEdgeID_in
):
    """
    Convert GCS coordinates with topology attributes to a pypolygon object.

    Args:

        dLongitude_center_in (float): The longitude of the center.
        dLatitude_center_in (float): The latitude of the center.
        aCoordinates_gcs_in (list): A list of GCS coordinates.
        aVertexID_in (list): Vertex IDs to assign to vertices.
        aEdgeID_in (list): Edge IDs to assign to edges.



    Returns:
        pypolygon: A pypolygon object with vertex and edge IDs assigned.
    """
    # Create points from coordinates (exclude the closing point)
    vertices = [
        pyvertex({"dLongitude_degree": float(lon), "dLatitude_degree": float(lat)})
        for lon, lat in aCoordinates_gcs_in[:-1]
    ]

    # Assign vertex IDs to points
    for i, vertex in enumerate(vertices):
        if i < len(aVertexID_in):
            vertex.lVertexID = int(aVertexID_in[i])

    # Create edges between consecutive points
    edges = [
        pyedge(vertices[i], vertices[(i + 1) % len(vertices)])
        for i in range(len(vertices))
    ]

    # Assign edge IDs to edges
    for i, edge in enumerate(edges):
        if i < len(aEdgeID_in):
            edge.lEdgeID = int(aEdgeID_in[i])

    # Add the closing point back to the points list
    vertices.append(vertices[0])

    return pymeshcell(dLongitude_center_in, dLatitude_center_in, edges, vertices)


