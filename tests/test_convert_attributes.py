import numpy as np
import pytest

from pyearthmesh.classes.meshcell import pymeshcell
from pyearthmesh.classes.mpas import pympas
from pyearthmesh.utility.convert_attributes import (
    convert_gcs_attributes_to_cell,
    convert_gcs_attributes_to_meshcell,
)


def test_convert_gcs_attributes_to_meshcell_mpas():
    # 6-vertex hexagon cell (unclosed, shape (6, 2))
    coords = np.array(
        [
            [100.0, 35.0],
            [101.0, 35.0],
            [101.5, 36.0],
            [101.0, 37.0],
            [100.0, 37.0],
            [99.5, 36.0],
        ]
    )
    vertex_ids = np.array([10, 11, 12, 13, 14, 15])
    edge_ids = np.array([101, 102, 103, 104, 105, 106])
    vertex_on_edge = np.array(
        [
            [10, 11],
            [11, 12],
            [12, 13],
            [13, 14],
            [14, 15],
            [15, 10],
        ]
    )

    cell = convert_gcs_attributes_to_meshcell(
        4,
        100.5,
        36.0,
        coords,
        vertex_ids,
        edge_ids,
        vertex_on_edge,
    )

    assert isinstance(cell, pympas)
    assert cell.nVertex == 6
    assert cell.nEdge == 6
    assert len(cell.aVertex) == 6
    assert len(cell.aEdge) == 6

    # Verify vertex IDs
    for i, v in enumerate(cell.aVertex):
        assert v.lVertexID == vertex_ids[i]

    # Verify edge IDs and endpoint vertex IDs from vertex_on_edge
    for j, e in enumerate(cell.aEdge):
        assert e.lEdgeID == edge_ids[j]
        assert e.pVertex_start.lVertexID == vertex_on_edge[j, 0]
        assert e.pVertex_end.lVertexID == vertex_on_edge[j, 1]

    # Check area and edge length calculation
    assert cell.dArea > 0.0
    cell.calculate_edge_length()
    assert cell.dLength > 0.0
    assert cell.dLength_edge == cell.dLength


def test_convert_gcs_attributes_to_meshcell_mpas_triangle():
    # 3-vertex triangle cell
    coords = np.array(
        [
            [100.0, 35.0],
            [101.0, 35.0],
            [100.5, 36.0],
        ]
    )
    vertex_ids = np.array([1, 2, 3])
    edge_ids = np.array([10, 20, 30])
    vertex_on_edge = np.array(
        [
            [1, 2],
            [2, 3],
            [3, 1],
        ]
    )

    cell = convert_gcs_attributes_to_meshcell(
        4,
        100.5,
        35.33,
        coords,
        vertex_ids,
        edge_ids,
        vertex_on_edge,
    )

    assert isinstance(cell, pympas)
    assert cell.nVertex == 3
    assert cell.nEdge == 3


def test_convert_gcs_attributes_to_meshcell_alias():
    # Alias convert_gcs_attributes_to_cell exists and functions identically
    assert convert_gcs_attributes_to_cell is convert_gcs_attributes_to_meshcell


def test_convert_gcs_attributes_to_meshcell_legacy_5_arg():
    coords = np.array(
        [
            [100.0, 35.0],
            [101.0, 35.0],
            [101.5, 36.0],
            [100.5, 36.0],
        ]
    )
    vertex_ids = np.array([1, 2, 3, 4])
    edge_ids = np.array([10, 20, 30, 40])

    cell = convert_gcs_attributes_to_meshcell(
        100.5,
        35.5,
        coords,
        vertex_ids,
        edge_ids,
    )
    assert cell is not None
    assert cell.nVertex == 4
    assert cell.nEdge == 4
