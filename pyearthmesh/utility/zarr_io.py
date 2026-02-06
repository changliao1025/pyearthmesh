"""
Zarr I/O utilities for mesh data.

This module provides classes and functions for reading and writing
mesh data in Zarr format, enabling efficient storage and access of
large meshes with 100+ million cells.
"""
import os
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np
from numpy.typing import NDArray

try:
    import zarr
    from numcodecs import Blosc
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    zarr = None
    Blosc = None

import logging

from pyearthmesh.classes.meshcell import pymeshcell
from pyearthmesh.classes.vertex import pyvertex
from pyearthmesh.classes.edge import pyedge
from pyearthmesh.utility.zarr_config import (
    ZARR_DEFAULTS,
    get_chunk_size_recommendation,
    get_compression_profile,
    get_mesh_config
)

logger = logging.getLogger(__name__)


def check_zarr_available() -> None:
    """Check if Zarr is available and raise error if not."""
    if not ZARR_AVAILABLE:
        raise ImportError(
            "Zarr is not installed. Please install it with: "
            "pip install zarr numcodecs"
        )


class ZarrMeshWriter:
    """
    Writer class for storing mesh data in Zarr format.

    This class handles the conversion of pymeshcell objects into
    a structured Zarr store with efficient chunking and compression.
    """

    def __init__(
        self,
        output_path: str,
        mesh_type: str = "dggrid",
        chunk_size: Optional[int] = None,
        compression_profile: str = "balanced",
        overwrite: bool = False
    ):
        """
        Initialize Zarr mesh writer.

        Args:
            output_path: Path to output Zarr store
            mesh_type: Type of mesh (dggrid, mpas, tin, etc.)
            chunk_size: Number of cells per chunk (auto if None)
            compression_profile: Compression profile (fast, balanced, maximum)
            overwrite: Whether to overwrite existing store
        """
        check_zarr_available()

        self.output_path = output_path
        self.mesh_type = mesh_type
        self.chunk_size = chunk_size
        self.compression_profile = compression_profile

        # Initialize Zarr store
        if overwrite and os.path.exists(output_path):
            import shutil
            shutil.rmtree(output_path)

        self.store = zarr.DirectoryStore(output_path)
        self.root = zarr.group(store=self.store, overwrite=overwrite)

        # Mesh configuration
        self.mesh_config = get_mesh_config(mesh_type)

        # Compression settings
        comp_opts = get_compression_profile(compression_profile)
        self.compressor = Blosc(
            cname=comp_opts["cname"],
            clevel=comp_opts["clevel"],
            shuffle=comp_opts["shuffle"]
        )

        # Statistics
        self.n_cells_written = 0
        self.initialized = False

    def initialize_arrays(
        self,
        n_cells: int,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize Zarr arrays for mesh data.

        Args:
            n_cells: Total number of cells
            attributes: Global attributes (metadata)
        """
        # Auto-determine chunk size if not specified
        if self.chunk_size is None:
            self.chunk_size = get_chunk_size_recommendation(n_cells)

        logger.info(f"Initializing Zarr arrays for {n_cells} cells with chunk size {self.chunk_size}")

        # Store global attributes
        self.root.attrs.update({
            "mesh_type": self.mesh_type,
            "n_cells": n_cells,
            "chunk_size": self.chunk_size,
            "compression_profile": self.compression_profile,
            "pyearthmesh_version": "0.1.0",
            "format_version": "1.0",
            **(attributes or {})
        })

        # Cell IDs
        self.root.create_dataset(
            "cell_ids",
            shape=(n_cells,),
            chunks=(self.chunk_size,),
            dtype=ZARR_DEFAULTS["dtype_mapping"]["cell_id"],
            compressor=self.compressor
        )

        # Coordinates group
        coords = self.root.create_group("coordinates")
        coords.create_dataset(
            "center_lon",
            shape=(n_cells,),
            chunks=(self.chunk_size,),
            dtype=ZARR_DEFAULTS["dtype_mapping"]["coordinates"],
            compressor=self.compressor
        )
        coords.create_dataset(
            "center_lat",
            shape=(n_cells,),
            chunks=(self.chunk_size,),
            dtype=ZARR_DEFAULTS["dtype_mapping"]["coordinates"],
            compressor=self.compressor
        )

        # Vertex coordinates (ragged - use max vertices)
        max_vertices = self.mesh_config["max_vertices"]
        coords.create_dataset(
            "vertices_lon",
            shape=(n_cells, max_vertices),
            chunks=(self.chunk_size, max_vertices),
            dtype=ZARR_DEFAULTS["dtype_mapping"]["coordinates"],
            fill_value=np.nan,
            compressor=self.compressor
        )
        coords.create_dataset(
            "vertices_lat",
            shape=(n_cells, max_vertices),
            chunks=(self.chunk_size, max_vertices),
            dtype=ZARR_DEFAULTS["dtype_mapping"]["coordinates"],
            fill_value=np.nan,
            compressor=self.compressor
        )

        # Geometry group
        geom = self.root.create_group("geometry")
        geom.create_dataset(
            "n_vertices",
            shape=(n_cells,),
            chunks=(self.chunk_size,),
            dtype=ZARR_DEFAULTS["dtype_mapping"]["counts"],
            compressor=self.compressor
        )
        geom.create_dataset(
            "n_edges",
            shape=(n_cells,),
            chunks=(self.chunk_size,),
            dtype=ZARR_DEFAULTS["dtype_mapping"]["counts"],
            compressor=self.compressor
        )
        geom.create_dataset(
            "area",
            shape=(n_cells,),
            chunks=(self.chunk_size,),
            dtype=ZARR_DEFAULTS["dtype_mapping"]["properties"],
            compressor=self.compressor
        )
        geom.create_dataset(
            "perimeter",
            shape=(n_cells,),
            chunks=(self.chunk_size,),
            dtype=ZARR_DEFAULTS["dtype_mapping"]["properties"],
            compressor=self.compressor
        )

        # Topology group
        topo = self.root.create_group("topology")
        max_neighbors = self.mesh_config["max_neighbors"]
        topo.create_dataset(
            "n_neighbors",
            shape=(n_cells,),
            chunks=(self.chunk_size,),
            dtype=ZARR_DEFAULTS["dtype_mapping"]["counts"],
            compressor=self.compressor
        )
        topo.create_dataset(
            "neighbor_ids",
            shape=(n_cells, max_neighbors),
            chunks=(self.chunk_size, max_neighbors),
            dtype=ZARR_DEFAULTS["dtype_mapping"]["neighbors"],
            fill_value=ZARR_DEFAULTS["fill_values"]["neighbor_id"],
            compressor=self.compressor
        )
        topo.create_dataset(
            "neighbor_distances",
            shape=(n_cells, max_neighbors),
            chunks=(self.chunk_size, max_neighbors),
            dtype=ZARR_DEFAULTS["dtype_mapping"]["properties"],
            fill_value=np.nan,
            compressor=self.compressor
        )

        # Properties group
        props = self.root.create_group("properties")
        props.create_dataset(
            "elevation_mean",
            shape=(n_cells,),
            chunks=(self.chunk_size,),
            dtype=ZARR_DEFAULTS["dtype_mapping"]["properties"],
            fill_value=ZARR_DEFAULTS["fill_values"]["elevation"],
            compressor=self.compressor
        )

        # Flags group
        flags = self.root.create_group("flags")
        flags.create_dataset(
            "flag_intersected",
            shape=(n_cells,),
            chunks=(self.chunk_size,),
            dtype=ZARR_DEFAULTS["dtype_mapping"]["flags"],
            fill_value=ZARR_DEFAULTS["fill_values"]["flag"],
            compressor=self.compressor
        )
        flags.create_dataset(
            "flag_coast",
            shape=(n_cells,),
            chunks=(self.chunk_size,),
            dtype=ZARR_DEFAULTS["dtype_mapping"]["flags"],
            fill_value=0,
            compressor=self.compressor
        )

        self.initialized = True

    def write_cells(
        self,
        cells: List[pymeshcell],
        start_index: int = 0,
        show_progress: bool = True
    ) -> None:
        """
        Write mesh cells to Zarr arrays.

        Args:
            cells: List of pymeshcell objects
            start_index: Starting index for writing
            show_progress: Whether to show progress
        """
        if not self.initialized:
            raise RuntimeError("Arrays not initialized. Call initialize_arrays() first.")

        n_cells = len(cells)
        end_index = start_index + n_cells

        if show_progress:
            logger.info(f"Writing {n_cells} cells to Zarr (indices {start_index} to {end_index})")

        # Prepare arrays for batch writing
        cell_ids = np.zeros(n_cells, dtype=np.int64)
        center_lons = np.zeros(n_cells, dtype=np.float32)
        center_lats = np.zeros(n_cells, dtype=np.float32)

        max_vertices = self.mesh_config["max_vertices"]
        vertices_lon = np.full((n_cells, max_vertices), np.nan, dtype=np.float32)
        vertices_lat = np.full((n_cells, max_vertices), np.nan, dtype=np.float32)
        n_vertices = np.zeros(n_cells, dtype=np.int32)
        n_edges = np.zeros(n_cells, dtype=np.int32)
        areas = np.zeros(n_cells, dtype=np.float32)
        perimeters = np.zeros(n_cells, dtype=np.float32)

        max_neighbors = self.mesh_config["max_neighbors"]
        n_neighbors = np.zeros(n_cells, dtype=np.int32)
        neighbor_ids = np.full((n_cells, max_neighbors), -1, dtype=np.int64)
        neighbor_distances = np.full((n_cells, max_neighbors), np.nan, dtype=np.float32)

        elevations = np.full(n_cells, -9999.0, dtype=np.float32)
        flag_intersected = np.full(n_cells, -1, dtype=np.int8)
        flag_coast = np.zeros(n_cells, dtype=np.int8)

        # Extract data from cells
        for i, cell in enumerate(cells):
            cell_ids[i] = cell.lCellID
            center_lons[i] = cell.dLongitude_center_degree
            center_lats[i] = cell.dLatitude_center_degree

            # Vertices
            nv = min(cell.nVertex, max_vertices)
            n_vertices[i] = nv
            for j in range(nv):
                vertices_lon[i, j] = cell.aVertex[j].dLongitude_degree
                vertices_lat[i, j] = cell.aVertex[j].dLatitude_degree

            # Geometry
            n_edges[i] = cell.nEdge
            areas[i] = cell.dArea
            perimeters[i] = cell.dLength

            # Topology
            if cell.aNeighbor is not None:
                nn = min(len(cell.aNeighbor), max_neighbors)
                n_neighbors[i] = nn
                neighbor_ids[i, :nn] = cell.aNeighbor[:nn]

                # Handle distances if available
                if cell.aNeighbor_distance is not None and len(cell.aNeighbor_distance) > 0:
                    neighbor_distances[i, :nn] = cell.aNeighbor_distance[:nn]

            # Properties
            elevations[i] = cell.dElevation_mean

            # Flags
            flag_intersected[i] = cell.iFlag_intersected
            flag_coast[i] = cell.iFlag_coast

        # Write to Zarr arrays
        self.root["cell_ids"][start_index:end_index] = cell_ids
        self.root["coordinates/center_lon"][start_index:end_index] = center_lons
        self.root["coordinates/center_lat"][start_index:end_index] = center_lats
        self.root["coordinates/vertices_lon"][start_index:end_index] = vertices_lon
        self.root["coordinates/vertices_lat"][start_index:end_index] = vertices_lat
        self.root["geometry/n_vertices"][start_index:end_index] = n_vertices
        self.root["geometry/n_edges"][start_index:end_index] = n_edges
        self.root["geometry/area"][start_index:end_index] = areas
        self.root["geometry/perimeter"][start_index:end_index] = perimeters
        self.root["topology/n_neighbors"][start_index:end_index] = n_neighbors
        self.root["topology/neighbor_ids"][start_index:end_index] = neighbor_ids
        self.root["topology/neighbor_distances"][start_index:end_index] = neighbor_distances
        self.root["properties/elevation_mean"][start_index:end_index] = elevations
        self.root["flags/flag_intersected"][start_index:end_index] = flag_intersected
        self.root["flags/flag_coast"][start_index:end_index] = flag_coast

        self.n_cells_written += n_cells

        if show_progress and self.n_cells_written % 1000000 == 0:
            logger.info(f"Written {self.n_cells_written:,} cells so far...")

    def finalize(self) -> None:
        """Finalize writing and close store."""
        logger.info(f"Finalized Zarr store with {self.n_cells_written:,} cells at {self.output_path}")
        self.store.close()


class ZarrMeshReader:
    """
    Reader class for loading mesh data from Zarr format.

    This class provides efficient access to mesh data with support for
    spatial filtering and lazy loading.
    """

    def __init__(self, zarr_path: str):
        """
        Initialize Zarr mesh reader.

        Args:
            zarr_path: Path to Zarr store
        """
        check_zarr_available()

        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

        self.zarr_path = zarr_path
        self.store = zarr.DirectoryStore(zarr_path)
        self.root = zarr.open_group(store=self.store, mode="r")

        # Load metadata
        self.mesh_type = self.root.attrs.get("mesh_type", "unknown")
        self.n_cells = self.root.attrs.get("n_cells", 0)
        self.mesh_config = get_mesh_config(self.mesh_type)

    def get_metadata(self) -> Dict[str, Any]:
        """Get mesh metadata."""
        return dict(self.root.attrs)

    def read_cells(
        self,
        cell_ids: Optional[List[int]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        max_cells: Optional[int] = None
    ) -> List[pymeshcell]:
        """
        Read mesh cells from Zarr store.

        Args:
            cell_ids: Specific cell IDs to read
            bbox: Bounding box (lon_min, lat_min, lon_max, lat_max)
            start_index: Start index for sequential read
            end_index: End index for sequential read
            max_cells: Maximum number of cells to read

        Returns:
            List of pymeshcell objects
        """
        # Determine indices to read
        if cell_ids is not None:
            indices = self._find_cell_indices(cell_ids)
        elif bbox is not None:
            indices = self._find_bbox_indices(bbox)
        elif start_index is not None:
            if end_index is None:
                end_index = self.n_cells
            indices = np.arange(start_index, min(end_index, self.n_cells))
        else:
            # Read all cells
            indices = np.arange(self.n_cells)

        # Apply max_cells limit
        if max_cells is not None and len(indices) > max_cells:
            indices = indices[:max_cells]

        logger.info(f"Reading {len(indices)} cells from Zarr")

        # Read data for selected indices
        cells = self._read_cells_by_indices(indices)

        return cells

    def _read_cells_by_indices(self, indices: NDArray[np.int64]) -> List[pymeshcell]:
        """
        Read cells at specific indices and reconstruct pymeshcell objects.

        Args:
            indices: Array indices to read

        Returns:
            List of pymeshcell objects
        """
        cells = []

        # Read in chunks to avoid memory issues
        chunk_size = 100000
        for i in range(0, len(indices), chunk_size):
            chunk_indices = indices[i:i+chunk_size]

            # Read arrays for this chunk
            cell_ids = self.root["cell_ids"][chunk_indices]
            center_lons = self.root["coordinates/center_lon"][chunk_indices]
            center_lats = self.root["coordinates/center_lat"][chunk_indices]
            vertices_lon = self.root["coordinates/vertices_lon"][chunk_indices]
            vertices_lat = self.root["coordinates/vertices_lat"][chunk_indices]
            n_vertices = self.root["geometry/n_vertices"][chunk_indices]
            n_edges = self.root["geometry/n_edges"][chunk_indices]
            areas = self.root["geometry/area"][chunk_indices]
            perimeters = self.root["geometry/perimeter"][chunk_indices]
            n_neighbors = self.root["topology/n_neighbors"][chunk_indices]
            neighbor_ids = self.root["topology/neighbor_ids"][chunk_indices]
            neighbor_distances = self.root["topology/neighbor_distances"][chunk_indices]
            elevations = self.root["properties/elevation_mean"][chunk_indices]
            flag_intersected = self.root["flags/flag_intersected"][chunk_indices]
            flag_coast = self.root["flags/flag_coast"][chunk_indices]

            # Reconstruct pymeshcell objects
            for j in range(len(chunk_indices)):
                try:
                    # Create vertices
                    nv = int(n_vertices[j])
                    vertices = []
                    for k in range(nv):
                        if not np.isnan(vertices_lon[j, k]):
                            vertex = pyvertex({
                                "dLongitude_degree": float(vertices_lon[j, k]),
                                "dLatitude_degree": float(vertices_lat[j, k])
                            })
                            vertices.append(vertex)

                    # Create edges (simplified - just connecting consecutive vertices)
                    edges = []
                    for k in range(len(vertices)):
                        next_k = (k + 1) % len(vertices)
                        edge = pyedge(vertices[k], vertices[next_k])
                        edges.append(edge)

                    # Create cell
                    if len(vertices) >= 3 and len(edges) >= 3:
                        cell = pymeshcell(
                            float(center_lons[j]),
                            float(center_lats[j]),
                            edges,
                            vertices
                        )

                        # Set cell attributes
                        cell.lCellID = int(cell_ids[j])
                        cell.nVertex = nv
                        cell.nEdge = int(n_edges[j])
                        cell.dArea = float(areas[j])
                        cell.dLength = float(perimeters[j])
                        cell.dElevation_mean = float(elevations[j])
                        cell.iFlag_intersected = int(flag_intersected[j])
                        cell.iFlag_coast = int(flag_coast[j])

                        # Set topology
                        nn = int(n_neighbors[j])
                        if nn > 0:
                            cell.nNeighbor = nn
                            cell.aNeighbor = [int(neighbor_ids[j, k]) for k in range(nn) if neighbor_ids[j, k] != -1]
                            cell.aNeighbor_distance = [float(neighbor_distances[j, k]) for k in range(nn) if not np.isnan(neighbor_distances[j, k])]

                        cells.append(cell)

                except Exception as e:
                    logger.warning(f"Failed to reconstruct cell at index {chunk_indices[j]}: {e}")
                    continue

        return cells

    def _find_cell_indices(self, cell_ids: List[int]) -> NDArray[np.int64]:
        """Find array indices for given cell IDs."""
        all_ids = self.root["cell_ids"][:]
        mask = np.isin(all_ids, cell_ids)
        return np.where(mask)[0]

    def _find_bbox_indices(self, bbox: Tuple[float, float, float, float]) -> NDArray[np.int64]:
        """Find array indices for cells within bounding box."""
        lon_min, lat_min, lon_max, lat_max = bbox

        center_lons = self.root["coordinates/center_lon"][:]
        center_lats = self.root["coordinates/center_lat"][:]

        mask = (
            (center_lons >= lon_min) &
            (center_lons <= lon_max) &
            (center_lats >= lat_min) &
            (center_lats <= lat_max)
        )

        return np.where(mask)[0]


# Convenience functions

def write_mesh_to_zarr(
    mesh_cells: List[pymeshcell],
    output_path: str,
    mesh_type: str = "dggrid",
    chunk_size: Optional[int] = None,
    compression_profile: str = "balanced",
    attributes: Optional[Dict[str, Any]] = None,
    overwrite: bool = False
) -> None:
    """
    Write mesh cells to Zarr format.

    Args:
        mesh_cells: List of pymeshcell objects
        output_path: Path to output Zarr store
        mesh_type: Type of mesh
        chunk_size: Cells per chunk (auto if None)
        compression_profile: Compression profile (fast, balanced, maximum)
        attributes: Additional metadata
        overwrite: Whether to overwrite existing store
    """
    writer = ZarrMeshWriter(
        output_path=output_path,
        mesh_type=mesh_type,
        chunk_size=chunk_size,
        compression_profile=compression_profile,
        overwrite=overwrite
    )

    n_cells = len(mesh_cells)
    writer.initialize_arrays(n_cells, attributes)
    writer.write_cells(mesh_cells)
    writer.finalize()


def read_mesh_from_zarr(
    zarr_path: str,
    cell_ids: Optional[List[int]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    lazy: bool = False,
    max_cells: Optional[int] = None
) -> Union[List[pymeshcell], zarr.Group]:
    """
    Read mesh cells from Zarr format.

    Args:
        zarr_path: Path to Zarr store
        cell_ids: Specific cell IDs to read
        bbox: Bounding box filter
        lazy: If True, return Zarr group for lazy access
        max_cells: Maximum number of cells to read

    Returns:
        List of pymeshcell objects or Zarr group
    """
    reader = ZarrMeshReader(zarr_path)

    if lazy:
        return reader.root
    else:
        return reader.read_cells(cell_ids=cell_ids, bbox=bbox, max_cells=max_cells)
