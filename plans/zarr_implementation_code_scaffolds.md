# Zarr Implementation Code Scaffolds

This document provides detailed code scaffolds and implementation guidance for the Zarr support in PyEarthMesh.

## Module Structure

```
pyearthmesh/
├── utility/
│   ├── zarr_io.py              # Core Zarr I/O (NEW)
│   ├── zarr_config.py          # Configuration and constants (NEW)
│   ├── zarr_spatial_index.py  # Spatial indexing utilities (NEW)
│   └── convert_mesh_formats.py # Format conversion (NEW)
└── meshes/
    └── structured/
        └── dggs/
            └── dggrid/
                └── create_dggrid_mesh.py  # Updated to support Zarr
```

## 1. Configuration Module: `zarr_config.py`

```python
"""
Zarr configuration and constants for mesh storage.
"""
from typing import Dict, Any
import numpy as np

# Default Zarr configuration
ZARR_DEFAULTS: Dict[str, Any] = {
    "chunk_size": 500000,  # cells per chunk
    "compression": "blosc",
    "compression_opts": {
        "cname": "zstd",  # zstd for coordinates/properties, lz4 for IDs
        "clevel": 5,
        "shuffle": 2  # bit-shuffle for better compression
    },
    "dtype_mapping": {
        "cell_id": np.int64,
        "coordinates": np.float32,
        "neighbors": np.int64,
        "properties": np.float32,
        "flags": np.int8,
        "counts": np.int32
    },
    "fill_values": {
        "neighbor_id": -1,
        "elevation": -9999.0,
        "flag": -1
    }
}

# Mesh type specific configurations
MESH_TYPE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "dggrid": {
        "max_vertices": 8,  # Hexagons have 6, but allow flexibility
        "max_neighbors": 8,
        "expected_neighbor_count": 6
    },
    "mpas": {
        "max_vertices": 10,
        "max_neighbors": 10,
        "expected_neighbor_count": 6
    },
    "tin": {
        "max_vertices": 3,
        "max_neighbors": 3,
        "expected_neighbor_count": 3
    },
    "square": {
        "max_vertices": 4,
        "max_neighbors": 8,  # Including diagonals
        "expected_neighbor_count": 4
    }
}

# Coordinate reference system defaults
CRS_DEFAULTS = {
    "default": "EPSG:4326",
    "wgs84": "EPSG:4326",
    "web_mercator": "EPSG:3857"
}

# Compression profiles for different use cases
COMPRESSION_PROFILES = {
    "fast": {
        "cname": "lz4",
        "clevel": 3,
        "shuffle": 1
    },
    "balanced": {
        "cname": "zstd",
        "clevel": 5,
        "shuffle": 2
    },
    "maximum": {
        "cname": "zstd",
        "clevel": 9,
        "shuffle": 2
    }
}

def get_chunk_size_recommendation(n_cells: int) -> int:
    """
    Recommend chunk size based on total cell count.

    Args:
        n_cells: Total number of cells in mesh

    Returns:
        Recommended chunk size
    """
    if n_cells < 1_000_000:
        return 100_000
    elif n_cells < 10_000_000:
        return 500_000
    elif n_cells < 100_000_000:
        return 1_000_000
    else:
        return 2_000_000

def get_compression_profile(profile: str = "balanced") -> Dict[str, Any]:
    """
    Get compression configuration for specified profile.

    Args:
        profile: One of 'fast', 'balanced', 'maximum'

    Returns:
        Compression configuration dictionary
    """
    if profile not in COMPRESSION_PROFILES:
        profile = "balanced"
    return COMPRESSION_PROFILES[profile]
```

## 2. Core Zarr I/O Module: `zarr_io.py`

```python
"""
Zarr I/O utilities for mesh data.
"""
import os
from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np
import zarr
from numcodecs import Blosc
import logging

from pyearthmesh.classes.meshcell import pymeshcell
from pyearthmesh.utility.zarr_config import (
    ZARR_DEFAULTS,
    MESH_TYPE_CONFIGS,
    get_chunk_size_recommendation,
    get_compression_profile
)

logger = logging.getLogger(__name__)


class ZarrMeshWriter:
    """
    Writer class for storing mesh data in Zarr format.
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
        self.mesh_config = MESH_TYPE_CONFIGS.get(mesh_type, MESH_TYPE_CONFIGS["dggrid"])

        # Compression settings
        comp_opts = get_compression_profile(compression_profile)
        self.compressor = Blosc(
            cname=comp_opts["cname"],
            clevel=comp_opts["clevel"],
            shuffle=comp_opts["shuffle"]
        )

        # Statistics
        self.n_cells_written = 0

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
        self.root["properties/elevation_mean"][start_index:end_index] = elevations
        self.root["flags/flag_intersected"][start_index:end_index] = flag_intersected
        self.root["flags/flag_coast"][start_index:end_index] = flag_coast

        self.n_cells_written += n_cells

        if show_progress and self.n_cells_written % 1000000 == 0:
            logger.info(f"Written {self.n_cells_written:,} cells so far...")

    def finalize(self) -> None:
        """Finalize writing and close store."""
        logger.info(f"Finalized Zarr store with {self.n_cells_written:,} cells")
        self.store.close()


class ZarrMeshReader:
    """
    Reader class for loading mesh data from Zarr format.
    """

    def __init__(self, zarr_path: str):
        """
        Initialize Zarr mesh reader.

        Args:
            zarr_path: Path to Zarr store
        """
        self.zarr_path = zarr_path
        self.store = zarr.DirectoryStore(zarr_path)
        self.root = zarr.open_group(store=self.store, mode="r")

        # Load metadata
        self.mesh_type = self.root.attrs.get("mesh_type", "unknown")
        self.n_cells = self.root.attrs.get("n_cells", 0)

    def get_metadata(self) -> Dict[str, Any]:
        """Get mesh metadata."""
        return dict(self.root.attrs)

    def read_cells(
        self,
        cell_ids: Optional[List[int]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None
    ) -> List[pymeshcell]:
        """
        Read mesh cells from Zarr store.

        Args:
            cell_ids: Specific cell IDs to read
            bbox: Bounding box (lon_min, lat_min, lon_max, lat_max)
            start_index: Start index for sequential read
            end_index: End index for sequential read

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
            indices = np.arange(start_index, end_index)
        else:
            # Read all cells
            indices = np.arange(self.n_cells)

        logger.info(f"Reading {len(indices)} cells from Zarr")

        # Read data for selected indices
        # This is simplified - actual implementation needs to reconstruct pymeshcell objects
        cells = []

        # For demonstration: read in chunks to avoid memory issues
        chunk_size = 100000
        for i in range(0, len(indices), chunk_size):
            chunk_indices = indices[i:i+chunk_size]
            # Read arrays for this chunk
            # ... (implementation details)

        return cells

    def _find_cell_indices(self, cell_ids: List[int]) -> np.ndarray:
        """Find array indices for given cell IDs."""
        all_ids = self.root["cell_ids"][:]
        mask = np.isin(all_ids, cell_ids)
        return np.where(mask)[0]

    def _find_bbox_indices(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
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
    lazy: bool = False
) -> Union[List[pymeshcell], zarr.Group]:
    """
    Read mesh cells from Zarr format.

    Args:
        zarr_path: Path to Zarr store
        cell_ids: Specific cell IDs to read
        bbox: Bounding box filter
        lazy: If True, return Zarr group for lazy access

    Returns:
        List of pymeshcell objects or Zarr group
    """
    reader = ZarrMeshReader(zarr_path)

    if lazy:
        return reader.root
    else:
        return reader.read_cells(cell_ids=cell_ids, bbox=bbox)
```

## 3. Format Conversion Module: `convert_mesh_formats.py`

```python
"""
Utilities for converting between mesh formats.
"""
import os
from typing import Optional, List
import logging
from osgeo import ogr
import zarr

from pyearthmesh.utility.zarr_io import ZarrMeshWriter, ZarrMeshReader
from pyearthmesh.classes.meshcell import pymeshcell
from pyearthmesh.utility.convert_coordinates import convert_gcs_coordinates_to_meshcell

logger = logging.getLogger(__name__)


def convert_gpkg_to_zarr(
    gpkg_file: str,
    zarr_path: str,
    mesh_type: str = "dggrid",
    chunk_size: Optional[int] = None,
    compression_profile: str = "balanced",
    batch_size: int = 100000
) -> None:
    """
    Convert GPKG mesh file to Zarr format.

    Args:
        gpkg_file: Input GPKG file path
        zarr_path: Output Zarr store path
        mesh_type: Type of mesh
        chunk_size: Cells per chunk
        compression_profile: Compression profile
        batch_size: Number of cells to read per batch
    """
    logger.info(f"Converting {gpkg_file} to Zarr format at {zarr_path}")

    # Open GPKG
    dataset = ogr.Open(gpkg_file, 0)
    if dataset is None:
        raise ValueError(f"Could not open {gpkg_file}")

    layer = dataset.GetLayer(0)
    n_features = layer.GetFeatureCount()

    logger.info(f"Found {n_features} features in GPKG")

    # Initialize Zarr writer
    writer = ZarrMeshWriter(
        output_path=zarr_path,
        mesh_type=mesh_type,
        chunk_size=chunk_size,
        compression_profile=compression_profile,
        overwrite=True
    )

    # Extract metadata
    spatial_ref = layer.GetSpatialRef()
    attributes = {
        "source_file": os.path.basename(gpkg_file),
        "source_format": "gpkg",
        "crs": spatial_ref.ExportToWkt() if spatial_ref else "EPSG:4326"
    }

    writer.initialize_arrays(n_features, attributes)

    # Read and write in batches
    layer.ResetReading()
    batch_cells = []
    total_written = 0

    for feature in layer:
        # Convert OGR feature to pymeshcell
        cell = _ogr_feature_to_meshcell(feature)
        if cell is not None:
            batch_cells.append(cell)

        if len(batch_cells) >= batch_size:
            writer.write_cells(batch_cells, start_index=total_written)
            total_written += len(batch_cells)
            batch_cells = []

            if total_written % 1000000 == 0:
                logger.info(f"Converted {total_written:,} / {n_features:,} cells")

    # Write remaining cells
    if batch_cells:
        writer.write_cells(batch_cells, start_index=total_written)
        total_written += len(batch_cells)

    writer.finalize()
    dataset = None

    logger.info(f"Successfully converted {total_written:,} cells to Zarr")


def _ogr_feature_to_meshcell(feature: ogr.Feature) -> Optional[pymeshcell]:
    """
    Convert OGR feature to pymeshcell object.

    Args:
        feature: OGR feature

    Returns:
        pymeshcell object or None if conversion fails
    """
    try:
        geometry = feature.GetGeometryRef()
        if geometry is None:
            return None

        # Extract geometry coordinates
        # This is a simplified version - actual implementation needs more detail
        from pyearth.gis.location.get_geometry_coordinates import get_geometry_coordinates
        coords = get_geometry_coordinates(geometry)

        # Create meshcell using convert_gcs_coordinates_to_meshcell
        cell = convert_gcs_coordinates_to_meshcell(
            coords,
            dLongitude_center_degree=coords[:, 0].mean(),
            dLatitude_center_degree=coords[:, 1].mean()
        )

        # Copy attributes from feature
        for i in range(feature.GetFieldCount()):
            field_name = feature.GetFieldDefnRef(i).GetName()
            field_value = feature.GetField(i)
            # Map field names to cell attributes
            # ... (implementation details)

        return cell

    except Exception as e:
        logger.warning(f"Failed to convert feature to meshcell: {e}")
        return None


def convert_zarr_to_gpkg(
    zarr_path: str,
    gpkg_file: str,
    cell_ids: Optional[List[int]] = None
) -> None:
    """
    Convert Zarr mesh to GPKG format.

    Args:
        zarr_path: Input Zarr store path
        gpkg_file: Output GPKG file path
        cell_ids: Optional list of cell IDs to export
    """
    logger.info(f"Converting Zarr {zarr_path} to GPKG {gpkg_file}")

    # Read from Zarr
    reader = ZarrMeshReader(zarr_path)
    cells = reader.read_cells(cell_ids=cell_ids)

    # Create GPKG and write cells
    # This would use existing GPKG writing functionality
    # ... (implementation details)

    logger.info(f"Successfully exported {len(cells)} cells to GPKG")


def convert_to_zarr(
    input_file: str,
    output_path: str,
    source_format: str = "auto",
    **zarr_kwargs
) -> None:
    """
    Convert mesh file to Zarr format (auto-detects format).

    Args:
        input_file: Input mesh file
        output_path: Output Zarr store path
        source_format: Source format (auto, gpkg, netcdf)
        **zarr_kwargs: Additional arguments for Zarr writer
    """
    # Auto-detect format
    if source_format == "auto":
        ext = os.path.splitext(input_file)[1].lower()
        format_map = {
            ".gpkg": "gpkg",
            ".nc": "netcdf",
            ".geojson": "geojson"
        }
        source_format = format_map.get(ext, "unknown")

    # Convert based on format
    if source_format == "gpkg":
        convert_gpkg_to_zarr(input_file, output_path, **zarr_kwargs)
    elif source_format == "netcdf":
        # Implement NetCDF to Zarr conversion
        raise NotImplementedError("NetCDF to Zarr conversion not yet implemented")
    else:
        raise ValueError(f"Unsupported source format: {source_format}")
```

## 4. Integration with DGGRID: Updated `create_dggrid_mesh.py`

Add the following to the existing file:

```python
# Add at top of file
from pyearthmesh.utility.convert_mesh_formats import convert_to_zarr

# Modify function signature
def create_dggrid_mesh(
    iFlag_global,
    iFlag_save_mesh,
    sFilename_mesh,
    sWorkspace_output,
    iResolution_index_in=None,
    sDggrid_type_in=None,
    iFlag_antarctic_in=None,
    iFlag_arctic_in=None,
    sFilename_boundary_in=None,
    # NEW PARAMETERS
    output_format="gpkg",  # gpkg, zarr, or both
    zarr_output_path=None,
    zarr_chunk_size=None,
    zarr_compression="balanced"
):
    """
    Create DGGRID mesh with optional Zarr output.

    New Args:
        output_format: Output format - 'gpkg', 'zarr', or 'both'
        zarr_output_path: Path for Zarr output (auto-generated if None)
        zarr_chunk_size: Chunk size for Zarr (auto if None)
        zarr_compression: Compression profile (fast, balanced, maximum)
    """

    # ... existing code to generate GPKG ...

    # After GPKG is created, optionally convert to Zarr
    if output_format in ["zarr", "both"]:
        if zarr_output_path is None:
            zarr_output_path = sWorkspace_output + slash + "mesh.zarr"

        print(f"Converting mesh to Zarr format at {zarr_output_path}")

        convert_to_zarr(
            input_file=sFilename_mesh,
            output_path=zarr_output_path,
            source_format="gpkg",
            mesh_type=sDggrid_type.lower(),
            chunk_size=zarr_chunk_size,
            compression_profile=zarr_compression
        )

        print(f"Zarr mesh created at: {zarr_output_path}")

    return
```

## 5. Example Usage Scripts

### Example: Create Large Mesh with Zarr

```python
"""
Example: Create 1km resolution ISEA3H mesh with Zarr output.
"""
import os
from pyearthmesh.meshes.structured.dggs.dggrid.create_dggrid_mesh import (
    create_dggrid_mesh,
    dggrid_find_index_by_resolution,
    copy_dggrid_binaries_to_output
)

# Configuration
dResolution_meter = 1000  # 1 km
sWorkspace_output = "/path/to/output/isea3h_1km"
os.makedirs(sWorkspace_output, exist_ok=True)

# Find resolution index
sDggrid_type = "ISEA3H"
iResolution_index = dggrid_find_index_by_resolution(sDggrid_type, dResolution_meter)
print(f"Using resolution index: {iResolution_index}")

# Copy DGGRID binaries
copy_dggrid_binaries_to_output(sWorkspace_output)

# Create mesh with Zarr output
create_dggrid_mesh(
    iFlag_global=1,
    iFlag_save_mesh=1,
    sFilename_mesh=os.path.join(sWorkspace_output, "mesh.gpkg"),
    sWorkspace_output=sWorkspace_output,
    iResolution_index_in=iResolution_index,
    sDggrid_type_in=sDggrid_type,
    output_format="zarr",  # Output to Zarr only
    zarr_output_path=os.path.join(sWorkspace_output, "mesh.zarr"),
    zarr_chunk_size=1000000,  # 1M cells per chunk
    zarr_compression="balanced"
)

print("Mesh creation complete!")
```

### Example: Read Zarr Mesh Subset

```python
"""
Example: Read subset of mesh from Zarr.
"""
from pyearthmesh.utility.zarr_io import read_mesh_from_zarr

# Read cells in bounding box (Continental USA)
zarr_path = "/path/to/output/isea3h_1km/mesh.zarr"
bbox = (-125, 24, -66, 49)  # (lon_min, lat_min, lon_max, lat_max)

print(f"Reading cells in bounding box {bbox}")
cells = read_mesh_from_zarr(
    zarr_path=zarr_path,
    bbox=bbox,
    lazy=False
)

print(f"Loaded {len(cells)} cells from Zarr")

# Access cell data
for i, cell in enumerate(cells[:5]):
    print(f"Cell {i}: ID={cell.lCellID}, "
          f"Center=({cell.dLongitude_center_degree:.4f}, "
          f"{cell.dLatitude_center_degree:.4f}), "
          f"Area={cell.dArea:.2f} m²")
```

### Example: Convert Existing GPKG to Zarr

```python
"""
Example: Convert existing GPKG mesh to Zarr format.
"""
from pyearthmesh.utility.convert_mesh_formats import convert_to_zarr

input_gpkg = "/path/to/existing/mesh.gpkg"
output_zarr = "/path/to/output/mesh.zarr"

print(f"Converting {input_gpkg} to Zarr...")

convert_to_zarr(
    input_file=input_gpkg,
    output_path=output_zarr,
    source_format="gpkg",
    mesh_type="dggrid",
    chunk_size=500000,
    compression_profile="maximum"
)

print(f"Conversion complete! Zarr store at: {output_zarr}")
```

## 6. Testing Strategy

### Unit Tests

```python
"""
tests/test_zarr_io.py
"""
import pytest
import numpy as np
import tempfile
import os
from pyearthmesh.utility.zarr_io import (
    ZarrMeshWriter,
    ZarrMeshReader,
    write_mesh_to_zarr,
    read_mesh_from_zarr
)
from pyearthmesh.classes.meshcell import pymeshcell


def create_mock_cell(cell_id, lon, lat):
    """Create mock mesh cell for testing."""
    # Simplified - actual implementation needs proper vertex/edge setup
    pass


def test_zarr_write_read_roundtrip():
    """Test writing and reading mesh cells."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = os.path.join(tmpdir, "test.zarr")

        # Create test cells
        test_cells = [
            create_mock_cell(i, -180 + i, 0)
            for i in range(100)
        ]

        # Write to Zarr
        write_mesh_to_zarr(
            mesh_cells=test_cells,
            output_path=zarr_path,
            mesh_type="dggrid"
        )

        # Read back
        read_cells = read_mesh_from_zarr(zarr_path)

        # Verify
        assert len(read_cells) == len(test_cells)
        # Add more assertions...


def test_zarr_bbox_query():
    """Test bounding box query."""
    # Implementation...
    pass


def test_zarr_compression_profiles():
    """Test different compression profiles."""
    # Implementation...
    pass
```

## Next Steps

1. Implement core [`zarr_io.py`](../pyearthmesh/utility/zarr_io.py) module
2. Create [`zarr_config.py`](../pyearthmesh/utility/zarr_config.py) with configuration
3. Implement [`convert_mesh_formats.py`](../pyearthmesh/utility/convert_mesh_formats.py)
4. Update [`create_dggrid_mesh.py`](../pyearthmesh/meshes/structured/dggs/dggrid/create_dggrid_mesh.py)
5. Add comprehensive tests
6. Benchmark performance with real data
7. Update documentation

