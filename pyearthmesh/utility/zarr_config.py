"""
Zarr configuration and constants for mesh storage.

This module provides configuration parameters, data type mappings, and
compression settings for storing mesh data in Zarr format.
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
    },
    "hexagon": {
        "max_vertices": 6,
        "max_neighbors": 6,
        "expected_neighbor_count": 6
    },
    "latlon": {
        "max_vertices": 4,
        "max_neighbors": 4,
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


def get_mesh_config(mesh_type: str) -> Dict[str, Any]:
    """
    Get mesh type configuration.

    Args:
        mesh_type: Type of mesh (dggrid, mpas, tin, etc.)

    Returns:
        Mesh configuration dictionary
    """
    mesh_type_lower = mesh_type.lower()
    if mesh_type_lower not in MESH_TYPE_CONFIGS:
        # Default to dggrid if unknown
        return MESH_TYPE_CONFIGS["dggrid"]
    return MESH_TYPE_CONFIGS[mesh_type_lower]
