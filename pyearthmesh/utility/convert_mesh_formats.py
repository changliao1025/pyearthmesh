"""
Utilities for converting between mesh formats.

This module provides functions to convert mesh data between different
formats including any GDAL-supported vector format and Zarr.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from osgeo import ogr, osr
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_driver_from_filename
from pyearth.gis.location.get_geometry_coordinates import get_geometry_coordinates

from pyearthmesh.classes.edge import pyedge
from pyearthmesh.classes.meshcell import pymeshcell
from pyearthmesh.classes.vertex import pyvertex
from pyearthmesh.utility.zarr_io import (
    ZarrMeshReader,
    ZarrMeshWriter,
    check_zarr_available,
)

logger = logging.getLogger(__name__)


def convert_vector_to_zarr(
    input_file: str,
    zarr_path: str,
    mesh_type: str = "dggrid",
    chunk_size: Optional[int] = None,
    compression_profile: str = "balanced",
    batch_size: int = 100000,
) -> None:
    """
    Convert any GDAL-supported vector mesh file to Zarr format.

    Args:
        input_file: Input vector file path (GPKG, GeoJSON, Shapefile, etc.)
        zarr_path: Output Zarr store path
        mesh_type: Type of mesh
        chunk_size: Cells per chunk
        compression_profile: Compression profile
        batch_size: Number of cells to read per batch

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If file cannot be opened
    """
    check_zarr_available()

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logger.info(f"Converting {input_file} to Zarr format at {zarr_path}")

    # Open vector file using GDAL
    dataset = ogr.Open(input_file, 0)
    if dataset is None:
        raise ValueError(f"Could not open {input_file}")

    layer = dataset.GetLayer(0)
    n_features = layer.GetFeatureCount()

    logger.info(f"Found {n_features} features in {input_file}")

    # Initialize Zarr writer
    writer = ZarrMeshWriter(
        output_path=zarr_path,
        mesh_type=mesh_type,
        chunk_size=chunk_size,
        compression_profile=compression_profile,
        overwrite=True,
    )

    # Extract metadata
    spatial_ref = layer.GetSpatialRef()
    attributes = {
        "source_file": os.path.basename(input_file),
        "source_format": os.path.splitext(input_file)[1].lstrip("."),
        "crs": spatial_ref.ExportToWkt() if spatial_ref else "EPSG:4326",
    }

    writer.initialize_arrays(n_features, attributes)

    # Read and write in batches
    layer.ResetReading()
    batch_cells = []
    total_written = 0
    failed_count = 0

    for feature in layer:
        # Convert OGR feature to pymeshcell
        cell = _ogr_feature_to_meshcell(feature)
        if cell is not None:
            batch_cells.append(cell)
        else:
            failed_count += 1

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

    if failed_count > 0:
        logger.warning(f"Failed to convert {failed_count} features")

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
        coords = get_geometry_coordinates(geometry)
        if coords is None or len(coords) < 3:
            return None

        vertices = []
        for i in range(len(coords) - 1):
            vertex = pyvertex(
                {
                    "dLongitude_degree": float(coords[i, 0]),
                    "dLatitude_degree": float(coords[i, 1]),
                }
            )
            vertices.append(vertex)

        edges = []
        for i in range(len(vertices)):
            edge = pyedge(vertices[i], vertices[(i + 1) % len(vertices)])
            edges.append(edge)

        # Calculate center
        center_lon = float(np.mean(coords[:-1, 0]))
        center_lat = float(np.mean(coords[:-1, 1]))

        # Create meshcell
        cell = pymeshcell(
            center_lon,
            center_lat,
            edges,
            vertices,
        )

        # Try to extract cell ID from feature
        fid = feature.GetFID()
        if fid is not None:
            cell.lCellID = fid

        # Try to extract other attributes
        for i in range(feature.GetFieldCount()):
            field_defn = feature.GetFieldDefnRef(i)
            field_name = field_defn.GetName().lower()
            field_value = feature.GetField(i)

            if field_value is None:
                continue

            # Map common field names
            if field_name in ("cellid", "cell_id", "id"):
                cell.lCellID = int(field_value)
            elif field_name in ("area", "darea"):
                cell.dArea = float(field_value)
            elif field_name in ("elevation", "delevation_mean", "elev"):
                cell.dElevation_mean = float(field_value)

        return cell

    except Exception as e:
        logger.debug(f"Failed to convert feature: {e}")
        return None


def convert_zarr_to_vector(
    zarr_path: str,
    output_file: str,
    cell_ids: Optional[List[int]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """
    Convert Zarr mesh to vector format.

    The output format is automatically determined from the file extension.

    Args:
        zarr_path: Input Zarr store path
        output_file: Output file path (.gpkg, .geojson, .shp, etc.)
        cell_ids: Optional list of cell IDs to export
        bbox: Optional bounding box (lon_min, lat_min, lon_max, lat_max)

    Raises:
        FileNotFoundError: If Zarr store doesn't exist
        ValueError: If driver cannot be found or file cannot be created
    """
    check_zarr_available()

    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    logger.info(f"Converting Zarr {zarr_path} to vector at {output_file}")

    reader = ZarrMeshReader(zarr_path)
    cells = reader.read_cells(cell_ids=cell_ids, bbox=bbox)

    if len(cells) == 0:
        logger.warning("No cells found to convert")
        return

    # Get driver for output format
    driver = get_vector_driver_from_filename(output_file)
    if driver is None:
        raise ValueError(f"Could not determine driver for {output_file}")

    # Remove existing file
    if os.path.exists(output_file):
        driver.DeleteDataSource(output_file)

    # Create data source
    data_source = driver.CreateDataSource(output_file)
    if data_source is None:
        raise ValueError(f"Could not create output file: {output_file}")

    # Create layer with WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    layer = data_source.CreateLayer("mesh", srs, ogr.wkbPolygon)

    # Add fields
    layer.CreateField(ogr.FieldDefn("cellid", ogr.OFTInteger64))
    layer.CreateField(ogr.FieldDefn("lon", ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn("lat", ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn("area", ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn("elevation", ogr.OFTReal))

    # Add features
    for cell in cells:
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("cellid", int(cell.lCellID))
        feature.SetField("lon", float(cell.dLongitude_center_degree))
        feature.SetField("lat", float(cell.dLatitude_center_degree))
        feature.SetField("area", float(cell.dArea))
        feature.SetField("elevation", float(cell.dElevation_mean))

        # Create polygon geometry
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for vertex in cell.aVertex:
            ring.AddPoint(vertex.dLongitude_degree, vertex.dLatitude_degree)
        # Close the ring
        if len(cell.aVertex) > 0:
            ring.AddPoint(
                cell.aVertex[0].dLongitude_degree, cell.aVertex[0].dLatitude_degree
            )

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        feature.SetGeometry(poly)

        layer.CreateFeature(feature)
        feature = None

    data_source = None
    logger.info(f"Successfully converted {len(cells)} cells to {output_file}")


def convert_to_zarr(
    input_file: str,
    output_path: str,
    mesh_type: str = "dggrid",
    **zarr_kwargs: Any,
) -> None:
    """
    Convert any GDAL-supported vector mesh file to Zarr format.

    Args:
        input_file: Input mesh file (GPKG, GeoJSON, Shapefile, etc.)
        output_path: Output Zarr store path
        mesh_type: Type of mesh (dggrid, mpas, tin, etc.)
        **zarr_kwargs: Additional arguments passed to convert_vector_to_zarr
    """
    convert_vector_to_zarr(input_file, output_path, mesh_type=mesh_type, **zarr_kwargs)


def convert_from_zarr(
    zarr_path: str,
    output_file: str,
    cell_ids: Optional[List[int]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """
    Convert Zarr to any GDAL-supported vector format.

    The output format is automatically determined from the file extension.

    Args:
        zarr_path: Input Zarr store path
        output_file: Output file path (.gpkg, .geojson, .shp, etc.)
        cell_ids: Optional list of cell IDs to export
        bbox: Optional bounding box (lon_min, lat_min, lon_max, lat_max)
    """
    convert_zarr_to_vector(zarr_path, output_file, cell_ids=cell_ids, bbox=bbox)
