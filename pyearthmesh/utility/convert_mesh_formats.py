"""
Utilities for converting between mesh formats.

This module provides functions to convert mesh data between different
formats including any GDAL-supported vector format and Zarr.
"""
import os
from typing import Optional, List, Dict, Any
import logging
from osgeo import ogr, osr
import numpy as np

from pyearthmesh.utility.zarr_io import (
    ZarrMeshWriter,
    ZarrMeshReader,
    check_zarr_available
)
from pyearthmesh.classes.meshcell import pymeshcell
from pyearthmesh.classes.vertex import pyvertex
from pyearthmesh.classes.edge import pyedge
from pyearth.gis.location.get_geometry_coordinates import get_geometry_coordinates
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_driver_from_filename

logger = logging.getLogger(__name__)


def convert_vector_to_zarr(
    input_file: str,
    zarr_path: str,
    mesh_type: str = "dggrid",
    chunk_size: Optional[int] = None,
    compression_profile: str = "balanced",
    batch_size: int = 100000
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
        overwrite=True
    )

    # Extract metadata
    spatial_ref = layer.GetSpatialRef()
    attributes = {
        "source_file": os.path.basename(input_file),
        "source_format": os.path.splitext(input_file)[1].lstrip('.'),
        "crs": spatial_ref.ExportToWkt() if spatial_ref else "EPSG:4326"
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
      

  
        vertices = []
        for i in range(l

                "dLongitude_degree": float(coords[i, 0]),
           
            })
            vertices.append(vertex)

        
   
        f
            edge = pyedge(vertices[i], verti
            ed

        # Calculate cente
        center_lon


        # Create meshcell
        cell =
            center_
            center_
            edges,
     
 

        # Try to extract cell ID from feature
    
        if fid is not None:
            cell.lC

        # Try to extract other attributes

            field_defn = feature.GetFieldDefnRef(i)
            field_name = field
            field_value = feature.GetField(i)

            if field_value is None:
    

            # Map common field names


            elif field_name
                cell
            elif fi
                cell.dElevatio

        return cell

    except Exce
        logger.debug(f"Failed to convert f
        return None


def co
    zarr_p
    out
    cell_ids: Optional[List[int]] = None,
    bbox: Optional[tu

    """
 

    The output format is automatically determined from the file extension.

    Args:
        zarr_path: Input Zarr 
        output_file: Output fil
        cell_ids: Optional list of 
        bbox: Optional bounding box (lon_min, la

    Raises:
        FileNotFoundError: If Zarr store

    check_z

    if not os.path.exists(zarr_path):
        raise F

    logger.info(f"Converting Zarr {zarr_pat

    # 
    reader = ZarrMeshReade


    if len(cells) == 0:
        logge
        return

    # Get driver for output format
    driver = get_ve

        raise ValueError(f"Could not determine driver for {output_file}")

    # Remove existing file
    if
        driver.DeleteDataSource(outpu

    # Create data sour
    data_source = driver.CreateDataSource(output_file)
 


    # Create layer with WGS84 
    srs = osr.SpatialReference()
    srs

    layer = data_source.CreateLayer("mesh", srs, ogr.wkbPolygon

    # Add fields
    layer.C
    layer.CreateField(ogr.Fi
    layer.CreateField(ogr.Fiel
    layer.CreateField(ogr.Fi

    # Add features
    for ce
        feature = ogr.Feature(layer.GetLayerDefn
        feature.SetFie
     

        feature.SetFie

        # Create polygon geometry
    
        for vertex
            ring.AddPoint(vertex.dLongitude_degree, 
        # Close the ring
        if len(cell.aVertex) > 0:
          

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ri
     

        layer.CreateFeature(feature)
        feature

    data_source = None

    log


def convert_to_zarr(
 
    output_path: str
    mesh_type: str = 
    **zarr_kwargs
) 

    Convert any GDAL-supp

    Args:
        input_file: Input mesh f
        output_path: Output Zarr store path
    
        **zarr_kwargs: Addit

    Raises:
        FileNotFoundErr
    """
    c


def convert_from_zarr(
    

    cell_ids: Optional[List[int]] = None,

) -> None:
    """
    Convert Zarr to any GDAL-supported vector forma

    The output format is automatically deter

    Args:
        zarr_path

        cell_ids: Optional list of cell IDs 
        bbox: Optional bounding box (lon_min, lat_min, lon_max, lat_max)

    Rais

    """
    convert_zarr_to
