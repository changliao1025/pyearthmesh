from pyearth.gis.geometry.extract_unique_vertices_and_connectivity import (
    extract_unique_vertices_and_connectivity,
)
from pyearth.gis.geometry.calculate_polygon_area import calculate_polygon_area
from pyearth.gis.geometry.international_date_line_utility import (
    split_international_date_line_polygon_coordinates,
    check_cross_international_date_line_polygon,
)
from pyearth.gis.location.get_geometry_coordinates import get_geometry_coordinates
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_driver_from_filename
import os
import logging
import traceback
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
from numpy.typing import NDArray
from osgeo import gdal, ogr

def setup_logger(module_name: str):
    # Use the module name to create a unique log file
    log_file = f"{module_name}.log"

    # Create a logger for this module
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Create a file handler for the log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create a console handler (optional)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define the log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger(__name__.split(".")[-1])

def check_geometry_validity(
    sFilename_source_mesh: str, iFlag_verbose_in: bool = False
) -> bool:
    """
    Comprehensive check of all geometries in a mesh vector file.

    Consolidates all polygon geometry validation including:
    - Coordinate range validation (-180 to 180 for lon, -90 to 90 for lat)
    - OGR geometry validity checks
    - International Date Line crossing detection
    - Multipolygon part validation
    - Minimum vertex count checks

    Args:
        sFilename_source_mesh (str): Path to the source mesh vector file
        iFlag_verbose_in (bool): If True, print detailed progress messages

    Returns:
        bool: True if all geometries are valid, False otherwise
    """
    try:
        pDataset = ogr.Open(sFilename_source_mesh, 0)  # Read-only
        if pDataset is None:
            logger.error(f"Failed to open file: {sFilename_source_mesh}")
            return False

        if iFlag_verbose_in:
            logger.info(f"Successfully opened mesh file: {sFilename_source_mesh}")

        # Get the first layer
        pLayer = pDataset.GetLayer(0)
        if pLayer is None:
            logger.error("Failed to get layer from the dataset.")
            pDataset = None
            return False

        # Get layer information
        pLayerDefn = pLayer.GetLayerDefn()
        if pLayerDefn is None:
            logger.error("Failed to get layer definition.")
            pDataset = None
            return False

        nFeatures = pLayer.GetFeatureCount()
        if nFeatures == 0:
            logger.warning("Layer contains no features.")
            pDataset = None
            return False

        if iFlag_verbose_in:
            logger.info(f"Validating geometries for {nFeatures} features...")

        # Process features with comprehensive validation
        pLayer.ResetReading()
        iFeature_index = 0
        invalid_geometry_count = 0
        valid_geometry_count = 0

        for pFeature in pLayer:
            if pFeature is None:
                invalid_geometry_count += 1
                iFeature_index += 1
                continue

            pGeometry = pFeature.GetGeometryRef()
            if pGeometry is None:
                logger.warning(f"Feature {iFeature_index} has no geometry")
                invalid_geometry_count += 1
                iFeature_index += 1
                continue

            # Skip GDAL geometry validation as it cannot handle IDL-crossing cells
            sGeometry_type = pGeometry.GetGeometryName()
            if sGeometry_type == "POLYGON":
                if not _validate_polygon_geometry(
                    pGeometry, iFeature_index, iFlag_verbose_in
                ):
                    invalid_geometry_count += 1
                else:
                    valid_geometry_count += 1

            elif sGeometry_type == "MULTIPOLYGON":
                if not _validate_multipolygon_geometry(
                    pGeometry, iFeature_index, iFlag_verbose_in
                ):
                    invalid_geometry_count += 1
                else:
                    valid_geometry_count += 1

            elif sGeometry_type in ["POINT", "LINESTRING"]:
                logger.warning(
                    f"Feature {iFeature_index}: Geometry type {sGeometry_type} not supported for mesh processing"
                )
                invalid_geometry_count += 1

            else:
                logger.warning(
                    f"Feature {iFeature_index}: Unknown geometry type {sGeometry_type}"
                )
                invalid_geometry_count += 1

            iFeature_index += 1

        # Cleanup
        pDataset = None

        # Report validation results
        total_features = valid_geometry_count + invalid_geometry_count
        success_rate = (
            (valid_geometry_count / total_features * 100) if total_features > 0 else 0
        )

        if iFlag_verbose_in or invalid_geometry_count > 0:
            logger.info(f"Geometry validation summary:")
            logger.info(f"  - Total features processed: {total_features}")
            logger.info(f"  - Valid geometries: {valid_geometry_count}")
            logger.info(f"  - Invalid geometries: {invalid_geometry_count}")
            logger.info(f"  - Success rate: {success_rate:.1f}%")

        if invalid_geometry_count > 0:
            logger.warning(
                "Found invalid geometries. The program will attempt to fix them."
            )
            return False

        if iFlag_verbose_in:
            logger.info("All geometries passed validation")

        return True

    except Exception as e:
        logger.error(f"Error in check_geometry_validity: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def _fix_geometry_coordinates_recursive(geometry: "ogr.Geometry") -> None:
    """
    Recursively fix coordinates in complex geometries.

    Args:
        geometry (ogr.Geometry): OGR Geometry object to fix in-place
    """
    geom_count = geometry.GetGeometryCount()

    if geom_count > 0:
        # Recurse through sub-geometries
        for i in range(geom_count):
            sub_geom = geometry.GetGeometryRef(i)
            _fix_geometry_coordinates_recursive(sub_geom)
    else:
        # Fix coordinates in this geometry
        point_count = geometry.GetPointCount()
        for i in range(point_count):
            x, y, z = geometry.GetPoint(i)
            if abs(x - 180.0) < 1e-10:
                x = 180.0 - 1e-8  # Nudge to just under 180°
            if abs(x + 180.0) < 1e-10:
                x = -180.0 + 1e-8  # Nudge to just above -180°
            # Normalize longitude using modular arithmetic
            normalized_x = ((x + 180) % 360) - 180
            # Use SetPoint_2D to ensure 2D geometry
            geometry.SetPoint_2D(i, normalized_x, y)


def fix_longitude_range_gdal(
    geometry: "ogr.Geometry", in_place: bool = False
) -> "ogr.Geometry":
    """
    Fix longitude values using GDAL geometry operations.
    Normalizes longitude coordinates to [-180, 180] range.

    Args:
        geometry (ogr.Geometry): OGR Geometry object to fix
        in_place (bool, optional): If True, modify geometry in place (faster).
            Default is False.

    Returns:
        ogr.Geometry: OGR Geometry object with normalized longitude coordinates
    """
    if geometry is None:
        return geometry

    # Optionally clone the geometry to avoid modifying the original
    fixed_geometry = geometry if in_place else geometry.Clone()

    # Get geometry type
    geom_type = fixed_geometry.GetGeometryName()

    if geom_type in [
        "POLYGON",
        "MULTIPOLYGON",
        "LINESTRING",
        "MULTILINESTRING",
        "POINT",
        "MULTIPOINT",
    ]:
        # For complex geometries, iterate through all geometry parts
        if geom_type.startswith("MULTI") or geom_type == "POLYGON":
            _fix_geometry_coordinates_recursive(fixed_geometry)
        else:
            # For simple geometries, fix coordinates directly using batch processing
            point_count = fixed_geometry.GetPointCount()
            if point_count > 0:
                # Process points in batches for better performance
                for i in range(point_count):
                    x, y, z = fixed_geometry.GetPoint(i)
                    # Normalize longitude using modular arithmetic
                    normalized_x = ((x + 180) % 360) - 180
                    # Avoid exact +180° by nudging to just under 180°
                    if abs(normalized_x - 180.0) < 1e-10:
                        normalized_x = 180.0 - 1e-8
                    # Use SetPoint_2D to ensure 2D geometry
                    fixed_geometry.SetPoint_2D(i, normalized_x, y)

    # Ensure the final geometry is 2D
    fixed_geometry.FlattenTo2D()
    return fixed_geometry


def create_geometry_from_coordinates(
    aCoord: NDArray[np.floating], geometry_type: str
) -> Optional["ogr.Geometry"]:
    """
    Create an OGR Geometry object from coordinate array based on specified geometry type.

    Args:
        aCoord (NDArray[np.floating]): Array of coordinates with shape (n, 2) or (n, 3)
        geometry_type (str): Type of geometry ('POLYGON', 'LINESTRING', 'POINT')

    Returns:
        Optional[ogr.Geometry]: OGR Geometry object, or None if geometry type is unsupported
    """
    if geometry_type == "POLYGON":
        pPolygon = ogr.Geometry(ogr.wkbPolygon)
        pLinearRing = ogr.Geometry(ogr.wkbLinearRing)
        for coord in aCoord:
            # Force 2D by only using x,y coordinates
            pLinearRing.AddPoint_2D(coord[0], coord[1])
        pLinearRing.CloseRings()
        pPolygon.AddGeometry(pLinearRing)
        # Ensure the polygon is 2D
        pPolygon.FlattenTo2D()
        return pPolygon
    elif geometry_type == "LINESTRING":
        pLineString = ogr.Geometry(ogr.wkbLineString)
        for coord in aCoord:
            pLineString.AddPoint(coord[0], coord[1])
        return pLineString
    elif geometry_type == "POINT":
        pPoint = ogr.Geometry(ogr.wkbPoint)
        pPoint.AddPoint(aCoord[0][0], aCoord[0][1])
        return pPoint
    else:
        logger.error(f"Unsupported geometry type for creation: {geometry_type}")
        return None

def fix_mesh_longitude_range_and_idl_crossing(
    sFilename_in: str, sFilename_out: str, handle_idl_crossing: bool = True
) -> bool:
    """
    Comprehensive GDAL-based function to fix longitude range issues and optionally handle
    International Date Line (IDL) crossing in vector files.

    This function combines the functionality of both fix_mesh_longitude_range_gdal and fix_idl_crossing
    into a single optimized function that can handle multiple layers and IDL crossing.

    Args:
        sFilename_in (str): Path to input vector file (any GDAL-supported format)
        sFilename_out (str): Path to output vector file
        handle_idl_crossing (bool, optional): Whether to check and split polygons crossing the IDL.
            Default is True.

    Returns:
        bool: True if successful, False otherwise
    """
    pDriver = None
    pDataset = None
    pDataset_out = None

    try:
        # Open source dataset
        pDriver = get_vector_driver_from_filename(sFilename_in)
        pDataset = pDriver.Open(sFilename_in, 0)
        if pDataset is None:
            logger.error(f"Could not open input file: {sFilename_in}")
            return False

        # Create output dataset
        pDriver_out = get_vector_driver_from_filename(sFilename_out)
        # delete output file if it already exists
        if os.path.exists(sFilename_out):
            pDriver_out.DeleteDataSource(sFilename_out)
        pDataset_out = pDriver_out.CreateDataSource(sFilename_out)
        if pDataset_out is None:
            logger.error(f"Could not create output file: {sFilename_out}")
            return False

        # Process each layer
        layer_count = pDataset.GetLayerCount()
        logger.info(f"Processing {layer_count} layer(s) from {sFilename_in}")

        total_processed = 0

        for layer_idx in range(layer_count):
            pLayer = pDataset.GetLayerByIndex(layer_idx)
            if pLayer is None:
                logger.warning(f"Could not get layer {layer_idx}")
                continue

            pLayerDefn = pLayer.GetLayerDefn()
            sSpatial_ref = pLayer.GetSpatialRef()
            layer_name = pLayer.GetName()
            nFeatures = pLayer.GetFeatureCount()

            logger.info(f"Processing layer '{layer_name}' with {nFeatures} features")

            # Create output layer with same schema
            pLayer_out = pDataset_out.CreateLayer(
                layer_name, sSpatial_ref, ogr.wkbUnknown
            )
            if pLayer_out is None:
                logger.error(f"Could not create output layer: {layer_name}")
                continue

            # Copy field definitions
            for iField in range(pLayerDefn.GetFieldCount()):
                field_defn = pLayerDefn.GetFieldDefn(iField)
                pLayer_out.CreateField(field_defn)

            # Process features with progress tracking
            processed_count = 0
            idl_crossing_count = 0

            for pFeature in pLayer:
                try:
                    geometry = pFeature.GetGeometryRef()
                    if geometry is None:
                        logger.warning(
                            f"Feature ID {pFeature.GetFID()} has no geometry, skipping..."
                        )
                        continue

                    geometry_type = geometry.GetGeometryName()

                    # check whether geometry contains poles
                    aCoord_origin = get_geometry_coordinates(geometry)
                    if np.min(np.abs(aCoord_origin[:, 1])) > 88.0:
                        continue
                    # Fix longitude coordinates using GDAL
                    fixed_geometry = fix_longitude_range_gdal(geometry)
                    # Ensure the geometry is 2D
                    if fixed_geometry is not None:
                        fixed_geometry.FlattenTo2D()
                    # Handle IDL crossing for polygon geometries if requested
                    if handle_idl_crossing and geometry_type in [
                        "POLYGON",
                        "MULTIPOLYGON",
                    ]:
                        # Check for IDL crossing after longitude normalization
                        aCoord = get_geometry_coordinates(fixed_geometry)
                        bCross_idl, aCoord_updated = (
                            check_cross_international_date_line_polygon(aCoord)
                        )

                        if bCross_idl:
                            idl_crossing_count += 1
                            logger.info(
                                f"Feature ID {pFeature.GetFID()} crosses the International Date Line. Splitting..."
                            )

                            [eastern_polygon, western_polygon] = (
                                split_international_date_line_polygon_coordinates(
                                    aCoord
                                )
                            )

                            # Create a multipolygon geometry (force 2D)
                            pGeometry_multi = ogr.Geometry(ogr.wkbMultiPolygon)

                            # Create eastern polygon
                            pPolygon_eastern = ogr.Geometry(ogr.wkbPolygon)
                            pLinearRing_eastern = ogr.Geometry(ogr.wkbLinearRing)
                            for coord in eastern_polygon:
                                # Force 2D by only using x,y coordinates
                                pLinearRing_eastern.AddPoint_2D(coord[0], coord[1])
                            pLinearRing_eastern.CloseRings()
                            pPolygon_eastern.AddGeometry(pLinearRing_eastern)
                            # Ensure the polygon is 2D
                            pPolygon_eastern.FlattenTo2D()
                            pGeometry_multi.AddGeometry(pPolygon_eastern)

                            # Create western polygon
                            pPolygon_western = ogr.Geometry(ogr.wkbPolygon)
                            pLinearRing_western = ogr.Geometry(ogr.wkbLinearRing)
                            for coord in western_polygon:
                                # Force 2D by only using x,y coordinates
                                pLinearRing_western.AddPoint_2D(coord[0], coord[1])
                            pLinearRing_western.CloseRings()
                            pPolygon_western.AddGeometry(pLinearRing_western)
                            # Ensure the polygon is 2D
                            pPolygon_western.FlattenTo2D()
                            pGeometry_multi.AddGeometry(pPolygon_western)

                            # Ensure the entire multipolygon is 2D
                            pGeometry_multi.FlattenTo2D()
                            final_geometry = pGeometry_multi
                        else:
                            if aCoord_updated is not None:
                                # Update fixed_geometry with adjusted coordinates to create a polygon, usually because of IDL
                                fixed_geometry = create_geometry_from_coordinates(
                                    aCoord_updated, geometry_type
                                )
                                final_geometry = fixed_geometry
                            else:
                                final_geometry = fixed_geometry
                    else:
                        final_geometry = fixed_geometry

                    # Ensure final geometry is always 2D
                    if final_geometry is not None:
                        final_geometry.FlattenTo2D()

                    # Create output feature
                    pFeature_out = ogr.Feature(pLayer_out.GetLayerDefn())
                    pFeature_out.SetGeometry(final_geometry)

                    # Copy all field values
                    for iField in range(pLayerDefn.GetFieldCount()):
                        sField_name = pLayerDefn.GetFieldDefn(iField).GetName()
                        pFeature_out.SetField(
                            sField_name, pFeature.GetField(sField_name)
                        )

                    pLayer_out.CreateFeature(pFeature_out)
                    pFeature_out = None

                    processed_count += 1
                    if processed_count % 1000 == 0:
                        logger.info(
                            f"Processed {processed_count}/{nFeatures} features in layer '{layer_name}'..."
                        )

                except Exception as e:
                    logger.error(
                        f"Error processing feature ID {pFeature.GetFID()} in layer '{layer_name}': {str(e)}"
                    )
                    continue

            if handle_idl_crossing and idl_crossing_count > 0:
                logger.info(
                    f"Layer '{layer_name}': {processed_count} features processed, {idl_crossing_count} IDL crossings handled"
                )
            else:
                logger.info(
                    f"Layer '{layer_name}': {processed_count} features processed"
                )

            total_processed += processed_count

        # Cleanup
        pDataset_out.FlushCache()
        pDataset_out = None
        pDataset = None

        logger.info(
            f"Successfully processed {total_processed} total features and created fixed file: {sFilename_out}"
        )
        return True

    except Exception as e:
        logger.error(f"Error in fix_mesh_longitude_range_and_idl: {str(e)}")
        return False

    finally:
        # Ensure proper cleanup
        if pDataset_out is not None:
            pDataset_out.FlushCache()
            pDataset_out = None
        if pDataset is not None:
            pDataset = None


def check_mesh_quality(sFilename_mesh_in: str, iFlag_verbose_in: bool = False) -> str:
    """
    Check mesh quality and fix if necessary.

    Args:
        sFilename_mesh_in (str): Path to the input mesh file
        iFlag_verbose_in (bool, optional): If True, print detailed progress messages.
            Default is False.

    Returns:
        str: Path to the validated/fixed mesh file
    """
    if not check_geometry_validity(
        sFilename_mesh_in, iFlag_verbose_in=iFlag_verbose_in
    ):
        # we need to fix the mesh using the IDL splitting utility
        # Make the filename adjustment more flexible to handle any format
        # Get the file extension and base name
        file_base, file_ext = os.path.splitext(sFilename_mesh_in)
        file_ext = file_ext.lstrip(".")
        sFilename_source_mesh_fixed = f"{file_base}_fixed.{file_ext}"
        fix_mesh_longitude_range_and_idl_crossing(
            sFilename_mesh_in, sFilename_source_mesh_fixed
        )
        return sFilename_source_mesh_fixed
    return sFilename_mesh_in


def _validate_polygon_geometry(
    pGeometry: "ogr.Geometry",
    feature_id: Union[int, str],
    iFlag_verbose_in: bool = False,
) -> bool:
    """
    Validate a single polygon geometry including coordinate range and IDL checks.

    Args:
        pGeometry: OGR Polygon geometry
        feature_id: Feature identifier for logging
        iFlag_verbose_in: Verbose logging flag

    Returns:
        bool: True if geometry is valid, False otherwise
    """
    try:
        # Get coordinates
        aCoord = get_geometry_coordinates(pGeometry)
        if aCoord is None or len(aCoord) < 3:
            logger.warning(
                f"Feature {feature_id}: Invalid or insufficient coordinates for polygon"
            )
            return False

        # Validate coordinate bounds
        lons = aCoord[:, 0]
        lats = aCoord[:, 1]

        # Check coordinate ranges
        if (
            np.any(lons < -180)
            or np.any(lons > 180)
            or np.any(lats < -90)
            or np.any(lats > 90)
        ):
            logger.warning(f"Feature {feature_id}: Coordinates outside valid range")
            logger.warning(f"  Longitude range: {lons.min():.3f} to {lons.max():.3f}")
            logger.warning(f"  Latitude range: {lats.min():.3f} to {lats.max():.3f}")
            return False

        # Check for International Date Line crossing (this is allowed but logged)
        iCross_idl, dummy = check_cross_international_date_line_polygon(aCoord)
        if iCross_idl:
            if iFlag_verbose_in:
                logger.info(
                    f"Feature {feature_id}: Polygon crosses International Date Line (valid)"
                )
            return False
        else:
            # use gdal geometry validity check only when it does not cross IDL
            if not pGeometry.IsValid():
                logger.warning(
                    f"Feature {feature_id}: Polygon geometry is invalid according to OGR"
                )
                return False
        return True

    except Exception as e:
        logger.warning(
            f"Feature {feature_id}: Error validating polygon geometry: {str(e)}"
        )
        return False



def _validate_multipolygon_geometry(
    pGeometry: "ogr.Geometry",
    feature_id: Union[int, str],
    iFlag_verbose_in: bool = False,
) -> bool:
    """
    Validate a multipolygon geometry by checking all constituent polygons.

    Args:
        pGeometry: OGR MultiPolygon geometry
        feature_id: Feature identifier for logging
        iFlag_verbose_in: Verbose logging flag

    Returns:
        bool: True if all parts are valid, False otherwise
    """
    try:
        valid_parts = 0
        total_parts = pGeometry.GetGeometryCount()

        if total_parts == 0:
            logger.warning(f"Feature {feature_id}: Multipolygon has no parts")
            return False

        for iPart in range(total_parts):
            pPolygon_part = pGeometry.GetGeometryRef(iPart)
            if pPolygon_part is None:
                logger.warning(
                    f"Feature {feature_id}: Multipolygon part {iPart} is None"
                )
                continue

            # Skip GDAL geometry validation as it cannot handle IDL-crossing cells

            # Validate coordinates of this part
            aCoord_part = get_geometry_coordinates(pPolygon_part)
            if aCoord_part is None or len(aCoord_part) < 3:
                logger.warning(
                    f"Feature {feature_id}: Multipolygon part {iPart} has insufficient coordinates"
                )
                continue

            # Check coordinate bounds for this part
            lons_part = aCoord_part[:, 0]
            lats_part = aCoord_part[:, 1]

            if (
                np.any(lons_part < -180)
                or np.any(lons_part > 180)
                or np.any(lats_part < -90)
                or np.any(lats_part > 90)
            ):
                logger.warning(
                    f"Feature {feature_id}: Multipolygon part {iPart} has coordinates outside valid range"
                )
                continue

            valid_parts += 1

        if valid_parts == 0:
            logger.warning(
                f"Feature {feature_id}: No valid parts found in multipolygon"
            )
            return False

        if iFlag_verbose_in and valid_parts < total_parts:
            logger.info(
                f"Feature {feature_id}: Multipolygon has {valid_parts}/{total_parts} valid parts"
            )

        return True

    except Exception as e:
        logger.warning(
            f"Feature {feature_id}: Error validating multipolygon geometry: {str(e)}"
        )
        return False
