"""Centralized mesh boundary determination and mesh generation module.

Unifies boundary resolution and mesh instantiation across all structured and
unstructured mesh types supported by PyEarthMesh (Hexagon, Square, LatLon, MPAS,
DGGRID, Triangular, TIN).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from osgeo import ogr
from pyearth.gis.spatialref.convert_between_degree_and_meter import (
    degree_to_meter,
    meter_to_degree,
)

from pyearthmesh.meshes.structured.cubicsphere.create_cubicsphere_mesh import (
    create_cubicsphere_mesh,
)
from pyearthmesh.meshes.structured.dggs.dggrid.create_dggrid_mesh import (
    create_dggrid_mesh,
)
from pyearthmesh.meshes.structured.hexagon.create_hexagon_mesh import (
    create_hexagon_mesh,
)
from pyearthmesh.meshes.structured.latlon.create_latlon_mesh import (
    create_latlon_mesh,
)
from pyearthmesh.meshes.structured.square.create_square_mesh import (
    create_square_mesh,
)
from pyearthmesh.meshes.structured.triangular.create_triangular_mesh import (
    create_triangular_mesh,
)
from pyearthmesh.meshes.unstructured.mpas.create_mpas_mesh import (
    create_mpas_mesh,
)
from pyearthmesh.meshes.unstructured.tin.create_tin_mesh import (
    create_tin_mesh,
)

logger = logging.getLogger(__name__)

MESH_TYPE_MAP: dict[str | int, str] = {
    1: "hexagon",
    2: "square",
    3: "latlon",
    4: "mpas",
    5: "dggrid",
    6: "triangular",
    7: "tin",
    8: "cubicsphere",
    9: "healpix",
    10: "isea",
    11: "rhealpix",
    "hexagon": "hexagon",
    "square": "square",
    "latlon": "latlon",
    "mpas": "mpas",
    "dggrid": "dggrid",
    "triangular": "triangular",
    "tin": "tin",
    "cubicsphere": "cubicsphere",
    "healpix": "healpix",
    "isea": "isea",
    "rhealpix": "rhealpix",
}


def determine_mesh_boundary(
    pBoundary_wkt: str | None = None,
    sFilename_mesh_boundary: str | None = None,
    iFlag_mesh_boundary: int = 0,
    dLongitude_left: float = -180.0,
    dLongitude_right: float = 180.0,
    dLatitude_bot: float = -90.0,
    dLatitude_top: float = 90.0,
    iFlag_global: int = 0,
) -> tuple[str | None, tuple[float, float, float, float] | None]:
    """Determine the appropriate boundary geometry (WKT) for mesh generation.

    Resolution order:
    1. Explicit ``pBoundary_wkt`` if provided.
    2. Vector boundary file (``sFilename_mesh_boundary``) if ``iFlag_mesh_boundary == 1``
       or if a valid vector file is provided.
    3. Rectangular bounding box constructed from coordinates
       (``dLongitude_left``, ``dLatitude_top``, ``dLongitude_right``, ``dLatitude_bot``)
       when not a global domain.
    4. None (representing a global or unrestricted mesh).

    Parameters
    ----------
    pBoundary_wkt : str, optional
        Pre-existing WKT boundary string.
    sFilename_mesh_boundary : str, optional
        Path to a boundary vector file (GeoJSON, Shapefile, etc.).
    iFlag_mesh_boundary : int, optional
        Flag indicating vector boundary usage (1 = enabled, 0 = disabled).
    dLongitude_left : float
        Western bounding longitude.
    dLongitude_right : float
        Eastern bounding longitude.
    dLatitude_bot : float
        Southern bounding latitude.
    dLatitude_top : float
        Northern bounding latitude.
    iFlag_global : int
        1 if mesh is global, 0 if regional.

    Returns
    -------
    tuple
        (boundary_wkt, (min_x, max_x, min_y, max_y) or None)
    """
    if pBoundary_wkt:
        return pBoundary_wkt, None

    # Option 1: Vector file boundary
    if iFlag_mesh_boundary == 1 or (
        sFilename_mesh_boundary and os.path.isfile(sFilename_mesh_boundary)
    ):
        if sFilename_mesh_boundary and os.path.isfile(sFilename_mesh_boundary):
            try:
                from pyearth.gis.gdal.read.vector.gdal_get_vector_boundary import (
                    gdal_get_vector_boundary,
                )

                wkt, extent = gdal_get_vector_boundary(sFilename_mesh_boundary)
                if wkt:
                    return wkt, extent
            except Exception as e:
                logger.warning(
                    "gdal_get_vector_boundary failed on %s: %s; trying MPASWorkflow fallback",
                    sFilename_mesh_boundary,
                    e,
                )
                try:
                    from pyearthmesh.config.workflows.mpas_workflow import (
                        MPASWorkflow,
                    )

                    wkt = MPASWorkflow._boundary_wkt_from_geojson(
                        sFilename_mesh_boundary
                    )
                    if wkt:
                        return wkt, None
                except Exception:
                    pass

    # Option 2: Coordinate bounding box for regional domain
    if not bool(iFlag_global):
        # Verify coordinates define a regional sub-box
        if not (
            float(dLongitude_left) == -180.0
            and float(dLongitude_right) == 180.0
            and float(dLatitude_bot) == -90.0
            and float(dLatitude_top) == 90.0
        ):
            pRing = ogr.Geometry(ogr.wkbLinearRing)
            pRing.AddPoint(float(dLongitude_left), float(dLatitude_top))
            pRing.AddPoint(float(dLongitude_right), float(dLatitude_top))
            pRing.AddPoint(float(dLongitude_right), float(dLatitude_bot))
            pRing.AddPoint(float(dLongitude_left), float(dLatitude_bot))
            pRing.AddPoint(float(dLongitude_left), float(dLatitude_top))
            pBoundary = ogr.Geometry(ogr.wkbPolygon)
            pBoundary.AddGeometry(pRing)
            wkt = pBoundary.ExportToWkt()
            extent = (
                float(dLongitude_left),
                float(dLongitude_right),
                float(dLatitude_bot),
                float(dLatitude_top),
            )
            return wkt, extent

    return None, None


def generate_mesh(
    case_or_type: Any = "mpas",
    *,
    sMesh_type: str | int | None = None,
    sFilename_mesh: str | None = None,
    # Boundary arguments
    iFlag_mesh_boundary: int | None = None,
    sFilename_mesh_boundary: str | None = None,
    pBoundary_wkt: str | None = None,
    # Extents
    dLongitude_left: float | None = None,
    dLongitude_right: float | None = None,
    dLatitude_bot: float | None = None,
    dLatitude_top: float | None = None,
    dLatitude_mean: float | None = None,
    dLongitude_mean: float | None = None,
    # Resolutions
    dResolution_degree: float | None = None,
    dResolution_meter: float | None = None,
    # Projected bounds
    dX_lowerleft: float | None = None,
    dY_lowerleft: float | None = None,
    dX_lowerright: float | None = None,
    dY_upperleft: float | None = None,
    pProjection_reference: Any = None,
    # Flags & options
    iFlag_global: int | None = None,
    iFlag_save_mesh: int | None = None,
    iFlag_rotation: int | None = None,
    iFlag_use_mesh_dem: int | None = None,
    iFlag_land_ocean_mask: int | None = None,
    sFilename_land_ocean_mask: str | None = None,
    iFlag_run_jigsaw: int | None = None,
    sWorkspace_jigsaw: str | None = None,
    sFilename_mpas_mesh_netcdf: str | None = None,
    sFilename_jigsaw_mesh_netcdf: str | None = None,
    aConfig_jigsaw: dict | None = None,
    iFlag_antarctic: int | None = None,
    iFlag_arctic: int | None = None,
    iFlag_antarctic_in: int | None = None,
    iFlag_arctic_in: int | None = None,
    # DGGRID parameters
    iResolution_index: int | None = None,
    sDggrid_type: str | None = None,
    sWorkspace_output: str | None = None,
    **kwargs: Any,
) -> list:
    """Central entry point to determine boundary and generate mesh cells.

    Can be invoked with a case instance (such as PyFlowline's ``flowlinecase`` or
    PyEarthMesh's ``BaseCase``), a configuration dictionary, or explicit keyword
    arguments.

    Parameters
    ----------
    case_or_type : Any
        Case object, config dictionary, or mesh type identifier.

    Returns
    -------
    list
        List of generated mesh cell objects (e.g. pympas, pyhexagon, pysquare,
        pylatlon, etc.).
    """
    # Extract properties from case object or dictionary if provided
    is_obj = hasattr(case_or_type, "__dict__") and not isinstance(
        case_or_type, (str, int, dict)
    )
    is_dict = isinstance(case_or_type, dict)

    def _get(key: str, default: Any = None) -> Any:
        if key in kwargs and kwargs[key] is not None:
            return kwargs[key]
        if is_dict:
            return case_or_type.get(key, default)
        if is_obj:
            return getattr(case_or_type, key, default)
        return default

    # Mesh type
    raw_mesh_type = (
        sMesh_type
        or _get("iMesh_type")
        or _get("sMesh_type")
        or (case_or_type if isinstance(case_or_type, (str, int)) else "mpas")
    )
    mesh_type_name = MESH_TYPE_MAP.get(raw_mesh_type, str(raw_mesh_type).lower())

    # Core parameters
    _sFilename_mesh = sFilename_mesh or _get("sFilename_mesh", "mesh.geojson")
    _iFlag_global = (
        iFlag_global if iFlag_global is not None else _get("iFlag_global", 0)
    )
    _iFlag_save_mesh = (
        iFlag_save_mesh if iFlag_save_mesh is not None else _get("iFlag_save_mesh", 1)
    )
    _iFlag_rotation = (
        iFlag_rotation if iFlag_rotation is not None else _get("iFlag_rotation", 0)
    )
    _iFlag_mesh_boundary = (
        iFlag_mesh_boundary
        if iFlag_mesh_boundary is not None
        else _get("iFlag_mesh_boundary", 0)
    )
    _sFilename_mesh_boundary = (
        sFilename_mesh_boundary
        or _get("sFilename_mesh_boundary_geojson")
        or _get("sFilename_mesh_boundary")
        or _get("sFilename_boundary")
    )
    _pBoundary_wkt = pBoundary_wkt or _get("pBoundary_wkt")

    # Extents
    _dLongitude_left = (
        dLongitude_left
        if dLongitude_left is not None
        else _get("dLongitude_left", -180.0)
    )
    _dLongitude_right = (
        dLongitude_right
        if dLongitude_right is not None
        else _get("dLongitude_right", 180.0)
    )
    _dLatitude_bot = (
        dLatitude_bot if dLatitude_bot is not None else _get("dLatitude_bot", -90.0)
    )
    _dLatitude_top = (
        dLatitude_top if dLatitude_top is not None else _get("dLatitude_top", 90.0)
    )

    # Determine boundary WKT
    resolved_wkt, extent = determine_mesh_boundary(
        pBoundary_wkt=_pBoundary_wkt,
        sFilename_mesh_boundary=_sFilename_mesh_boundary,
        iFlag_mesh_boundary=_iFlag_mesh_boundary,
        dLongitude_left=_dLongitude_left,
        dLongitude_right=_dLongitude_right,
        dLatitude_bot=_dLatitude_bot,
        dLatitude_top=_dLatitude_top,
        iFlag_global=_iFlag_global,
    )

    # Update extents if resolved from vector file
    if extent:
        _dLongitude_left, _dLongitude_right, _dLatitude_bot, _dLatitude_top = (
            extent[0],
            extent[1],
            extent[2],
            extent[3],
        )

    # Latitude / Longitude mean
    _dLatitude_mean = (
        dLatitude_mean
        if dLatitude_mean is not None
        else _get("dLatitude_mean", 0.5 * (_dLatitude_bot + _dLatitude_top))
    )
    _dLongitude_mean = (
        dLongitude_mean
        if dLongitude_mean is not None
        else _get("dLongitude_mean", 0.5 * (_dLongitude_left + _dLongitude_right))
    )

    # Resolution handling
    _dResolution_degree = (
        dResolution_degree
        if dResolution_degree is not None
        else _get("dResolution_degree")
    )
    _dResolution_meter = (
        dResolution_meter
        if dResolution_meter is not None
        else _get("dResolution_meter")
    )

    if _dResolution_degree is not None and _dResolution_meter is None:
        try:
            _dResolution_meter = degree_to_meter(_dResolution_degree, _dLatitude_mean)
        except Exception:
            pass
    elif _dResolution_meter is not None and _dResolution_degree is None:
        try:
            _dResolution_degree = meter_to_degree(_dResolution_meter, _dLatitude_mean)
        except Exception:
            pass

    # Projected bounds
    _dX_lowerleft = dX_lowerleft if dX_lowerleft is not None else _get("dX_lowerleft")
    _dY_lowerleft = dY_lowerleft if dY_lowerleft is not None else _get("dY_lowerleft")
    _dX_lowerright = (
        dX_lowerright if dX_lowerright is not None else _get("dX_lowerright")
    )
    _dY_upperleft = dY_upperleft if dY_upperleft is not None else _get("dY_upperleft")
    _pProjection_reference = (
        pProjection_reference
        if pProjection_reference is not None
        else _get("pProjection_reference")
    )

    # Polar flags
    _iFlag_antarctic = (
        iFlag_antarctic_in
        if iFlag_antarctic_in is not None
        else (
            iFlag_antarctic
            if iFlag_antarctic is not None
            else _get("iFlag_antarctic", 0)
        )
    )
    _iFlag_arctic = (
        iFlag_arctic_in
        if iFlag_arctic_in is not None
        else (iFlag_arctic if iFlag_arctic is not None else _get("iFlag_arctic", 0))
    )

    # Other options
    _iFlag_use_mesh_dem = (
        iFlag_use_mesh_dem
        if iFlag_use_mesh_dem is not None
        else _get("iFlag_use_mesh_dem", 0)
    )
    _iFlag_land_ocean_mask = (
        iFlag_land_ocean_mask
        if iFlag_land_ocean_mask is not None
        else _get("iFlag_land_ocean_mask", 0)
    )
    _sFilename_land_ocean_mask = (
        sFilename_land_ocean_mask
        or _get("sFilename_coastline_boundary_geojson")
        or _get("sFilename_land_ocean_mask")
    )
    _iFlag_run_jigsaw = (
        iFlag_run_jigsaw
        if iFlag_run_jigsaw is not None
        else _get("iFlag_run_jigsaw", 0)
    )
    _sWorkspace_jigsaw = sWorkspace_jigsaw or _get("sWorkspace_jigsaw")
    _sFilename_mpas_mesh_netcdf = sFilename_mpas_mesh_netcdf or _get(
        "sFilename_mpas_mesh_netcdf"
    )
    _sFilename_jigsaw_mesh_netcdf = sFilename_jigsaw_mesh_netcdf or _get(
        "sFilename_jigsaw_mesh_netcdf"
    )
    _aConfig_jigsaw = aConfig_jigsaw or _get("aConfig_jigsaw") or _get("jigsaw")
    _iResolution_index = (
        iResolution_index
        if iResolution_index is not None
        else _get("iResolution_index", 10)
    )
    _sDggrid_type = sDggrid_type or _get("sDggrid_type", "ISEA3H")
    _sWorkspace_output = sWorkspace_output or _get("sWorkspace_output", "./output")

    # ──────────────────────────────────────────────────────────────────────────
    # Generate requested mesh type
    # ──────────────────────────────────────────────────────────────────────────
    if mesh_type_name == "hexagon":
        if _dResolution_meter is None:
            _dResolution_meter = degree_to_meter(_dResolution_degree, _dLatitude_mean)
        dArea = np.power(_dResolution_meter, 2.0)
        dLength_edge = np.sqrt(2.0 * dArea / (3.0 * np.sqrt(3.0)))
        if _iFlag_rotation == 0:
            dX_spacing = dLength_edge * np.sqrt(3.0)
            dY_spacing = dLength_edge * 1.5
            ncolumn = int((_dX_lowerright - _dX_lowerleft) / dX_spacing)
            nrow = int((_dY_upperleft - _dY_lowerleft) / dY_spacing)
        else:
            dX_spacing = dLength_edge * 1.5
            dY_spacing = dLength_edge * np.sqrt(3.0)
            ncolumn = int((_dX_lowerright - _dX_lowerleft) / dX_spacing) + 1
            nrow = int((_dY_upperleft - _dY_lowerleft) / dY_spacing)

        return create_hexagon_mesh(
            _iFlag_rotation,
            _dX_lowerleft,
            _dY_lowerleft,
            _dResolution_meter,
            ncolumn,
            nrow,
            _sFilename_mesh,
            _pProjection_reference,
            resolved_wkt,
        )

    elif mesh_type_name == "square":
        ncolumn = int((_dX_lowerright - _dX_lowerleft) / _dResolution_meter)
        nrow = int((_dY_upperleft - _dY_lowerleft) / _dResolution_meter)
        return create_square_mesh(
            _dX_lowerleft,
            _dY_lowerleft,
            _dResolution_meter,
            ncolumn,
            nrow,
            _sFilename_mesh,
            _pProjection_reference,
            resolved_wkt,
        )

    elif mesh_type_name == "latlon":
        ncolumn = int((_dLongitude_right - _dLongitude_left) / _dResolution_degree)
        nrow = int((_dLatitude_top - _dLatitude_bot) / _dResolution_degree)
        return create_latlon_mesh(
            _dLongitude_left,
            _dLatitude_bot,
            _dResolution_degree,
            ncolumn,
            nrow,
            _sFilename_mesh,
            resolved_wkt,
        )

    elif mesh_type_name == "mpas":
        boundary_for_mpas = (
            None if (_iFlag_antarctic == 1 or _iFlag_arctic == 1) else resolved_wkt
        )
        mask_for_mpas = (
            _sFilename_land_ocean_mask if _iFlag_land_ocean_mask == 1 else None
        )
        return create_mpas_mesh(
            _sFilename_mesh,
            iFlag_global_in=_iFlag_global,
            iFlag_use_mesh_dem_in=_iFlag_use_mesh_dem,
            iFlag_save_mesh_in=_iFlag_save_mesh,
            iFlag_run_jigsaw_in=_iFlag_run_jigsaw,
            iFlag_antarctic_in=_iFlag_antarctic,
            iFlag_arctic_in=_iFlag_arctic,
            sWorkspace_jigsaw_in=_sWorkspace_jigsaw,
            sFilename_mpas_mesh_netcdf_in=_sFilename_mpas_mesh_netcdf,
            sFilename_jigsaw_mesh_netcdf_in=_sFilename_jigsaw_mesh_netcdf,
            sFilename_land_ocean_mask_in=mask_for_mpas,
            aConfig_jigsaw_in=_aConfig_jigsaw,
            pBoundary_in=boundary_for_mpas,
            sFilename_mesh_boundary_in=_sFilename_mesh_boundary,
        )

    elif mesh_type_name == "dggrid":
        dggrid_dir = os.path.join(_sWorkspace_output, "dggrid")
        os.makedirs(dggrid_dir, exist_ok=True)
        boundary_file = _sFilename_mesh_boundary if _iFlag_mesh_boundary == 1 else None
        if boundary_file:
            return create_dggrid_mesh(
                _iFlag_global,
                _iFlag_save_mesh,
                _sFilename_mesh,
                dggrid_dir,
                iResolution_index_in=_iResolution_index,
                iFlag_antarctic_in=_iFlag_antarctic,
                iFlag_arctic_in=_iFlag_arctic,
                sDggrid_type_in=_sDggrid_type,
                sFilename_boundary_in=boundary_file,
            )
        else:
            return create_dggrid_mesh(
                _iFlag_global,
                _iFlag_save_mesh,
                _sFilename_mesh,
                dggrid_dir,
                iResolution_index_in=_iResolution_index,
                sDggrid_type_in=_sDggrid_type,
            )

    elif mesh_type_name in ("triangular", "structured_tin"):
        dArea = np.power(_dResolution_meter, 2.0)
        dLength_edge = np.sqrt(4.0 * dArea / np.sqrt(3.0))
        dX_shift = 0.5 * dLength_edge
        dY_shift = 0.5 * dLength_edge * np.sqrt(3.0)
        ncolumn = int((_dX_lowerright - _dX_lowerleft) / dX_shift)
        nrow = int((_dY_upperleft - _dY_lowerleft) / dY_shift)
        return create_triangular_mesh(
            _dX_lowerleft,
            _dY_lowerleft,
            _dResolution_meter,
            ncolumn,
            nrow,
            _sFilename_mesh,
            _pProjection_reference,
        )

    elif mesh_type_name in ("tin", "unstructured_tin"):
        boundary_for_tin = (
            None if (_iFlag_antarctic == 1 or _iFlag_arctic == 1) else resolved_wkt
        )
        return create_tin_mesh(
            _iFlag_global,
            _iFlag_use_mesh_dem,
            _iFlag_save_mesh,
            _sFilename_jigsaw_mesh_netcdf,
            _sFilename_mesh,
            iFlag_run_jigsaw_in=_iFlag_run_jigsaw,
            iFlag_antarctic_in=_iFlag_antarctic,
            iFlag_arctic_in=_iFlag_arctic,
            sWorkspace_jigsaw_in=_sWorkspace_jigsaw,
            pBoundary_in=boundary_for_tin,
        )

    elif mesh_type_name == "cubicsphere":
        cubicsphere_dir = os.path.join(_sWorkspace_output, "tempestremap")
        os.makedirs(cubicsphere_dir, exist_ok=True)
        return create_cubicsphere_mesh(
            _iFlag_global,
            _iFlag_save_mesh,
            _dResolution_meter,
            _sFilename_mesh,
            cubicsphere_dir,
            pBoundary_in=resolved_wkt,
        )

    elif mesh_type_name == "healpix":
        from pyearthmesh.meshes.structured.dggs.healpix.create_healpix_mesh import (
            create_healpix_mesh,
        )

        output_ws = _sWorkspace_output or os.path.dirname(_sFilename_mesh) or "./output"
        return create_healpix_mesh(
            _dResolution_meter,
            _sFilename_mesh,
            output_ws,
            iFlag_netcdf=_get("iFlag_netcdf", 0),
            pBoundary_in=resolved_wkt,
            iFlag_use_tiles=_get("iFlag_use_tiles", 0),
            nTile=_get("nTile"),
            tile_mode=_get("tile_mode", "simple"),
            zone_level=_get("zone_level"),
            tiles_per_base=_get("tiles_per_base"),
        )

    elif mesh_type_name == "isea":
        from pyearthmesh.meshes.structured.dggs.isea.create_isea_mesh import (
            create_isea_mesh,
        )

        output_ws = _sWorkspace_output or os.path.dirname(_sFilename_mesh) or "./output"
        return create_isea_mesh(
            _dResolution_meter,
            _get("sISEA_type", "ISEA3H"),
            _sFilename_mesh,
            output_ws,
            iFlag_netcdf=_get("iFlag_netcdf", 0),
            pBoundary_in=resolved_wkt,
            iFlag_use_tiles=_get("iFlag_use_tiles", 0),
            nTile=_get("nTile"),
            tile_mode=_get("tile_mode", "simple"),
            zone_level=_get("zone_level"),
            tiles_per_base=_get("tiles_per_base"),
        )

    elif mesh_type_name == "rhealpix":
        from pyearthmesh.meshes.structured.dggs.healpix.create_rhealpix_mesh import (
            create_rhealpix_mesh,
        )

        output_ws = _sWorkspace_output or os.path.dirname(_sFilename_mesh) or "./output"
        return create_rhealpix_mesh(
            _dResolution_meter,
            _sFilename_mesh,
            output_ws,
            iFlag_netcdf=_get("iFlag_netcdf", 0),
            pBoundary_in=resolved_wkt,
            iFlag_use_tiles=_get("iFlag_use_tiles", 0),
            nTile=_get("nTile"),
            tile_mode=_get("tile_mode", "simple"),
            zone_level=_get("zone_level"),
            tiles_per_base=_get("tiles_per_base"),
        )

    else:
        raise ValueError(f"Unsupported mesh type: {mesh_type_name}")


__all__ = [
    "determine_mesh_boundary",
    "generate_mesh",
]
