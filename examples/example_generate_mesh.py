#!/usr/bin/env python3
"""Unified Mesh Boundary Determination and Generation Example.

Demonstrates using ``generate_mesh`` and ``determine_mesh_boundary`` to:
1. Automatically determine boundary WKT from vector boundaries or coordinates.
2. Generate meshes across different mesh types (Lat-Lon, MPAS, Hexagon, Square)
   using the unified approach.
"""

from pathlib import Path

from pyearthmesh.meshes.generate_mesh import (
    determine_mesh_boundary,
    generate_mesh,
)


def main() -> None:
    output_dir = Path("./output/unified_mesh_examples").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PyEarthMesh: Unified Boundary Determination & Mesh Generation")
    print("=" * 80)

    # ── 1. Boundary determination from bounding box ───────────────────────────
    print("\n1. Determining boundary from regional coordinates...")
    wkt, extent = determine_mesh_boundary(
        dLongitude_left=100.0,
        dLongitude_right=105.0,
        dLatitude_bot=30.0,
        dLatitude_top=35.0,
        iFlag_global=0,
    )
    wkt_preview = wkt[:50] if wkt else "None"
    print(f"   * Resolved WKT: {wkt_preview}...")
    print(f"   * Bounding extent: {extent}")

    # ── 2. Unified regional Lat-Lon mesh generation ───────────────────────────
    print("\n2. Generating regional Lat-Lon mesh...")
    latlon_output = str(output_dir / "regional_latlon.geojson")
    cells = generate_mesh(
        sMesh_type="latlon",
        sFilename_mesh=latlon_output,
        dLongitude_left=100.0,
        dLongitude_right=105.0,
        dLatitude_bot=30.0,
        dLatitude_top=35.0,
        dResolution_degree=0.5,
        iFlag_global=0,
    )
    print(f"   * Generated {len(cells)} Lat-Lon cells -> {latlon_output}")

    # ── 3. Unified Square mesh generation ─────────────────────────────────────
    print("\n3. Generating local Square mesh...")
    from osgeo import osr

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    proj_wkt = srs.ExportToWkt()

    square_output = str(output_dir / "local_square.geojson")
    cells_square = generate_mesh(
        sMesh_type="square",
        sFilename_mesh=square_output,
        dX_lowerleft=100.0,
        dY_lowerleft=30.0,
        dX_lowerright=102.0,
        dY_upperleft=32.0,
        dResolution_meter=0.5,
        pBoundary_wkt=wkt,
        pProjection_reference=proj_wkt,
    )
    print(f"   * Generated {len(cells_square)} Square cells -> {square_output}")

    print("\n" + "=" * 80)
    print("Unified mesh generation examples completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
