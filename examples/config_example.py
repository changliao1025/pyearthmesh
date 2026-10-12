#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example demonstrating the modern two-tier workflow configuration and compilation.

Shows how user-facing workflow specifications are compiled into mesh-native
configurations in memory, and demonstrates the low-level backend validation
provided by ConfigManager.
"""

from pyearthmesh.config.config_manager import compile_workflow
from pyearthmesh.classes.meshconfig import ConfigManager

print("=" * 80)
print("pyearthmesh Configuration Architecture Example")
print("=" * 80)

# Example 1: Compile a JIGSAW workflow specification into native parameters
print("\n1. Compiling JIGSAW workflow specification...")
jigsaw_spec = {
    "workflow": {"mesh_type": "mpas", "case_index": 1, "model": "jigsaw"},
    "resolutions": {"ocean_km": 100, "land_km": 30.0},
    "paths": {"output_workspace": "./output/jigsaw_run"},
    "jigsaw": {"ncolumn_space": 720, "nrow_space": 360},
}
jigsaw_native = compile_workflow(jigsaw_spec)
print(f"   * Native mesh type:  {jigsaw_native['mesh_type']}")
print(f"   * Resolution ocean:  {jigsaw_native['dResolution_ocean']} km")
print(f"   * Grid dimensions:   {jigsaw_native['ncolumn_space']}x{jigsaw_native['nrow_space']}")

# Example 2: Compile an MPAS workflow specification
print("\n2. Compiling MPAS workflow specification...")
mpas_spec = {
    "workflow": {"mesh_type": "mpas", "case_index": 2},
    "resolutions": {"ocean_km": 100, "land_km": 12.5, "river_network_km": 5},
    "paths": {"output_workspace": "./output/mpas_run"},
}
mpas_native = compile_workflow(mpas_spec)
print(f"   * Mesh type:         {mpas_native['mesh_type']}")
print(f"   * Run JIGSAW flag:   {mpas_native['iFlag_run_jigsaw']}")
print(f"   * Land resolution:   {mpas_native['dResolution_land']} km")

# Example 3: Compile a HEALPix workflow specification
print("\n3. Compiling HEALPix workflow specification...")
healpix_spec = {
    "workflow": {"mesh_type": "healpix"},
    "resolutions": {"meter": 50000.0},
    "paths": {"output_workspace": "./output/healpix_run", "output_mesh": "healpix.geojson"},
    "features": {"use_tiles": False, "tile_mode": "simple"},
}
healpix_native = compile_workflow(healpix_spec)
print(f"   * Mesh type:         {healpix_native['mesh_type']}")
print(f"   * Resolution meter:  {healpix_native['dResolution_meter']}")
print(f"   * Output mesh:       {healpix_native['sFilename_mesh']}")

# Example 4: Compile an ISEA workflow specification
print("\n4. Compiling ISEA workflow specification...")
isea_spec = {
    "workflow": {"mesh_type": "isea"},
    "resolutions": {"meter": 10000.0},
    "paths": {"output_workspace": "./output/isea_run", "output_mesh": "isea.geojson"},
    "features": {"isea_type": "ISEA3H"},
}
isea_native = compile_workflow(isea_spec)
print(f"   * Mesh type:         {isea_native['mesh_type']}")
print(f"   * ISEA type:         {isea_native['sISEA_type']}")

# Example 5: Compile a Lat-Lon workflow specification
print("\n5. Compiling Lat-Lon workflow specification...")
latlon_spec = {
    "mesh_type": "latlon",
    "resolution": {"degree": 0.5},
    "region": {"bbox": {"left": -180.0, "right": 180.0, "bottom": -90.0, "top": 90.0}},
    "paths": {"output_workspace": "./output/latlon_run"},
}
latlon_native = compile_workflow(latlon_spec)
print(f"   * Mesh type:         {latlon_native['mesh_type']}")
print(f"   * Grid dimensions:   {latlon_native['ncolumn']} columns x {latlon_native['nrow']} rows")

# Example 6: Query available mesh types from backend registry
print("\n6. Available mesh types registered in ConfigManager:")
for mesh_type in ConfigManager.get_available_mesh_types():
    print(f"   - {mesh_type}")

# Example 7: Backend default configuration inspection
print("\n7. Inspecting raw backend defaults for HEALPix...")
default_healpix = ConfigManager("healpix").get_default_config()
print(f"   * Default parameter count: {len(default_healpix)}")

# Example 8: Validation example (correctly catching errors)
print("\n8. Testing backend validation...")
try:
    healpix_manager = ConfigManager("healpix")
    invalid_config = {"nside": 7, "mesh_type": "healpix"}  # 7 is not a power of 2
    healpix_manager.validate_config(invalid_config)
except ValueError as e:
    print(f"   * Validation correctly caught error: {e}")

print("\n" + "=" * 80)
print("All workflow compilation and validation examples completed successfully!")
print("=" * 80)
