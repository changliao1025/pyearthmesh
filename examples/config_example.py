#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example demonstrating the new ConfigManager architecture
Shows how to create configurations for different mesh types
"""

import os
import json
from pathlib import Path
from pyearthmesh.classes.config import ConfigManager

# Set up output directory
output_dir = Path("./test_configs")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("ConfigManager Example - Creating configurations for different mesh types")
print("=" * 80)

# Example 1: Create a JIGSAW configuration
print("\n1. Creating JIGSAW configuration...")
jigsaw_manager = ConfigManager("jigsaw")
jigsaw_config = jigsaw_manager.create_template_config(
    output_dir / "jigsaw_config.json",
    custom_values={
        "ncolumn_space": 720,
        "nrow_space": 360,
        "dResolution_land": 30.0,
        "sWorkspace_output": "./output/jigsaw_run",
    }
)
print(f"   * JIGSAW config created with {len(jigsaw_config)} parameters")
print(f"   * Resolution: {jigsaw_config['ncolumn_space']}x{jigsaw_config['nrow_space']}")

# Example 2: Create an MPAS configuration
print("\n2. Creating MPAS configuration...")
mpas_manager = ConfigManager("mpas")
mpas_config = mpas_manager.create_template_config(
    output_dir / "mpas_config.json",
    custom_values={
        "sFilename_mpas_mesh_netcdf": "./data/mpas_mesh.nc",
        "iFlag_run_jigsaw": 1,
        "sWorkspace_output": "./output/mpas_run",
    }
)
print(f"   * MPAS config created with {len(mpas_config)} parameters")
print(f"   * Run JIGSAW: {mpas_config['iFlag_run_jigsaw']}")

# Example 3: Create a HEALPix configuration
print("\n3. Creating HEALPix configuration...")
healpix_manager = ConfigManager("healpix")
healpix_config = healpix_manager.create_template_config(
    output_dir / "healpix_config.json",
    custom_values={
        "nside": 16,
        "nest": True,
        "sWorkspace_output": "./output/healpix_run",
    }
)
print(f"   * HEALPix config created with {len(healpix_config)} parameters")
print(f"   * nside: {healpix_config['nside']}, nest: {healpix_config['nest']}")

# Example 4: Create an ISEA configuration
print("\n4. Creating ISEA configuration...")
isea_manager = ConfigManager("isea")
isea_config = isea_manager.create_template_config(
    output_dir / "isea_config.json",
    custom_values={
        "resolution": 9,
        "aperture": 3,
        "sWorkspace_output": "./output/isea_run",
    }
)
print(f"   * ISEA config created with {len(isea_config)} parameters")
print(f"   * Resolution: {isea_config['resolution']}, Aperture: {isea_config['aperture']}")

# Example 5: Create a Lat-Lon configuration
print("\n5. Creating Lat-Lon configuration...")
latlon_manager = ConfigManager("latlon")
latlon_config = latlon_manager.create_template_config(
    output_dir / "latlon_config.json",
    custom_values={
        "dLongitude_degree": 0.5,
        "dLatitude_degree": 0.5,
        "dLongitude_left": -180.0,
        "dLongitude_right": 180.0,
        "dLatitude_bot": -90.0,
        "dLatitude_top": 90.0,
        "sWorkspace_output": "./output/latlon_run",
    }
)
print(f"   * Lat-Lon config created with {len(latlon_config)} parameters")
print(f"   * Resolution: {latlon_config['dLongitude_degree']} deg x {latlon_config['dLatitude_degree']} deg")

# Example 6: Load and validate a configuration
print("\n6. Loading and validating configuration...")
loaded_config = ConfigManager.load_config(output_dir / "jigsaw_config.json")
print(f"   * Loaded config for mesh type: {loaded_config['mesh_type']}")
print(f"   * Configuration validated successfully")

# Example 7: Show available mesh types
print("\n7. Available mesh types:")
available_types = ConfigManager.get_available_mesh_types()
for mesh_type in available_types:
    print(f"   - {mesh_type}")

# Example 8: Using ConfigManager without specifying mesh type initially
print("\n8. Using ConfigManager with dynamic mesh type...")
dynamic_manager = ConfigManager()
config_healpix = dynamic_manager.get_default_config("healpix")
print(f"   * Got default config for HEALPix with {len(config_healpix)} parameters")

# Example 9: Validation example (will catch errors)
print("\n9. Testing configuration validation...")
try:
    healpix_manager_test = ConfigManager("healpix")
    invalid_config = {"nside": 7, "mesh_type": "healpix"}  # 7 is not a power of 2
    healpix_manager_test.validate_config(invalid_config)
except ValueError as e:
    print(f"   * Validation correctly caught error: {e}")

print("\n" + "=" * 80)
print("All examples completed successfully!")
print(f"Configuration files saved to: {output_dir.absolute()}")
print("=" * 80)
