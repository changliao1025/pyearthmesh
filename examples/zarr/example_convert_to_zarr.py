"""
Example: Convert existing mesh to Zarr format and read it back.

This example demonstrates how to:
1. Convert an existing GPKG/GeoJSON mesh to Zarr format
2. Read a subset of cells from Zarr
3. Export a subset back to vector format
"""
import os
from pyearthmesh.utility.convert_mesh_formats import (
    convert_to_zarr,
    convert_from_zarr
)
from pyearthmesh.utility.zarr_io import read_mesh_from_zarr

# Configuration
sWorkspace_input = "/compyfs/liao313/04model/pyearthmesh/isea3h/10km"
sWorkspace_output = "/compyfs/liao313/04model/pyearthmesh/isea3h/10km/zarr_test"

os.makedirs(sWorkspace_output, exist_ok=True)

# Input mesh file (can be GPKG, GeoJSON, Shapefile, etc.)
sFilename_input_mesh = os.path.join(sWorkspace_input, "dggrid", "isea3h_mesh.GPKG")

# Output Zarr store
sFilename_zarr = os.path.join(sWorkspace_output, "isea3h_mesh.zarr")

print("=" * 80)
print("Step 1: Converting mesh to Zarr format")
print("=" * 80)

# Convert to Zarr
convert_to_zarr(
    input_file=sFilename_input_mesh,
    output_path=sFilename_zarr,
    mesh_type="dggrid",
    chunk_size=500000,  # 500K cells per chunk
    compression_profile="balanced"  # or "fast" or "maximum"
)

print(f"\nZarr store created at: {sFilename_zarr}")

# Check Zarr store size
import subprocess
result = subprocess.run(
    ["du", "-sh", sFilename_zarr],
    capture_output=True,
    text=True
)
print(f"Zarr store size: {result.stdout.strip()}")

print("\n" + "=" * 80)
print("Step 2: Reading subset from Zarr")
print("=" * 80)

# Read cells in bounding box (example: Continental USA)
bbox = (-125, 24, -66, 49)  # (lon_min, lat_min, lon_max, lat_max)

print(f"\nReading cells in bounding box: {bbox}")
cells = read_mesh_from_zarr(
    zarr_path=sFilename_zarr,
    bbox=bbox,
    lazy=False
)

print(f"Loaded {len(cells)} cells from Zarr")

# Display info about first few cells
print("\nFirst 5 cells:")
for i, cell in enumerate(cells[:5]):
    print(f"  Cell {i}: ID={cell.lCellID}, "
          f"Center=({cell.dLongitude_center_degree:.4f}, "
          f"{cell.dLatitude_center_degree:.4f}), "
          f"Area={cell.dArea:.2f} m²")

print("\n" + "=" * 80)
print("Step 3: Exporting subset to vector format")
print("=" * 80)

# Export subset to GPKG for visualization
sFilename_subset = os.path.join(sWorkspace_output, "usa_subset.gpkg")

print(f"\nExporting to: {sFilename_subset}")
convert_from_zarr(
    zarr_path=sFilename_zarr,
    output_file=sFilename_subset,
    bbox=bbox
)

print(f"Subset exported successfully!")

# Can also export to other formats
sFilename_geojson = os.path.join(sWorkspace_output, "usa_subset.geojson")
convert_from_zarr(
    zarr_path=sFilename_zarr,
    output_file=sFilename_geojson,
    bbox=bbox
)
print(f"Also exported to GeoJSON: {sFilename_geojson}")

print("\n" + "=" * 80)
print("Step 4: Demonstrating lazy loading")
print("=" * 80)

# Open Zarr for lazy access (doesn't load all data into memory)
import zarr
zarr_group = zarr.open_group(sFilename_zarr, mode='r')

print(f"\nZarr store attributes:")
for key, value in zarr_group.attrs.items():
    print(f"  {key}: {value}")

print(f"\nZarr arrays:")
for name in zarr_group.array_keys():
    array = zarr_group[name]
    print(f"  {name}: shape={array.shape}, dtype={array.dtype}, "
          f"chunks={array.chunks}, nbytes={array.nbytes / 1e6:.2f} MB")

print("\n" + "=" * 80)
print("Complete! Zarr mesh operations demonstrated successfully.")
print("=" * 80)
