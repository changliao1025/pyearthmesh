import geopandas as gpd
import pandas as pd
import numpy as np
import os
import sys
from shapely.strtree import STRtree
from shapely.geometry import Point
from concurrent.futures import ProcessPoolExecutor

from retired import mesh

def extract_land_meshcells_vectorized(mesh_path, land_path, output_path,
                        iFlag_multi_tile=0,
                        sExtension_in=None,
                        sExtension_out=None,
                        iParallel_in=0,
                        nWorker_in=None):
    """
    Efficiently filters millions of mesh cells against land polygons using STRtree spatial indexing.

    Supports both single-file and multi-tile processing modes with optional parallel processing.

    Parameters:
    -----------
    mesh_path : str
        Single-file mode (iFlag_multi_tile=0): Path to input mesh file
        Multi-tile mode (iFlag_multi_tile=1): Path to input workspace directory containing mesh tile files
    land_path : str
        Path to land polygons file (parquet, gpkg, geojson, etc.)
    output_path : str
        Single-file mode (iFlag_multi_tile=0): Path to output file for filtered mesh cells
        Multi-tile mode (iFlag_multi_tile=1): Path to output workspace directory for filtered mesh files
    iFlag_multi_tile : int, optional
        0 = single-file mode (default), 1 = multi-tile mode (workspace processing)
    sExtension_in : str, optional
        Multi-tile mode only: File extension to match for input files (e.g., '.parquet', '.geojson')
        Required when iFlag_multi_tile=1
    sExtension_out : str, optional
        Multi-tile mode only: File extension for output files (defaults to same as input)
    iParallel_in : int, optional
        Multi-tile mode only: 0 = sequential processing (default), 1 = parallel processing
    nWorker_in : int, optional
        Multi-tile mode only: Number of worker processes for parallel processing.
        None = use default (number of CPU cores). Ignored if iParallel_in=0.

    Returns:
    --------
    int or None :
        Single-file mode: Number of mesh cells saved
        Multi-tile mode: None
    """
    # Multi-tile mode
    if iFlag_multi_tile:
        return _filter_mesh_by_land_multi_tile(
            sWorkspace_in=mesh_path,
            sFilename_land=land_path,
            sWorkspace_out=output_path,
            sExtension_in=sExtension_in,
            sExtension_out=sExtension_out,
            iParallel_in=iParallel_in,
            nWorker_in=nWorker_in
        )

    # Single-file mode
    return _filter_mesh_by_land_single_file(mesh_path, land_path, output_path)


def _filter_mesh_by_land_single_file(mesh_path, land_path, output_path):
    """
    Internal function: Filter mesh cells from a single mesh file using land polygons.

    Parameters:
    -----------
    mesh_path : str
        Path to input mesh file
    land_path : str
        Path to land polygons file
    output_path : str
        Path to output file for filtered mesh cells

    Returns:
    --------
    int : Number of mesh cells saved
    """
    print(f"Processing: {mesh_path}")
    print(f"Loading land polygons from: {land_path}")
    if os.path.exists(output_path):
        #remove first
        os.remove(output_path)

    # OPTIMIZATION 1: Use pyogrio for faster land loading
    land = gpd.read_file(land_path, engine="pyogrio")
    print(f"Loaded {len(land)} land polygons")

    print(f"Loading mesh cells from: {mesh_path}")
    # For large datasets, use pyogrio engine if available for 10x faster loading
    mesh = gpd.read_file(mesh_path, engine="pyogrio")
    print(f"Loaded {len(mesh)} mesh cells")

    # STEP 1: Calculate Centroids
    # Testing a point inside a polygon is significantly faster than
    # testing if two polygons overlap.
    print("Calculating centroids...")
    #centroids = mesh.geometry.centroid
    # Fix 1: Pass numpy array, not GeoSeries
    centroids_arr = mesh.geometry.centroid.values  # numpy array of Shapely Points

    # STEP 2: Spatial Index (The "Magic" part)
    # We build an STRTree of the land polygons
    print("Building spatial index...")
    tree = STRtree(land.geometry.values)

    # STEP 3: Query the index
    # Find which centroids are potentially within the bounding boxes of land
    print("Querying spatial index...")

    # Fix 2: Use 'within' — semantically correct for point-in-polygon,
    # and avoids the boundary-intersection path that 'intersects' also checks
    possible_indices = tree.query(centroids_arr, predicate='within')

    # The result is [query_geometry_idx, tree_geometry_idx]. Since the
    # query geometries are mesh centroids, possible_indices[0] contains the
    # mesh indices that matched land polygons in the tree.
    # OPTIMIZATION 2: Direct set conversion with np.fromiter (faster than list conversion)
    print(f"Extracting unique indices from {len(possible_indices[0])} intersections...")
    land_mesh_indices = np.fromiter(set(possible_indices[0]), dtype=np.int64)

    print(f"Filtering: Found {len(land_mesh_indices)} cells on land.")

    # OPTIMIZATION 3: Use boolean indexing instead of iloc (faster for large datasets)
    mask = np.zeros(len(mesh), dtype=bool)
    mask[land_mesh_indices] = True
    land_mesh = mesh[mask]

    print(f"Saving to {output_path}...")
    if len(land_mesh) == 0:
        return 0

    # Auto-detect format based on file extension
    file_ext = os.path.splitext(output_path)[1].lower()
    if file_ext == '.parquet':
        land_mesh.to_parquet(output_path)
    elif file_ext == '.gpkg':
        land_mesh.to_file(output_path, driver="GPKG")
    elif file_ext in ['.shp', '.geojson', '.json']:
        land_mesh.to_file(output_path)
    else:
        # Default to parquet for efficiency with large datasets
        land_mesh.to_parquet(output_path)

    print(f"Saved {len(land_mesh)} mesh cells to {output_path}")
    print("Done!")

    return len(land_mesh)


def _filter_single_tile_worker(task):
    """
    Internal worker function: Filter mesh cells for a single tile file.

    Designed for use with ProcessPoolExecutor.map().

    Parameters:
    -----------
    task : tuple
        Tuple of (sFile_mesh_in, sFilename_land, sWorkspace_out, sExtension_out)

    Returns:
    --------
    str : Path to output file
    """
    sFile_mesh_in, sFilename_land, sWorkspace_out, sExtension_out = task

    # Generate output filename
    sBasename = os.path.splitext(os.path.basename(sFile_mesh_in))[0]
    sFile_out = os.path.join(sWorkspace_out, sBasename + '_land' + sExtension_out)

    print("=" * 80)
    print(f"Processing tile: {sFile_mesh_in}")
    print(f"Output file: {sFile_out}")
    print("=" * 80)

    # Call the single-file filtering function
    nSaved = _filter_mesh_by_land_single_file(
        sFile_mesh_in,
        sFilename_land,
        sFile_out
    )

    print(f"Completed filtering for {sFile_mesh_in}: {nSaved} cells saved")
    return sFile_out


def _filter_mesh_by_land_multi_tile(sWorkspace_in,
                                     sFilename_land,
                                     sWorkspace_out,
                                     sExtension_in,
                                     sExtension_out=None,
                                     iParallel_in=0,
                                     nWorker_in=None):
    """
    Internal implementation: Filter mesh cells from multiple tile files in a workspace.

    Parameters:
    -----------
    sWorkspace_in : str
        Input workspace directory containing mesh tile files
    sFilename_land : str
        Path to land polygons file
    sWorkspace_out : str
        Output workspace directory for filtered mesh files
    sExtension_in : str
        File extension to match for input files (e.g., '.parquet', '.geojson')
    sExtension_out : str, optional
        File extension for output files (defaults to same as input)
    iParallel_in : int, optional
        0 = sequential processing (default), 1 = parallel processing
    nWorker_in : int, optional
        Number of worker processes for parallel processing.
        None = use default (number of CPU cores). Ignored if iParallel_in=0.

    Returns:
    --------
    None
    """
    # Check existence of the input workspace
    if not os.path.exists(sWorkspace_in):
        print("The input workspace does not exist: " + sWorkspace_in)
        sys.exit(1)

    # Check existence of land file
    if not os.path.exists(sFilename_land):
        print("The land file does not exist: " + sFilename_land)
        sys.exit(1)

    # Create the output workspace if not exist
    if not os.path.exists(sWorkspace_out):
        os.makedirs(sWorkspace_out)

    # Default output extension to input extension if not specified
    if sExtension_out is None:
        sExtension_out = sExtension_in

    # Find all the matching files in the input workspace
    sWorkspace_in_abs = os.path.abspath(sWorkspace_in)
    sWorkspace_out_abs = os.path.abspath(sWorkspace_out)

    aTask = []
    for root, dirs, files in os.walk(sWorkspace_in):
        # Prevent os.walk from descending into the output workspace when it's
        # inside the input workspace by removing matching dirs in-place
        dirs[:] = [d for d in dirs if not os.path.abspath(os.path.join(root, d)).startswith(sWorkspace_out_abs)]

        # Skip processing if current root is the output workspace (safety)
        if os.path.abspath(root).startswith(sWorkspace_out_abs):
            continue

        # Sort files to ensure consistent processing order
        files.sort()

        # Exclude files that have '_land' in the name to avoid re-processing
        files = [f for f in files if '_land' not in f]

        for file in files:
            if file.endswith(sExtension_in):
                sFile_in = os.path.join(root, file)
                aTask.append((sFile_in, sFilename_land, sWorkspace_out, sExtension_out))

    nFile = len(aTask)
    print("=" * 80)
    print(f"Found {nFile} matching files with extension '{sExtension_in}' in: {sWorkspace_in}")
    print("=" * 80)

    if nFile == 0:
        print("No matching files found. Exiting.")
        return

    if iParallel_in and nFile > 1:
        print(f"Processing {nFile} files in parallel with {nWorker_in or 'default'} workers...")
        print("=" * 80)
        with ProcessPoolExecutor(max_workers=nWorker_in) as executor:
            for _ in executor.map(_filter_single_tile_worker, aTask):
                pass
    else:
        print(f"Processing {nFile} files sequentially...")
        print("=" * 80)
        for i, task in enumerate(aTask, 1):
            print(f"\n[{i}/{nFile}] Processing file {i} of {nFile}")
            _filter_single_tile_worker(task)

    print("\n" + "=" * 80)
    print("Land mesh cell filtering completed for all matching files in the workspace.")
    print(f"Output files saved to: {sWorkspace_out}")
    print("=" * 80)



