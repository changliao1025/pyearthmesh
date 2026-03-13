import os, sys
from osgeo import ogr, _osr, gdal
import shutil
from concurrent.futures import ProcessPoolExecutor
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_driver_from_filename
from rtree import index

# Progress logging interval for large mesh loops.
PROGRESS_INTERVAL = 100000

def _build_coastline_rtree(sFilename_coastline, aExtent_mesh=None):
    """
    Build R-tree spatial index for coastline geometries.

    Parameters:
    -----------
    sFilename_coastline : str
        Path to coastline file
    aExtent_mesh : tuple, optional
        Extent of the mesh (minX, maxX, minY, maxY). When provided, coastline
        features are filtered to this tile extent before R-tree construction.

    Returns:
    --------
    tuple : (rtree_idx, aCoastline_geometries)
        R-tree index and dictionary of coastline geometries (WKT strings if bSerialize_for_parallel=True)
    """
    pDataset_coastline = ogr.Open(sFilename_coastline, 0)
    if pDataset_coastline is None:
        raise ValueError('Cannot open coastline file: ' + sFilename_coastline)

    pLayer_coastline = pDataset_coastline.GetLayer(0)

    print('Building R-tree spatial index for coastline polygons...', flush=True)
    rtree_idx = index.Index()
    aCoastline_geometries = {}

    if aExtent_mesh is not None:
        min_x, max_x, min_y, max_y = aExtent_mesh
        if min_x <= max_x and min_y <= max_y:
            pLayer_coastline.SetSpatialFilterRect(min_x, min_y, max_x, max_y)
            print(
                f'Applying coastline spatial filter to tile extent: ({min_x}, {min_y}, {max_x}, {max_y})',
                flush=True,
            )
        else:
            print('Skipping coastline spatial filter because mesh extent is invalid', flush=True)

    pLayer_coastline.ResetReading()
    for idx, pFeature_coastline in enumerate(pLayer_coastline):
        pGeometry_coastline = pFeature_coastline.GetGeometryRef()
        if pGeometry_coastline is not None:
            # Get bounding box of coastline geometry
            envelope = pGeometry_coastline.GetEnvelope()  # returns (minX, maxX, minY, maxY)
            # Insert into R-tree with bbox (minX, minY, maxX, maxY)
            rtree_idx.insert(idx, (envelope[0], envelope[2], envelope[1], envelope[3]))
            # Store geometry - convert to WKT for parallel processing to avoid pickling issues
            aCoastline_geometries[idx] = pGeometry_coastline.Clone()

    pLayer_coastline.SetSpatialFilter(None)

    pDataset_coastline = None

    print(f'R-tree index built with {len(aCoastline_geometries)} coastline polygons', flush=True)
    return rtree_idx, aCoastline_geometries

def extract_land_meshcells(sFilename_mesh_in,
                           sFilename_coastline,
                           sFilename_mesh_out,
                           sFilename_mesh_touch_out=None,
                           iFlag_multi_tile=0,
                           sExtension_in=None,
                           sExtension_out=None,
                           iFlag_parallel=0,
                           nWorker_in=None
                           ):
    """
    Extract land mesh cells from mesh file(s) using coastline data.

    This function supports both single-file and multi-tile processing modes.

    Parameters:
    -----------
    sFilename_mesh_in : str
        Single-file mode (iFlag_multi_tile=0): Path to input mesh file
        Multi-tile mode (iFlag_multi_tile=1): Path to input workspace directory containing mesh tile files
    sFilename_coastline : str
        Path to coastline file
    sFilename_mesh_out : str
        Single-file mode (iFlag_multi_tile=0): Path to output file for land mesh cells
        Multi-tile mode (iFlag_multi_tile=1): Path to output workspace directory for extracted land mesh files
    sFilename_mesh_touch_out : str, optional
        Single-file mode: Path to output file for cells that only touch coastlines
        Multi-tile mode: If provided, separate touch files will be created for each tile
    iFlag_multi_tile : int, optional
        0 = single-file mode (default), 1 = multi-tile mode (workspace processing)
    sExtension_in : str, optional
        Multi-tile mode only: File extension to match for input files (e.g., '.geojson', '.shp')
        Required when iFlag_multi_tile=1
    sExtension_out : str, optional
        Multi-tile mode only: File extension for output files (defaults to same as input)
    iFlag_parallel : int, optional
        Multi-tile mode only: 0 = sequential processing (default), 1 = parallel processing
    nWorker_in : int, optional
        Multi-tile mode only: Number of worker processes for parallel execution (default: None, uses system default)


    Returns:
    --------
    int or None :
        Single-file mode: Number of mesh cells saved
        Multi-tile mode: None
    """
    # Multi-tile mode
    if iFlag_multi_tile:
        return _extract_land_meshcells_multi_tile_impl(
            sWorkspace_in=sFilename_mesh_in,
            sWorkspace_out=sFilename_mesh_out,
            sExtension_in=sExtension_in,
            sFilename_coastline=sFilename_coastline,
            sExtension_out=sExtension_out,
            bExtract_touch_in=(sFilename_mesh_touch_out is not None),
            iFlag_parallel=iFlag_parallel,
            nWorker_in=nWorker_in
        )

    # Single-file mode (original implementation)
    return _extract_land_meshcells_single_file(
        sFilename_mesh=sFilename_mesh_in,
        sFilename_coastline=sFilename_coastline,
        sFilename_mesh_land_out=sFilename_mesh_out,
        sFilename_mesh_touch_out=sFilename_mesh_touch_out
    )

def _extract_land_meshcells_single_file(sFilename_mesh,
                                        sFilename_coastline,
                                        sFilename_mesh_land_out,
                                        sFilename_mesh_touch_out=None   ):
    """
    Internal function: Extract land mesh cells from a single mesh file.

    Parameters:
    -----------
    sFilename_mesh : str
        Path to input mesh file
    sFilename_coastline : str
        Path to coastline file
    sFilename_mesh_land_out : str
        Path to output file for land mesh cells
    sFilename_mesh_touch_out : str, optional
        Path to output file for cells that only touch coastlines


    Returns:
    --------
    int : Number of mesh cells saved
    """
    pDriver_out = get_vector_driver_from_filename(sFilename_mesh_land_out)
    if pDriver_out is None:
        raise ValueError('Cannot get driver for output file: ' + sFilename_mesh_land_out)

    #delete output file if it already exists
    if os.path.exists(sFilename_mesh_land_out):
        os.remove(sFilename_mesh_land_out)

    # Create output datasource for all land cells
    pDataset_out = pDriver_out.CreateDataSource(sFilename_mesh_land_out)
    if pDataset_out is None:
        raise ValueError('Cannot create output file: ' + sFilename_mesh_land_out)

    # Create optional output datasource for touch-only cells
    pDataset_touch_out = None
    pDriver_touch_out = None
    if sFilename_mesh_touch_out is not None:
        pDriver_touch_out = get_vector_driver_from_filename(sFilename_mesh_touch_out)
        if pDriver_touch_out is None:
            raise ValueError('Cannot get driver for touch output file: ' + sFilename_mesh_touch_out)
        if os.path.exists(sFilename_mesh_touch_out):
            os.remove(sFilename_mesh_touch_out)
        pDataset_touch_out = pDriver_touch_out.CreateDataSource(sFilename_mesh_touch_out)
        if pDataset_touch_out is None:
            raise ValueError('Cannot create touch output file: ' + sFilename_mesh_touch_out)

    #read the mesh
    pDataset_mesh = ogr.Open(sFilename_mesh, 0)
    if pDataset_mesh is None:
        raise ValueError('Cannot open mesh file: ' + sFilename_mesh)

    # Get layers
    pLayer_mesh = pDataset_mesh.GetLayer(0)
    aExtent_mesh = pLayer_mesh.GetExtent(force=1)  # (minX, maxX, minY, maxY)

    # Get spatial reference and layer definition from mesh
    pSpatialRef = pLayer_mesh.GetSpatialRef()
    pLayerDefn_mesh = pLayer_mesh.GetLayerDefn()

    # Create output layer with same schema as input mesh
    pLayer_out = pDataset_out.CreateLayer('mesh_land', pSpatialRef, ogr.wkbPolygon)
    if pLayer_out is None:
        raise ValueError('Cannot create output layer')

    # Copy field definitions from mesh layer
    for i in range(pLayerDefn_mesh.GetFieldCount()):
        pFieldDefn = pLayerDefn_mesh.GetFieldDefn(i)
        pLayer_out.CreateField(pFieldDefn)

    # Create optional touch-only output layer
    pLayer_touch_out = None
    if pDataset_touch_out is not None:
        pLayer_touch_out = pDataset_touch_out.CreateLayer('mesh_touch', pSpatialRef, ogr.wkbPolygon)
        if pLayer_touch_out is None:
            raise ValueError('Cannot create touch output layer')
        # Copy field definitions
        for i in range(pLayerDefn_mesh.GetFieldCount()):
            pFieldDefn = pLayerDefn_mesh.GetFieldDefn(i)
            pLayer_touch_out.CreateField(pFieldDefn)

    # Build R-tree spatial index if not provided
    rtree_idx, aCoastline_geometries = _build_coastline_rtree(sFilename_coastline, aExtent_mesh)

    # Loop through all mesh cells
    pLayer_mesh.ResetReading()
    nSaved = 0
    nTouch = 0
    nTotal = pLayer_mesh.GetFeatureCount()

    for i, pFeature_mesh in enumerate(pLayer_mesh):
        if i % PROGRESS_INTERVAL == 0:
            print(f'Processing mesh cell {i}/{nTotal}...')

        pGeometry_mesh = pFeature_mesh.GetGeometryRef()
        if pGeometry_mesh is None:
            continue

        # Handle multipart geometries (e.g., cells crossing antimeridian)
        aGeometry_parts = []
        geom_type = pGeometry_mesh.GetGeometryType()

        if geom_type == ogr.wkbMultiPolygon or geom_type == ogr.wkbMultiPolygon25D:
            # Split multipolygon into individual parts
            for j in range(pGeometry_mesh.GetGeometryCount()):
                pGeom_part = pGeometry_mesh.GetGeometryRef(j)
                if pGeom_part is not None:
                    aGeometry_parts.append(pGeom_part)
        else:
            # Single polygon
            aGeometry_parts.append(pGeometry_mesh)

        # Track status for each part
        bAnyIntersects = False
        bAnyTouches = False
        bAllPartsWithin = True  # Assume all parts are within until proven otherwise

        # Process each part of the geometry
        for pGeom_part in aGeometry_parts:
            # Get bounding box of mesh cell part
            envelope_mesh = pGeom_part.GetEnvelope()  # (minX, maxX, minY, maxY)
            bbox_mesh = (envelope_mesh[0], envelope_mesh[2], envelope_mesh[1], envelope_mesh[3])

            # Use R-tree to find candidate coastline polygons that intersect the bbox
            candidate_indices = list(rtree_idx.intersection(bbox_mesh))

            bPartWithin = False
            for idx in candidate_indices:
                coastline_data = aCoastline_geometries[idx]

                # Handle both WKT strings (from parallel processing) and geometry objects
                if isinstance(coastline_data, str):
                    pGeometry_coastline = ogr.CreateGeometryFromWkt(coastline_data)
                else:
                    pGeometry_coastline = coastline_data

                # Most candidates do not intersect; short-circuit those quickly.
                if not pGeom_part.Intersects(pGeometry_coastline):
                    continue

                bAnyIntersects = True

                # Only evaluate Within for intersecting candidates.
                if pGeom_part.Within(pGeometry_coastline):
                    bPartWithin = True
                    break

                bAnyTouches = True

            # If this part is not within any coastline, then not all parts are within
            if not bPartWithin:
                bAllPartsWithin = False

        # Determine final status:
        # - bIntersects: true if any part intersects
        # - bWithin: true only if ALL parts are within
        # - bTouches: true if any part touches AND not all parts are within
        bIntersects = bAnyIntersects
        bWithin = bAllPartsWithin and bAnyIntersects
        bTouches = bAnyTouches and not bAllPartsWithin

        # If it intersects or is within a coastline, save to main output
        if bIntersects:
            pFeature_out = ogr.Feature(pLayer_out.GetLayerDefn())
            pFeature_out.SetGeometry(pGeometry_mesh.Clone())

            # Copy all attributes
            for j in range(pLayerDefn_mesh.GetFieldCount()):
                pFeature_out.SetField(j, pFeature_mesh.GetField(j))

            pLayer_out.CreateFeature(pFeature_out)
            pFeature_out = None
            nSaved += 1

            # If it only touches (not within) and we have a touch output file, save it there too
            if bTouches and not bWithin and pLayer_touch_out is not None:
                pFeature_touch_out = ogr.Feature(pLayer_touch_out.GetLayerDefn())
                pFeature_touch_out.SetGeometry(pGeometry_mesh.Clone())

                # Copy all attributes
                for j in range(pLayerDefn_mesh.GetFieldCount()):
                    pFeature_touch_out.SetField(j, pFeature_mesh.GetField(j))

                pLayer_touch_out.CreateFeature(pFeature_touch_out)
                pFeature_touch_out = None
                nTouch += 1

    print(f'Saved {nSaved} mesh cells out of {nTotal} total cells that touch or are within coastlines', flush=True)
    if pLayer_touch_out is not None:
        print(f'Saved {nTouch} mesh cells that only touch coastlines (excluding within)', flush=True)

    # Cleanup
    pDataset_mesh = None
    pDataset_coastline = None
    pDataset_out = None
    if pDataset_touch_out is not None:
        pDataset_touch_out = None

    return nSaved

def _extract_single_tile_worker(task):
    """Internal worker function: Extract land mesh cells for a single tile file.

    Each worker builds its own R-tree from the coastline file to avoid pickling issues.
    """
    sFile_mesh_in, sFilename_coastline, sWorkspace_out, sExtension_out, bExtract_touch = task

    # Generate output filenames
    sBasename = os.path.splitext(os.path.basename(sFile_mesh_in))[0]
    sFile_land_out = os.path.join(sWorkspace_out, sBasename + '_land' + sExtension_out)

    sFile_touch_out = None
    if bExtract_touch:
        sFile_touch_out = os.path.join(sWorkspace_out, sBasename + '_touch' + sExtension_out)

    print(f"Extracting land mesh cells from: {sFile_mesh_in}", flush=True)
    print(f"  Output land file: {sFile_land_out}", flush=True)
    if sFile_touch_out:
        print(f"  Output touch file: {sFile_touch_out}", flush=True)

    # Call the single-file extraction function
    # R-tree will be built internally since we don't pass it
    nSaved = _extract_land_meshcells_single_file(
        sFile_mesh_in,
        sFilename_coastline,
        sFile_land_out,
        sFile_touch_out
    )

    print(f"Completed extraction for {sFile_mesh_in}: {nSaved} cells saved", flush=True)
    return sFile_land_out

def _extract_land_meshcells_multi_tile_impl(sWorkspace_in,
                                             sWorkspace_out,
                                             sExtension_in,
                                             sFilename_coastline,
                                             sExtension_out=None,
                                             bExtract_touch_in=False,
                                             iFlag_parallel=0,
                                             nWorker_in=None):
    """
    Internal implementation: Extract land mesh cells from multiple tile files in a workspace.

    Parameters:
    -----------
    sWorkspace_in : str
        Input workspace directory containing mesh tile files
    sWorkspace_out : str
        Output workspace directory for extracted land mesh files
    sExtension_in : str
        File extension to match for input files (e.g., '.geojson', '.shp')
    sFilename_coastline : str
        Path to coastline file used for extraction
    sExtension_out : str, optional
        File extension for output files (defaults to same as input)
    bExtract_touch_in : bool, optional
        Whether to create separate output files for cells that only touch coastlines (default: False)
    iFlag_parallel : int, optional
        Whether to use parallel processing (0=sequential, 1=parallel, default: 0)
    nWorker_in : int, optional
        Number of worker processes for parallel execution (default: None, uses system default)

    Returns:
    --------
    None
    """
    # Check existence of the input workspace
    if not os.path.exists(sWorkspace_in):
        print("The input workspace does not exist: " + sWorkspace_in)
        sys.exit(1)

    # Check existence of coastline file
    if not os.path.exists(sFilename_coastline):
        print("The coastline file does not exist: " + sFilename_coastline)
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
        # Exclude files that have '_land' or '_touch' in the name to avoid re-processing
        files = [f for f in files if '_land' not in f and '_touch' not in f]
        for file in files:
            if file.endswith(sExtension_in):
                sFile_in = os.path.join(root, file)
                # Each worker will build its own R-tree to avoid pickling issues
                aTask.append((sFile_in, sFilename_coastline, sWorkspace_out, sExtension_out,
                             bExtract_touch_in))

    nFile = len(aTask)
    print(f"Found {nFile} matching files in the input workspace: {sWorkspace_in}")

    if nFile == 0:
        print("No matching files found. Exiting.")
        return

    # Process files either in parallel or sequentially
    if iFlag_parallel and nFile > 1:
        print(f"Processing {nFile} files in parallel with {nWorker_in or 'default'} workers...")
        print("Note: Each worker builds its own R-tree index from the coastline file.")
        print("This adds overhead but avoids GDAL/OGR multiprocessing issues.")
        # Use chunksize=1 to ensure proper distribution of work across workers
        # GDAL/OGR has global state issues, so we process one file at a time per worker
        with ProcessPoolExecutor(max_workers=nWorker_in) as executor:
            results = list(executor.map(_extract_single_tile_worker, aTask, chunksize=1))
        print(f"Processed {len(results)} files successfully.")
    else:
        print(f"Processing {nFile} files sequentially...")
        for task in aTask:
            _extract_single_tile_worker(task)

    print("=" * 80)
    print("Land mesh cell extraction completed for all matching files in the workspace.")
    print("=" * 80)

def extract_land_meshcells_multi_tile(sWorkspace_in,
                                       sWorkspace_out,
                                       sExtension_in,
                                       sFilename_coastline,
                                       sExtension_out=None,
                                       bExtract_touch_in=False,
                                       iFlag_parallel=0,
                                       nWorker_in=None):
    """
    DEPRECATED: Use extract_land_meshcells() with iFlag_multi_tile=1 instead.

    This function is kept for backward compatibility.
    Extract land mesh cells from multiple tile files in a workspace.

    Parameters:
    -----------
    sWorkspace_in : str
        Input workspace directory containing mesh tile files
    sWorkspace_out : str
        Output workspace directory for extracted land mesh files
    sExtension_in : str
        File extension to match for input files (e.g., '.geojson', '.shp')
    sFilename_coastline : str
        Path to coastline file used for extraction
    sExtension_out : str, optional
        File extension for output files (defaults to same as input)
    bExtract_touch_in : bool, optional
        Whether to create separate output files for cells that only touch coastlines (default: False)
    iFlag_parallel : int, optional
        Whether to use parallel processing (0=sequential, 1=parallel, default: 0)
    nWorker_in : int, optional
        Number of worker processes for parallel execution (default: None, uses system default)

    Returns:
    --------
    None
    """
    return extract_land_meshcells(
        sFilename_mesh_in=sWorkspace_in,
        sFilename_coastline=sFilename_coastline,
        sFilename_mesh_out=sWorkspace_out,
        sFilename_mesh_touch_out='_touch_' if bExtract_touch_in else None,
        iFlag_multi_tile=1,
        sExtension_in=sExtension_in,
        sExtension_out=sExtension_out,
        iFlag_parallel=iFlag_parallel,
        nWorker_in=nWorker_in
    )