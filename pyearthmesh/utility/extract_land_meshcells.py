import os
from osgeo import ogr, _osr, gdal
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_driver_from_filename

from rtree import index
def extract_land_meshcells(sFilename_mesh, sFilename_coastline, sFilename_mesh_land_out, sFilename_mesh_touch_out=None):


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

    pDataset_coastline = ogr.Open(sFilename_coastline, 0)
    if pDataset_coastline is None:
        raise ValueError('Cannot open coastline file: ' + sFilename_coastline)

    # Get layers
    pLayer_mesh = pDataset_mesh.GetLayer(0)
    pLayer_coastline = pDataset_coastline.GetLayer(0)

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

    # Build R-tree spatial index for coastline geometries
    print('Building R-tree spatial index for coastline polygons...')
    rtree_idx = index.Index()
    aCoastline_geometries = {}

    pLayer_coastline.ResetReading()
    for idx, pFeature_coastline in enumerate(pLayer_coastline):
        pGeometry_coastline = pFeature_coastline.GetGeometryRef()
        if pGeometry_coastline is not None:
            # Get bounding box of coastline geometry
            envelope = pGeometry_coastline.GetEnvelope()  # returns (minX, maxX, minY, maxY)
            # Insert into R-tree with bbox (minX, minY, maxX, maxY)
            rtree_idx.insert(idx, (envelope[0], envelope[2], envelope[1], envelope[3]))
            # Store geometry by index
            aCoastline_geometries[idx] = pGeometry_coastline.Clone()

    print(f'R-tree index built with {len(aCoastline_geometries)} coastline polygons')

    # Loop through all mesh cells
    pLayer_mesh.ResetReading()
    nSaved = 0
    nTouch = 0
    nTotal = pLayer_mesh.GetFeatureCount()

    for i, pFeature_mesh in enumerate(pLayer_mesh):
        if i % 1000 == 0:
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
            bPartIntersects = False

            for idx in candidate_indices:
                pGeometry_coastline = aCoastline_geometries[idx]

                # Check Within first (completely inside)
                if pGeom_part.Within(pGeometry_coastline):
                    bPartWithin = True
                    bPartIntersects = True
                    bAnyIntersects = True
                    break
                # Check if it intersects (touches)
                elif pGeom_part.Intersects(pGeometry_coastline):
                    bPartIntersects = True
                    bAnyIntersects = True
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

    print(f'Saved {nSaved} mesh cells out of {nTotal} total cells that touch or are within coastlines')
    if pLayer_touch_out is not None:
        print(f'Saved {nTouch} mesh cells that only touch coastlines (excluding within)')

    # Cleanup
    pDataset_mesh = None
    pDataset_coastline = None
    pDataset_out = None
    if pDataset_touch_out is not None:
        pDataset_touch_out = None

    return nSaved