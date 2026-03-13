import os, stat, platform
import numpy as np
from pathlib import Path
import subprocess
import datetime
from shutil import copy2
from osgeo import osr, ogr, gdal
import healpy as hp
import json
from typing import List, Dict, Tuple, Optional
from pyearth.system.define_global_variables import earth_radius
from pyearth.gis.location.get_geometry_coordinates import get_geometry_coordinates
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_driver_from_filename
from pyearthmesh.utility.convert_coordinates import convert_gcs_coordinates_to_meshcell


pDate = datetime.datetime.today()
sDate_default = (
    "{:04d}".format(pDate.year)
    + "{:02d}".format(pDate.month)
    + "{:02d}".format(pDate.day)
)

slash = os.sep

#this module use the dggal tool:
#https://github.com/ecere/dggal


class HEALPixTileStructure:
    """
    Hierarchical tile structure based on HEALPix tessellation.

    Uses HEALPix's natural 12 base pixels as top-level tiles,
    with optional hierarchical subdivision.
    """

    def __init__(self, level: int, tile_scheme: str = 'hierarchical',
                 tiles_per_base: Optional[int] = None):
        """
        Initialize HEALPix tile structure.

        Parameters:
        -----------
        level : int
            HEALPix refinement level (nside = 2^level)
        tile_scheme : str
            'base' (12 tiles), 'hierarchical' (subdivided), 'adaptive'
        tiles_per_base : int, optional
            Number of tiles per base pixel for hierarchical scheme
        """
        self.level = level
        self.nside = 2 ** level
        self.npix = hp.nside2npix(self.nside)
        self.base_tiles = 12
        self.tile_scheme = tile_scheme

        if tile_scheme == 'base':
            self.tiles_per_base = 1
            self.total_tiles = 12
        elif tile_scheme == 'hierarchical':
            if tiles_per_base is None:
                self.tiles_per_base = self._calculate_optimal_tiles_per_base()
            else:
                self.tiles_per_base = tiles_per_base
            self.total_tiles = 12 * self.tiles_per_base
        else:
            self.tiles_per_base = tiles_per_base or 1
            self.total_tiles = 12 * self.tiles_per_base

        self.tile_metadata = {}
        self._build_tile_structure()

    def _calculate_optimal_tiles_per_base(self) -> int:
        """Calculate optimal tiles per base pixel (target ~75k cells/tile)."""
        target_cells_per_tile = 75000
        cells_per_base = self.npix // 12
        tiles_per_base = max(1, int(np.ceil(cells_per_base / target_cells_per_tile)))

        # Round to nearest power of 4 for clean subdivision
        if tiles_per_base > 1:
            power = int(np.ceil(np.log(tiles_per_base) / np.log(4)))
            tiles_per_base = 4 ** power

        return tiles_per_base

    def _build_tile_structure(self):
        """Build tile structure and metadata."""
        if self.tile_scheme == 'base':
            self._build_base_tile_structure()
        else:
            self._build_hierarchical_tile_structure()

    def _build_base_tile_structure(self):
        """Build 12 base tiles."""
        pixels_per_base = self.npix // 12
        for base_pix in range(12):
            pix_start = base_pix * pixels_per_base
            pix_end = (base_pix + 1) * pixels_per_base

            self.tile_metadata[base_pix] = {
                'tile_id': base_pix,
                'base_pixel': base_pix,
                'pix_range': (pix_start, pix_end),
                'n_pixels': pixels_per_base
            }

    def _build_hierarchical_tile_structure(self):
        """Build hierarchical tile structure with subdivision."""
        pixels_per_subtile = self.npix // (12 * self.tiles_per_base)

        tile_id = 0
        for base_pix in range(12):
            for sub_idx in range(self.tiles_per_base):
                pix_start = (base_pix * self.tiles_per_base + sub_idx) * pixels_per_subtile
                pix_end = pix_start + pixels_per_subtile

                self.tile_metadata[tile_id] = {
                    'tile_id': tile_id,
                    'base_pixel': base_pix,
                    'sub_index': sub_idx,
                    'pix_range': (pix_start, pix_end),
                    'n_pixels': pixels_per_subtile
                }
                tile_id += 1

    def get_tile_for_pixel(self, pix: int) -> int:
        """Get tile ID for a HEALPix pixel."""
        if self.tile_scheme == 'base':
            return pix // (self.npix // 12)
        else:
            pixels_per_tile = self.npix // self.total_tiles
            return pix // pixels_per_tile

    def get_tile_pixels(self, tile_id: int) -> np.ndarray:
        """Get all pixel indices for a tile."""
        pix_start, pix_end = self.tile_metadata[tile_id]['pix_range']
        return np.arange(pix_start, pix_end, dtype=np.int64)

class ZoneBasedTileStructure:
    """
    Zone-based tile structure where all cells in a zone form one tile.

    For mesh at level n, zones defined at level m (m < n).
    All level-n pixels within each level-m pixel form one tile.
    """

    def __init__(self, mesh_level: int, zone_level: int):
        """
        Initialize zone-based tile structure.

        Parameters:
        -----------
        mesh_level : int
            HEALPix level for mesh (fine resolution)
        zone_level : int
            HEALPix level for zones (coarse resolution)
        """
        if zone_level >= mesh_level:
            raise ValueError(f"zone_level ({zone_level}) must be < mesh_level ({mesh_level})")

        self.mesh_level = mesh_level
        self.zone_level = zone_level
        self.mesh_nside = 2 ** mesh_level
        self.zone_nside = 2 ** zone_level
        self.n_zones = hp.nside2npix(self.zone_nside)
        self.n_mesh_pixels = hp.nside2npix(self.mesh_nside)
        self.pixels_per_zone = self.n_mesh_pixels // self.n_zones

        print(f"Zone-based tiling: mesh_level={mesh_level}, zone_level={zone_level}")
        print(f"  Total zones: {self.n_zones}, Pixels per zone: {self.pixels_per_zone:,}")

    def get_zone_for_pixel(self, pix: int) -> int:
        """Get zone ID for a mesh pixel."""
        return pix // self.pixels_per_zone

    def get_pixels_in_zone(self, zone_id: int) -> np.ndarray:
        """Get all mesh pixels in a zone."""
        pix_start = zone_id * self.pixels_per_zone
        pix_end = (zone_id + 1) * self.pixels_per_zone
        return np.arange(pix_start, pix_end, dtype=np.int64)

def healpix_find_level_by_resolution(sHealpix_type, dResolution_meter):
    """
    Find the appropriate rHEALPix level for a given target resolution.

    For rHEALPix with aperture 9:
    - Level 0: ~4,759 km resolution
    - Each level divides cells by 3 (aperture 9 = 3²)
    - Resolution at level n ≈ R₀ / 3ⁿ

    Args:
        sHealpix_type (str): Type of HEALPix ('rHEALPix', 'HEALPix')
        dResolution_meter (float): Target resolution in meters

    Returns:
        int: The appropriate refinement level
    """
    # Earth's radius in meters
    dRadius_earth = earth_radius

    # rHEALPix aperture 9 (each level divides by 3)
    if sHealpix_type.lower() == 'rhealpix':
        # Base resolution at level 0 (approximate)
        # For rHEALPix, level 0 has 12 base cells on a sphere
        # Average cell edge length ≈ sqrt(4πR²/12) ≈ 4,759 km
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 12)

        # Aperture factor: each level divides by 3
        aperture_factor = 3.0

        # Find level where resolution_level0 / (aperture_factor^level) ≈ target_resolution
        # level = log(resolution_level0 / target_resolution) / log(aperture_factor)
        level = np.log(dResolution_level0 / dResolution_meter) / np.log(aperture_factor)

        # Round to nearest integer level
        iLevel = int(np.round(level))

        # Ensure non-negative level
        if iLevel < 0:
            iLevel = 0

    elif sHealpix_type.lower() == 'healpix':
        # HEALPix aperture 4 (each level divides by 2)
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 12)
        aperture_factor = 2.0

        level = np.log(dResolution_level0 / dResolution_meter) / np.log(aperture_factor)
        iLevel = int(np.round(level))

        if iLevel < 0:
            iLevel = 0
    else:
        raise ValueError(f"Unknown HEALPix type: {sHealpix_type}")

    return iLevel

def healpix_find_resolution_by_level(sHealpix_type, iLevel):
    """
    Find the resolution for a given rHEALPix level.

    For rHEALPix with aperture 9:
    - Level 0: ~4,759 km resolution
    - Each level divides cells by 3 (aperture 9 = 3²)
    - Resolution at level n ≈ R₀ / 3ⁿ

    Args:
        sHealpix_type (str): Type of HEALPix ('rHEALPix', 'HEALPix')
        iLevel (int): Refinement level

    Returns:
        float: The resolution in meters
    """
    # Earth's radius in meters
    dRadius_earth = earth_radius

    if sHealpix_type.lower() == 'rhealpix':
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 12)
        aperture_factor = 3.0

        dResolution = dResolution_level0 / (aperture_factor ** iLevel)

    elif sHealpix_type.lower() == 'healpix':
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 12)
        aperture_factor = 2.0

        dResolution = dResolution_level0 / (aperture_factor ** iLevel)

    else:
        raise ValueError(f"Unknown HEALPix type: {sHealpix_type}")

    return dResolution

def create_healpix_mesh(dResolution_meter_in,
    sFilename_mesh,
    sWorkspace_output,
    iFlag_netcdf = 0,
    pBoundary_in=None,
    iFlag_use_tiles=0,
    nTile=None,
    tile_mode='simple',
    zone_level=None,
    tiles_per_base=None):
    """
    Create a spherical mesh based on the HEALPix tessellation.

    Parameters
    ----------
    dResolution_meter_in : float
        The resolution parameter for the HEALPix mesh.
    sFilename_mesh : str
        The filename for the output mesh file (or base filename if using tiles).
    sWorkspace_output : str
        The output directory path.
    iFlag_netcdf : int, optional
        Flag to force NetCDF output (0=auto-detect from extension, 1=NetCDF), by default 0.
    pBoundary_in : object, optional
        Boundary constraint for the mesh (not yet implemented), by default None.
    iFlag_use_tiles : int, optional
        Flag to enable tiling (0=single file, 1=multiple tile files), by default 0.
    nTile : int, optional
        Number of tiles (legacy parameter). If None, automatically determined.
        Only used if iFlag_use_tiles=1 and tile_mode='simple', by default None.
    tile_mode : str, optional
        Tiling strategy: 'simple' (legacy), 'base' (12 tiles), 'hierarchical' (subdivided),
        'zone' (zone-based). Only used if iFlag_use_tiles=1, by default 'simple'.
    zone_level : int, optional
        Zone level for zone-based tiling (must be < mesh level).
        Only used if tile_mode='zone', by default None (auto-determined).
    tiles_per_base : int, optional
        Number of tiles per base pixel for hierarchical mode.
        Only used if tile_mode='hierarchical', by default None (auto-determined).

    Returns
    -------
    str or list or dict
        If single file: filename string
        If legacy tiling (tile_mode='simple'): list of filenames
        If advanced tiling: dict with 'filenames', 'tile_structure', 'metadata'
    """

    #find the level based on the resolution
    iLevel = healpix_find_level_by_resolution('HEALPix', dResolution_meter_in)
    print("Determined HEALPix level: " + str(iLevel))
    dResolution_actual = healpix_find_resolution_by_level('HEALPix', iLevel)
    if dResolution_actual < dResolution_meter_in:
        print(f"Warning: Actual resolution ({dResolution_actual:.1f} m) is finer than target resolution ({dResolution_meter_in:.1f} m). Consider using a coarser level for better performance.")
        #decrease by 1 level to get a coarser resolution
        iLevel = iLevel - 1
    else:
        print(f"Actual resolution at this level: {dResolution_actual:.1f} meters")

    #now generate the mesh using dggal
    # example command: dgg isea3h -crs ico grid 3 > isea3h-level3-isea.geojson

    nside = 2 ** iLevel  # nside is 2^level for HEALPix.

    # 1. Get total number of pixels
    npix = hp.nside2npix(nside)

    # 2. Get all pixels globally (no filtering)
    # HEALPix provides equal-area tessellation of the entire sphere
    pix_indices = np.arange(npix, dtype=np.int64)

    #get the file extension of the output file
    sFilename_mesh_full = os.path.join(sWorkspace_output, sFilename_mesh)
    sExtension = Path(sFilename_mesh_full).suffix.lower()
    if sExtension == '.nc':
        iFlag_netcdf = 1

    # Determine if we should use tiles
    n_cells = len(pix_indices)

    # Handle advanced tiling modes
    if iFlag_use_tiles == 1 and tile_mode in ['base', 'hierarchical', 'zone']:
        print(f"\nUsing advanced tiling mode: {tile_mode}")
        return _create_healpix_mesh_with_advanced_tiles(
            iLevel, nside, pix_indices, sFilename_mesh, sWorkspace_output,
            iFlag_netcdf, tile_mode, zone_level, tiles_per_base
        )

    # Auto-determine number of tiles if not specified (legacy mode)
    if iFlag_use_tiles == 1 and nTile is None:
        # Use approximately 100k cells per tile by default
        nTile = max(1, int(np.ceil(n_cells / 100000)))
        print(f"Auto-determined number of tiles: {nTile} (total cells: {n_cells})")

    # If not using tiles or only 1 tile, use original single-file approach
    if iFlag_use_tiles == 0 or nTile == 1:
        #remove the output file if it already exists
        if os.path.exists(sFilename_mesh_full):
            os.remove(sFilename_mesh_full)

        if iFlag_netcdf == 1:
            import netCDF4 as nc
            #save in netcdf format
            pDataset_out = nc.Dataset(sFilename_mesh_full, 'w', format='NETCDF4')
            pDataset_out.createDimension('cell', len(pix_indices))
            pDataset_out.createVariable('cellid', 'i4', ('cell',))
            pDataset_out.createVariable('lon', 'f4', ('cell',))
            pDataset_out.createVariable('lat', 'f4', ('cell',))
            #also save the vertices of the cells
            pDataset_out.createDimension('nv', 4) #careful, this assumes 4 vertices per cell, which is true for HEALPix but not for rHEALPix?
            pDataset_out.createVariable('xv', 'f4', ('cell', 'nv'))
            pDataset_out.createVariable('yv', 'f4', ('cell', 'nv'))

            #save the cell centers and vertices using a streaming method to avoid memory issues
            chunk_size = 10000  # Process 10,000 cells at a time

            for chunk_start in range(0, n_cells, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_cells)
                chunk_pix = pix_indices[chunk_start:chunk_end]
                chunk_len = len(chunk_pix)

                # Pre-allocate arrays for this chunk
                chunk_cellids = np.zeros(chunk_len, dtype=np.int32)
                chunk_lons = np.zeros(chunk_len, dtype=np.float32)
                chunk_lats = np.zeros(chunk_len, dtype=np.float32)
                chunk_xv = np.zeros((chunk_len, 4), dtype=np.float32)
                chunk_yv = np.zeros((chunk_len, 4), dtype=np.float32)

                # Process each cell in the chunk
                for i, pix in enumerate(chunk_pix):
                    # Get the 4 corners of the HEALPix cell
                    corners = hp.boundaries(nside, pix, step=1, nest=True)
                    # Convert vectors to Lat/Lon
                    # Note: hp.vec2ang with lonlat=True returns lon in [0, 360], not [-180, 180]
                    lons, lats = hp.vec2ang(corners.T, lonlat=True)
                    # Convert longitude from [0, 360] to [-180, 180]
                    # Handle 180° carefully to avoid splits: if most lons will be negative, convert 180 to -180
                    if np.all(lons >= 180):
                        lons = lons - 360
                    else:
                        # Count how many values are >180 (will become negative)
                        will_be_negative = np.sum(lons > 180)
                        will_be_positive = np.sum((lons < 180) & (lons >= 0))
                        at_boundary = np.sum(np.abs(lons - 180) < 1e-10)

                        # Convert all lons > 180
                        lons = np.where(lons > 180, lons - 360, lons)

                        # If more values will be negative, also convert 180 to -180
                        if will_be_negative > will_be_positive:
                            lons = np.where(np.abs(lons - 180) < 1e-10, -180.0, lons)

                    # Get cell center
                    center_lon, center_lat = hp.pix2ang(nside, pix, nest=True, lonlat=True)
                    # Convert center longitude from [0, 360] to [-180, 180]
                    center_lon = center_lon - 360 if center_lon > 180 else center_lon

                    # Store in chunk arrays
                    chunk_cellids[i] = pix
                    chunk_lons[i] = center_lon
                    chunk_lats[i] = center_lat
                    chunk_xv[i, :] = lons
                    chunk_yv[i, :] = lats

                # Write entire chunk to NetCDF at once
                pDataset_out.variables['cellid'][chunk_start:chunk_end] = chunk_cellids
                pDataset_out.variables['lon'][chunk_start:chunk_end] = chunk_lons
                pDataset_out.variables['lat'][chunk_start:chunk_end] = chunk_lats
                pDataset_out.variables['xv'][chunk_start:chunk_end, :] = chunk_xv
                pDataset_out.variables['yv'][chunk_start:chunk_end, :] = chunk_yv

                # Optional: print progress
                if (chunk_end % 50000 == 0) or (chunk_end == n_cells):
                    print(f"Processed {chunk_end}/{n_cells} cells ({100*chunk_end/n_cells:.1f}%)")

            # Close the NetCDF file
            pDataset_out.close()

        else:
            #use gis format for easy visualization
            pDriver = get_vector_driver_from_filename(sFilename_mesh_full)
            pDataset_out = pDriver.CreateDataSource(sFilename_mesh_full)

            pSRS = osr.SpatialReference()
            pSRS.ImportFromEPSG(4326)  # WGS84 Lat/Lon
            pLayer_out = pDataset_out.CreateLayer('healpix_mesh', srs=pSRS, geom_type=ogr.wkbPolygon)
            pLayer_out.CreateField(ogr.FieldDefn('cellid', ogr.OFTInteger))
            pLayer_out.CreateField(ogr.FieldDefn('lon', ogr.OFTReal))
            pLayer_out.CreateField(ogr.FieldDefn('lat', ogr.OFTReal))

            # Use transactions for better performance
            chunk_size = 1000  # Commit every 1000 features

            for chunk_start in range(0, n_cells, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_cells)
                chunk_pix = pix_indices[chunk_start:chunk_end]

                # Start transaction for this chunk
                pLayer_out.StartTransaction()

                for pix in chunk_pix:
                    # Get the 4 corners of the HEALPix cell
                    corners = hp.boundaries(nside, pix, step=1, nest=True)
                    # Convert vectors to Lat/Lon
                    lons, lats = hp.vec2ang(corners.T, lonlat=True)
                    # Convert longitude from [0, 360] to [-180, 180]
                    if np.all(lons >= 180):
                        lons = lons - 360
                    else:
                        # Count how many values are >180 (will become negative)
                        will_be_negative = np.sum(lons > 180)
                        will_be_positive = np.sum((lons < 180) & (lons >= 0))
                        at_boundary = np.sum(np.abs(lons - 180) < 1e-10)
                        # Convert all lons > 180
                        lons = np.where(lons > 180, lons - 360, lons)
                        # If more values will be negative, also convert 180 to -180
                        if will_be_negative > will_be_positive:
                            lons = np.where(np.abs(lons - 180) < 1e-10, -180.0, lons)

                    # Get cell center
                    center_lon, center_lat = hp.pix2ang(nside, pix, nest=True, lonlat=True)
                    # Convert center longitude from [0, 360] to [-180, 180]
                    center_lon = center_lon - 360 if center_lon > 180 else center_lon

                    # Create a polygon for this cell
                    ring = ogr.Geometry(ogr.wkbLinearRing)
                    for lon, lat in zip(lons, lats):
                        ring.AddPoint(lon, lat)
                    ring.CloseRings()

                    poly = ogr.Geometry(ogr.wkbPolygon)
                    poly.AddGeometry(ring)

                    # Create a feature for this cell
                    feature = ogr.Feature(pLayer_out.GetLayerDefn())
                    feature.SetField('cellid', int(pix))
                    feature.SetField('lon', float(center_lon))
                    feature.SetField('lat', float(center_lat))
                    feature.SetGeometry(poly)
                    pLayer_out.CreateFeature(feature)
                    feature = None

                # Commit transaction for this chunk
                pLayer_out.CommitTransaction()

                # Optional: print progress
                if (chunk_end % 10000 == 0) or (chunk_end == n_cells):
                    print(f"Processed {chunk_end}/{n_cells} cells ({100*chunk_end/n_cells:.1f}%)")

            pDataset_out.FlushCache()
            pDataset_out = None

        return sFilename_mesh

    else:
        # Use tiling approach - split cells into multiple files
        print(f"Creating {nTile} tile files...")

        # Calculate cells per tile
        cells_per_tile = int(np.ceil(n_cells / nTile))

        # Get base filename without extension
        sFilename_base = Path(sFilename_mesh).stem

        tile_filenames = []

        for iTile in range(nTile):
            # Calculate start and end indices for this tile
            tile_start = iTile * cells_per_tile
            tile_end = min((iTile + 1) * cells_per_tile, n_cells)
            tile_pix_indices = pix_indices[tile_start:tile_end]

            # Create tile filename
            sFilename_tile = f"{sFilename_base}_tile_{iTile:04d}{sExtension}"
            sFilename_tile_full = os.path.join(sWorkspace_output, sFilename_tile)

            # Remove existing tile file if it exists
            if os.path.exists(sFilename_tile_full):
                os.remove(sFilename_tile_full)

            tile_filenames.append(sFilename_tile)

            print(f"Creating tile {iTile+1}/{nTile}: {sFilename_tile} ({len(tile_pix_indices)} cells)")

            if iFlag_netcdf == 1:
                import netCDF4 as nc
                # Save tile in NetCDF format
                pDataset_tile = nc.Dataset(sFilename_tile_full, 'w', format='NETCDF4')
                pDataset_tile.createDimension('cell', len(tile_pix_indices))
                pDataset_tile.createVariable('cellid', 'i4', ('cell',))
                pDataset_tile.createVariable('lon', 'f4', ('cell',))
                pDataset_tile.createVariable('lat', 'f4', ('cell',))
                pDataset_tile.createDimension('nv', 4)
                pDataset_tile.createVariable('xv', 'f4', ('cell', 'nv'))
                pDataset_tile.createVariable('yv', 'f4', ('cell', 'nv'))

                # Add tile metadata
                pDataset_tile.tile_index = iTile
                pDataset_tile.total_tiles = nTile
                pDataset_tile.global_cell_start = tile_start
                pDataset_tile.global_cell_end = tile_end

                chunk_size = 10000
                tile_n_cells = len(tile_pix_indices)

                for chunk_start in range(0, tile_n_cells, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, tile_n_cells)
                    chunk_pix = tile_pix_indices[chunk_start:chunk_end]
                    chunk_len = len(chunk_pix)

                    chunk_cellids = np.zeros(chunk_len, dtype=np.int32)
                    chunk_lons = np.zeros(chunk_len, dtype=np.float32)
                    chunk_lats = np.zeros(chunk_len, dtype=np.float32)
                    chunk_xv = np.zeros((chunk_len, 4), dtype=np.float32)
                    chunk_yv = np.zeros((chunk_len, 4), dtype=np.float32)

                    for i, pix in enumerate(chunk_pix):
                        corners = hp.boundaries(nside, pix, step=1, nest=True)
                        lons, lats = hp.vec2ang(corners.T, lonlat=True)
                        # Convert longitude from [0, 360] to [-180, 180]
                        #if all equal or larger than 180
                        if np.all(lons >= 180):
                            lons = lons - 360
                        else:
                            lons = np.where(lons >= 180, lons - 360, lons)

                        center_lon, center_lat = hp.pix2ang(nside, pix, nest=True, lonlat=True)
                        # Convert center longitude from [0, 360] to [-180, 180]
                        center_lon = center_lon - 360 if center_lon > 180 else center_lon

                        chunk_cellids[i] = pix
                        chunk_lons[i] = center_lon
                        chunk_lats[i] = center_lat
                        chunk_xv[i, :] = lons
                        chunk_yv[i, :] = lats

                    pDataset_tile.variables['cellid'][chunk_start:chunk_end] = chunk_cellids
                    pDataset_tile.variables['lon'][chunk_start:chunk_end] = chunk_lons
                    pDataset_tile.variables['lat'][chunk_start:chunk_end] = chunk_lats
                    pDataset_tile.variables['xv'][chunk_start:chunk_end, :] = chunk_xv
                    pDataset_tile.variables['yv'][chunk_start:chunk_end, :] = chunk_yv

                pDataset_tile.close()

            else:
                # Save tile in vector format
                pDriver = get_vector_driver_from_filename(sFilename_tile_full)
                pDataset_tile = pDriver.CreateDataSource(sFilename_tile_full)

                pSRS = osr.SpatialReference()
                pSRS.ImportFromEPSG(4326)
                pLayer_tile = pDataset_tile.CreateLayer('healpix_mesh', srs=pSRS, geom_type=ogr.wkbPolygon)
                pLayer_tile.CreateField(ogr.FieldDefn('cellid', ogr.OFTInteger))
                pLayer_tile.CreateField(ogr.FieldDefn('lon', ogr.OFTReal))
                pLayer_tile.CreateField(ogr.FieldDefn('lat', ogr.OFTReal))
                pLayer_tile.CreateField(ogr.FieldDefn('tile_index', ogr.OFTInteger))

                chunk_size = 1000
                tile_n_cells = len(tile_pix_indices)

                for chunk_start in range(0, tile_n_cells, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, tile_n_cells)
                    chunk_pix = tile_pix_indices[chunk_start:chunk_end]

                    pLayer_tile.StartTransaction()

                    for pix in chunk_pix:
                        corners = hp.boundaries(nside, pix, step=1, nest=True)
                        lons, lats = hp.vec2ang(corners.T, lonlat=True)
                        # Convert longitude from [0, 360] to [-180, 180]
                        if np.all(lons >= 180):
                            lons = lons - 360
                        else:
                            lons = np.where(lons >= 180, lons - 360, lons)

                        center_lon, center_lat = hp.pix2ang(nside, pix, nest=True, lonlat=True)
                        # Convert center longitude from [0, 360] to [-180, 180]
                        center_lon = center_lon - 360 if center_lon > 180 else center_lon

                        ring = ogr.Geometry(ogr.wkbLinearRing)
                        for lon, lat in zip(lons, lats):
                            ring.AddPoint(lon, lat)
                        ring.CloseRings()

                        poly = ogr.Geometry(ogr.wkbPolygon)
                        poly.AddGeometry(ring)

                        feature = ogr.Feature(pLayer_tile.GetLayerDefn())
                        feature.SetField('cellid', int(pix))
                        feature.SetField('lon', float(center_lon))
                        feature.SetField('lat', float(center_lat))
                        feature.SetField('tile_index', iTile)
                        feature.SetGeometry(poly)
                        pLayer_tile.CreateFeature(feature)
                        feature = None

                    pLayer_tile.CommitTransaction()

                pDataset_tile.FlushCache()
                pDataset_tile = None

        print(f"Successfully created {nTile} tile files")
        return tile_filenames

def _create_healpix_mesh_with_advanced_tiles(iLevel, nside, pix_indices, sFilename_mesh,
                                             sWorkspace_output, iFlag_netcdf, tile_mode,
                                             zone_level, tiles_per_base):
    """
    Internal function to create HEALPix mesh with advanced tiling.

    Returns dict with 'filenames', 'tile_structure', 'metadata'.
    """
    sFilename_base = Path(sFilename_mesh).stem
    sExtension = Path(sFilename_mesh).suffix.lower()
    n_cells = len(pix_indices)

    # Create tile structure based on mode
    if tile_mode == 'zone':
        if zone_level is None:
            zone_level = max(0, iLevel - 11)  # Default: 8 levels coarser

        print(f"Creating zone-based tiles (zone_level={zone_level})...")
        tile_structure = ZoneBasedTileStructure(iLevel, zone_level)

        # Group pixels by zone
        zone_pixels = {}
        for pix in pix_indices:
            zone_id = tile_structure.get_zone_for_pixel(pix)
            if zone_id not in zone_pixels:
                zone_pixels[zone_id] = []
            zone_pixels[zone_id].append(pix)

        print(f"Pixels distributed across {len(zone_pixels)} zones")

        # Create tile files
        filenames = []
        for zone_id, zone_pix_list in zone_pixels.items():
            zone_pix_array = np.array(zone_pix_list, dtype=np.int64)
            sFilename_zone = f"{sFilename_base}_zone_{zone_id:05d}{sExtension}"
            sFilename_zone_full = os.path.join(sWorkspace_output, sFilename_zone)

            print(f"  Zone {zone_id}: {len(zone_pix_array):,} pixels -> {sFilename_zone}")
            _write_tile_file(sFilename_zone_full, zone_pix_array, nside, iFlag_netcdf)
            filenames.append(sFilename_zone)

        return {
            'filenames': filenames,
            'tile_structure': tile_structure,
            'metadata': {
                'level': iLevel,
                'nside': nside,
                'n_pixels': n_cells,
                'tile_mode': 'zone',
                'zone_level': zone_level,
                'n_zones': len(zone_pixels)
            }
        }

    else:  # 'base' or 'hierarchical'
        print(f"Creating {tile_mode} tile structure...")
        tile_structure = HEALPixTileStructure(iLevel, tile_scheme=tile_mode,
                                              tiles_per_base=tiles_per_base)

        # Group pixels by tile
        tile_pixels = {}
        for pix in pix_indices:
            tile_id = tile_structure.get_tile_for_pixel(pix)
            if tile_id not in tile_pixels:
                tile_pixels[tile_id] = []
            tile_pixels[tile_id].append(pix)

        print(f"Pixels distributed across {len(tile_pixels)} tiles")

        # Create tile files
        filenames = []
        for tile_id, tile_pix_list in tile_pixels.items():
            tile_pix_array = np.array(tile_pix_list, dtype=np.int64)

            # Get tile metadata for descriptive naming
            tile_meta = tile_structure.tile_metadata[tile_id]
            base_pixel = tile_meta['base_pixel']

            if tile_mode == 'base':
                # For base tiles: use base pixel number
                sFilename_tile = f"{sFilename_base}_base_{base_pixel:02d}{sExtension}"
            else:  # hierarchical
                # For hierarchical tiles: include base pixel and sub-index
                sub_index = tile_meta.get('sub_index', 0)
                sFilename_tile = f"{sFilename_base}_base_{base_pixel:02d}_sub_{sub_index:03d}{sExtension}"

            sFilename_tile_full = os.path.join(sWorkspace_output, sFilename_tile)

            print(f"  Tile {tile_id}: base_{base_pixel:02d}, {len(tile_pix_array):,} pixels -> {sFilename_tile}")
            _write_tile_file(sFilename_tile_full, tile_pix_array, nside, iFlag_netcdf)
            filenames.append(sFilename_tile)

        # Save tile metadata
        metadata_file = os.path.join(sWorkspace_output, f"{sFilename_base}_tile_metadata.json")
        _save_tile_metadata(tile_structure, metadata_file)

        return {
            'filenames': filenames,
            'tile_structure': tile_structure,
            'metadata': {
                'level': iLevel,
                'nside': nside,
                'n_pixels': n_cells,
                'tile_mode': tile_mode,
                'n_tiles': len(tile_pixels),
                'metadata_file': metadata_file
            }
        }

def _write_tile_file(filename, pix_indices, nside, iFlag_netcdf):
    """Write a single tile file."""
    if os.path.exists(filename):
        os.remove(filename)

    n_cells = len(pix_indices)

    if iFlag_netcdf == 1:
        import netCDF4 as nc
        pDataset = nc.Dataset(filename, 'w', format='NETCDF4')
        pDataset.createDimension('cell', n_cells)
        pDataset.createVariable('cellid', 'i4', ('cell',))
        pDataset.createVariable('lon', 'f4', ('cell',))
        pDataset.createVariable('lat', 'f4', ('cell',))
        pDataset.createDimension('nv', 4)
        pDataset.createVariable('xv', 'f4', ('cell', 'nv'))
        pDataset.createVariable('yv', 'f4', ('cell', 'nv'))

        # Write data in chunks
        chunk_size = 10000
        for chunk_start in range(0, n_cells, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_cells)
            chunk_pix = pix_indices[chunk_start:chunk_end]
            chunk_len = len(chunk_pix)

            chunk_cellids = np.zeros(chunk_len, dtype=np.int32)
            chunk_lons = np.zeros(chunk_len, dtype=np.float32)
            chunk_lats = np.zeros(chunk_len, dtype=np.float32)
            chunk_xv = np.zeros((chunk_len, 4), dtype=np.float32)
            chunk_yv = np.zeros((chunk_len, 4), dtype=np.float32)

            for i, pix in enumerate(chunk_pix):
                corners = hp.boundaries(nside, pix, step=1, nest=True)
                lons, lats = hp.vec2ang(corners.T, lonlat=True)
                if np.all(lons >= 180):
                    lons = lons - 360
                else:
                    lons = np.where(lons >= 180, lons - 360, lons)

                center_lon, center_lat = hp.pix2ang(nside, pix, nest=True, lonlat=True)
                center_lon = center_lon - 360 if center_lon > 180 else center_lon

                chunk_cellids[i] = pix
                chunk_lons[i] = center_lon
                chunk_lats[i] = center_lat
                chunk_xv[i, :] = lons
                chunk_yv[i, :] = lats

            pDataset.variables['cellid'][chunk_start:chunk_end] = chunk_cellids
            pDataset.variables['lon'][chunk_start:chunk_end] = chunk_lons
            pDataset.variables['lat'][chunk_start:chunk_end] = chunk_lats
            pDataset.variables['xv'][chunk_start:chunk_end, :] = chunk_xv
            pDataset.variables['yv'][chunk_start:chunk_end, :] = chunk_yv

        pDataset.close()
    else:
        # Vector format
        pDriver = get_vector_driver_from_filename(filename)
        pDataset = pDriver.CreateDataSource(filename)

        pSRS = osr.SpatialReference()
        pSRS.ImportFromEPSG(4326)
        pLayer = pDataset.CreateLayer('healpix_mesh', srs=pSRS, geom_type=ogr.wkbPolygon)
        pLayer.CreateField(ogr.FieldDefn('cellid', ogr.OFTInteger))
        pLayer.CreateField(ogr.FieldDefn('lon', ogr.OFTReal))
        pLayer.CreateField(ogr.FieldDefn('lat', ogr.OFTReal))

        chunk_size = 1000
        for chunk_start in range(0, n_cells, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_cells)
            chunk_pix = pix_indices[chunk_start:chunk_end]

            pLayer.StartTransaction()

            for pix in chunk_pix:
                corners = hp.boundaries(nside, pix, step=1, nest=True)
                lons, lats = hp.vec2ang(corners.T, lonlat=True)
                if np.all(lons >= 180):
                    lons = lons - 360
                else:
                    lons = np.where(lons >= 180, lons - 360, lons)

                center_lon, center_lat = hp.pix2ang(nside, pix, nest=True, lonlat=True)
                center_lon = center_lon - 360 if center_lon > 180 else center_lon

                ring = ogr.Geometry(ogr.wkbLinearRing)
                for lon, lat in zip(lons, lats):
                    ring.AddPoint(lon, lat)
                ring.CloseRings()

                poly = ogr.Geometry(ogr.wkbPolygon)
                poly.AddGeometry(ring)

                feature = ogr.Feature(pLayer.GetLayerDefn())
                feature.SetField('cellid', int(pix))
                feature.SetField('lon', float(center_lon))
                feature.SetField('lat', float(center_lat))
                feature.SetGeometry(poly)
                pLayer.CreateFeature(feature)
                feature = None

            pLayer.CommitTransaction()

        pDataset.FlushCache()
        pDataset = None

def _save_tile_metadata(tile_structure, filename):
    """Save tile metadata to JSON file."""
    metadata_json = {
        'structure_info': {
            'level': tile_structure.level,
            'nside': tile_structure.nside,
            'npix': tile_structure.npix,
            'tile_scheme': tile_structure.tile_scheme,
            'total_tiles': tile_structure.total_tiles,
            'tiles_per_base': tile_structure.tiles_per_base
        },
        'tiles': {}
    }

    for tile_id, meta in tile_structure.tile_metadata.items():
        tile_meta_json = {}
        for key, value in meta.items():
            if isinstance(value, (np.integer, np.floating)):
                tile_meta_json[key] = int(value) if isinstance(value, np.integer) else float(value)
            elif isinstance(value, tuple):
                tile_meta_json[key] = list(value)
            else:
                tile_meta_json[key] = value
        metadata_json['tiles'][str(tile_id)] = tile_meta_json

    with open(filename, 'w') as f:
        json.dump(metadata_json, f, indent=2)

    print(f"Tile metadata saved to: {filename}")


