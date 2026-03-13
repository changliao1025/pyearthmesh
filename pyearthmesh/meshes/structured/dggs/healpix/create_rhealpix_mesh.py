import os, stat, platform
import numpy as np
from pathlib import Path
import subprocess
import datetime
from shutil import copy2
from osgeo import osr, ogr, gdal
import json
from typing import List, Dict, Tuple, Optional

from dggal import *

# Initialize the application
app = Application(appGlobals=globals())
pydggal_setup(app)

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
#API documentation: https://dggal.org/docs/html/dggal/Classes/DGGRS.html


class RHEALPixTileStructure:
    """
    Hierarchical tile structure based on rHEALPix tessellation.

    Uses rHEALPix's natural 12 base pixels as top-level tiles,
    with optional hierarchical subdivision. Adapted for aperture 9.
    """

    def __init__(self, level: int, tile_scheme: str = 'hierarchical',
                 tiles_per_base: Optional[int] = None):
        """
        Initialize rHEALPix tile structure.

        Parameters:
        -----------
        level : int
            rHEALPix refinement level
        tile_scheme : str
            'base' (12 tiles), 'hierarchical' (subdivided), 'adaptive'
        tiles_per_base : int, optional
            Number of tiles per base pixel for hierarchical scheme
        """
        self.level = level
        self.aperture = 9  # rHEALPix uses aperture 9
        self.n_cells_per_level = 12 * (self.aperture ** level)
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
        cells_per_base = self.n_cells_per_level // 12
        tiles_per_base = max(1, int(np.ceil(cells_per_base / target_cells_per_tile)))

        # Round to nearest power of 9 for clean subdivision (aperture 9)
        if tiles_per_base > 1:
            power = int(np.ceil(np.log(tiles_per_base) / np.log(9)))
            tiles_per_base = 9 ** power

        return tiles_per_base

    def _build_tile_structure(self):
        """Build tile structure and metadata."""
        if self.tile_scheme == 'base':
            self._build_base_tile_structure()
        else:
            self._build_hierarchical_tile_structure()

    def _build_base_tile_structure(self):
        """Build 12 base tiles."""
        cells_per_base = self.n_cells_per_level // 12
        for base_idx in range(12):
            cell_start = base_idx * cells_per_base
            cell_end = (base_idx + 1) * cells_per_base

            self.tile_metadata[base_idx] = {
                'tile_id': base_idx,
                'base_pixel': base_idx,
                'cell_range': (cell_start, cell_end),
                'n_cells': cells_per_base
            }

    def _build_hierarchical_tile_structure(self):
        """Build hierarchical tile structure with subdivision."""
        cells_per_subtile = self.n_cells_per_level // (12 * self.tiles_per_base)

        tile_id = 0
        for base_idx in range(12):
            for sub_idx in range(self.tiles_per_base):
                cell_start = (base_idx * self.tiles_per_base + sub_idx) * cells_per_subtile
                cell_end = cell_start + cells_per_subtile

                self.tile_metadata[tile_id] = {
                    'tile_id': tile_id,
                    'base_pixel': base_idx,
                    'sub_index': sub_idx,
                    'cell_range': (cell_start, cell_end),
                    'n_cells': cells_per_subtile
                }
                tile_id += 1

    def get_tile_for_cell_index(self, cell_index: int) -> int:
        """Get tile ID for a rHEALPix cell based on its index in the sorted list."""
        if self.tile_scheme == 'base':
            return cell_index // (self.n_cells_per_level // 12)
        else:
            cells_per_tile = self.n_cells_per_level // self.total_tiles
            return cell_index // cells_per_tile

    def get_tile_cell_range(self, tile_id: int) -> Tuple[int, int]:
        """Get cell index range for a tile."""
        return self.tile_metadata[tile_id]['cell_range']


class RHEALPixZoneBasedTileStructure:
    """
    Zone-based tile structure where all cells in a zone form one tile.

    For mesh at level n, zones defined at level m (m < n).
    All level-n cells within each level-m cell form one tile.
    """

    def __init__(self, mesh_level: int, zone_level: int):
        """
        Initialize zone-based tile structure for rHEALPix.

        Parameters:
        -----------
        mesh_level : int
            rHEALPix level for mesh (fine resolution)
        zone_level : int
            rHEALPix level for zones (coarse resolution)
        """
        if zone_level >= mesh_level:
            raise ValueError(f"zone_level ({zone_level}) must be < mesh_level ({mesh_level})")

        self.mesh_level = mesh_level
        self.zone_level = zone_level
        self.aperture = 9
        self.n_zones = 12 * (self.aperture ** zone_level)
        self.n_mesh_cells = 12 * (self.aperture ** mesh_level)
        self.cells_per_zone = self.n_mesh_cells // self.n_zones

        print(f"Zone-based tiling: mesh_level={mesh_level}, zone_level={zone_level}")
        print(f"  Total zones: {self.n_zones}, Cells per zone: {self.cells_per_zone:,}")

    def get_zone_for_cell_index(self, cell_index: int) -> int:
        """Get zone ID for a mesh cell based on its index."""
        return cell_index // self.cells_per_zone

    def get_cell_range_in_zone(self, zone_id: int) -> Tuple[int, int]:
        """Get cell index range in a zone."""
        cell_start = zone_id * self.cells_per_zone
        cell_end = (zone_id + 1) * self.cells_per_zone
        return (cell_start, cell_end)

#dggrs = rHEALPix()

def generate_dggal_bash_script(sWorkspace_output, sCommand_in):
    os.chdir(sWorkspace_output)
    # detemine the system platform
    # Determine the appropriate executable name for the platform
    system = platform.system()

    if system == "Windows":
        # execute binary on Windows
        iFlag_unix = 0
    elif system == "Linux":
        # execute binary on Linux
        iFlag_unix = 1
    elif system == "Darwin":
        # execute binary on macOS
        iFlag_unix = 1
    else:
        # unsupported operating system
        print("Unsupported operating system: " + system)
        print("Please reach out to the developers for assistance.")
        # generate the bash/batch script
    if iFlag_unix == 1:
        sFilename_bash = os.path.join(str(Path(sWorkspace_output)), "run_dggal.sh")
        ofs = open(sFilename_bash, "w")
        sLine = "#!/bin/bash\n"
        ofs.write(sLine)
        sLine = "cd " + sWorkspace_output + "\n"
        ofs.write(sLine)
        sLine = sCommand_in + "\n"
        ofs.write(sLine)
        ofs.close()
        os.chmod(sFilename_bash, stat.S_IRWXU)
    else:
        sFilename_bash = os.path.join(str(Path(sWorkspace_output)), "run_dggal.bat")
        ofs = open(sFilename_bash, "w")
        sLine = "cd " + sWorkspace_output + "\n"
        ofs.write(sLine)
        sLine = sCommand_in + "\n"
        ofs.write(sLine)
        ofs.close()
        os.chmod(sFilename_bash, stat.S_IRWXU)

    return sFilename_bash

def rhealpix_find_level_by_resolution(sHealpix_type, dResolution_meter):
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
    dggrs = ISEA3H()
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

def rhealpix_find_resolution_by_level(sHealpix_type, iLevel):
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


def create_rhealpix_mesh(dResolution_meter_in,
    sFilename_mesh,
    sWorkspace_output,
    iFlag_netcdf=0,
    pBoundary_in=None,
    iFlag_use_tiles=0,
    nTile=None,
    tile_mode='simple',
    zone_level=None,
    tiles_per_base=None):
    """
    Create a spherical mesh based on the rHEALPix tessellation.

    Parameters
    ----------
    dResolution_meter_in : float
        The resolution parameter for the rHEALPix mesh.
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
    iLevel = rhealpix_find_level_by_resolution('rHEALPix', dResolution_meter_in)
    print("Determined rHEALPix level: " + str(iLevel))
    dResolution_actual = rhealpix_find_resolution_by_level('rHEALPix', iLevel)
    if dResolution_actual < dResolution_meter_in:
        print(f"Warning: Actual resolution ({dResolution_actual:.1f} m) is finer than target resolution ({dResolution_meter_in:.1f} m). Consider using a coarser level for better performance.")
        #decrease by 1 level to get a coarser resolution
        iLevel = iLevel - 1
    else:
        print(f"Actual resolution at this level: {dResolution_actual:.1f} meters")

    # Initialize DGGRS for rHEALPix
    dggrs = rHEALPix()

    # Get all cell IDs at this resolution
    # For rHEALPix, we generate cells programmatically
    n_cells = 12 * (9 ** iLevel)
    print(f"Total cells to generate: {n_cells:,}")

    #get the file extension of the output file
    sFilename_mesh_full = os.path.join(sWorkspace_output, sFilename_mesh)
    sExtension = Path(sFilename_mesh_full).suffix.lower()
    if sExtension == '.nc':
        iFlag_netcdf = 1

    # Handle advanced tiling modes
    if iFlag_use_tiles == 1 and tile_mode in ['base', 'hierarchical', 'zone']:
        print(f"\nUsing advanced tiling mode: {tile_mode}")
        return _create_rhealpix_mesh_with_advanced_tiles(
            iLevel, dggrs, sFilename_mesh, sWorkspace_output,
            iFlag_netcdf, tile_mode, zone_level, tiles_per_base, n_cells
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
            pDataset_out.createDimension('cell', n_cells)
            pDataset_out.createVariable('cellid', 'i4', ('cell',))
            pDataset_out.createVariable('lon', 'f4', ('cell',))
            pDataset_out.createVariable('lat', 'f4', ('cell',))

            #save the cell centers using a streaming method to avoid memory issues
            chunk_size = 10000  # Process 10,000 cells at a time

            for chunk_start in range(0, n_cells, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_cells)
                chunk_len = chunk_end - chunk_start

                # Pre-allocate arrays for this chunk
                chunk_cellids = np.zeros(chunk_len, dtype=np.int32)
                chunk_lons = np.zeros(chunk_len, dtype=np.float32)
                chunk_lats = np.zeros(chunk_len, dtype=np.float32)

                # Process each cell in the chunk
                for i in range(chunk_len):
                    cell_id = chunk_start + i
                    # Get cell geometry from dggrs
                    #cell_geom = dggrs.cell_to_geometry(cell_id)
                    centroid = cell_geom.Centroid()

                    chunk_cellids[i] = cell_id
                    chunk_lons[i] = centroid.GetX()
                    chunk_lats[i] = centroid.GetY()

                # Write entire chunk to NetCDF at once
                pDataset_out.variables['cellid'][chunk_start:chunk_end] = chunk_cellids
                pDataset_out.variables['lon'][chunk_start:chunk_end] = chunk_lons
                pDataset_out.variables['lat'][chunk_start:chunk_end] = chunk_lats

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
            pLayer_out = pDataset_out.CreateLayer('rhealpix_mesh', srs=pSRS, geom_type=ogr.wkbPolygon)
            pLayer_out.CreateField(ogr.FieldDefn('cellid', ogr.OFTInteger))
            pLayer_out.CreateField(ogr.FieldDefn('lon', ogr.OFTReal))
            pLayer_out.CreateField(ogr.FieldDefn('lat', ogr.OFTReal))

            # Use transactions for better performance
            chunk_size = 1000  # Commit every 1000 features

            for chunk_start in range(0, n_cells, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_cells)

                # Start transaction for this chunk
                pLayer_out.StartTransaction()

                for cell_id in range(chunk_start, chunk_end):
                    #Convert cell ID to zone
                    zone = dggrs.getZoneFromTextID(cell_id)
                    if zone == nullZone:
                        print(f"Error: Invalid cell ID '{cell_id}'")
                        return None
                    # Get WGS84 vertices (lon/lat coordinates)
                    # The second parameter (0) is the refinement level
                    vertices = dggrs.getZoneRefinedWGS84Vertices(zone, 0)
                    centroid = dggrs.getZoneWGS84Centroid(zone)

                    cell_geom = ogr.Geometry(ogr.wkbPolygon)
                    ring = ogr.Geometry(ogr.wkbLinearRing)
                    for vertex in vertices:
                        lon, lat = vertex
                        ring.AddPoint(lon, lat)
                    ring.CloseRings()
                    cell_geom.AddGeometry(ring)

                    # Create a feature for this cell
                    feature = ogr.Feature(pLayer_out.GetLayerDefn())
                    feature.SetField('cellid', int(cell_id))
                    feature.SetField('lon', float(centroid.lon))
                    feature.SetField('lat', float(centroid.lat))
                    feature.SetGeometry(cell_geom)
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

            # Create tile filename
            sFilename_tile = f"{sFilename_base}_tile_{iTile:04d}{sExtension}"
            sFilename_tile_full = os.path.join(sWorkspace_output, sFilename_tile)

            # Remove existing tile file if it exists
            if os.path.exists(sFilename_tile_full):
                os.remove(sFilename_tile_full)

            tile_filenames.append(sFilename_tile)

            print(f"Creating tile {iTile+1}/{nTile}: {sFilename_tile} ({tile_end - tile_start} cells)")

            _write_tile_file(dggrs, sFilename_tile_full, iFlag_netcdf,
                           tile_start, tile_end, iTile, nTile)

        print(f"Successfully created {nTile} tile files")
        return tile_filenames


def _create_rhealpix_mesh_with_advanced_tiles(iLevel, dggrs, sFilename_mesh,
                                              sWorkspace_output, iFlag_netcdf, tile_mode,
                                              zone_level, tiles_per_base, n_cells):
    """
    Internal function to create rHEALPix mesh with advanced tiling.

    Returns dict with 'filenames', 'tile_structure', 'metadata'.
    """
    sFilename_base = Path(sFilename_mesh).stem
    sExtension = Path(sFilename_mesh).suffix.lower()

    # Create tile structure based on mode
    if tile_mode == 'zone':
        if zone_level is None:
            zone_level = max(0, iLevel - 2)  # Default: 2 levels coarser

        print(f"Creating zone-based tiles (zone_level={zone_level})...")
        tile_structure = RHEALPixZoneBasedTileStructure(iLevel, zone_level)

        # Group cells by zone
        zone_cells = {}
        for cell_id in range(n_cells):
            zone_id = tile_structure.get_zone_for_cell_index(cell_id)
            if zone_id not in zone_cells:
                zone_cells[zone_id] = []
            zone_cells[zone_id].append(cell_id)

        print(f"Cells distributed across {len(zone_cells)} zones")

        # Create tile files
        filenames = []
        for zone_id, cell_list in zone_cells.items():
            sFilename_zone = f"{sFilename_base}_zone_{zone_id:05d}{sExtension}"
            sFilename_zone_full = os.path.join(sWorkspace_output, sFilename_zone)

            print(f"  Zone {zone_id}: {len(cell_list):,} cells -> {sFilename_zone}")
            _write_tile_from_cell_list(dggrs, cell_list, sFilename_zone_full, iFlag_netcdf)
            filenames.append(sFilename_zone)

        return {
            'filenames': filenames,
            'tile_structure': tile_structure,
            'metadata': {
                'level': iLevel,
                'n_cells': n_cells,
                'tile_mode': 'zone',
                'zone_level': zone_level,
                'n_zones': len(zone_cells)
            }
        }

    else:  # 'base' or 'hierarchical'
        print(f"Creating {tile_mode} tile structure...")
        tile_structure = RHEALPixTileStructure(iLevel, tile_scheme=tile_mode,
                                              tiles_per_base=tiles_per_base)

        # Group cells by tile
        tile_cells = {}
        for cell_id in range(n_cells):
            tile_id = tile_structure.get_tile_for_cell_index(cell_id)
            if tile_id not in tile_cells:
                tile_cells[tile_id] = []
            tile_cells[tile_id].append(cell_id)

        print(f"Cells distributed across {len(tile_cells)} tiles")

        # Create tile files
        filenames = []
        for tile_id, cell_list in tile_cells.items():
            sFilename_tile = f"{sFilename_base}_tile_{tile_id:04d}{sExtension}"
            sFilename_tile_full = os.path.join(sWorkspace_output, sFilename_tile)

            print(f"  Tile {tile_id}: {len(cell_list):,} cells -> {sFilename_tile}")
            _write_tile_from_cell_list(dggrs, cell_list, sFilename_tile_full, iFlag_netcdf)
            filenames.append(sFilename_tile)

        # Save tile metadata
        metadata_file = os.path.join(sWorkspace_output, f"{sFilename_base}_tile_metadata.json")
        _save_tile_metadata(tile_structure, metadata_file)

        return {
            'filenames': filenames,
            'tile_structure': tile_structure,
            'metadata': {
                'level': iLevel,
                'n_cells': n_cells,
                'tile_mode': tile_mode,
                'n_tiles': len(tile_cells),
                'metadata_file': metadata_file
            }
        }


def _write_tile_file(dggrs, filename, iFlag_netcdf, start_idx, end_idx, tile_index, total_tiles):
    """Write a single tile file from a range of cell indices."""
    if os.path.exists(filename):
        os.remove(filename)

    n_cells = end_idx - start_idx

    if iFlag_netcdf == 1:
        import netCDF4 as nc
        pDataset = nc.Dataset(filename, 'w', format='NETCDF4')
        pDataset.createDimension('cell', n_cells)
        pDataset.createVariable('cellid', 'i4', ('cell',))
        pDataset.createVariable('lon', 'f4', ('cell',))
        pDataset.createVariable('lat', 'f4', ('cell',))

        # Add tile metadata
        pDataset.tile_index = tile_index
        pDataset.total_tiles = total_tiles
        pDataset.global_cell_start = start_idx
        pDataset.global_cell_end = end_idx

        # Write data
        for i, cell_id in enumerate(range(start_idx, end_idx)):
            zone = dggrs.getZoneFromTextID(cell_id)
            if zone == nullZone:
                print(f"Error: Invalid cell ID '{cell_id}'")
                return None
            # Get WGS84 vertices (lon/lat coordinates)
            # The second parameter (0) is the refinement level
            vertices = dggrs.getZoneRefinedWGS84Vertices(zone, 0)
            centroid = dggrs.getZoneWGS84Centroid(zone)
            cell_geom = ogr.Geometry(ogr.wkbPolygon)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for vertex in vertices:
                lon, lat = vertex
                ring.AddPoint(lon, lat)
            ring.CloseRings()
            cell_geom.AddGeometry(ring)

            pDataset.variables['cellid'][i] = cell_id
            pDataset.variables['lon'][i] = centroid.lon
            pDataset.variables['lat'][i] = centroid.lat

        pDataset.close()
    else:
        # Vector format
        pDriver = get_vector_driver_from_filename(filename)
        pDataset = pDriver.CreateDataSource(filename)

        pSRS = osr.SpatialReference()
        pSRS.ImportFromEPSG(4326)
        pLayer = pDataset.CreateLayer('rhealpix_mesh', srs=pSRS, geom_type=ogr.wkbPolygon)
        pLayer.CreateField(ogr.FieldDefn('cellid', ogr.OFTInteger))
        pLayer.CreateField(ogr.FieldDefn('lon', ogr.OFTReal))
        pLayer.CreateField(ogr.FieldDefn('lat', ogr.OFTReal))
        pLayer.CreateField(ogr.FieldDefn('tile_index', ogr.OFTInteger))

        for cell_id in range(start_idx, end_idx):
            cell_geom = dggrs.cell_to_geometry(cell_id)
            centroid = cell_geom.Centroid()

            feature = ogr.Feature(pLayer.GetLayerDefn())
            feature.SetField('cellid', int(cell_id))
            feature.SetField('lon', float(centroid.GetX()))
            feature.SetField('lat', float(centroid.GetY()))
            feature.SetField('tile_index', tile_index)
            feature.SetGeometry(cell_geom)
            pLayer.CreateFeature(feature)
            feature = None

        pDataset.FlushCache()
        pDataset = None


def _write_tile_from_cell_list(dggrs, cell_list, filename, iFlag_netcdf):
    """Write a tile file from a list of cell IDs."""
    if os.path.exists(filename):
        os.remove(filename)

    n_cells = len(cell_list)

    if iFlag_netcdf == 1:
        import netCDF4 as nc
        pDataset = nc.Dataset(filename, 'w', format='NETCDF4')
        pDataset.createDimension('cell', n_cells)
        pDataset.createVariable('cellid', 'i4', ('cell',))
        pDataset.createVariable('lon', 'f4', ('cell',))
        pDataset.createVariable('lat', 'f4', ('cell',))

        for i, cell_id in enumerate(cell_list):
            #cell_geom = dggrs.cell_to_geometry(cell_id)
            cell_geom = dggrs.getZoneWGS84Vertices(cell_id)
            centroid = cell_geom.Centroid()

            pDataset.variables['cellid'][i] = cell_id
            pDataset.variables['lon'][i] = centroid.GetX()
            pDataset.variables['lat'][i] = centroid.GetY()

        pDataset.close()
    else:
        # Vector format
        pDriver = get_vector_driver_from_filename(filename)
        pDataset = pDriver.CreateDataSource(filename)

        pSRS = osr.SpatialReference()
        pSRS.ImportFromEPSG(4326)
        pLayer = pDataset.CreateLayer('rhealpix_mesh', srs=pSRS, geom_type=ogr.wkbPolygon)
        pLayer.CreateField(ogr.FieldDefn('cellid', ogr.OFTInteger))
        pLayer.CreateField(ogr.FieldDefn('lon', ogr.OFTReal))
        pLayer.CreateField(ogr.FieldDefn('lat', ogr.OFTReal))

        for cell_id in cell_list:
            zone = dggrs.getZoneFromTextID(str(cell_id))
            if zone == nullZone:
                print(f"Error: Invalid cell ID '{cell_id}'")
                return None
            # Get WGS84 vertices (lon/lat coordinates)
            # The second parameter (0) is the refinement level
            vertices = dggrs.getZoneRefinedWGS84Vertices(zone, 0)
            centroid = dggrs.getZoneWGS84Centroid(zone)
            cell_geom = ogr.Geometry(ogr.wkbPolygon)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for vertex in vertices:
                lon, lat = vertex
                ring.AddPoint(lon, lat)
            ring.CloseRings()
            cell_geom.AddGeometry(ring)

            feature = ogr.Feature(pLayer.GetLayerDefn())
            feature.SetField('cellid', int(cell_id))
            feature.SetField('lon', float(centroid.lon))
            feature.SetField('lat', float(centroid.lat))
            feature.SetGeometry(cell_geom)
            pLayer.CreateFeature(feature)
            feature = None

        pDataset.FlushCache()
        pDataset = None


def _save_tile_metadata(tile_structure, filename):
    """Save tile metadata to JSON file."""
    metadata_json = {
        'structure_info': {
            'level': tile_structure.level,
            'aperture': tile_structure.aperture,
            'n_cells': tile_structure.n_cells_per_level,
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
