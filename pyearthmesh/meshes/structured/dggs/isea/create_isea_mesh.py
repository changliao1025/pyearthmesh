import os, stat, platform
import numpy as np
from pathlib import Path
import subprocess
import datetime
from shutil import copy2
from osgeo import osr, ogr, gdal
from dggal import Application, DGGRS
import json
from typing import List, Dict, Tuple, Optional

from pyearth.system.define_global_variables import earth_radius
from pyearth.gis.location.get_geometry_coordinates import get_geometry_coordinates
from pyearth.gis.gdal.gdal_vector_format_support import get_vector_driver_from_filename
from pyearthmesh.utility.convert_coordinates import convert_gcs_coordinates_to_meshcell


from dggal import *
# Initialize the application
app = Application(appGlobals=globals())
pydggal_setup(app)


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


class ISEATileStructure:
    """
    Hierarchical tile structure based on ISEA tessellation.

    Uses ISEA's natural 20 base triangles as top-level tiles,
    with optional hierarchical subdivision. Supports aperture 3 and 4.
    """

    def __init__(self, level: int, aperture: int, tile_scheme: str = 'hierarchical',
                 tiles_per_base: Optional[int] = None):
        """
        Initialize ISEA tile structure.

        Parameters:
        -----------
        level : int
            ISEA refinement level
        aperture : int
            ISEA aperture (3 for ISEA3H, 4 for ISEA4H)
        tile_scheme : str
            'base' (20 tiles), 'hierarchical' (subdivided), 'adaptive'
        tiles_per_base : int, optional
            Number of tiles per base triangle for hierarchical scheme
        """
        self.level = level
        self.aperture = aperture  # 3 for ISEA3H, 4 for ISEA4H
        self.n_cells_per_level = 20 * (self.aperture ** level)
        self.base_tiles = 20  # ISEA has 20 base triangles (icosahedron)
        self.tile_scheme = tile_scheme

        if tile_scheme == 'base':
            self.tiles_per_base = 1
            self.total_tiles = 20
        elif tile_scheme == 'hierarchical':
            if tiles_per_base is None:
                self.tiles_per_base = self._calculate_optimal_tiles_per_base()
            else:
                self.tiles_per_base = tiles_per_base
            self.total_tiles = 20 * self.tiles_per_base
        else:
            self.tiles_per_base = tiles_per_base or 1
            self.total_tiles = 20 * self.tiles_per_base

        self.tile_metadata = {}
        self._build_tile_structure()

    def _calculate_optimal_tiles_per_base(self) -> int:
        """Calculate optimal tiles per base triangle (target ~75k cells/tile)."""
        target_cells_per_tile = 75000
        cells_per_base = self.n_cells_per_level // 20
        tiles_per_base = max(1, int(np.ceil(cells_per_base / target_cells_per_tile)))

        # Round to nearest power of aperture for clean subdivision
        if tiles_per_base > 1:
            power = int(np.ceil(np.log(tiles_per_base) / np.log(self.aperture)))
            tiles_per_base = self.aperture ** power

        return tiles_per_base

    def _build_tile_structure(self):
        """Build tile structure and metadata."""
        if self.tile_scheme == 'base':
            self._build_base_tile_structure()
        else:
            self._build_hierarchical_tile_structure()

    def _build_base_tile_structure(self):
        """Build 20 base tiles."""
        cells_per_base = self.n_cells_per_level // 20
        for base_idx in range(20):
            cell_start = base_idx * cells_per_base
            cell_end = (base_idx + 1) * cells_per_base

            self.tile_metadata[base_idx] = {
                'tile_id': base_idx,
                'base_triangle': base_idx,
                'cell_range': (cell_start, cell_end),
                'n_cells': cells_per_base
            }

    def _build_hierarchical_tile_structure(self):
        """Build hierarchical tile structure with subdivision."""
        cells_per_subtile = self.n_cells_per_level // (20 * self.tiles_per_base)

        tile_id = 0
        for base_idx in range(20):
            for sub_idx in range(self.tiles_per_base):
                cell_start = (base_idx * self.tiles_per_base + sub_idx) * cells_per_subtile
                cell_end = cell_start + cells_per_subtile

                self.tile_metadata[tile_id] = {
                    'tile_id': tile_id,
                    'base_triangle': base_idx,
                    'sub_index': sub_idx,
                    'cell_range': (cell_start, cell_end),
                    'n_cells': cells_per_subtile
                }
                tile_id += 1

    def get_tile_for_cell_index(self, cell_index: int) -> int:
        """Get tile ID for an ISEA cell based on its index in the sorted list."""
        if self.tile_scheme == 'base':
            return cell_index // (self.n_cells_per_level // 20)
        else:
            cells_per_tile = self.n_cells_per_level // self.total_tiles
            return cell_index // cells_per_tile

    def get_tile_cell_range(self, tile_id: int) -> Tuple[int, int]:
        """Get cell index range for a tile."""
        return self.tile_metadata[tile_id]['cell_range']


class ISEAZoneBasedTileStructure:
    """
    Zone-based tile structure where all cells in a zone form one tile.

    For mesh at level n, zones defined at level m (m < n).
    All level-n cells within each level-m cell form one tile.
    """

    def __init__(self, mesh_level: int, zone_level: int, aperture: int):
        """
        Initialize zone-based tile structure for ISEA.

        Parameters:
        -----------
        mesh_level : int
            ISEA level for mesh (fine resolution)
        zone_level : int
            ISEA level for zones (coarse resolution)
        aperture : int
            ISEA aperture (3 or 4)
        """
        if zone_level >= mesh_level:
            raise ValueError(f"zone_level ({zone_level}) must be < mesh_level ({mesh_level})")

        self.mesh_level = mesh_level
        self.zone_level = zone_level
        self.aperture = aperture
        self.n_zones = 20 * (self.aperture ** zone_level)
        self.n_mesh_cells = 20 * (self.aperture ** mesh_level)
        self.cells_per_zone = self.n_mesh_cells // self.n_zones

        print(f"Zone-based tiling: mesh_level={mesh_level}, zone_level={zone_level}, aperture={aperture}")
        print(f"  Total zones: {self.n_zones}, Cells per zone: {self.cells_per_zone:,}")

    def get_zone_for_cell_index(self, cell_index: int) -> int:
        """Get zone ID for a mesh cell based on its index."""
        return cell_index // self.cells_per_zone

    def get_cell_range_in_zone(self, zone_id: int) -> Tuple[int, int]:
        """Get cell index range in a zone."""
        cell_start = zone_id * self.cells_per_zone
        cell_end = (zone_id + 1) * self.cells_per_zone
        return (cell_start, cell_end)


def isea_find_level_by_resolution(sISEA_type, dResolution_meter):
    """
    Find the appropriate ISEA level for a given target resolution.

    For ISEA3H (aperture 3):
    - Level 0: ~4,759 km resolution
    - Each level increases cell count by 3 (aperture 3)
    - Cell area decreases by factor of 3, so edge length decreases by √3
    - Resolution at level n ≈ R₀ / √3ⁿ

    For ISEA4H (aperture 4):
    - Each level increases cell count by 4 (aperture 4)
    - Cell area decreases by factor of 4, so edge length decreases by 2
    - Resolution at level n ≈ R₀ / 2ⁿ

    Args:
        sISEA_type (str): Type of ISEA ('ISEA3H', 'ISEA4H')
        dResolution_meter (float): Target resolution in meters

    Returns:
        int: The appropriate refinement level
    """
    # Earth's radius in meters
    dRadius_earth = earth_radius

    if sISEA_type.lower() == 'isea3h':
        # Base resolution at level 0 (approximate)
        # For ISEA3H with 20 base triangles on icosahedron
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 20)
        # Aperture 3: edge length factor is sqrt(3)
        edge_factor = np.sqrt(3.0)
        level = np.log(dResolution_level0 / dResolution_meter) / np.log(edge_factor)
        iLevel = int(np.round(level))
        # Ensure non-negative level
        if iLevel < 0:
            iLevel = 0

    elif sISEA_type.lower() == 'isea4h':
        # Base resolution at level 0 (approximate)
        # For ISEA4H with 20 base triangles on icosahedron
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 20)
        # Aperture 4: edge length factor is 2
        edge_factor = 2.0
        level = np.log(dResolution_level0 / dResolution_meter) / np.log(edge_factor)
        iLevel = int(np.round(level))
        # Ensure non-negative level
        if iLevel < 0:
            iLevel = 0
    else:
        raise ValueError(f"Unknown ISEA type: {sISEA_type}")

    return iLevel

def isea_find_resolution_by_level(sISEA_type, iLevel):
    """
    Find the resolution for a given ISEA level.

    For ISEA3H (aperture 3):
    - Level 0: ~4,759 km resolution
    - Each level increases cell count by 3 (aperture 3)
    - Cell area decreases by factor of 3, so edge length decreases by √3
    - Resolution at level n ≈ R₀ / √3ⁿ

    For ISEA4H (aperture 4):
    - Each level increases cell count by 4 (aperture 4)
    - Cell area decreases by factor of 4, so edge length decreases by 2
    - Resolution at level n ≈ R₀ / 2ⁿ

    Args:
        sISEA_type (str): Type of ISEA ('ISEA3H', 'ISEA4H')
        iLevel (int): Refinement level

    Returns:
        float: The resolution in meters
    """
    # Earth's radius in meters
    dRadius_earth = earth_radius

    if sISEA_type.lower() == 'isea3h':
        # Base resolution at level 0 (approximate)
        # For ISEA3H with 20 base triangles on icosahedron
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 20)
        # Aperture 3: edge length factor is sqrt(3)
        edge_factor = np.sqrt(3.0)
        dResolution = dResolution_level0 / (edge_factor ** iLevel)
    elif sISEA_type.lower() == 'isea4h':
        # Base resolution at level 0 (approximate)
        # For ISEA4H with 20 base triangles on icosahedron
        dResolution_level0 = np.sqrt(4 * np.pi * dRadius_earth**2 / 20)
        # Aperture 4: edge length factor is 2
        edge_factor = 2.0
        dResolution = dResolution_level0 / (edge_factor ** iLevel)
    else:
        raise ValueError(f"Unknown ISEA type: {sISEA_type}")
    return dResolution

def create_isea_mesh(dResolution_meter_in,
                     sISEA_type,
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
    Create a spherical mesh based on the ISEA tessellation.

    Parameters
    ----------
    dResolution_meter_in : float
        The resolution parameter for the ISEA mesh.
    sISEA_type : str
        Type of ISEA grid ('ISEA3H' or 'ISEA4H')
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
        Tiling strategy: 'simple' (legacy), 'base' (20 tiles), 'hierarchical' (subdivided),
        'zone' (zone-based). Only used if iFlag_use_tiles=1, by default 'simple'.
    zone_level : int, optional
        Zone level for zone-based tiling (must be < mesh level).
        Only used if tile_mode='zone', by default None (auto-determined).
    tiles_per_base : int, optional
        Number of tiles per base triangle for hierarchical mode.
        Only used if tile_mode='hierarchical', by default None (auto-determined).

    Returns
    -------
    str or list or dict
        If single file: filename string
        If legacy tiling (tile_mode='simple'): list of filenames
        If advanced tiling: dict with 'filenames', 'tile_structure', 'metadata'
    """

    #find the level based on the resolution
    iLevel = isea_find_level_by_resolution(sISEA_type, dResolution_meter_in)
    try:
        dggrs_temp = DGGRS(sISEA_type, resolution=0)
        iLevel = dggrs_temp.getLevelFromMetersPerSubZone(dResolution_meter_in, 0)
    except RuntimeError as e:
        if 'null pointer' in str(e):
            print('Warning: dggal getLevelFromMetersPerSubZone unavailable; using analytical level estimate.')
        else:
            raise
    print("Determined ISEA level: " + str(iLevel))
    dResolution_actual = isea_find_resolution_by_level(sISEA_type, iLevel)
    print("Actual resolution at this level: " + str(dResolution_actual) + " meters )")

    # Determine aperture from ISEA type
    if sISEA_type.lower() == 'isea3h':
        aperture = 3
    elif sISEA_type.lower() == 'isea4h':
        aperture = 4
    else:
        raise ValueError(f"Unknown ISEA type: {sISEA_type}")

    # Initialize DGGRS for ISEA
    dggrs = DGGRS(sISEA_type, resolution=iLevel)

    # Get all cell IDs at this resolution
    # For ISEA, we generate cells programmatically
    n_cells = 20 * (aperture ** iLevel)
    print(f"Total cells to generate: {n_cells:,}")

    #get the file extension of the output file
    sFilename_mesh_full = os.path.join(sWorkspace_output, sFilename_mesh)
    sExtension = Path(sFilename_mesh_full).suffix.lower()
    if sExtension == '.nc':
        iFlag_netcdf = 1

    # Handle advanced tiling modes
    if iFlag_use_tiles == 1 and tile_mode in ['base', 'hierarchical', 'zone']:
        print(f"\nUsing advanced tiling mode: {tile_mode}")
        return _create_isea_mesh_with_advanced_tiles(
            iLevel, aperture, dggrs, sFilename_mesh, sWorkspace_output,
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
                    cell_geom = dggrs.cell_to_geometry(cell_id)
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
            pLayer_out = pDataset_out.CreateLayer('isea_mesh', srs=pSRS, geom_type=ogr.wkbPolygon)
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
                    # Get cell geometry from dggrs
                    cell_geom = dggrs.cell_to_geometry(cell_id)
                    centroid = cell_geom.Centroid()

                    # Create a feature for this cell
                    feature = ogr.Feature(pLayer_out.GetLayerDefn())
                    feature.SetField('cellid', int(cell_id))
                    feature.SetField('lon', float(centroid.GetX()))
                    feature.SetField('lat', float(centroid.GetY()))
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


def _create_isea_mesh_with_advanced_tiles(iLevel, aperture, dggrs, sFilename_mesh,
                                          sWorkspace_output, iFlag_netcdf, tile_mode,
                                          zone_level, tiles_per_base, n_cells):
    """
    Internal function to create ISEA mesh with advanced tiling.

    Returns dict with 'filenames', 'tile_structure', 'metadata'.
    """
    sFilename_base = Path(sFilename_mesh).stem
    sExtension = Path(sFilename_mesh).suffix.lower()

    # Create tile structure based on mode
    if tile_mode == 'zone':
        if zone_level is None:
            zone_level = max(0, iLevel - 2)  # Default: 2 levels coarser

        print(f"Creating zone-based tiles (zone_level={zone_level})...")
        tile_structure = ISEAZoneBasedTileStructure(iLevel, zone_level, aperture)

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
                'aperture': aperture,
                'n_cells': n_cells,
                'tile_mode': 'zone',
                'zone_level': zone_level,
                'n_zones': len(zone_cells)
            }
        }

    else:  # 'base' or 'hierarchical'
        print(f"Creating {tile_mode} tile structure...")
        tile_structure = ISEATileStructure(iLevel, aperture, tile_scheme=tile_mode,
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
                'aperture': aperture,
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
            cell_geom = dggrs.cell_to_geometry(cell_id)
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
        pLayer = pDataset.CreateLayer('isea_mesh', srs=pSRS, geom_type=ogr.wkbPolygon)
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
            cell_geom = dggrs.cell_to_geometry(cell_id)
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
        pLayer = pDataset.CreateLayer('isea_mesh', srs=pSRS, geom_type=ogr.wkbPolygon)
        pLayer.CreateField(ogr.FieldDefn('cellid', ogr.OFTInteger))
        pLayer.CreateField(ogr.FieldDefn('lon', ogr.OFTReal))
        pLayer.CreateField(ogr.FieldDefn('lat', ogr.OFTReal))

        for cell_id in cell_list:
            cell_geom = dggrs.cell_to_geometry(cell_id)
            centroid = cell_geom.Centroid()

            feature = ogr.Feature(pLayer.GetLayerDefn())
            feature.SetField('cellid', int(cell_id))
            feature.SetField('lon', float(centroid.GetX()))
            feature.SetField('lat', float(centroid.GetY()))
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
