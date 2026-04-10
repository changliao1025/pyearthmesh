import os
import json
from pathlib import Path
from abc import ABC, abstractmethod


class BaseMeshConfig(ABC):
    """Abstract base class for mesh-specific configurations"""

    @abstractmethod
    def get_default_config(self):
        """Returns a dictionary with default configuration values for this mesh type"""
        pass

    @abstractmethod
    def validate_config(self, config):
        """Validates the configuration for this mesh type"""
        pass

    def get_mesh_type(self):
        """Returns the mesh type identifier"""
        return self.__class__.__name__.replace('Config', '').lower()

    @staticmethod
    def _sort_config_for_output(config):
        """Sort a config dict by data type then alphabetically within each type group.

        The ordering of type groups is:
        1. bool   – flag parameters (iFlag_*, etc.)
        2. int    – integer parameters
        3. float  – floating-point parameters
        4. str    – string parameters
        5. list   – list parameters
        6. None   – parameters whose value is None (null in JSON)

        Within each group keys are sorted alphabetically (case-insensitive).

        Args:
            config (dict): The configuration dictionary to sort.

        Returns:
            dict: A new ordered dictionary with keys sorted as described.
        """
        def _type_rank(value):
            # bool must come before int because bool is a subclass of int
            if isinstance(value, bool):
                return 0
            if isinstance(value, int):
                return 1
            if isinstance(value, float):
                return 2
            if isinstance(value, str):
                return 3
            if isinstance(value, list):
                return 4
            return 5  # None and anything else

        return dict(
            sorted(
                config.items(),
                key=lambda kv: (_type_rank(kv[1]), kv[0].lower()),
            )
        )

    def create_template_config(self, output_filename, custom_values=None):
        """Creates a configuration file with default values, optionally customized.

        This base implementation merges defaults with any user-provided ``custom_values``
        and writes the result as a JSON file.  Subclasses may override this method to
        add mesh-specific logic while still calling ``super().create_template_config()``
        if desired.

        The keys in the output JSON are sorted by data type (bool → int → float →
        str → list → null) and then alphabetically within each type group.

        Args:
            output_filename (str or Path): Path to save the configuration file.
            custom_values (dict, optional): Key/value pairs that override defaults.

        Returns:
            dict: The final merged configuration that was written to disk.
        """
        config = self.get_default_config()
        if custom_values:
            config.update(custom_values)
        self.validate_config(config)
        config = self._sort_config_for_output(config)
        output_path = Path(output_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_filename, "w") as f:
            json.dump(config, f, indent=4)
        return config


class JigsawConfig(BaseMeshConfig):
    """Configuration manager for JIGSAW mesh generation"""

    def get_default_config(self):
        """Returns a dictionary with all default JIGSAW configuration values"""
        return {
            # Grid resolution parameters
            "ncolumn_space": 360,  # Number of columns in spacing grid (longitude)
            "nrow_space": 180,  # Number of rows in spacing grid (latitude)
            "dSpac_value": 100.0,  # Default spacing value
            # Feature flags for geometry generation
            "iFlag_geom": False,  # Enable geometry generation
            "iFlag_spac": False,  # Enable spacing function generation
            "iFlag_init": False,  # Enable initialization mesh
            "iFlag_opts": False,  # Enable custom options
            # Environment type flags
            "iFlag_spac_ocean": False,  # Apply ocean-specific spacing
            "iFlag_spac_land": True,  # Apply land-specific spacing
            # Point feature geometry and spacing flags
            "iFlag_geom_dam": False,  # Include dam geometries
            "iFlag_spac_dam": False,  # Apply dam-specific spacing
            "iFlag_geom_city": False,  # Include city geometries
            "iFlag_spac_city": False,  # Apply city-specific spacing
            # Line feature geometry and spacing flags
            "iFlag_geom_river_network": False,  # Include river network geometries
            "iFlag_spac_river_network": False,  # Apply river network-specific spacing
            "iFlag_geom_coastline": False,  # Include coastline geometries
            "iFlag_spac_coastline": False,  # Apply coastline-specific spacing
            # Polygon feature geometry and spacing flags
            "iFlag_geom_watershed_boundary": False,  # Include watershed boundary geometries
            "iFlag_spac_watershed_boundary": False,  # Apply watershed boundary-specific spacing
            "iFlag_geom_lake_boundary": False,  # Include lake boundary geometries
            "iFlag_spac_lake_boundary": False,  # Apply lake boundary-specific spacing
            # Resolution parameters for different features (in degrees)
            "dResolution_land": 45.0,  # Land feature spacing
            "dResolution_dam": 4.0,  # Dam feature spacing
            "dResolution_city": 4.0,  # City feature spacing
            "dResolution_river_network": 4.0,  # River network feature spacing
            "dResolution_coastline": 4.0,  # Coastline feature spacing
            "dResolution_watershed_boundary": 4.0,  # Watershed boundary feature spacing
            "dResolution_lake_boundary": 4.0,  # Lake boundary feature spacing
            # Mesh type identifiers
            "geom_mshID": "ellipsoid-mesh",  # Geometry mesh type
            "spac_mshID": "ellipsoid-grid",  # Spacing grid type
            # Earth/sphere parameters
            "FULL_SPHERE_RADIUS": 6371.0,  # Earth radius in km
            # Gradient limiting
            "dhdx_lim": 0.25,  # Gradient limit for mesh sizing
            # File paths for features
            "sFilename_dam_vector": None,  # Dam vector file
            "sFilename_dam_raster": None,  # Dam raster file
            "sFilename_city_vector": None,  # City vector file
            "sFilename_city_raster": None,  # City raster file
            "sFilename_river_network_vector": None,  # River network vector file
            "sFilename_river_network_raster": None,  # River network raster file
            "sFilename_coastline_vector": None,  # Coastline vector file
            "sFilename_coastline_raster": None,  # Coastline raster file
            "sFilename_watershed_boundary_vector": None,  # Watershed boundary vector file
            "sFilename_watershed_boundary_raster": None,  # Watershed boundary raster file
            "sFilename_lake_boundary_vector": None,  # Lake boundary vector file
            "sFilename_lake_boundary_raster": None,  # Lake boundary raster file
            # Mesh sizing parameters
            "hfun_hmax": "inf",  # Max. refinement function value
            "hfun_hmin": 0.0,  # Min. refinement function value
            "hfun_scal": "absolute",  # Scaling type: "relative" or "absolute"
            "mesh_dims": 2,  # Mesh dimension (2 for surface)
            "bisection": -1,  # Bisection method (-1 for heuristic)
            # Optimization parameters
            "optm_qlim": 0.95,  # Quality limit for optimization
            "optm_iter": 32,  # Number of optimization iterations
            "optm_qtol": 1.0e-05,  # Quality tolerance
            # Core mesh sizing and quality parameters
            "mesh_rad2": 1.5,  # Max. radius-edge ratio
            "mesh_rad3": 2.0,  # Max. radius-circumsphere ratio for tetras
            "mesh_eps1": 0.333,  # Min. mesh quality threshold
            "mesh_eps2": 0.333,  # Min. mesh quality threshold for tetra
            "mesh_top": 1,  # Mesh topology (1 for manifold surface)
            "mesh_iter": 3,  # Mesh iteration limit
            # Verbosity and iterations
            "verbosity": 0,  # Verbosity level (0-3)
            # File paths (will be populated based on workspace)
            "geom_file": None,  # Input geometry file
            "hfun_file": None,  # Input mesh-size file
            "mesh_file": None,  # Output mesh file
            # Algorithm selection
            "mesh_kern": "delfront",  # Meshing kernel: "delfront" or "delaunay"
            "optm_kern": "odt+dqdx",  # Optimisation kernel
            # Region boundary
            "geom_feat": True,  # Detect sharp features in geometry
            # Output options
            "mesh_type": "euclidean-mesh",  # Mesh type (euclidean-mesh or ellipsoid-mesh)
            "output_formats": ["vtk", "gmsh"],  # Output formats to generate
        }

    def validate_config(self, config):
        """Validates JIGSAW configuration"""
        required_keys = ["ncolumn_space", "nrow_space", "dSpac_value"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required JIGSAW configuration key: {key}")
        return True


class MPASConfig(BaseMeshConfig):
    """Configuration manager for MPAS mesh generation"""

    def get_default_config(self):
        """Returns a dictionary with default MPAS configuration values"""
        return {
            # MPAS-specific parameters
            "sFilename_mpas_mesh_netcdf": None,  # MPAS mesh NetCDF file
            "sFilename_jigsaw_mesh_netcdf": None,  # JIGSAW mesh NetCDF file
            "sFilename_land_ocean_mask": None,  # Land-ocean mask file
            "pBoundary_wkt": None,  # Boundary in WKT format
            "iFlag_run_jigsaw": 1,  # Flag to run JIGSAW for mesh generation
            # Common mesh parameters
            "iFlag_global": 1,  # Global mesh flag
            "iFlag_save_mesh": 1,  # Save mesh flag
            "mesh_type": "mpas",  # Mesh type identifier
        }

    def validate_config(self, config):
        """Validates MPAS configuration"""
        # MPAS can work with minimal configuration
        return True

    def create_template_config(self, output_filename, custom_values=None):
        """Creates a configuration file for MPAS, merging JIGSAW defaults when
        ``iFlag_run_jigsaw`` is truthy (either from defaults or ``custom_values``).

        The merge order is:
        1. JIGSAW defaults (base layer, only when jigsaw is enabled)
        2. MPAS defaults (override JIGSAW keys where they conflict)
        3. ``custom_values`` supplied by the caller (highest priority)

        The ``mesh_type`` key is always forced to ``"mpas"`` so that the
        resulting file is unambiguously identified as an MPAS configuration.

        Keys in the output JSON are sorted by data type (bool → int → float →
        str → list → null) then alphabetically within each type group.

        Args:
            output_filename (str or Path): Path to save the configuration file.
            custom_values (dict, optional): Key/value pairs that override defaults.

        Returns:
            dict: The final merged configuration that was written to disk.
        """
        mpas_defaults = self.get_default_config()

        # Determine whether jigsaw should be included.  Check custom_values
        # first (caller intent), then fall back to the MPAS default (which is 1).
        if custom_values and "iFlag_run_jigsaw" in custom_values:
            run_jigsaw = bool(custom_values["iFlag_run_jigsaw"])
        else:
            run_jigsaw = bool(mpas_defaults.get("iFlag_run_jigsaw", 0))

        if run_jigsaw:
            # Start with JIGSAW defaults as the base layer
            jigsaw_defaults = JigsawConfig().get_default_config()
            # Remove the jigsaw-specific mesh_type so MPAS value wins
            jigsaw_defaults.pop("mesh_type", None)
            config = jigsaw_defaults
            # MPAS defaults take precedence over JIGSAW defaults
            config.update(mpas_defaults)
        else:
            config = mpas_defaults.copy()

        # Caller-supplied values have the highest priority
        if custom_values:
            config.update(custom_values)

        # Always ensure mesh_type is "mpas"
        config["mesh_type"] = "mpas"

        self.validate_config(config)
        config = self._sort_config_for_output(config)

        output_path = Path(output_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_filename, "w") as f:
            json.dump(config, f, indent=4)

        return config


class HealpixConfig(BaseMeshConfig):
    """Configuration manager for HEALPix mesh generation.

    HEALPix meshes are generated directly via the healpy library and do not
    use JIGSAW.  The config keys here correspond to the parameters accepted by
    ``create_healpix_mesh``.
    """

    def get_default_config(self):
        """Returns a dictionary with default HEALPix configuration values"""
        return {
            # HEALPix resolution — passed as dResolution_meter_in
            "dResolution_meter": 100000.0,  # Target resolution in metres
            # Output file
            "sFilename_mesh": "healpix_mesh.geojson",  # Output mesh filename
            # Tiling options
            "iFlag_netcdf": 0,       # 0 = vector format, 1 = NetCDF
            "iFlag_use_tiles": 0,    # 0 = single file, 1 = tiled output
            "nTile": None,           # Number of tiles (None = auto)
            "tile_mode": "simple",   # 'simple', 'base', 'hierarchical', 'zone'
            "zone_level": None,      # Zone level for zone-based tiling
            "tiles_per_base": None,  # Tiles per base pixel (hierarchical mode)
            # Boundary
            "pBoundary_wkt": None,   # Boundary polygon in WKT (None = global)
            # Common flags
            "iFlag_global": 1,       # Global mesh flag
            "mesh_type": "healpix",  # Mesh type identifier
        }

    def validate_config(self, config):
        """Validates HEALPix configuration"""
        if "dResolution_meter" in config and config["dResolution_meter"] is not None:
            if config["dResolution_meter"] <= 0:
                raise ValueError(
                    f"dResolution_meter must be positive, got {config['dResolution_meter']}"
                )
        return True


class ISEAConfig(BaseMeshConfig):
    """Configuration manager for ISEA mesh generation.

    ISEA meshes are generated via the dggal library and do not use JIGSAW.
    The config keys here correspond to the parameters accepted by
    ``create_isea_mesh``.
    """

    def get_default_config(self):
        """Returns a dictionary with default ISEA configuration values"""
        return {
            # ISEA resolution — passed as dResolution_meter_in
            "dResolution_meter": 100000.0,  # Target resolution in metres
            # ISEA grid type: 'ISEA3H' (aperture 3) or 'ISEA4H' (aperture 4)
            "sISEA_type": "ISEA3H",
            # Output file
            "sFilename_mesh": "isea_mesh.geojson",  # Output mesh filename
            # Tiling options
            "iFlag_netcdf": 0,       # 0 = vector format, 1 = NetCDF
            "iFlag_use_tiles": 0,    # 0 = single file, 1 = tiled output
            "nTile": None,           # Number of tiles (None = auto)
            "tile_mode": "simple",   # 'simple', 'base', 'hierarchical', 'zone'
            "zone_level": None,      # Zone level for zone-based tiling
            "tiles_per_base": None,  # Tiles per base triangle (hierarchical mode)
            # Boundary
            "pBoundary_wkt": None,   # Boundary polygon in WKT (None = global)
            # Common flags
            "iFlag_global": 1,       # Global mesh flag
            "mesh_type": "isea",     # Mesh type identifier
        }

    def validate_config(self, config):
        """Validates ISEA configuration"""
        if "sISEA_type" in config and config["sISEA_type"] is not None:
            if config["sISEA_type"].upper() not in ("ISEA3H", "ISEA4H"):
                raise ValueError(
                    f"sISEA_type must be 'ISEA3H' or 'ISEA4H', got {config['sISEA_type']!r}"
                )
        return True


class LatLonConfig(BaseMeshConfig):
    """Configuration manager for Lat-Lon mesh generation.

    Lat-Lon meshes are generated directly via GDAL and do not use JIGSAW.
    The config keys here correspond to the parameters accepted by
    ``create_latlon_mesh``.
    """

    def get_default_config(self):
        """Returns a dictionary with default Lat-Lon configuration values"""
        return {
            # Grid resolution
            "dResolution_degree": 1.0,   # Cell resolution in degrees
            # Domain extent
            "dLongitude_left": -180.0,   # Left boundary longitude
            "dLongitude_right": 180.0,   # Right boundary longitude
            "dLatitude_bot": -90.0,      # Bottom boundary latitude
            "dLatitude_top": 90.0,       # Top boundary latitude
            # Derived grid dimensions (computed from extent / resolution)
            "ncolumn": 360,              # Number of columns
            "nrow": 180,                 # Number of rows
            # Output file
            "sFilename_mesh": "latlon_mesh.geojson",  # Output mesh filename
            # Boundary
            "pBoundary_wkt": None,       # Boundary polygon in WKT (None = auto bbox)
            # Common flags
            "iFlag_global": 1,           # Global mesh flag
            "mesh_type": "latlon",       # Mesh type identifier
        }

    def validate_config(self, config):
        """Validates Lat-Lon configuration"""
        if "dLongitude_left" in config and "dLongitude_right" in config:
            if config["dLongitude_left"] >= config["dLongitude_right"]:
                raise ValueError("Longitude left must be less than longitude right")
        if "dLatitude_bot" in config and "dLatitude_top" in config:
            if config["dLatitude_bot"] >= config["dLatitude_top"]:
                raise ValueError("Latitude bottom must be less than latitude top")
        return True

    # No create_template_config override needed — base class implementation is sufficient.
    # LatLon meshes do not use JIGSAW.


class SquareConfig(BaseMeshConfig):
    """Configuration manager for Square (regular grid) mesh generation.

    Square meshes are generated via GDAL and use a projected coordinate system.
    The config keys correspond to the parameters accepted by ``create_square_mesh``.
    """

    def get_default_config(self):
        """Returns a dictionary with default Square mesh configuration values"""
        return {
            # Grid origin (projected coordinates, metres)
            "dX_left": -180.0,           # Left boundary X coordinate (degrees or metres)
            "dY_bot": -90.0,             # Bottom boundary Y coordinate (degrees or metres)
            # Cell resolution
            "dResolution_meter": 100000.0,  # Cell size in metres
            # Grid dimensions
            "ncolumn": 360,              # Number of columns
            "nrow": 180,                 # Number of rows
            # Output file
            "sFilename_mesh": "square_mesh.geojson",  # Output mesh filename
            # Projection reference (WKT string or EPSG code)
            "pProjection_reference": None,  # Projection WKT (None = WGS84)
            # Boundary
            "pBoundary_wkt": None,       # Boundary polygon in WKT (None = bounding box)
            # Common flags
            "iFlag_global": 1,           # Global mesh flag
            "mesh_type": "square",       # Mesh type identifier
        }

    def validate_config(self, config):
        """Validates Square mesh configuration"""
        if "dResolution_meter" in config and config["dResolution_meter"] is not None:
            if config["dResolution_meter"] <= 0:
                raise ValueError(
                    f"dResolution_meter must be positive, got {config['dResolution_meter']}"
                )
        if "ncolumn" in config and config["ncolumn"] is not None:
            if config["ncolumn"] <= 0:
                raise ValueError(f"ncolumn must be a positive integer, got {config['ncolumn']}")
        if "nrow" in config and config["nrow"] is not None:
            if config["nrow"] <= 0:
                raise ValueError(f"nrow must be a positive integer, got {config['nrow']}")
        return True


class HexagonConfig(BaseMeshConfig):
    """Configuration manager for Hexagon mesh generation.

    Hexagon meshes are generated via GDAL and use a projected coordinate system.
    The config keys correspond to the parameters accepted by ``create_hexagon_mesh``.
    """

    def get_default_config(self):
        """Returns a dictionary with default Hexagon mesh configuration values"""
        return {
            # Rotation flag: 0 = flat-top, 1 = pointy-top
            "iFlag_rotation": 0,         # Hexagon orientation flag
            # Grid origin (projected coordinates, metres)
            "dX_left": -180.0,           # Left boundary X coordinate (degrees or metres)
            "dY_bot": -90.0,             # Bottom boundary Y coordinate (degrees or metres)
            # Cell resolution
            "dResolution_meter": 100000.0,  # Cell size in metres (inscribed circle radius)
            # Grid dimensions
            "ncolumn": 360,              # Number of columns
            "nrow": 180,                 # Number of rows
            # Output file
            "sFilename_mesh": "hexagon_mesh.geojson",  # Output mesh filename
            # Projection reference (WKT string or EPSG code)
            "pProjection_reference": None,  # Projection WKT (None = WGS84)
            # Boundary
            "pBoundary_wkt": None,       # Boundary polygon in WKT (None = bounding box)
            # Common flags
            "iFlag_global": 1,           # Global mesh flag
            "mesh_type": "hexagon",      # Mesh type identifier
        }

    def validate_config(self, config):
        """Validates Hexagon mesh configuration"""
        if "iFlag_rotation" in config and config["iFlag_rotation"] is not None:
            if config["iFlag_rotation"] not in (0, 1):
                raise ValueError(
                    f"iFlag_rotation must be 0 (flat-top) or 1 (pointy-top), "
                    f"got {config['iFlag_rotation']}"
                )
        if "dResolution_meter" in config and config["dResolution_meter"] is not None:
            if config["dResolution_meter"] <= 0:
                raise ValueError(
                    f"dResolution_meter must be positive, got {config['dResolution_meter']}"
                )
        return True


class TriangularConfig(BaseMeshConfig):
    """Configuration manager for Triangular mesh generation.

    Triangular meshes are generated via GDAL and use a projected coordinate system.
    The config keys correspond to the parameters accepted by ``create_triangular_mesh``.
    """

    def get_default_config(self):
        """Returns a dictionary with default Triangular mesh configuration values"""
        return {
            # Grid origin (projected coordinates, metres)
            "dX_left": -180.0,           # Left boundary X coordinate (degrees or metres)
            "dY_bot": -90.0,             # Bottom boundary Y coordinate (degrees or metres)
            # Cell resolution
            "dResolution_meter": 100000.0,  # Cell size in metres (equivalent area)
            # Grid dimensions
            "ncolumn": 360,              # Number of columns
            "nrow": 180,                 # Number of rows
            # Output file
            "sFilename_mesh": "triangular_mesh.geojson",  # Output mesh filename
            # Projection reference (WKT string or EPSG code)
            "pProjection_reference": None,  # Projection WKT (None = WGS84)
            # Boundary
            "pBoundary_wkt": None,       # Boundary polygon in WKT (None = bounding box)
            # Common flags
            "iFlag_global": 1,           # Global mesh flag
            "mesh_type": "triangular",   # Mesh type identifier
        }

    def validate_config(self, config):
        """Validates Triangular mesh configuration"""
        if "dResolution_meter" in config and config["dResolution_meter"] is not None:
            if config["dResolution_meter"] <= 0:
                raise ValueError(
                    f"dResolution_meter must be positive, got {config['dResolution_meter']}"
                )
        if "ncolumn" in config and config["ncolumn"] is not None:
            if config["ncolumn"] <= 0:
                raise ValueError(f"ncolumn must be a positive integer, got {config['ncolumn']}")
        if "nrow" in config and config["nrow"] is not None:
            if config["nrow"] <= 0:
                raise ValueError(f"nrow must be a positive integer, got {config['nrow']}")
        return True


class CubicSphereConfig(BaseMeshConfig):
    """Configuration manager for Cubic Sphere mesh generation.

    Cubic sphere meshes are generated via the ``GenerateCSMesh`` external tool.
    The config keys correspond to the parameters accepted by ``create_cubicsphere_mesh``.
    """

    def get_default_config(self):
        """Returns a dictionary with default Cubic Sphere mesh configuration values"""
        return {
            # Resolution in metres (converted to km internally)
            "dResolution_meter": 100000.0,  # Target cell resolution in metres
            # Output file
            "sFilename_mesh": "cubicsphere_mesh.geojson",  # Output mesh filename
            # Workspace for intermediate files (GenerateCSMesh output)
            "sWorkspace_output": None,   # Working directory for mesh generation
            # Boundary
            "pBoundary_wkt": None,       # Boundary polygon in WKT (None = global)
            # Common flags
            "iFlag_global": 1,           # Global mesh flag
            "iFlag_save_mesh": 1,        # Save mesh to file flag
            "mesh_type": "cubicsphere",  # Mesh type identifier
        }

    def validate_config(self, config):
        """Validates Cubic Sphere mesh configuration"""
        if "dResolution_meter" in config and config["dResolution_meter"] is not None:
            if config["dResolution_meter"] <= 0:
                raise ValueError(
                    f"dResolution_meter must be positive, got {config['dResolution_meter']}"
                )
        return True


class TINConfig(BaseMeshConfig):
    """Configuration manager for TIN (Triangulated Irregular Network) mesh generation.

    TIN meshes are generated via JIGSAW and share most configuration keys with
    ``JigsawConfig``.  The ``create_template_config`` override merges JIGSAW
    defaults as the base layer, then applies TIN-specific overrides on top.
    """

    def get_default_config(self):
        """Returns a dictionary with default TIN configuration values"""
        return {
            # TIN-specific parameters
            "sFilename_jigsaw_mesh_netcdf": None,  # JIGSAW mesh NetCDF file
            "pBoundary_wkt": None,                 # Boundary in WKT format
            "iFlag_run_jigsaw": 1,                 # Flag to run JIGSAW for mesh generation
            "iFlag_antarctic": 0,                  # Include Antarctic region
            "iFlag_arctic": 0,                     # Include Arctic region
            # Common mesh parameters
            "iFlag_global": 1,                     # Global mesh flag
            "iFlag_save_mesh": 1,                  # Save mesh flag
            "mesh_type": "tin",                    # Mesh type identifier
        }

    def validate_config(self, config):
        """Validates TIN configuration"""
        return True

    def create_template_config(self, output_filename, custom_values=None):
        """Creates a configuration file for TIN, merging JIGSAW defaults when
        ``iFlag_run_jigsaw`` is truthy.

        The merge order is:
        1. JIGSAW defaults (base layer, only when jigsaw is enabled)
        2. TIN defaults (override JIGSAW keys where they conflict)
        3. ``custom_values`` supplied by the caller (highest priority)

        Keys in the output JSON are sorted by data type (bool → int → float →
        str → list → null) then alphabetically within each type group.

        Args:
            output_filename (str or Path): Path to save the configuration file.
            custom_values (dict, optional): Key/value pairs that override defaults.

        Returns:
            dict: The final merged configuration that was written to disk.
        """
        tin_defaults = self.get_default_config()

        if custom_values and "iFlag_run_jigsaw" in custom_values:
            run_jigsaw = bool(custom_values["iFlag_run_jigsaw"])
        else:
            run_jigsaw = bool(tin_defaults.get("iFlag_run_jigsaw", 0))

        if run_jigsaw:
            jigsaw_defaults = JigsawConfig().get_default_config()
            jigsaw_defaults.pop("mesh_type", None)
            config = jigsaw_defaults
            config.update(tin_defaults)
        else:
            config = tin_defaults.copy()

        if custom_values:
            config.update(custom_values)

        config["mesh_type"] = "tin"

        self.validate_config(config)
        config = self._sort_config_for_output(config)

        output_path = Path(output_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_filename, "w") as f:
            json.dump(config, f, indent=4)

        return config


class DGGRIDConfig(BaseMeshConfig):
    """Configuration manager for DGGRID mesh generation.

    DGGRID meshes are generated via the external ``dggrid`` binary.
    The config keys correspond to the parameters accepted by ``create_dggrid_mesh``.
    """

    def get_default_config(self):
        """Returns a dictionary with default DGGRID mesh configuration values"""
        return {
            # DGGRID grid type: 'ISEA3H' (aperture 3) or 'ISEA4H' (aperture 4)
            "sDggrid_type": "ISEA3H",    # DGGRID grid type identifier
            # Resolution index (integer level; higher = finer)
            "iResolution_index": 9,      # Resolution level index
            # Maximum cells per output file (None = single file)
            "lMax_cells_per_output_file": None,
            # Output file
            "sFilename_mesh": "dggrid_mesh.geojson",  # Output mesh filename
            # Workspace for intermediate DGGRID files
            "sWorkspace_output": None,   # Working directory for mesh generation
            # Optional boundary shapefile (None = global)
            "sFilename_boundary": None,  # Boundary shapefile path
            # Region flags
            "iFlag_antarctic": 0,        # Include Antarctic region
            "iFlag_arctic": 0,           # Include Arctic region
            # Common flags
            "iFlag_global": 1,           # Global mesh flag
            "iFlag_save_mesh": 1,        # Save mesh to file flag
            "mesh_type": "dggrid",       # Mesh type identifier
        }

    def validate_config(self, config):
        """Validates DGGRID configuration"""
        if "sDggrid_type" in config and config["sDggrid_type"] is not None:
            if config["sDggrid_type"].upper() not in ("ISEA3H", "ISEA4H"):
                raise ValueError(
                    f"sDggrid_type must be 'ISEA3H' or 'ISEA4H', got {config['sDggrid_type']!r}"
                )
        if "iResolution_index" in config and config["iResolution_index"] is not None:
            if not isinstance(config["iResolution_index"], int) or config["iResolution_index"] < 0:
                raise ValueError(
                    f"iResolution_index must be a non-negative integer, "
                    f"got {config['iResolution_index']}"
                )
        return True


class RHEALPixConfig(BaseMeshConfig):
    """Configuration manager for rHEALPix mesh generation.

    rHEALPix meshes are generated via the dggal library (aperture 9).
    The config keys correspond to the parameters accepted by ``create_rhealpix_mesh``.
    """

    def get_default_config(self):
        """Returns a dictionary with default rHEALPix configuration values"""
        return {
            # rHEALPix resolution — passed as dResolution_meter_in
            "dResolution_meter": 100000.0,  # Target resolution in metres
            # Output file
            "sFilename_mesh": "rhealpix_mesh.geojson",  # Output mesh filename
            # Workspace for intermediate files
            "sWorkspace_output": None,   # Working directory for mesh generation
            # Tiling options
            "iFlag_netcdf": 0,           # 0 = vector format, 1 = NetCDF
            "iFlag_use_tiles": 0,        # 0 = single file, 1 = tiled output
            "nTile": None,               # Number of tiles (None = auto)
            "tile_mode": "simple",       # 'simple', 'base', 'hierarchical', 'zone'
            "zone_level": None,          # Zone level for zone-based tiling
            "tiles_per_base": None,      # Tiles per base pixel (hierarchical mode)
            # Boundary
            "pBoundary_wkt": None,       # Boundary polygon in WKT (None = global)
            # Common flags
            "iFlag_global": 1,           # Global mesh flag
            "mesh_type": "rhealpix",     # Mesh type identifier
        }

    def validate_config(self, config):
        """Validates rHEALPix configuration"""
        if "dResolution_meter" in config and config["dResolution_meter"] is not None:
            if config["dResolution_meter"] <= 0:
                raise ValueError(
                    f"dResolution_meter must be positive, got {config['dResolution_meter']}"
                )
        return True


class ConfigManager:
    """General configuration manager that delegates to specific mesh config classes"""

    # Registry of available mesh configurations
    _mesh_configs = {
        "jigsaw": JigsawConfig,
        "mpas": MPASConfig,
        "healpix": HealpixConfig,
        "isea": ISEAConfig,
        "latlon": LatLonConfig,
        "square": SquareConfig,
        "hexagon": HexagonConfig,
        "triangular": TriangularConfig,
        "cubicsphere": CubicSphereConfig,
        "tin": TINConfig,
        "dggrid": DGGRIDConfig,
        "rhealpix": RHEALPixConfig,
    }

    def __init__(self, mesh_type=None):
        """Initialize ConfigManager with optional mesh type

        Args:
            mesh_type (str, optional): Type of mesh configuration to use
        """
        self.mesh_type = mesh_type
        self._config_instance = None

        if mesh_type:
            self._config_instance = self._get_config_instance(mesh_type)

    @classmethod
    def register_mesh_config(cls, mesh_type, config_class):
        """Register a new mesh configuration class

        Args:
            mesh_type (str): Identifier for the mesh type
            config_class (BaseMeshConfig): Configuration class for the mesh type
        """
        if not issubclass(config_class, BaseMeshConfig):
            raise TypeError("config_class must be a subclass of BaseMeshConfig")
        cls._mesh_configs[mesh_type.lower()] = config_class

    @classmethod
    def get_available_mesh_types(cls):
        """Returns list of available mesh types"""
        return list(cls._mesh_configs.keys())

    def _get_config_instance(self, mesh_type):
        """Get configuration instance for specified mesh type"""
        mesh_type_lower = mesh_type.lower()
        if mesh_type_lower not in self._mesh_configs:
            raise ValueError(
                f"Unknown mesh type: {mesh_type}. "
                f"Available types: {', '.join(self.get_available_mesh_types())}"
            )
        return self._mesh_configs[mesh_type_lower]()

    def get_default_config(self, mesh_type=None):
        """Get default configuration for specified mesh type

        Args:
            mesh_type (str, optional): Type of mesh. Uses instance mesh_type if not provided.

        Returns:
            dict: Default configuration dictionary
        """
        if mesh_type is None:
            if self.mesh_type is None:
                raise ValueError("mesh_type must be specified")
            mesh_type = self.mesh_type

        config_instance = self._get_config_instance(mesh_type)
        config = config_instance.get_default_config()

        # Add common parameters
        config.update({
            "mesh_type": mesh_type,
            "iCase_index": 1,
            "iFlag_standalone": 1,
            "iFlag_global": config.get("iFlag_global", 1),
            "sDate": None,
            "sWorkspace_output": None,
            "sFilename_model_configuration": None,
        })

        return config

    def validate_config(self, config, mesh_type=None):
        """Validate configuration for specified mesh type

        Args:
            config (dict): Configuration dictionary to validate
            mesh_type (str, optional): Type of mesh. Uses instance mesh_type if not provided.

        Returns:
            bool: True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        if mesh_type is None:
            mesh_type = config.get("mesh_type", self.mesh_type)

        if mesh_type is None:
            raise ValueError("mesh_type must be specified in config or as parameter")

        config_instance = self._get_config_instance(mesh_type)
        return config_instance.validate_config(config)

    def create_template_config(self, output_filename, mesh_type=None, custom_values=None):
        """Create a configuration file with default values, optionally customized.

        Keys in the output JSON are sorted by data type (bool → int → float →
        str → list → null) then alphabetically within each type group.

        Args:
            output_filename (str or Path): Path to save the configuration file
            mesh_type (str, optional): Type of mesh. Uses instance mesh_type if not provided.
            custom_values (dict, optional): Dictionary of values to override defaults

        Returns:
            dict: The created configuration
        """
        if mesh_type is None:
            if self.mesh_type is None:
                raise ValueError("mesh_type must be specified")
            mesh_type = self.mesh_type

        # Get defaults for the specified mesh type
        config = self.get_default_config(mesh_type)

        # Apply customizations
        if custom_values:
            for key, value in custom_values.items():
                if isinstance(value, Path):
                    value = str(value)
                config[key] = value

        # Validate the configuration
        self.validate_config(config, mesh_type)

        # Sort by data type then alphabetically
        config = BaseMeshConfig._sort_config_for_output(config)

        # Ensure directory exists
        output_path = Path(output_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(output_filename, "w") as f:
            json.dump(config, f, indent=4)

        return config

    @staticmethod
    def load_config(filename):
        """Load a configuration from a JSON file

        Args:
            filename (str or Path): Path to the configuration file

        Returns:
            dict: The loaded configuration
        """
        with open(filename, "r") as f:
            config = json.load(f)

        # Validate if mesh_type is specified
        if "mesh_type" in config:
            manager = ConfigManager(config["mesh_type"])
            manager.validate_config(config)

        return config

    @staticmethod
    def save_config(config, filename):
        """Save a configuration to a JSON file

        Args:
            config (dict): Configuration dictionary to save
            filename (str or Path): Path to save the configuration file
        """
        # Validate if mesh_type is specified
        if "mesh_type" in config:
            manager = ConfigManager(config["mesh_type"])
            manager.validate_config(config)

        # Ensure directory exists
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(filename, "w") as f:
            json.dump(config, f, indent=4)



