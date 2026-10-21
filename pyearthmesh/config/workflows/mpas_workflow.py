"""MPAS mesh workflow.

Translates the single sectioned workflow config into the flat parameters that
:class:`~pyearthmesh.meshes.unstructured.mpas.mpascase.mpascase` already consumes,
builds the case object **in memory** (no intermediate config file), and delegates
``setup`` / ``run`` / HPC-job generation to it.
"""

import os
from typing import Any

from pyearthmesh.config.workflows.base import BaseWorkflow
from pyearthmesh.meshes.unstructured.mpas.mpascase import MPASCase, mpascase

__all__ = ["MPASWorkflow", "MPASCase", "mpascase"]


class MPASWorkflow(BaseWorkflow):
    """Workflow class for the ``mpas`` mesh type."""

    mesh_type = "mpas"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """MPAS-flavoured defaults for the sectioned workflow config."""
        return {
            "workflow": {
                "date": None,
                "mesh_type": "mpas",
                "case_index": 1,
                "model": "jigsaw",
                "simplify_hydrosheds_river_network": False,
                "process_coastline": False,
                "debug": True,
                "largest_outlets": 10,
                "run_on_hpc": False,
            },
            "resolutions": {
                "ocean_km": 100,
                "land_km": 12.5,
                "river_network_km": 5,
                "coastline_km": 10,
                "lake_boundary_km": 3,
                "watershed_boundary_km": 3,
            },
            "paths": {
                "input_workspace": "",
                "output_workspace": "./output",
                "hydrosheds_rivers": "",
                "region_geometry": "",
                "dam_vector": "",
            },
            "jigsaw": {
                "standalone": True,
                "create_directory": True,
                "hours": 5,
                "slurm": "slurm",
            },
        }

    @classmethod
    def translate_frontend_to_backend(cls, spec, native_config):
        """Compile a generic workflow spec into the native MPAS config dict."""
        if hasattr(spec, "to_dict"):
            spec_data = spec.to_dict()
        elif isinstance(spec, dict):
            spec_data = spec
        else:
            raise TypeError("spec must be a WorkflowSpec or dict")

        config = dict(native_config)

        case_index = spec_data.get("case_index", 1)
        date = spec_data.get("date")
        paths = spec_data.get("paths", {}) or {}
        region = spec_data.get("region", {}) or {}
        resolution = spec_data.get("resolution", {}) or {}
        execution = spec_data.get("execution", {}) or {}
        features = spec_data.get("features", {}) or {}

        config["mesh_type"] = "mpas"
        config["iCase_index"] = int(case_index)
        config["sDate"] = date
        config["verbosity"] = 1 if execution.get("debug") else 0

        model_name = str(execution.get("model", "jigsaw")).lower()
        config["sModel"] = model_name
        config["iFlag_run_jigsaw"] = 1 if model_name == "jigsaw" else 0

        coastline = bool(features.get("coastline", False))
        config["iFlag_geom_coastline"] = coastline
        config["iFlag_spac_coastline"] = coastline

        output_workspace = paths.get("output_workspace") or paths.get(
            "sWorkspace_output"
        )
        if output_workspace is not None:
            config["sWorkspace_output"] = output_workspace

        region_geometry = region.get("geometry") or paths.get("region_geometry")
        boundary_wkt = cls._boundary_wkt_from_geojson(region_geometry)
        config["pBoundary_wkt"] = boundary_wkt
        config["iFlag_global"] = 0 if boundary_wkt else 1

        # Resolution mapping (all JIGSAW / MPAS resolutions are in kilometres)
        def _get_res(
            key_km: str, key_native: str, default: float | None = None
        ) -> float | None:
            val = resolution.get(key_km)
            if val is None:
                val = resolution.get(key_native)
            if val is None:
                return default
            return float(val)

        ocean_res = _get_res("ocean_km", "dResolution_ocean", 100.0)
        config["dResolution_ocean"] = ocean_res
        config["dSpac_value"] = ocean_res

        land_res = _get_res("land_km", "dResolution_land", 45.0)
        config["dResolution_land"] = land_res

        river_res = _get_res("river_network_km", "dResolution_river_network", 4.0)
        config["dResolution_river_network"] = river_res

        coast_res = _get_res("coastline_km", "dResolution_coastline", 4.0)
        config["dResolution_coastline"] = coast_res

        lake_res = _get_res("lake_boundary_km", "dResolution_lake_boundary", 4.0)
        config["dResolution_lake_boundary"] = lake_res

        wshed_res = _get_res(
            "watershed_boundary_km", "dResolution_watershed_boundary", 4.0
        )
        config["dResolution_watershed_boundary"] = wshed_res

        dam_res = _get_res("dam_km", "dResolution_dam", None)
        if dam_res is not None:
            config["dResolution_dam"] = dam_res
        elif "dResolution_dam" not in config:
            config["dResolution_dam"] = 4.0

        city_res = _get_res("city_km", "dResolution_city", None)
        if city_res is not None:
            config["dResolution_city"] = city_res
        elif "dResolution_city" not in config:
            config["dResolution_city"] = 4.0

        # JIGSAW-specific flags and numerical controls use their native names so
        # the sectioned workflow can express the complete reference workflow.
        # Resolution keys are excluded here so user-specified resolutions in km
        # (converted to degrees above) remain the single authoritative source.
        excluded_keys = {
            "standalone",
            "create_directory",
            "hours",
            "slurm",
            "dSpac_value",
            "dResolution_land",
            "dResolution_ocean",
            "dResolution_river_network",
            "dResolution_coastline",
            "dResolution_lake_boundary",
            "dResolution_watershed_boundary",
            "dResolution_dam",
            "dResolution_city",
        }
        jigsaw = features.get("jigsaw", {}) or {}
        for key, value in jigsaw.items():
            if key not in excluded_keys:
                config[key] = value

        river_vector = paths.get("hydrosheds_rivers") or features.get(
            "hydrosheds_rivers"
        )
        if river_vector:
            config["sFilename_river_network_vector"] = river_vector
            config["iFlag_geom_river_network"] = True
            config["iFlag_spac_river_network"] = True

        dam_vector = paths.get("dam_vector") or features.get("dam_vector")
        if dam_vector:
            config["sFilename_dam_vector"] = dam_vector
            config["iFlag_geom_dam"] = True
            config["iFlag_spac_dam"] = True

        # Forward optional JIGSAW feature files to the native filenames used by
        # run_jigsaw.
        feature_files = {
            "city_vector": "sFilename_city_vector",
            "city_raster": "sFilename_city_raster",
            "river_network_vector": "sFilename_river_network_vector",
            "river_network_raster": "sFilename_river_network_raster",
            "coastline_vector": "sFilename_coastline_vector",
            "coastline_raster": "sFilename_coastline_raster",
            "watershed_boundary_vector": "sFilename_watershed_boundary_vector",
            "watershed_boundary_raster": "sFilename_watershed_boundary_raster",
            "lake_boundary_vector": "sFilename_lake_boundary_vector",
            "lake_boundary_raster": "sFilename_lake_boundary_raster",
        }
        for path_key, config_key in feature_files.items():
            value = paths.get(path_key) or features.get(path_key)
            if value:
                config[config_key] = value

        return config

    def validate(self) -> None:
        mesh_type = str(self.workflow.get("mesh_type", "")).lower()
        if mesh_type != "mpas":
            raise ValueError(
                f"MPASWorkflow requires workflow.mesh_type == 'mpas', got {mesh_type!r}"
            )
        if not self.paths.get("output_workspace"):
            raise ValueError("paths.output_workspace is required")

    def build_case(self):
        """Construct the MPAS case adapter from the compiled native config."""
        from pyearthmesh.meshes.unstructured.mpas.mpascase import mpascase

        sFilename_config = self.config.get("config_file")
        self.aConfig_mesh["sFilename_model_configuration"] = (
            str(sFilename_config) if sFilename_config else None
        )

        # These values control case construction rather than mesh translation.
        jig = self.jigsaw
        self.aConfig_mesh["iFlag_standalone"] = 1 if jig.get("standalone", True) else 0
        iFlag_create_directory = 1 if jig.get("create_directory", True) else 0

        self.case = mpascase(
            self.aConfig_mesh,
            iFlag_create_directory_in=iFlag_create_directory,
            sWorkspace_output_in=self.aConfig_mesh["sWorkspace_output"],
        )
        return self.case

    def create_hpc_job(self, slurm=None, hours=None):
        """Delegate HPC script generation to the underlying case adapter.

        If ``slurm`` or ``hours`` are not provided, values from the workflow's
        ``jigsaw`` configuration section (defaulting to ``'slurm'`` and 5 hours)
        are used automatically.
        """
        if slurm is None:
            slurm = self.jigsaw.get("slurm", "slurm")
        if hours is None:
            hours = self.jigsaw.get("hours", 5)
        if hasattr(self.case, "create_hpc_job"):
            return self.case.create_hpc_job(slurm=slurm, hours=hours)
        return self.case._mpas_create_hpc_job(sSlurm_in=slurm, hours_in=hours)

    @staticmethod
    def _boundary_wkt_from_geojson(sFilename_region):
        """Read a region GeoJSON and return its boundary as a WKT string.

        Returns ``None`` when no region is supplied (global mesh).  Multiple
        features are unioned into a single boundary.  The dataset is closed only
        *after* exporting the WKT (GDAL geometries crash once their dataset is
        closed).
        """
        if not sFilename_region:
            return None
        if not os.path.isfile(sFilename_region):
            print(
                "  ! region_geometry not found, falling back to a global mesh: "
                + str(sFilename_region)
            )
            return None

        from osgeo import ogr

        ds = ogr.Open(sFilename_region)
        if ds is None:
            print(
                "  ! region_geometry could not be opened, falling back to a global "
                "mesh: " + str(sFilename_region)
            )
            return None

        layer = ds.GetLayer()
        geoms = []
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is not None:
                geoms.append(geom.Clone())

        if not geoms:
            ds = None
            return None

        boundary = geoms[0]
        for geom in geoms[1:]:
            boundary = boundary.Union(geom)

        wkt = boundary.ExportToWkt()
        ds = None  # close AFTER exporting WKT
        return wkt
