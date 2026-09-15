"""MPAS mesh workflow.

Translates the single sectioned workflow config into the flat parameters that
:class:`~pyearthmesh.meshes.unstructured.mpas.mpascase.mpascase` already consumes,
builds the case object **in memory** (no intermediate config file), and delegates
``setup`` / ``run`` / HPC-job generation to it.
"""

import os
from typing import Any

from pyearthmesh.config.workflows.base import BaseWorkflow


class MPASWorkflow(BaseWorkflow):
    """Workflow class for the ``mpas`` mesh type."""

    mesh_type = "mpas"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """MPAS-flavoured defaults for the sectioned workflow config."""
        return {
            "workflow": {"date": None, "mesh_type": "mpas", "case_index": 1,
                         "model": "jigsaw", "simplify_hydrosheds_river_network": False,
                         "process_coastline": False, "debug": True, "largest_outlets": 10,
                         "run_on_hpc": False},
            "resolutions": {"ocean_km": 100, "land_km": 12.5, "river_network_km": 5,
                            "coastline_km": 10},
            "paths": {"input_workspace": "", "output_workspace": "./output",
                      "hydrosheds_rivers": "", "region_geometry": "", "dam_vector": ""},
            "jigsaw": {"standalone": True, "create_directory": True, "hours": 5,
                       "slurm": "slurm"},
        }

    def validate(self) -> None:
        mesh_type = str(self.workflow.get("mesh_type", "")).lower()
        if mesh_type != "mpas":
            raise ValueError(
                f"MPASWorkflow requires workflow.mesh_type == 'mpas', got {mesh_type!r}"
            )
        if not self.paths.get("output_workspace"):
            raise ValueError("paths.output_workspace is required")

    def build_case(self):
        """Assemble the flat mesh config and construct the ``mpascase`` object."""
        # Heavy imports are deferred so that importing this module stays cheap and
        # free of any dependency on config_manager (avoids circular imports).
        from pyearthmesh.classes.meshconfig import JigsawConfig, MPASConfig
        from pyearthmesh.meshes.unstructured.mpas.mpascase import mpascase

        # Base layer mirrors MPASConfig.create_template_config's merge order:
        # JIGSAW defaults first, then MPAS defaults on top.
        aConfig = JigsawConfig().get_default_config()
        aConfig.pop("mesh_type", None)
        aConfig.update(MPASConfig().get_default_config())

        # --- resolutions (km -> degrees) -------------------------------------
        res = self.resolutions
        aConfig["dSpac_value"] = self._km_to_degree(res.get("ocean_km", 100))
        aConfig["dResolution_land"] = self._km_to_degree(res.get("land_km", 100))
        aConfig["dResolution_river_network"] = self._km_to_degree(
            res.get("river_network_km", 10)
        )
        aConfig["dResolution_coastline"] = self._km_to_degree(
            res.get("coastline_km", 10)
        )

        # --- workflow section -------------------------------------------------
        wf = self.workflow
        sModel = str(wf.get("model", "jigsaw")).lower()
        aConfig["mesh_type"] = "mpas"
        aConfig["sModel"] = sModel
        aConfig["iCase_index"] = int(wf.get("case_index", 1))
        aConfig["sDate"] = wf.get("date")
        aConfig["verbosity"] = 1 if wf.get("debug") else 0
        aConfig["iFlag_run_jigsaw"] = 1 if sModel == "jigsaw" else 0

        bCoastline = bool(wf.get("process_coastline", False))
        aConfig["iFlag_geom_coastline"] = bCoastline
        aConfig["iFlag_spac_coastline"] = bCoastline

        # --- paths section ----------------------------------------------------
        paths = self.paths
        aConfig["sWorkspace_output"] = paths.get("output_workspace")

        sFilename_config = self.config.get("config_file")
        aConfig["sFilename_model_configuration"] = (
            str(sFilename_config) if sFilename_config else None
        )

        sRiver = paths.get("hydrosheds_rivers") or None
        if sRiver:
            aConfig["sFilename_river_network_vector"] = sRiver
            aConfig["iFlag_geom_river_network"] = True
            aConfig["iFlag_spac_river_network"] = True

        sDam = paths.get("dam_vector") or None
        if sDam:
            aConfig["sFilename_dam_vector"] = sDam
            aConfig["iFlag_geom_dam"] = True
            aConfig["iFlag_spac_dam"] = True

        # --- boundary (region_geometry GeoJSON -> pBoundary_wkt WKT string) ---
        pBoundary_wkt = self._boundary_wkt_from_geojson(paths.get("region_geometry"))
        aConfig["pBoundary_wkt"] = pBoundary_wkt
        aConfig["iFlag_global"] = 0 if pBoundary_wkt else 1

        # --- jigsaw / execution section --------------------------------------
        jig = self.jigsaw
        aConfig["iFlag_standalone"] = 1 if jig.get("standalone", True) else 0
        iFlag_create_directory = 1 if jig.get("create_directory", True) else 0

        self.aConfig_mesh = aConfig
        self.case = mpascase(
            aConfig,
            iFlag_create_directory_in=iFlag_create_directory,
            sWorkspace_output_in=aConfig["sWorkspace_output"],
        )
        return self.case

    def create_hpc_job(self, slurm=None, hours=10):
        """Delegate HPC script generation to the underlying ``mpascase``."""
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
