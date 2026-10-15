"""Lat-Lon mesh workflow and frontend-to-native compiler."""

import os
from pathlib import Path
from typing import Any

from pyearthmesh.config.workflows.base import BaseCase, BaseWorkflow


class LatLonCase(BaseCase):
    """Small runtime adapter for the structured Lat-Lon mesh generator."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    def setup(self):
        output_filename = self.mesh_file
        Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
        return output_filename

    def run(self):
        from pyearthmesh.meshes.structured.latlon.create_latlon_mesh import (
            create_latlon_mesh,
        )

        return create_latlon_mesh(
            self.config["dLongitude_left"],
            self.config["dLatitude_bot"],
            self.config["dResolution_degree"],
            self.config["ncolumn"],
            self.config["nrow"],
            self.config["sFilename_mesh"],
            self.config.get("pBoundary_wkt"),
        )


class LatLonWorkflow(BaseWorkflow):
    """Workflow compiler and runtime adapter for regular Lat-Lon meshes."""

    mesh_type = "latlon"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "workflow": {"date": None, "mesh_type": "latlon", "case_index": 1,
                         "model": "latlon", "debug": False, "run_on_hpc": False},
            "resolutions": {"degree": 1.0},
            "paths": {"output_workspace": "./output", "region_geometry": ""},
        }

    @classmethod
    def translate_frontend_to_backend(cls, spec, native_config):
        if hasattr(spec, "to_dict"):
            spec = spec.to_dict()
        if not isinstance(spec, dict):
            raise TypeError("spec must be a WorkflowSpec or dict")

        config = dict(native_config)
        resolution = spec.get("resolution", {}) or {}
        region = spec.get("region", {}) or {}
        paths = spec.get("paths", {}) or {}

        degree = (
            resolution.get("degree")
            or resolution.get("resolution_degree")
            or resolution.get("ocean_degree")
        )
        if degree is None and resolution.get("ocean_km") is not None:
            degree = cls._km_to_degree(float(resolution["ocean_km"]))
        degree = float(degree or 1.0)
        if degree <= 0:
            raise ValueError("Lat-Lon resolution must be positive")

        bounds = region.get("bbox", {}) or {}
        longitude_left = float(region.get("longitude_left") or bounds.get("left", -180.0))
        longitude_right = float(region.get("longitude_right") or bounds.get("right", 180.0))
        latitude_bot = float(region.get("latitude_bot") or bounds.get("bottom", -90.0))
        latitude_top = float(region.get("latitude_top") or bounds.get("top", 90.0))

        config["mesh_type"] = "latlon"
        config["dResolution_degree"] = degree
        config["dLongitude_left"] = longitude_left
        config["dLongitude_right"] = longitude_right
        config["dLatitude_bot"] = latitude_bot
        config["dLatitude_top"] = latitude_top
        config["ncolumn"] = int(round((longitude_right - longitude_left) / degree))
        config["nrow"] = int(round((latitude_top - latitude_bot) / degree))
        config["pBoundary_wkt"] = region.get("wkt")
        config["iFlag_global"] = int(not bool(config["pBoundary_wkt"]))

        output_workspace = paths.get("output_workspace") or "./output"
        output_filename = paths.get("output_mesh") or "latlon_mesh.geojson"
        config["sWorkspace_output"] = output_workspace
        config["sFilename_mesh"] = os.path.join(output_workspace, output_filename)
        return config

    def validate(self) -> None:
        mesh_type = str(self.workflow.get("mesh_type", "")).lower()
        if mesh_type != "latlon":
            raise ValueError(
                f"LatLonWorkflow requires workflow.mesh_type == 'latlon', got {mesh_type!r}"
            )
        if not self.paths.get("output_workspace"):
            raise ValueError("paths.output_workspace is required")

    def build_case(self):
        self.case = LatLonCase(self.aConfig_mesh)
        return self.case


__all__ = ["LatLonCase", "LatLonWorkflow"]
