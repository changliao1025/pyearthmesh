"""rHEALPix workflow and frontend-to-native compiler."""

import os
from pathlib import Path
from typing import Any

from pyearthmesh.config.workflows.base import BaseCase, BaseWorkflow


class RHEALPixCase(BaseCase):
    """Runtime adapter for the structured rHEALPix mesh generator."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    def setup(self):
        Path(self.output_workspace).mkdir(parents=True, exist_ok=True)
        return self.output_workspace

    def run(self):
        from pyearthmesh.meshes.structured.dggs.healpix.create_rhealpix_mesh import (
            create_rhealpix_mesh,
        )

        result = create_rhealpix_mesh(
            self.config["dResolution_meter"],
            self.config["sFilename_mesh"],
            self.config["sWorkspace_output"],
            iFlag_netcdf=self.config.get("iFlag_netcdf", 0),
            pBoundary_in=self.config.get("pBoundary_wkt"),
            iFlag_use_tiles=self.config.get("iFlag_use_tiles", 0),
            nTile=self.config.get("nTile"),
            tile_mode=self.config.get("tile_mode", "simple"),
            zone_level=self.config.get("zone_level"),
            tiles_per_base=self.config.get("tiles_per_base"),
        )

        if self.config.get("convert_to_parquet"):
            if self.config.get("iFlag_use_tiles", 0) == 1:
                from pyearthmesh.meshes.structured.dggs.convert_vector_format_multi_tile import (
                    convert_multi_tile_vector_format,
                )

                sWorkspace_in = self.config["sWorkspace_output"]
                sWorkspace_out = os.path.join(sWorkspace_in, "convert")
                Path(sWorkspace_out).mkdir(parents=True, exist_ok=True)
                sExtension_in = Path(self.config["sFilename_mesh"]).suffix.lower()
                convert_multi_tile_vector_format(
                    sWorkspace_in, sWorkspace_out, sExtension_in, ".parquet"
                )
            else:
                from pyearth.toolbox.conversion.convert_vector_format import (
                    convert_vector_format,
                )

                sFilename_mesh_full = os.path.join(
                    self.config["sWorkspace_output"], self.config["sFilename_mesh"]
                )
                sFilename_parquet = os.path.join(
                    self.config["sWorkspace_output"],
                    Path(self.config["sFilename_mesh"]).stem + ".parquet",
                )
                convert_vector_format(
                    sFilename_mesh_full, sFilename_parquet, use_ogr2ogr=True
                )

        return result


class RHEALPixWorkflow(BaseWorkflow):
    """Workflow compiler and runtime adapter for rHEALPix meshes."""

    mesh_type = "rhealpix"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "workflow": {
                "date": None,
                "mesh_type": "rhealpix",
                "case_index": 1,
                "model": "rhealpix",
                "debug": False,
                "run_on_hpc": False,
            },
            "resolutions": {"meter": 100000.0},
            "paths": {
                "output_workspace": "./output",
                "output_mesh": "rhealpix_mesh.geojson",
            },
            "features": {
                "use_tiles": False,
                "tile_mode": "simple",
                "convert_to_parquet": False,
            },
        }

    @classmethod
    def translate_frontend_to_backend(cls, spec, native_config):
        if hasattr(spec, "to_dict"):
            spec = spec.to_dict()
        if not isinstance(spec, dict):
            raise TypeError("spec must be a WorkflowSpec or dict")

        config = dict(native_config)
        resolution = spec.get("resolution", {}) or {}
        paths = spec.get("paths", {}) or {}
        features = spec.get("features", {}) or {}
        region = spec.get("region", {}) or {}

        resolution_meter = resolution.get("meter") or resolution.get("resolution_meter")
        if resolution_meter is None and resolution.get("ocean_km") is not None:
            resolution_meter = float(resolution["ocean_km"]) * 1000.0
        if resolution_meter is None:
            raise ValueError("rHEALPix resolution requires resolution.meter")
        if float(resolution_meter) <= 0:
            raise ValueError("rHEALPix resolution must be positive")

        config["mesh_type"] = "rhealpix"
        config["dResolution_meter"] = float(resolution_meter)
        config["sFilename_mesh"] = paths.get("output_mesh") or "rhealpix_mesh.geojson"
        config["sWorkspace_output"] = paths.get("output_workspace") or "./output"
        config["pBoundary_wkt"] = region.get("wkt")
        config["iFlag_global"] = int(not bool(config["pBoundary_wkt"]))

        use_tiles = bool(features.get("use_tiles", False))
        config["iFlag_use_tiles"] = int(use_tiles)
        config["tile_mode"] = features.get("tile_mode", "simple")
        config["nTile"] = features.get("nTile")
        config["zone_level"] = features.get("zone_level")
        config["tiles_per_base"] = features.get("tiles_per_base")
        config["iFlag_netcdf"] = int(
            str(config["sFilename_mesh"]).lower().endswith(".nc")
        )
        config["convert_to_parquet"] = bool(features.get("convert_to_parquet", False))
        return config

    def validate(self) -> None:
        mesh_type = str(self.workflow.get("mesh_type", "")).lower()
        if mesh_type != "rhealpix":
            raise ValueError(
                f"RHEALPixWorkflow requires workflow.mesh_type == 'rhealpix', got {mesh_type!r}"
            )
        if not self.paths.get("output_workspace"):
            raise ValueError("paths.output_workspace is required")

    def build_case(self):
        self.case = RHEALPixCase(self.aConfig_mesh)
        return self.case


__all__ = ["RHEALPixCase", "RHEALPixWorkflow"]
