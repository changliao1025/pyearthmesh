"""ISEA workflow and frontend-to-native compiler."""

import os
from pathlib import Path
from typing import Any

from pyearthmesh.config.workflows.base import BaseCase, BaseWorkflow


class ISEACase(BaseCase):
    """Runtime adapter for the structured ISEA mesh generator."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    def setup(self):
        Path(self.output_workspace).mkdir(parents=True, exist_ok=True)
        return self.output_workspace

    def run(self):
        from pyearthmesh.meshes.structured.dggs.isea.create_isea_mesh import (
            create_isea_mesh,
        )

        result = create_isea_mesh(
            self.config["dResolution_meter"],
            self.config.get("sISEA_type", "ISEA3H"),
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
            if self.config.get("check_mesh_quality"):
                from pyearthmesh.utility.mesh_utility import check_mesh_quality

                check_mesh_quality(sFilename_parquet)

        return result


class ISEAWorkflow(BaseWorkflow):
    """Workflow compiler and runtime adapter for ISEA meshes."""

    mesh_type = "isea"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "workflow": {
                "date": None,
                "mesh_type": "isea",
                "case_index": 1,
                "model": "isea",
                "debug": False,
                "run_on_hpc": False,
            },
            "resolutions": {"meter": 100000.0},
            "paths": {
                "output_workspace": "./output",
                "output_mesh": "isea_mesh.geojson",
            },
            "features": {
                "isea_type": "ISEA3H",
                "use_tiles": False,
                "tile_mode": "simple",
                "convert_to_parquet": False,
                "check_mesh_quality": False,
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
            raise ValueError("ISEA resolution requires resolution.meter")
        if float(resolution_meter) <= 0:
            raise ValueError("ISEA resolution must be positive")

        isea_type = (
            features.get("isea_type")
            or features.get("sISEA_type")
            or config.get("sISEA_type")
            or "ISEA3H"
        )
        isea_type = str(isea_type).upper()
        if isea_type not in ("ISEA3H", "ISEA4H"):
            raise ValueError(
                f"sISEA_type must be 'ISEA3H' or 'ISEA4H', got {isea_type!r}"
            )

        config["mesh_type"] = "isea"
        config["sISEA_type"] = isea_type
        config["dResolution_meter"] = float(resolution_meter)
        config["sFilename_mesh"] = paths.get("output_mesh") or "isea_mesh.geojson"
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
        config["check_mesh_quality"] = bool(features.get("check_mesh_quality", False))
        return config

    def validate(self) -> None:
        mesh_type = str(self.workflow.get("mesh_type", "")).lower()
        if mesh_type != "isea":
            raise ValueError(
                f"ISEAWorkflow requires workflow.mesh_type == 'isea', got {mesh_type!r}"
            )
        if not self.paths.get("output_workspace"):
            raise ValueError("paths.output_workspace is required")

    def build_case(self):
        self.case = ISEACase(self.aConfig_mesh)
        return self.case


__all__ = ["ISEACase", "ISEAWorkflow"]
