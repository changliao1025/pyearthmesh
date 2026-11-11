"""Abstract base class for per-mesh-type workflow classes.

A workflow object is constructed from the dict returned by
``load_workflow_config`` (already path-resolved).  Subclasses implement the
mesh-specific defaults, validation, and translation of the sectioned config into
the underlying case object; the base class provides shared orchestration that
delegates to that case object.
"""

from abc import ABC, abstractmethod
from typing import Any


class WorkflowSpec:
    """Mesh-agnostic front-end specification for a run.

    This is the first layer in the two-tier compiler architecture: a user-facing,
    mesh-generic specification that describes the intent of a run without
    committing to the backend-native config keys required by a specific mesh
    generator.
    """

    def __init__(
        self,
        mesh_type: str,
        case_index: int = 1,
        date: str | None = None,
        region: dict[str, Any] | None = None,
        resolution: dict[str, Any] | None = None,
        paths: dict[str, Any] | None = None,
        flags: dict[str, Any] | None = None,
        execution: dict[str, Any] | None = None,
        features: dict[str, Any] | None = None,
    ):
        self.mesh_type = self._normalize_mesh_type(mesh_type)
        self.case_index = int(case_index)
        self.date = date
        self.region = dict(region or {})
        self.resolution = dict(resolution or {})
        self.paths = dict(paths or {})
        self.flags = dict(flags or {})
        self.execution = dict(execution or {})
        self.features = dict(features or {})

        if self.case_index < 1:
            raise ValueError("case_index must be a positive integer")

    @staticmethod
    def _normalize_mesh_type(mesh_type: str) -> str:
        if mesh_type is None:
            raise ValueError("mesh_type is required")

        mesh_type_str = str(mesh_type).strip().lower()
        if not mesh_type_str:
            raise ValueError("mesh_type cannot be empty")

        try:
            from pyearthmesh.classes.meshconfig import ConfigManager

            available = set(ConfigManager.get_available_mesh_types())
        except Exception:
            available = set()

        if available and mesh_type_str not in available:
            raise ValueError(
                f"Unsupported mesh_type {mesh_type_str!r}. "
                f"Available types: {sorted(available)}"
            )

        return mesh_type_str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowSpec":
        if not isinstance(data, dict):
            raise ValueError("WorkflowSpec input must be a dictionary")

        return cls(
            mesh_type=data.get("mesh_type", "mpas"),
            case_index=data.get("case_index", 1),
            date=data.get("date"),
            region=data.get("region"),
            resolution=data.get("resolution"),
            paths=data.get("paths"),
            flags=data.get("flags"),
            execution=data.get("execution"),
            features=data.get("features"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mesh_type": self.mesh_type,
            "case_index": self.case_index,
            "date": self.date,
            "region": self.region,
            "resolution": self.resolution,
            "paths": self.paths,
            "flags": self.flags,
            "execution": self.execution,
            "features": self.features,
        }


from pyearthmesh.classes.base_case import BaseCase


class BaseWorkflow(ABC):
    """Turn a loaded, sectioned workflow config into a runnable mesh case."""

    #: mesh type identifier used by the factory registry
    mesh_type: str | None = None

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.workflow = config.get("workflow", {})
        self.resolutions = config.get("resolutions", {})
        self.flags = config.get("flags", {})
        self.paths = config.get("paths", {})
        self.region = config.get("region", {})
        self.features = config.get("features", {})
        self.jigsaw = config.get("jigsaw", {})
        #: underlying case object (e.g. mpascase), built by ``build_case``
        self.case = None
        #: flat mesh config assembled for the case object (for inspection/debug)
        self.aConfig_mesh = None

        self.validate()
        self.aConfig_mesh = self.compile_frontend_to_backend()
        self.build_case()

    @classmethod
    def compile_from_spec(cls, spec: WorkflowSpec | dict[str, Any]) -> dict[str, Any]:
        """Compile a front-end workflow spec into the backend-native config.

        This is the generic two-tier compiler boundary: the user-facing spec is
        translated into the native config dict expected by the case-building code
        for the selected mesh family.
        """
        if isinstance(spec, dict):
            spec = WorkflowSpec.from_dict(spec)
        if not isinstance(spec, WorkflowSpec):
            raise TypeError("spec must be a WorkflowSpec or a dictionary")

        mesh_type = str(spec.mesh_type).lower()
        if cls.mesh_type is not None and mesh_type != str(cls.mesh_type).lower():
            raise ValueError(
                f"{cls.__name__} expects mesh_type {cls.mesh_type!r}, got {mesh_type!r}"
            )

        from pyearthmesh.classes.meshconfig import ConfigManager

        native_config = ConfigManager(mesh_type).get_default_config(mesh_type)
        native_config.update(cls.translate_frontend_to_backend(spec, native_config))
        return native_config

    @classmethod
    def resolve_boundary_and_flags(
        cls, spec: WorkflowSpec | dict[str, Any], native_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Centrally resolve boundary geometry, extents, and mesh flags.

        Calculates dLatitude_mean and dLongitude_mean dynamically from extents.
        """
        if hasattr(spec, "to_dict"):
            spec_data = spec.to_dict()
        elif isinstance(spec, dict):
            spec_data = spec
        else:
            raise TypeError("spec must be a WorkflowSpec or dict")

        flags = spec_data.get("flags", {}) or {}
        region = spec_data.get("region", {}) or {}
        paths = spec_data.get("paths", {}) or {}
        features = spec_data.get("features", {}) or {}
        bounds = region.get("bbox", {}) or {}

        lon_left = region.get("longitude_left")
        if lon_left is None:
            lon_left = bounds.get("left")

        lon_right = region.get("longitude_right")
        if lon_right is None:
            lon_right = bounds.get("right")

        lat_bot = region.get("latitude_bot")
        if lat_bot is None:
            lat_bot = bounds.get("bottom")

        lat_top = region.get("latitude_top")
        if lat_top is None:
            lat_top = bounds.get("top")

        boundary_file = (
            paths.get("mesh_boundary")
            or paths.get("mesh_boundary_file")
            or paths.get("boundary_file")
            or paths.get("boundary_vector")
            or region.get("boundary_file")
            or paths.get("region_geometry")
            or region.get("geometry")
        )
        wkt = region.get("wkt")

        i_flag_mesh_boundary = int(bool(flags.get("mesh_boundary", False)))
        i_flag_global = int(bool(flags.get("global", False)))

        from pyearthmesh.meshes.generate_mesh import determine_mesh_boundary

        resolved_wkt, extent = determine_mesh_boundary(
            pBoundary_wkt=wkt,
            sFilename_mesh_boundary=boundary_file,
            iFlag_mesh_boundary=i_flag_mesh_boundary,
            dLongitude_left=float(lon_left) if lon_left is not None else -180.0,
            dLongitude_right=float(lon_right) if lon_right is not None else 180.0,
            dLatitude_bot=float(lat_bot) if lat_bot is not None else -90.0,
            dLatitude_top=float(lat_top) if lat_top is not None else 90.0,
            iFlag_global=i_flag_global,
        )

        if extent:
            lon_left, lon_right, lat_bot, lat_top = extent

        lat_mean = (
            0.5 * (float(lat_bot) + float(lat_top))
            if lat_bot is not None and lat_top is not None
            else 0.0
        )
        lon_mean = (
            0.5 * (float(lon_left) + float(lon_right))
            if lon_left is not None and lon_right is not None
            else 0.0
        )

        native_config["pBoundary_wkt"] = resolved_wkt
        native_config["sFilename_mesh_boundary"] = boundary_file or None
        native_config["sFilename_boundary"] = boundary_file or None
        native_config["iFlag_mesh_boundary"] = i_flag_mesh_boundary

        if "global" in flags:
            native_config["iFlag_global"] = 1 if flags["global"] else 0
        else:
            native_config["iFlag_global"] = (
                0
                if (
                    resolved_wkt
                    or extent
                    or (
                        lon_left is not None
                        and lon_right is not None
                        and lat_bot is not None
                        and lat_top is not None
                        and (
                            float(lon_left) != -180.0
                            or float(lon_right) != 180.0
                            or float(lat_bot) != -90.0
                            or float(lat_top) != 90.0
                        )
                    )
                )
                else 1
            )

        if lon_left is not None:
            native_config["dLongitude_left"] = float(lon_left)
        if lon_right is not None:
            native_config["dLongitude_right"] = float(lon_right)
        if lat_bot is not None:
            native_config["dLatitude_bot"] = float(lat_bot)
        if lat_top is not None:
            native_config["dLatitude_top"] = float(lat_top)
        native_config["dLatitude_mean"] = float(lat_mean)
        native_config["dLongitude_mean"] = float(lon_mean)

        if "save_mesh" in flags:
            native_config["iFlag_save_mesh"] = 1 if flags["save_mesh"] else 0
        else:
            native_config["iFlag_save_mesh"] = 1

        if "rotation" in flags:
            native_config["iFlag_rotation"] = 1 if flags["rotation"] else 0

        if "use_mesh_dem" in flags:
            native_config["iFlag_use_mesh_dem"] = 1 if flags["use_mesh_dem"] else 0

        if "use_tiles" in flags or "use_tiles" in features:
            native_config["iFlag_use_tiles"] = (
                1 if (flags.get("use_tiles") or features.get("use_tiles")) else 0
            )

        if "netcdf" in flags:
            native_config["iFlag_netcdf"] = 1 if flags["netcdf"] else 0

        if "check_mesh_quality" in flags or "check_mesh_quality" in features:
            native_config["check_mesh_quality"] = bool(
                flags.get("check_mesh_quality", features.get("check_mesh_quality"))
            )

        if "convert_to_parquet" in flags or "convert_to_parquet" in features:
            native_config["convert_to_parquet"] = bool(
                flags.get("convert_to_parquet", features.get("convert_to_parquet"))
            )

        return native_config

    @classmethod
    def translate_frontend_to_backend(
        cls, spec: WorkflowSpec, native_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Translate a generic workflow spec into a mesh-native config dict.

        Subclasses override this to handle mesh-specific translation.
        """
        return {}

    def compile_frontend_to_backend(self) -> dict[str, Any]:
        """Compile the current workflow config to the native backend config."""
        if not isinstance(self.config, dict):
            raise TypeError("Workflow config must be a dictionary")

        spec_data = {
            "mesh_type": self.workflow.get("mesh_type", self.mesh_type),
            "case_index": self.workflow.get("case_index", 1),
            "date": self.workflow.get("date"),
            "region": {
                **self.region,
                "geometry": self.region.get("geometry")
                or self.paths.get("region_geometry"),
            },
            "resolution": {
                "kind": "km",
                **self.resolutions,
            },
            "paths": self.paths,
            "flags": self.flags,
            "execution": {
                "debug": self.workflow.get("debug", False),
                "model": self.workflow.get("model", "jigsaw"),
                "run_on_hpc": self.workflow.get("run_on_hpc", False),
            },
            "features": {
                **(self.features or {}),
                "coastline": self.workflow.get("process_coastline", False),
                "simplify_hydrosheds_river_network": self.workflow.get(
                    "simplify_hydrosheds_river_network", False
                ),
                "jigsaw": self.jigsaw,
            },
        }
        return self.compile_from_spec(spec_data)

    # ------------------------------------------------------------------
    # Mesh-specific hooks
    # ------------------------------------------------------------------
    @classmethod
    @abstractmethod
    def default_config(cls) -> dict[str, Any]:
        """Return the mesh-specific default (sectioned) workflow config."""

    @abstractmethod
    def validate(self) -> None:
        """Raise ``ValueError`` on missing/invalid/incompatible configuration."""

    @abstractmethod
    def build_case(self):
        """Translate the sectioned config into the underlying case object."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _km_to_degree(km: float) -> float:
        """Convert a distance in kilometres to degrees.

        Uses the latitude approximation (1 degree ~= 111.32 km).  Kept in one
        place so the factor is documented and consistent.
        """
        return float(km) / 111.32

    # ------------------------------------------------------------------
    # Properties (encapsulate case/config paths)
    # ------------------------------------------------------------------
    @property
    def output_workspace(self) -> str:
        """Return the resolved output workspace directory for this workflow."""
        if self.case is not None:
            if hasattr(self.case, "output_workspace") and self.case.output_workspace:
                return str(self.case.output_workspace)
            if hasattr(self.case, "sWorkspace_output") and self.case.sWorkspace_output:
                return str(self.case.sWorkspace_output)
            if hasattr(self.case, "config") and isinstance(self.case.config, dict):
                ws = self.case.config.get("sWorkspace_output")
                if ws:
                    return str(ws)
        if self.aConfig_mesh and isinstance(self.aConfig_mesh, dict):
            ws = self.aConfig_mesh.get("sWorkspace_output")
            if ws:
                return str(ws)
        return str(self.paths.get("output_workspace", ""))

    @property
    def mesh_file(self) -> str:
        """Return the resolved path to the output mesh file for this workflow."""
        if self.case is not None:
            if hasattr(self.case, "mesh_file") and self.case.mesh_file:
                return str(self.case.mesh_file)
            if hasattr(self.case, "sFilename_mesh") and self.case.sFilename_mesh:
                return str(self.case.sFilename_mesh)
            if hasattr(self.case, "config") and isinstance(self.case.config, dict):
                mf = self.case.config.get("sFilename_mesh")
                if mf:
                    return str(mf)
        if self.aConfig_mesh and isinstance(self.aConfig_mesh, dict):
            mf = self.aConfig_mesh.get("sFilename_mesh")
            if mf:
                return str(mf)
        return str(self.paths.get("output_mesh", ""))

    # ------------------------------------------------------------------
    # Orchestration (delegates to the case object by default)
    # ------------------------------------------------------------------
    def setup(self):
        """Prepare the workspace for the run."""
        return self.case.setup()

    def run(self):
        """Execute mesh generation."""
        return self.case.run()

    def create_hpc_job(self, slurm=None, hours=10):
        """Generate HPC (SLURM) job scripts for the run."""
        if self.case is not None:
            if hasattr(self.case, "create_hpc_job"):
                return self.case.create_hpc_job(slurm=slurm, hours=hours)
            if hasattr(self.case, "_mpas_create_hpc_job"):
                return self.case._mpas_create_hpc_job(sSlurm_in=slurm, hours_in=hours)
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_hpc_job"
        )
