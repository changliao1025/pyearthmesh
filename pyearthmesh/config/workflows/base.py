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
        execution: dict[str, Any] | None = None,
        features: dict[str, Any] | None = None,
    ):
        self.mesh_type = self._normalize_mesh_type(mesh_type)
        self.case_index = int(case_index)
        self.date = date
        self.region = dict(region or {})
        self.resolution = dict(resolution or {})
        self.paths = dict(paths or {})
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
            "execution": self.execution,
            "features": self.features,
        }


class BaseWorkflow(ABC):
    """Turn a loaded, sectioned workflow config into a runnable mesh case."""

    #: mesh type identifier used by the factory registry
    mesh_type = None

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.workflow = config.get("workflow", {})
        self.resolutions = config.get("resolutions", {})
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
                "geometry": self.region.get("geometry") or self.paths.get("region_geometry"),
            },
            "resolution": {
                "kind": "km",
                **self.resolutions,
            },
            "paths": self.paths,
            "execution": {
                "debug": self.workflow.get("debug", False),
                "model": self.workflow.get("model", "jigsaw"),
                "run_on_hpc": self.workflow.get("run_on_hpc", False),
            },
            "features": {
                **self.features,
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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_hpc_job"
        )
