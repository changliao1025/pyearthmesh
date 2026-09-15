"""Abstract base class for per-mesh-type workflow classes.

A workflow object is constructed from the dict returned by
``load_workflow_config`` (already path-resolved).  Subclasses implement the
mesh-specific defaults, validation, and translation of the sectioned config into
the underlying case object; the base class provides shared orchestration that
delegates to that case object.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseWorkflow(ABC):
    """Turn a loaded, sectioned workflow config into a runnable mesh case."""

    #: mesh type identifier used by the factory registry
    mesh_type = None

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.workflow = config.get("workflow", {})
        self.resolutions = config.get("resolutions", {})
        self.paths = config.get("paths", {})
        self.jigsaw = config.get("jigsaw", {})
        #: underlying case object (e.g. mpascase), built by ``build_case``
        self.case = None
        #: flat mesh config assembled for the case object (for inspection/debug)
        self.aConfig_mesh = None

        self.validate()
        self.build_case()

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
