"""Base adapter class for all mesh cases."""

import os
from abc import ABC, abstractmethod
from typing import Any


class BaseCase(ABC):
    """Abstract base runtime adapter for mesh case execution.

    Provides a uniform interface across all mesh types (structured, DGGS,
    unstructured):
    - ``self.config`` stores the compiled native mesh configuration dict.
    - ``setup()`` prepares the output workspace directories.
    - ``run()`` executes the generator and returns the resulting mesh.
    - ``output_workspace`` returns the resolved output directory.
    - ``mesh_file`` returns the primary output file path.
    - ``create_hpc_job()`` generates batch scripts for clusters (where supported).
    """

    config: dict[str, Any] = {}

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)

    @property
    def output_workspace(self) -> str:
        """Resolved output directory for this case."""
        ws = (
            getattr(self, "sWorkspace_output", None)
            or self.config.get("sWorkspace_output")
            or ""
        )
        return str(ws)

    @property
    def mesh_file(self) -> str:
        """Resolved path to the primary output mesh file."""
        mf = (
            getattr(self, "sFilename_mesh", None)
            or self.config.get("sFilename_mesh")
            or ""
        )
        return str(mf)

    @abstractmethod
    def setup(self):
        """Prepare the workspace directories for mesh generation."""
        pass

    @abstractmethod
    def run(self):
        """Execute mesh generation and return the resulting mesh or cells."""
        pass

    def create_hpc_job(self, slurm=None, hours=10):
        """Generate HPC job scripts for execution on a cluster."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_hpc_job"
        )


__all__ = ["BaseCase"]
