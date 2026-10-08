"""Per-mesh-type workflow classes.

Each mesh type owns a class that knows how to translate the single sectioned
workflow config into its underlying case object and drive the run.  The registry
and ``create_workflow`` factory live in
:mod:`pyearthmesh.config.config_manager`.
"""

from pyearthmesh.config.workflows.base import BaseWorkflow, WorkflowSpec
from pyearthmesh.config.workflows.healpix_workflow import HealpixWorkflow
from pyearthmesh.config.workflows.isea_workflow import ISEACase, ISEAWorkflow
from pyearthmesh.config.workflows.latlon_workflow import LatLonWorkflow
from pyearthmesh.config.workflows.mpas_workflow import MPASWorkflow
from pyearthmesh.config.workflows.rhealpix_workflow import RHEALPixCase, RHEALPixWorkflow

# Backward-compatible alias for the older module name.
MPASWorkflowModuleAlias = MPASWorkflow

__all__ = [
    "BaseWorkflow",
    "WorkflowSpec",
    "HealpixWorkflow",
    "ISEACase",
    "ISEAWorkflow",
    "LatLonWorkflow",
    "MPASWorkflow",
    "MPASWorkflowModuleAlias",
    "RHEALPixCase",
    "RHEALPixWorkflow",
]
