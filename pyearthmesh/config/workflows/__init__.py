"""Per-mesh-type workflow classes.

Each mesh type owns a class that knows how to translate the single sectioned
workflow config into its underlying case object and drive the run.  The registry
and ``create_workflow`` factory live in
:mod:`pyearthmesh.config.config_manager`.
"""

from pyearthmesh.config.workflows.base import BaseCase, BaseWorkflow, WorkflowSpec
from pyearthmesh.config.workflows.healpix_workflow import HealpixCase, HealpixWorkflow
from pyearthmesh.config.workflows.isea_workflow import ISEACase, ISEAWorkflow
from pyearthmesh.config.workflows.latlon_workflow import LatLonCase, LatLonWorkflow
from pyearthmesh.config.workflows.mpas_workflow import MPASCase, MPASWorkflow
from pyearthmesh.config.workflows.rhealpix_workflow import RHEALPixCase, RHEALPixWorkflow

# Backward-compatible alias for the older module name.
MPASWorkflowModuleAlias = MPASWorkflow

__all__ = [
    "BaseCase",
    "BaseWorkflow",
    "WorkflowSpec",
    "HealpixCase",
    "HealpixWorkflow",
    "ISEACase",
    "ISEAWorkflow",
    "LatLonCase",
    "LatLonWorkflow",
    "MPASCase",
    "MPASWorkflow",
    "MPASWorkflowModuleAlias",
    "RHEALPixCase",
    "RHEALPixWorkflow",
]
