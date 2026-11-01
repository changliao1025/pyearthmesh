import json
import platform
from pathlib import Path
from typing import Any

# ==============================================================================
# Workflow configuration loader and dumper
#
# This is the single, sectioned, human-editable configuration schema. It is the
# one and only source of truth for a run; there is no intermediate flat config
# file any more.
#
# Primary format is YAML (.yaml / .yml) to allow inline and block comments (#),
# making workflow specifications transparent and user-friendly. JSON (.json)
# is also supported for programmatic workflows and backward compatibility.
#
# ``load_workflow_config`` resolves platform-specific and relative paths and
# normalises the run date.
# ==============================================================================

DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "workflow": {
        "date": None,
        "mesh_type": "mpas",
        "case_index": 1,
        "model": "jigsaw",
        "simplify_hydrosheds_river_network": False,
        "process_coastline": False,
        "debug": True,
        "largest_outlets": 10,
        "run_on_hpc": False,
    },
    "resolutions": {
        "degree": None,
        "meter": None,
        "ocean_km": 100,
        "land_km": 100,
        "river_network_km": 10,
        "coastline_km": 10,
        "lake_boundary_km": 10,
        "watershed_boundary_km": 10,
        "dam_km": None,
        "city_km": None,
    },
    "paths": {
        "input_workspace": "",
        "output_workspace": "./output",
        "hydrosheds_rivers": "",
        "region_geometry": "",
        "dam_vector": "",
        "city_vector": "",
        "city_raster": "",
        "river_network_vector": "",
        "river_network_raster": "",
        "coastline_vector": "",
        "coastline_raster": "",
        "watershed_boundary_vector": "",
        "watershed_boundary_raster": "",
        "lake_boundary_vector": "",
        "lake_boundary_raster": "",
        "output_mesh": "",
    },
    "region": {
        "bbox": {},
        "wkt": "",
        "longitude_left": None,
        "longitude_right": None,
        "latitude_bot": None,
        "latitude_top": None,
    },
    "features": {
        "coastline": False,
        "simplify_hydrosheds_river_network": False,
        "use_tiles": False,
        "tile_mode": "simple",
        "nTile": None,
        "zone_level": None,
        "tiles_per_base": None,
        "isea_type": "ISEA3H",
        "convert_to_parquet": False,
        "check_mesh_quality": False,
        "endorheic_lake": True,
    },
    "jigsaw": {
        "standalone": True,
        "create_directory": True,
        "hours": 5,
        "slurm": "slurm",
        "ncolumn_space": 360,
        "nrow_space": 180,
        "iFlag_geom": False,
        "iFlag_spac": False,
        "iFlag_init": False,
        "iFlag_opts": False,
        "iFlag_spac_ocean": False,
        "iFlag_spac_land": True,
        "iFlag_geom_dam": False,
        "iFlag_spac_dam": False,
        "iFlag_geom_city": False,
        "iFlag_spac_city": False,
        "iFlag_geom_river_network": False,
        "iFlag_spac_river_network": False,
        "iFlag_geom_coastline": False,
        "iFlag_spac_coastline": False,
        "iFlag_geom_watershed_boundary": False,
        "iFlag_spac_watershed_boundary": False,
        "iFlag_geom_lake_boundary": False,
        "iFlag_spac_lake_boundary": False,
        "dhdx_lim": 0.25,
        "hfun_hmax": "inf",
        "hfun_hmin": 0.0,
        "hfun_scal": "absolute",
        "mesh_dims": 2,
        "bisection": -1,
        "optm_qlim": 0.95,
        "optm_iter": 32,
        "optm_qtol": 1.0e-05,
        "mesh_rad2": 1.5,
        "mesh_rad3": 2.0,
        "mesh_eps1": 0.333,
        "mesh_eps2": 0.333,
        "mesh_top": 1,
        "mesh_iter": 3,
        "geom_mshID": "ellipsoid-mesh",
        "spac_mshID": "ellipsoid-grid",
        "FULL_SPHERE_RADIUS": 6371.0,
        "verbosity": 0,
        "mesh_kern": "delfront",
        "optm_kern": "odt+dqdx",
        "geom_feat": True,
    },
}


def _merge(
    defaults: dict[str, Any], values: dict[str, Any], section: str
) -> dict[str, Any]:
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {section} setting(s): {', '.join(sorted(unknown))}")
    result = defaults.copy()
    result.update(values)
    return result


def _select_platform_path(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise ValueError("Path values must be strings or platform-specific objects")

    current_platform = platform.system()
    selected = value.get(current_platform, value.get("default", ""))
    if not isinstance(selected, str):
        raise ValueError(f"Path for {current_platform} must be a string")
    return selected


def _resolve_config_path(filename: str | Path) -> Path:
    path = Path(filename).expanduser().resolve()
    if path.is_dir():
        for candidate_name in (
            "workflow_config.yaml",
            "workflow_config.yml",
            "workflow_config.json",
        ):
            candidate = path / candidate_name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No workflow configuration file (workflow_config.yaml, workflow_config.yml, "
            f"or workflow_config.json) found in directory: {path}"
        )
    if not path.is_file():
        for ext in (".yaml", ".yml", ".json"):
            candidate = path.with_suffix(ext)
            if candidate.is_file():
                return candidate
    return path


def _parse_config_file(path: Path) -> dict[str, Any]:
    ext = path.suffix.lower()
    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to parse YAML workflow configuration files. "
                "Please install pyyaml with `pip install pyyaml` or `conda install pyyaml`."
            ) from exc
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    elif ext == ".json":
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
    else:
        # Fallback: attempt YAML first (YAML parser also accepts JSON and comments), then JSON
        try:
            import yaml

            with path.open(encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except Exception:
            with path.open(encoding="utf-8") as f:
                raw = json.load(f)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Workflow configuration in {path} must contain a mapping/dictionary object"
        )
    return raw


def load_workflow_config(filename: str | Path) -> dict[str, Any]:
    """Load a workflow YAML or JSON file, apply defaults, and resolve relative paths.

    Supports:
        - YAML (.yaml, .yml) with inline/block comments (#)
        - JSON (.json) for backward compatibility
        - Directory paths containing workflow_config.yaml / .yml / .json
        - Path stems without file extension
    """
    resolved_path = _resolve_config_path(filename)
    raw_dict = _parse_config_file(resolved_path)

    config: dict[str, Any] = {
        section: _merge(DEFAULT_CONFIG[section], raw_dict.get(section, {}), section)
        for section in DEFAULT_CONFIG
    }
    config["config_file"] = resolved_path
    for key, value in config["paths"].items():
        if key == "output_mesh":
            continue
        value = _select_platform_path(value)
        if value:
            p = Path(value).expanduser()
            if not p.is_absolute():
                p = resolved_path.parent / p
            config["paths"][key] = str(p.resolve())
        else:
            config["paths"][key] = ""
    if not config["workflow"]["date"]:
        from datetime import datetime

        config["workflow"]["date"] = datetime.now().strftime("%Y%m%d")
    return config


def dump_workflow_config(
    config: dict[str, Any],
    filename: str | Path,
    fmt: str | None = None,
) -> None:
    """Dump a workflow configuration dictionary to a YAML or JSON file.

    Args:
        config (dict[str, Any]): Workflow configuration dictionary.
        filename (str | Path): Destination file path.
        fmt (str | None): Target format ('yaml', 'yml', or 'json'). If None,
            inferred from filename suffix (defaults to 'yaml').
    """
    dest = Path(filename).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if fmt is None:
        ext = dest.suffix.lower()
        if ext in (".yaml", ".yml"):
            fmt = "yaml"
        elif ext == ".json":
            fmt = "json"
        else:
            fmt = "yaml"
    else:
        fmt = fmt.lower()

    # Clean serializable dictionary: exclude internal 'config_file' and convert Path to str
    export_dict: dict[str, Any] = {}
    for section, sec_data in config.items():
        if section == "config_file":
            continue
        if isinstance(sec_data, dict):
            export_dict[section] = {
                k: str(v) if isinstance(v, Path) else v for k, v in sec_data.items()
            }
        else:
            export_dict[section] = (
                str(sec_data) if isinstance(sec_data, Path) else sec_data
            )

    if fmt in ("yaml", "yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to write YAML workflow configuration files. "
                "Please install pyyaml with `pip install pyyaml`."
            ) from exc
        with dest.open("w", encoding="utf-8") as f:
            yaml.safe_dump(export_dict, f, sort_keys=False, default_flow_style=False)
    elif fmt == "json":
        with dest.open("w", encoding="utf-8") as f:
            json.dump(export_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported format {fmt!r}. Expected 'yaml' or 'json'.")


# ==============================================================================
# Workflow factory
#
# Each mesh type owns a class in the ``pyearthmesh/config/workflows/`` subpackage.
# ``create_workflow`` loads the single JSON (if given a path), selects the class
# registered for ``workflow.mesh_type``, and returns an instantiated workflow
# whose underlying case object has already been built in memory.
# ==============================================================================

from pyearthmesh.config.workflows import (  # noqa: E402
    HealpixWorkflow,
    ISEAWorkflow,
    LatLonWorkflow,
    MPASWorkflow,
    RHEALPixWorkflow,
)
from pyearthmesh.config.workflows.base import WorkflowSpec  # noqa: E402

_WORKFLOWS = {
    "healpix": HealpixWorkflow,
    "isea": ISEAWorkflow,
    "latlon": LatLonWorkflow,
    "mpas": MPASWorkflow,
    "rhealpix": RHEALPixWorkflow,
}


def register_workflow(mesh_type: str, workflow_class) -> None:
    """Register a workflow class for a mesh type."""
    _WORKFLOWS[mesh_type.lower()] = workflow_class


def get_available_workflow_types():
    """Return the list of mesh types that have a registered workflow class."""
    return list(_WORKFLOWS.keys())


def _workflow_spec_from_config(config: dict[str, Any], mesh_type: str) -> WorkflowSpec:
    """Normalize a loaded sectioned workflow config into the frontend spec."""
    workflow = config.get("workflow", {}) or {}
    paths = config.get("paths", {}) or {}
    region = dict(config.get("region", {}) or {})
    region.setdefault("geometry", paths.get("region_geometry"))

    return WorkflowSpec.from_dict(
        {
            "mesh_type": mesh_type,
            "case_index": workflow.get("case_index", 1),
            "date": workflow.get("date"),
            "region": region,
            "resolution": config.get("resolutions", {}) or {},
            "paths": paths,
            "execution": {
                "debug": workflow.get("debug", False),
                "model": workflow.get("model", "jigsaw"),
                "run_on_hpc": workflow.get("run_on_hpc", False),
            },
            "features": {
                **(config.get("features", {}) or {}),
                "jigsaw": config.get("jigsaw", {}) or {},
                "coastline": workflow.get("process_coastline", False),
                "simplify_hydrosheds_river_network": workflow.get(
                    "simplify_hydrosheds_river_network", False
                ),
            },
        }
    )


def compile_workflow(spec_or_config):
    """Compile a generic workflow spec or config dict into the backend-native config.

    This is the two-tier compiler boundary: user-facing workflow data is converted
    into the mesh-family-specific config expected by the backend generator.
    """
    if isinstance(spec_or_config, (str, Path)):
        spec_or_config = load_workflow_config(spec_or_config)

    if isinstance(spec_or_config, dict):
        mesh_type = str(
            spec_or_config.get("workflow", {}).get("mesh_type")
            or spec_or_config.get("mesh_type")
            or ""
        ).lower()
        if not mesh_type:
            raise ValueError("mesh_type is required to compile a workflow")
        workflow_cls = _WORKFLOWS.get(mesh_type)
        if workflow_cls is None:
            raise ValueError(
                f"Unknown mesh_type {mesh_type!r}. Available: {', '.join(sorted(_WORKFLOWS))}"
            )
        if "workflow" in spec_or_config:
            return workflow_cls.compile_from_spec(
                _workflow_spec_from_config(spec_or_config, mesh_type)
            )
        return workflow_cls.compile_from_spec(spec_or_config)

    if isinstance(spec_or_config, WorkflowSpec):
        mesh_type = spec_or_config.mesh_type
        workflow_cls = _WORKFLOWS.get(mesh_type)
        if workflow_cls is None:
            raise ValueError(
                f"Unknown mesh_type {mesh_type!r}. Available: {', '.join(sorted(_WORKFLOWS))}"
            )
        return workflow_cls.compile_from_spec(spec_or_config)

    raise TypeError("Expected a workflow spec, config dictionary, or path")


def create_workflow(config_or_path):
    """Build the workflow object for a loaded config dict or a YAML/JSON file path.

    Args:
        config_or_path (dict | str | Path): Either an already-loaded config
            (as returned by ``load_workflow_config``), a path to a workflow
            YAML/JSON file, or a directory containing a workflow configuration.

    Returns:
        BaseWorkflow: An instantiated workflow object with its case built.

    Raises:
        ValueError: If ``workflow.mesh_type`` has no registered workflow class.
    """
    if isinstance(config_or_path, (str, Path)):
        config = load_workflow_config(config_or_path)
    else:
        config = config_or_path

    mesh_type = str(config.get("workflow", {}).get("mesh_type", "")).lower()
    if mesh_type not in _WORKFLOWS:
        raise ValueError(
            f"Unknown mesh_type {mesh_type!r}. "
            f"Available: {', '.join(sorted(_WORKFLOWS))}"
        )
    return _WORKFLOWS[mesh_type](config)
