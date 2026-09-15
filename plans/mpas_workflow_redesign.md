# MPAS Workflow Redesign — Design Document

> **Revision 3.** The System A "flat mesh-config template" pipeline is **dropped**.
> A **single JSON file** (`workflow_config.json`) is the one and only source of truth,
> and **each mesh type owns a class** that interprets that JSON and drives the run.
> `workflow_config.py` is **merged into `config_manager.py`**, which becomes the single
> consolidated configuration module (loader + registry + factory).

## 1. Purpose

Redesign the mesh-generation workflow so that:

- **One** human-editable JSON file — [`examples/mpas/workflow_config.json`](../examples/mpas/workflow_config.json)
  — fully describes a run (no intermediate generated config, no hardcoded values).
- [`pyearthmesh/config/config_manager.py`](../pyearthmesh/config/config_manager.py)
  becomes the **single consolidated config module**: it absorbs
  [`workflow_config.py`](../pyearthmesh/config/workflow_config.py) (`load_workflow_config`,
  `DEFAULT_CONFIG`, `_merge`, `_select_platform_path`) and hosts the workflow registry and
  `create_workflow` factory. `load_workflow_config` remains the loader that resolves
  platform-specific and relative paths and normalises the run date. `workflow_config.py`
  is deleted after the merge.
- **Each mesh type has its own class** (e.g. `MPASWorkflow`) that validates the config
  for that mesh type, builds the underlying case object, and orchestrates
  `setup()` / `run()` / HPC-job generation.
- [`examples/mpas/mpas_example.py`](../examples/mpas/mpas_example.py) becomes a thin
  script: *load JSON → build the workflow class → execute*.

---

## 2. Current State

The repository has **two independent configuration systems**.

### 2.1 System A — flat mesh-config pipeline (to be DROPPED)

- **Producer:** `create_template_configuration_file(...)` in
  [`pyearthmesh/config/config_manager.py`](../pyearthmesh/config/config_manager.py),
  delegating to `MPASConfig.create_template_config` in
  [`pyearthmesh/classes/meshconfig.py`](../pyearthmesh/classes/meshconfig.py).
- **Artefact:** an intermediate *flat* JSON (`mpas_configuration.json`, plus a
  `mpas_configuration_copy.json` duplicate) with keys like `iCase_index`,
  `iFlag_run_jigsaw`, `dResolution_land`, `pBoundary_wkt`, `sWorkspace_output`.
- **Consumer:** `read_configuration_file(...)` builds an
  [`mpascase`](../pyearthmesh/meshes/unstructured/mpas/mpascase.py).

**Why drop it:** it introduces a second, generated config file (a second source of
truth), a copy-and-reload workaround, and a flat vocabulary that duplicates what the
sectioned workflow JSON already expresses.

### 2.2 System B — sectioned workflow config (to be PROMOTED to the single source)

- **Data:** `workflow_config.json` — sections `workflow`, `resolutions`, `paths`,
  `jigsaw`.
- **Loader:** `load_workflow_config(filename)` — merges `DEFAULT_CONFIG`, rejects
  unknown keys, resolves platform-specific path objects, makes relative paths absolute
  relative to the JSON, fills in `date`. Currently in `workflow_config.py`; **moved into
  `config_manager.py`** by this redesign (§5.4).
- **Status today:** `load_workflow_config` is **never called** (verified by search).

### 2.3 Problems to solve

1. Two sources of truth; the generated flat file is redundant.
2. Run-critical values (case index, date, workspace, HPC settings) are hardcoded in the
   example.
3. No class owns the end-to-end lifecycle for a mesh type — logic is split between the
   example script, `config_manager`, and `mpascase`.
4. Unit mismatch: workflow `resolutions` are in **km**, JIGSAW `dResolution_*` defaults
   are in **degrees** (see comments in `JigsawConfig.get_default_config`).
5. `paths.region_geometry` is a GeoJSON path, but `mpascase` expects `pBoundary_wkt`
   (a WKT *string*, consumed via `ogr.CreateGeometryFromWkt`).

---

## 3. Goals

- **G1.** A single JSON file drives the whole run; nothing run-critical is hardcoded.
- **G2.** Remove the System A artefacts: `create_template_configuration_file`,
  the intermediate `mpas_configuration.json` / `mpas_configuration_copy.json`, and
  `read_configuration_file` as the primary entry point.
- **G3.** Introduce a per-mesh-type **workflow class** hierarchy with a registry and a
  factory, so `mesh_type` in the JSON selects the class.
- **G4.** Each class owns: schema defaults, validation, translation of the sectioned
  config into the parameters its case/creator needs, and run orchestration.
- **G5.** Derive `pBoundary_wkt` from `paths.region_geometry` inside the MPAS class.
- **G6.** Drive local-vs-HPC execution and HPC settings (`hours`, `slurm`) from config.
- **G7.** Keep the design mesh-agnostic: MPAS is the first implementation, but JIGSAW,
  HEALPix, ISEA, LatLon, etc. slot into the same registry.

## 4. Non-Goals

- Not changing `create_mpas_mesh` / JIGSAW internals (only how they are *fed*).
- Not adding new mesh types in this pass (the registry must accommodate them later).

---

## 5. Proposed Architecture

```
workflow_config.json   (ONE file, sectioned, single source of truth)
        │
        ▼
load_workflow_config()                 [pyearthmesh/config/config_manager.py]
   • resolve platform/relative paths    – loader stays thin & generic
   • fill run date                       (merged in from workflow_config.py)
        │  returns dict {workflow, resolutions, paths, jigsaw, config_file}
        ▼
create_workflow(config)                ◄── NEW factory (registry lookup on mesh_type)
        │
        ▼
BaseWorkflow (ABC)
   ├── MPASWorkflow        ◄── per-mesh-type class
   ├── JigsawWorkflow
   ├── HealpixWorkflow
   └── ...
        │  each class:
        │   • default_config() / validate()      (mesh-specific schema)
        │   • build_case()  → translate sections → case object (in memory)
        │   • setup() / run() / create_hpc_job()
        ▼
   MPASWorkflow.run()            (local: setup + run)
   MPASWorkflow.create_hpc_job() (HPC: generate scripts from jigsaw.hours/slurm)
```

**Key idea:** the single JSON is loaded once, handed to a factory that picks the
mesh-type class, and that class translates the sectioned config **in memory** into
whatever its underlying case object / creator function needs. No intermediate config
file is ever written.

### 5.1 `BaseWorkflow` interface

```python
class BaseWorkflow(ABC):
    mesh_type: str = None

    def __init__(self, config: dict):
        self.config = config                 # already path-resolved by the loader
        self.validate()

    @classmethod
    @abstractmethod
    def default_config(cls) -> dict: ...      # mesh-specific defaults (replaces *Config)

    @abstractmethod
    def validate(self) -> None: ...           # raise on missing/invalid/incompatible keys

    @abstractmethod
    def build_case(self): ...                 # construct the underlying case object

    def setup(self): ...                       # default: delegate to case object
    def run(self): ...                         # default: delegate to case object
    def create_hpc_job(self, slurm, hours): ...# default: delegate to case object
```

### 5.2 `MPASWorkflow`

- `mesh_type = "mpas"`.
- `default_config()` — MPAS/JIGSAW-relevant defaults (reuse the values already defined
  in `MPASConfig` / `JigsawConfig` so that work is not lost, but now expressed against
  the sectioned schema).
- `build_case()` — translate the sectioned config (§6) into the flat dict that
  [`mpascase`](../pyearthmesh/meshes/unstructured/mpas/mpascase.py) already consumes,
  derive `pBoundary_wkt` (§7), apply km→degree conversion (§8), then construct
  `mpascase` **in memory** (no file round-trip). `mpascase` internals stay unchanged.
- `setup()/run()/create_hpc_job()` — delegate to the `mpascase` instance.

### 5.3 Registry + factory

```python
_WORKFLOWS = {
    "mpas":    MPASWorkflow,
    "jigsaw":  JigsawWorkflow,
    # "healpix": HealpixWorkflow, ...
}

def create_workflow(config_or_path):
    cfg = (load_workflow_config(config_or_path)
           if isinstance(config_or_path, (str, Path)) else config_or_path)
    mesh_type = cfg["workflow"]["mesh_type"].lower()
    if mesh_type not in _WORKFLOWS:
        raise ValueError(f"Unknown mesh_type {mesh_type!r}; "
                         f"available: {sorted(_WORKFLOWS)}")
    return _WORKFLOWS[mesh_type](cfg)
```

> This registry **replaces** the `ConfigManager._mesh_configs` role for run
> orchestration. Decide whether `ConfigManager` is retired entirely or kept only as a
> defaults library reused by the workflow classes (see §10).

### 5.4 Consolidated config module (`config_manager.py`)

`workflow_config.py` is merged into `config_manager.py` so there is exactly one config
module. After the merge, `config_manager.py` contains:

```python
# --- moved verbatim from workflow_config.py ---
DEFAULT_CONFIG = {...}
def _merge(defaults, values, section): ...
def _select_platform_path(value): ...
def load_workflow_config(filename): ...

# --- new orchestration (replaces System A) ---
_WORKFLOWS = {"mpas": MPASWorkflow, "jigsaw": JigsawWorkflow, ...}
def create_workflow(config_or_path): ...
```

Consequences:
- `workflow_config.py` is **deleted**; all imports point to
  `pyearthmesh.config.config_manager`.
- System A functions (`create_template_configuration_file`, `read_configuration_file`)
  are removed from `config_manager.py` (or kept as a thin shim — see §10 Decision 2).
- `BaseWorkflow` / `MPASWorkflow` live in a `pyearthmesh/config/workflows/` subpackage
  (see §5.5), imported by the registry here; `config_manager.py` stays focused on the
  loader + registry + factory.

### 5.5 Module layout (workflows/ subpackage)

```
pyearthmesh/config/
├── config_manager.py          # load_workflow_config + DEFAULT_CONFIG/_merge/
│                              #   _select_platform_path + _WORKFLOWS registry
│                              #   + create_workflow factory (imports the classes below)
└── workflows/
    ├── __init__.py            # re-exports BaseWorkflow, MPASWorkflow, …
    ├── base.py                # BaseWorkflow (ABC) — §5.1
    └── mpas.py                # MPASWorkflow — §5.2 (boundary §7, units §8)
```

- `config_manager.py` imports the workflow classes from `pyearthmesh.config.workflows`
  to populate `_WORKFLOWS`; the classes do **not** import `config_manager` (avoids a
  circular import — the factory calls `load_workflow_config`, the classes receive an
  already-loaded dict).
- Adding a new mesh type = drop a module in `workflows/` and register it in `_WORKFLOWS`.

---

## 6. Field Mapping (inside each workflow class)

`build_case()` maps the sectioned config onto the parameters the case object needs.

### 6.1 `workflow` section

| workflow_config.json | Target (mpascase / mesh config) | Notes |
|---|---|---|
| `date` | `sDate` | normalised by the loader |
| `mesh_type` | selects the workflow class | `'mpas'` |
| `case_index` | `iCase_index` | |
| `model` | `sModel` | `'jigsaw'` |
| `debug` | `verbosity` | `True → 1`, `False → 0` |
| `process_coastline` | `iFlag_geom_coastline` / `iFlag_spac_coastline` | bool → 0/1 |
| `simplify_hydrosheds_river_network` | *(pre-processing flag)* | river prep step, not mesh config |
| `largest_outlets` | *(pre-processing flag)* | outlet-selection step |

### 6.2 `resolutions` section (km → degrees)

| workflow_config.json | Target | Conversion |
|---|---|---|
| `land_km` | `dResolution_land` | km → degrees (§8) |
| `river_network_km` | `dResolution_river_network` | km → degrees |
| `coastline_km` | `dResolution_coastline` | km → degrees |
| `ocean_km` | `dSpac_value` (or a dedicated ocean key) | km → degrees |

### 6.3 `paths` section

| workflow_config.json | Target | Notes |
|---|---|---|
| `output_workspace` | `sWorkspace_output` | already absolute (loader) |
| `input_workspace` | base for derived input paths | |
| `region_geometry` | `pBoundary_wkt` | GeoJSON → WKT string (§7) |
| `hydrosheds_rivers` | `sFilename_river_network_vector` | |
| `dam_vector` | `sFilename_dam_vector` | empty → `None` |

### 6.4 `jigsaw` section

| workflow_config.json | Target | Notes |
|---|---|---|
| `standalone` | `iFlag_standalone` | bool → 0/1 |
| `create_directory` | `iFlag_create_directory` | bool → 0/1 |
| `hours` | `hours` for `create_hpc_job` | |
| `slurm` | `slurm` for `create_hpc_job` | |

> **Execution mode.** The current example hardcodes `iFlag_hpc = 1`. Add an explicit
> key (e.g. `workflow.run_on_hpc`) or infer it. See §10.

---

## 7. Boundary Handling (`region_geometry` → `pBoundary_wkt`)

`mpascase.run()` passes `self.pBoundary_wkt` into `create_mpas_mesh`, which calls
`ogr.CreateGeometryFromWkt(...)`. `MPASWorkflow.build_case()` must therefore supply a
**WKT string** (not a path, not a live geometry — per the GDAL "geometry crashes after
dataset close" gotcha already noted in the code).

```python
def _boundary_wkt_from_geojson(sFilename_region):
    if not sFilename_region:
        return None
    from osgeo import ogr
    ds = ogr.Open(sFilename_region)
    layer = ds.GetLayer()
    geoms = [feat.GetGeometryRef().Clone() for feat in layer]
    boundary = geoms[0] if len(geoms) == 1 else _union_all(geoms)
    wkt = boundary.ExportToWkt()
    ds = None            # close AFTER exporting WKT
    return wkt
```

- Empty / missing `region_geometry` → `pBoundary_wkt = None` → global mesh
  (`iFlag_global = 1`).

---

## 8. Unit Conversion (km → degrees)

JIGSAW `dResolution_*` are in **degrees**; workflow `resolutions` are in **km**.
Convert inside `MPASWorkflow` (one place, documented):

- **Option A — latitude approximation:** `degrees = km / 111.32`.
- **Option B — reuse an existing project helper** (e.g. in `convert_coordinates` /
  `mesh_utility`) for consistency.

See §10 for the decision.

---

## 9. Revised `mpas_example.py` (target)

```python
import os
from pyearthmesh.config.config_manager import load_workflow_config, create_workflow

# 1. One file, one load — nothing hardcoded
sFilename_workflow = os.path.join(os.path.dirname(__file__), "workflow_config.json")
cfg = load_workflow_config(sFilename_workflow)

# 2. Factory picks MPASWorkflow from cfg["workflow"]["mesh_type"]
workflow = create_workflow(cfg)          # validates + builds the mpascase in memory

# 3. Execute — mode & HPC settings come from config
jigsaw_cfg = cfg["jigsaw"]
if cfg["workflow"].get("run_on_hpc"):
    workflow.create_hpc_job(slurm=jigsaw_cfg["slurm"], hours=jigsaw_cfg["hours"])
else:
    workflow.setup()
    aMpas = workflow.run()
    if aMpas is not None:
        print(f"MPAS mesh created: {len(aMpas)} cells -> {workflow.case.sFilename_mesh}")
```

The `create_template_configuration_file` call, the `mpas_configuration.json` /
`mpas_configuration_copy.json` files, both `read_configuration_file` calls, and the
`change_json_key_value` patch all disappear.

### 9.1 HPC job script generation must change

`mpascase._mpas_create_hpc_job` currently writes a `run_mpas.py` that does
`read_configuration_file(sFilename_configuration_in)` against the flat config. Since
System A is dropped, the generated driver must instead reference the **single workflow
JSON** and the factory:

```python
# generated run_mpas.py (new)
from pyearthmesh.config.config_manager import load_workflow_config, create_workflow
cfg = load_workflow_config("<abs path to workflow_config.json>")
wf = create_workflow(cfg)
wf.setup()
wf.run()
```

The workflow JSON must be copied into (or reachable from) the job directory. Decide
whether this logic lives in `BaseWorkflow.create_hpc_job` (shared) with mesh-specific
hooks, so every mesh type gets consistent HPC scripts.

---

## 10. Open Decisions

1. **Fate of `ConfigManager` / `classes/meshconfig.py`.** Retire entirely, or keep the
   `*Config.get_default_config()` values as a defaults library reused by the workflow
   classes? (Recommendation: reuse the defaults, retire the template-file generation.)
2. **System A functions inside `config_manager.py`.** After merging `workflow_config.py`
   in, remove `create_template_configuration_file` and `read_configuration_file`, or keep
   `read_configuration_file` as a thin backward-compatible shim over `create_workflow`?
3. **Execution-mode key.** Explicit `run_on_hpc` boolean vs inference.
4. **km → degree factor.** Option A (111.32) vs Option B (existing helper).
5. **`ocean_km` target.** `dSpac_value` vs a dedicated ocean-resolution key.
6. **Region dissolve.** Is `region_geometry` a single polygon or must features be unioned?
7. **Workflow-class location — RESOLVED.** Use a `pyearthmesh/config/workflows/`
   subpackage for the classes (§5.5); `config_manager.py` keeps only the loader +
   registry + factory and imports the classes from the subpackage.
8. **HPC script ownership.** Shared `BaseWorkflow.create_hpc_job` vs per-class override.
9. **Pre-processing flags.** Where `simplify_hydrosheds_river_network`,
   `process_coastline`, `largest_outlets` are consumed.

---

## 11. Implementation Steps

1. **Merge `workflow_config.py` into `config_manager.py`** (§5.4): move `DEFAULT_CONFIG`,
   `_merge`, `_select_platform_path`, `load_workflow_config`; update imports; delete
   `workflow_config.py`.
2. **Create the `pyearthmesh/config/workflows/` subpackage** (§5.5): `__init__.py` and
   `base.py` with `BaseWorkflow` (ABC, §5.1).
3. **Add `MPASWorkflow`** in `workflows/mpas.py`: `default_config()`, `validate()`,
   `build_case()` (mapping §6, boundary §7, units §8), and `setup()/run()/create_hpc_job()`
   delegating to `mpascase`.
4. **Add registry + `create_workflow` factory** in `config_manager.py`, importing the
   classes from `workflows/` (§5.3, §5.5).
5. **Update `load_workflow_config` / `DEFAULT_CONFIG`** for any new keys (e.g.
   `run_on_hpc`) so `_merge` validation passes.
6. **Update HPC script generation** (§9.1) to reference the single JSON + factory.
7. **Rewrite `mpas_example.py`** to the target in §9; delete all System A calls and
   hardcoded values.
8. **Remove/deprecate System A**: `create_template_configuration_file`,
   `read_configuration_file` (or shim), intermediate JSON handling, and the
   `change_json_key_value` patch in the example — per Decisions 1 & 2.
9. **Smoke test** both modes (HPC generates scripts only; local runs setup + run).

---

## 12. Acceptance Criteria

- [ ] Editing only `workflow_config.json` changes case index, date, workspace,
      resolutions, boundary, and HPC settings — with no source edits.
- [ ] Exactly one config file is read; **no** intermediate `mpas_configuration.json`
      (or copy) is produced.
- [ ] `create_workflow` selects `MPASWorkflow` from `workflow.mesh_type`; an unknown
      type raises a clear error.
- [ ] `MPASWorkflow.build_case()` produces a valid `mpascase` **in memory**.
- [ ] `region_geometry` yields a valid `pBoundary_wkt`; empty yields a global mesh.
- [ ] km→degree conversion is centralised in the MPAS class and documented.
- [ ] Generated HPC `run_mpas.py` loads the single workflow JSON via the factory.
- [ ] `workflow_config.py` no longer exists; `load_workflow_config` and `create_workflow`
      are both importable from `pyearthmesh.config.config_manager`.
- [ ] `create_mpas_mesh` / JIGSAW internals are unchanged.
