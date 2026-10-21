# Two-Tier Compiler Pattern for Multi-Mesh Workflow Architecture

> **Status: ACTIVE SPECIFICATION (CORE ARCHITECTURE IMPLEMENTED)**
> - **Core Compiler Boundary:** Implemented via [`compile_workflow()`](../pyearthmesh/config/config_manager.py) and [`WorkflowSpec`](../pyearthmesh/config/workflows/base.py).
> - **Implemented Workflow Adapters:** `MPASWorkflow`, `LatLonWorkflow`, `HealpixWorkflow`, `ISEAWorkflow`, `RHEALPixWorkflow`.
> - **Test Suite:** Passing in [`tests/test_compile_workflow.py`](../tests/test_compile_workflow.py) and [`tests/test_workflow_spec.py`](../tests/test_workflow_spec.py).
> - **Future Extensions:** Workflow classes for square, hexagon, tin, dggrid meshes can be added using the same compiler boundary.

## Purpose

This document proposes a two-tier compiler architecture for `pyearthmesh` that is designed for the real scope of the package: multiple mesh families, not only MPAS.

The package currently supports multiple mesh types, including:

- structured or regular-grid families such as lat-lon, square, hexagon, triangular
- spherical/pixel-based families such as HEALPix, rHEALPix, Cubic Sphere
- unstructured families such as JIGSAW, MPAS, TIN, DGGRID, ISEA

A robust design must therefore be mesh-family aware rather than assuming a single backend-specific workflow.

The design follows the principle:

- Tier 1: high-level domain specification
- Tier 2: compiled backend execution model

This keeps the user-facing workflow readable while preserving compatibility with many engine-specific mesh generators and runtime execution patterns.

---

## 1. Design goal

The package should support multiple mesh families using a shared workflow model and a common compiler boundary.

The key requirement is not merely “many mesh types,” but a clean separation of concerns:

1. The user specifies the domain and mesh intent.
2. The framework selects the mesh family and compiles that intent into a canonical internal mesh specification.
3. The backend executes that compiled model.

This is the scientific-modeling equivalent of a frontend + compiler + backend pipeline.

---

## 2. Mesh families in this project

The architecture should be designed around the fact that the package already groups mesh types into distinct families.

### 2.1 Structured / regular-grid meshes

Examples:
- latlon
- square
- hexagon
- triangular

These typically encode:
- grid origin
- resolution in degrees or metres
- domain bounds
- regular cell layout
- optional region boundary

### 2.2 Spherical / pixel meshes

Examples:
- healpix
- rhealpix
- cubicsphere

These typically encode:
- cell resolution in metres
- tiling mode or pre-defined spherical partitioning
- global or regional boundary conditions
- output format and tiling policy

### 2.3 Unstructured / geometry-driven meshes

Examples:
- jigsaw
- mpas
- tin
- dggrid
- isea

These typically encode:
- boundary geometry in WKT or vector data
- feature flags for coastline, rivers, dams, etc.
- grid generation rules and region constraints
- backend-specific optional preprocessing

This variety is the reason the architecture cannot be MPAS-only or even JIGSAW-only. The compiler layer must be generic enough to support all of these patterns.

---

## 3. The core idea: a generic two-tier workflow

We define a two-tier system:

### Tier 1: Frontend / domain specification

This layer expresses the scientific and workflow intent in a human-readable form.

Examples:
- mesh type: `mpas`, `latlon`, `healpix`, `isea`
- region geometry: GeoJSON boundary
- resolution policy: degrees, metres, or km
- feature processing: coastline, rivers, dams
- output workspace and case metadata
- HPC policy
- date and case index

This layer should be declarative and mesh-agnostic. It should answer:

> “What does the user want the mesh to represent?”

### Tier 2: Backend / compiled execution model

This layer translates the specification into the exact internal model that the chosen backend understands.

Examples:
- flat config keys for a specific generator
- boundary WKT for MPAS or other geometry-driven mesh families
- regular-grid parameters for Lat-Lon
- tiling and spherical partition settings for HEALPix
- runtime job metadata and output file naming

This layer should answer:

> “How does the chosen backend actually run this?”

A runtime object such as `meshcase` or the per-mesh case object should be a thin execution wrapper around the compiled model, not the source of truth for the domain specification.

---

## 4. Why the current design feels redundant

The current code path mixes three layers together:

1. high-level workflow dict
2. compiled flat config dict
3. runtime case object state

This is especially visible in the MPAS path, but it is not limited to MPAS. The same pattern would happen for other mesh types if we keep building them through raw generator-specific config dicts.

Typical duplicates include:

- `workflow.mesh_type` and `aConfig["mesh_type"]`
- `paths.output_workspace` and `sWorkspace_output`
- `workflow.case_index` and `iCase_index`
- `workflow.date` and `sDate`
- `workflow.model` and `sModel`

The duplication is not random; it reflects a transition between:

- legacy flat config style
- sectioned workflow style
- runtime execution objects

The clean architecture should make these layers explicit and prevent the same fact from being redeclared multiple times.

---

## 5. Proposed architecture for the multi-mesh repository

### 5.1 Layered structure

```text
User workflow JSON
        |
        v
Frontend mesh specification
        |
        v
Mesh-family compiler registry
        |
        +--> regular-grid compiler
        +--> spherical-grid compiler
        +--> unstructured-mesh compiler
        |
        v
Canonical MeshSpec / CaseSpec
        |
        v
Backend adapter / runtime case
        |
        +--> LatLon case
        +--> HEALPix case
        +--> MPAS case
        +--> JIGSAW case
        +--> etc.
```

### 5.2 Responsibilities by layer

#### Frontend

Responsible for:
- accepting a JSON or Python dict specification
- validating section keys and values
- resolving platform-specific paths
- defaulting optional values
- preserving scientific meaning across mesh families

This layer should remain user-facing and easy to edit by hand.

#### Mesh-family compiler

Responsible for:
- selecting the correct mesh family or backend
- validating compatibility between the high-level spec and the actual mesh family
- converting user intent into a canonical internal mesh spec
- generating mesh-family-specific parameters and flags

This is the true compiler stage.

#### Backend runtime

Responsible for:
- creating output directories and file names
- invoking mesh-generation APIs or executables
- creating HPC job scripts when needed
- capturing execution metadata
- interacting with backend-specific APIs

This layer should be mostly procedural and operational.

---

## 6. Canonical internal model

We should introduce a canonical internal representation that is mesh-family neutral but rich enough to support all generator types.

A good shape is:

```python
class MeshSpec:
    mesh_type: str
    family: str            # "regular", "spherical", "unstructured"
    case_index: int
    date: str | None
    output_workspace: str
    region_geometry: str | None
    boundary_wkt: str | None
    resolutions: dict[str, float]
    feature_flags: dict[str, bool]
    execution: dict[str, Any]
    paths: dict[str, str]
    backend_params: dict[str, Any]
```

This object is the compiler output and should be the single authoritative compiled representation.

Important: `backend_params` is intentionally generic. For example:

- a regular-grid mesh may use `dResolution_degree`
- a spherical mesh may use `nside` or tiling metadata
- an unstructured mesh may use `pBoundary_wkt`, river flags, or coastline processing flags

The compiler is responsible for normalizing those values into a shared spec while preserving generator-specific data in `backend_params`.

---

## 7. Mesh-family compiler strategy

The project should not force every mesh type through a single one-size-fits-all workflow class. Instead, the workflow should be organized by family and/or backend.

### 7.1 Generic workflow registry

```python
_WORKFLOWS = {
    "mpas": MPASWorkflow,
    "jigsaw": JigsawWorkflow,
    "latlon": LatLonWorkflow,
    "healpix": HealpixWorkflow,
    "rhealpix": RHEALPixWorkflow,
    "isea": ISEAWorkflow,
    "square": SquareWorkflow,
    "hexagon": HexagonWorkflow,
    "triangular": TriangularWorkflow,
    "dggrid": DGGRIDWorkflow,
    "tin": TINWorkflow,
}
```

This keeps the API generic while allowing each mesh type or mesh family to own its own validation and translation logic.

### 7.2 Family-specific base classes

A more structured design is to allow a small family hierarchy:

```python
class BaseWorkflow(ABC):
    mesh_type: str
    family: str

class RegularGridWorkflow(BaseWorkflow):
    family = "regular"

class SphericalGridWorkflow(BaseWorkflow):
    family = "spherical"

class UnstructuredMeshWorkflow(BaseWorkflow):
    family = "unstructured"
```

Then mesh-specific classes inherit from the appropriate family base class.

This is a better fit for the actual diversity in the repository than a single flat workflow abstraction.

---

## 8. Recommended class responsibilities

### 8.1 Frontend config loader

File examples:
- `pyearthmesh/config/config_manager.py`

Responsibilities:
- load JSON
- merge defaults
- reject unknown keys
- resolve relative and platform-specific paths
- set default date if missing

This should remain generic and mesh-type neutral.

### 8.2 BaseWorkflow

File examples:
- `pyearthmesh/config/workflows/base.py`

Responsibilities:
- define the workflow interface
- shared helpers such as distance conversion
- common orchestration
- delegation to backend runtime objects

### 8.3 Mesh-family workflow classes

Examples:
- `MPASWorkflow` for unstructured geometry-driven meshes
- `LatLonWorkflow` for regular grids
- `HealpixWorkflow` for spherical pixel meshes
- `JigsawWorkflow` for JIGSAW-based mesh generation

Responsibilities:
- validate mesh-specific requirements
- compile the frontend config into `MeshSpec`
- attach generator-specific parameters in `backend_params`
- build the backend runtime case object

### 8.4 Runtime case classes

Examples:
- `meshcase`
- mesh-specific case classes for MPAS, JIGSAW, or other generators

Responsibilities:
- file layout
- case naming
- execution lifecycle
- output directories
- HPC job generation

These should consume the compiled spec instead of owning scientific intent.

---

## 9. Data flow in the multi-mesh design

### Ideal flow

```text
JSON workflow
  -> load_workflow_config
  -> WorkflowFactory.select(mesh_type)
  -> mesh-family compiler
  -> MeshSpec
  -> backend adapter
  -> runtime case object
  -> setup/run/HPC job generation
```

### Required property

The `MeshSpec` should be the only internally compiled representation. Everything else should read from it or derive from it.

---

## 10. Compatibility strategy

We do not need to remove all legacy behavior in one step. The migration can be incremental.

### Phase 1: introduce the compiler boundary

- keep the current JSON workflow input format
- add a canonical `MeshSpec`
- update workflow classes to build the compiled model
- keep current runtime case objects functional

### Phase 2: make `MeshSpec` canonical

- update backend execution to read from `MeshSpec`
- remove direct dependence on raw workflow dicts inside runtime code

### Phase 3: deprecate legacy flat config paths

- remove the old “intermediate generated config” pattern from the primary execution path
- keep compatibility shims only where necessary

---

## 11. Design principles

1. One source of truth for domain intent
2. One canonical compiled representation for all mesh types
3. Mesh-family-specific validation stays local to the relevant workflow class
4. Runtime objects should not re-define scientific intent
5. Backend-specific details stay in backend adapters
6. Adding a new mesh type should require a workflow implementation and registry entry, not a rewrite of the whole system

---

## 12. Acceptance criteria

The design is successful when:

- every supported mesh type has a mesh-specific or mesh-family-specific workflow class,
- JSON config remains the frontend specification,
- there is one canonical compiled representation,
- runtime execution objects do not duplicate user-level mesh semantics,
- new mesh types can be added via registry and compiler extension,
- no intermediate generated config is required for the primary execution path.

---

## 13. Recommended direction for this project

The correct architecture for `pyearthmesh` is not MPAS-centric. It is a multi-mesh compiler pipeline.

The right target is:

- frontend config manager
- mesh-family-aware workflow compiler
- canonical compiled mesh specification
- backend-specific runtime execution layer

That is the robust scientific-modeling pattern for a project that supports many mesh types.

## 8. Compatibility strategy

We do not need to remove all legacy behavior in one step. The migration can happen in phases.

### Phase 1: introduce the compiler boundary

- keep the existing JSON config format
- add a `MeshSpec` object
- update workflow classes to build the compiled model
- leave runtime case objects intact

### Phase 2: make `MeshSpec` canonical

- update backend execution to read from `MeshSpec`
- remove direct dependence on raw workflow dicts in runtime code

### Phase 3: deprecate legacy flat config paths

- remove the old “intermediate generated config” pattern from the workflow lifecycle
- keep compatibility shims only if needed for old scripts

---

## 9. Design principles

1. One source of truth for scientific intent
2. One compiled internal representation
3. Backend-specific details should stay in backend adapters
4. Runtime objects should not duplicate user-level config semantics
5. Mesh-specific logic should be isolated per mesh type
6. Support for more mesh types should be registry-driven, not hardcoded in workflow logic

---

## 10. Proposed implementation plan

### Step 1: define the compiler contract

Add a `MeshSpec` dataclass or simple class with the canonical fields.

### Step 2: refactor `BaseWorkflow`

Move workflow interpretation into a clear compile step:

```python
class BaseWorkflow:
    def compile(self) -> MeshSpec:
        ...
```

### Step 3: update `MPASWorkflow`

Refactor `build_case()` so it builds a `MeshSpec` first, then uses that to construct the runtime case object.

### Step 4: make runtime objects adapter-only

The runtime case should read from the compiled spec, not re-create independent science-level settings.

### Step 5: add a registry for future mesh types

The factory should register supported mesh classes without embedding all logic into a single `if/elif` chain.

---

## 11. Acceptance criteria

The design is successful when:

- each mesh type owns a workflow class,
- JSON config remains the front-end specification,
- there is a single canonical compiled representation,
- runtime execution objects do not redefine domain intent,
- adding a new mesh type requires only a mesh-specific workflow and backend adapter,
- there is no intermediate generated config file in the primary execution path.

---

## 12. Recommended direction for this project

The right target architecture for `pyearthmesh` is:

- Frontend config manager
- mesh-type workflow compiler
- canonical compiled mesh specification
- mesh-specific runtime execution backend

This is the correct analog of a scientific compiler pipeline and is the design that scales well across many mesh types.
