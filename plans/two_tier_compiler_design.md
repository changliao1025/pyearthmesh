# Two-Tier Compiler Pattern for Multi-Mesh Workflow Architecture

> **Status: ACTIVE SPECIFICATION (CORE ARCHITECTURE IMPLEMENTED & YAML-EXTENDED)**
> - **Core Compiler Boundary:** Implemented via [`compile_workflow()`](../pyearthmesh/config/config_manager.py) and [`WorkflowSpec`](../pyearthmesh/config/workflows/base.py).
> - **Specification Formats:** **YAML (primary, comment-enabled)** and **JSON (backward-compatible, programmatic)** via [`load_workflow_config()`](../pyearthmesh/config/config_manager.py) and [`dump_workflow_config()`](../pyearthmesh/config/config_manager.py).
> - **Implemented Workflow Adapters:** `MPASWorkflow`, `LatLonWorkflow`, `HealpixWorkflow`, `ISEAWorkflow`, `RHEALPixWorkflow`.
> - **Test Suite:** Passing in [`tests/test_compile_workflow.py`](../tests/test_compile_workflow.py) (including YAML loading with comments, extension resolution, roundtrip dumping, and YAML/JSON compile parity across all 5 mesh families) and [`tests/test_workflow_spec.py`](../tests/test_workflow_spec.py).
> - **Future Extensions:** Workflow classes for square, hexagon, tin, dggrid meshes can be added using the same compiler boundary.

---

## Purpose

This document specifies the two-tier compiler architecture for `pyearthmesh` designed to support multiple mesh families across structured, spherical, and unstructured spatial representations.

The package currently supports multiple mesh families:

- **Structured / Regular-Grid:** latlon, square, hexagon, triangular
- **Spherical / Pixel / DGGS:** HEALPix, rHEALPix, Cubic Sphere, ISEA3H / DGGRID
- **Unstructured / Geometry-Driven:** JIGSAW, MPAS, TIN

A robust design must be mesh-family aware rather than assuming a single backend-specific workflow.

The architecture enforces a strict separation:

- **Tier 1: High-Level Domain Specification (YAML / JSON):** User intent, spatial resolutions, boundary constraints, and execution controls.
- **Tier 2: Compiled Backend Execution Model:** Engine-specific native configuration, geometry generators, and runtime case orchestration (`setup()`, `run()`, `create_hpc_job()`).

---

## 1. Design Goal & Philosophy

Scientific workflow definitions must satisfy two competing demands:
1. **User Friendliness & Maintainability:** Domain scientists need readable configuration files that explain parameter choices (e.g. *why* a river buffer is 5 km vs. 100 km ocean, what coordinate projection is expected, or which SLURM queue is targeted) directly inline via `# comments`.
2. **Execution Rigor:** Diverse mesh generation engines (JIGSAW, MPAS-Tools, DGGRID, HEALPix libraries) require specific, non-uniform parameters, rasters, and boundary geometries.

The Two-Tier Compiler Pattern bridges this divide cleanly:

```text
User Intent (YAML with Comments)
        │
        ▼  [Tier 1: load_workflow_config / WorkflowSpec]
Domain Specification
        │
        ▼  [Compiler: compile_workflow()]
Backend-Native Configuration
        │
        ▼  [Tier 2: BaseWorkflow / BaseCase]
Execution & Mesh Generation (setup / run / HPC scripts)
```

---

## 2. Why YAML is the Primary Tier 1 Specification Format

While standard JSON is effective for machine-to-machine exchange, it possesses major shortcomings for scientific configuration:
- **No Comments Allowed:** Standard JSON strictly prohibits comments (`//` or `#`). Users cannot annotate parameters, record data sources, explain resolution tradeoffs, or temporarily toggle settings.
- **Syntax Rigidity:** Quoted keys, trailing commas, and strict bracket balancing make editing large configurations error-prone.

By adopting **YAML** as the primary human-facing format:
- **Inline and Block Comments (`# ...`):** Users can thoroughly document grids, parameter origins, and modeling decisions directly in the file.
- **Cleaner Syntax:** Whitespace-based indentation eliminates bracket clutter and improves readability.
- **Native Types & Multi-Line Strings:** Seamless representation of numbers, booleans, lists, mappings, and WKT boundary strings.
- **Full Backward Compatibility:** JSON is a formal syntactic subset of YAML; `pyearthmesh` seamlessly accepts both `.yaml`/`.yml` and `.json` files.

---

## 3. Mesh Families and Parameter Models

Mesh families have fundamentally different parameter requirements:

### 3.1 Structured / Regular-Grid Meshes (`latlon`, `square`, `hexagon`, `triangular`)
- Grid origin / domain bounding box (`dLongitude_left`, `dLatitude_top`, etc.)
- Resolution in degrees (`dResolution_degree`) or meters (`dResolution_meter`)
- Number of rows and columns (`nrow`, `ncolumn`)
- Optional polygon clipping mask

### 3.2 Spherical Pixel / DGGS Meshes (`healpix`, `rhealpix`, `isea`)
- Target cell width in meters (`dResolution_meter`)
- Partitioning / tiling mode (`tile_mode`: `'base'` vs `'simple'`, `use_tiles`: boolean)
- DGGS aperture specification (e.g., `ISEA3H`)
- Parquet tabular export and topological mesh quality checks

### 3.3 Unstructured Geometry-Driven Meshes (`mpas`, `jigsaw`, `tin`)
- Multi-scale spatial resolutions in kilometers (`ocean_km`, `land_km`, `river_network_km`, `coastline_km`, `lake_boundary_km`, `watershed_boundary_km`)
- Vector and raster guidance masks for rivers, lakes, coastlines, and basin divides
- JIGSAW generator tuning flags (`iFlag_geom`, `iFlag_spac`, `ncolumn_space`, `nrow_space`, etc.)
- HPC / SLURM execution parameters (partition, walltime hours)

---

## 4. Layered Architecture

```text
User Workflow File (workflow_config.yaml / workflow_config.json)
        │
        ▼
Frontend Config Loader: load_workflow_config()
  ├── Resolves relative paths against file location
  ├── Resolves OS-specific paths (e.g. Linux vs. default)
  ├── Strips / parses comments seamlessly
  ├── Merges with DEFAULT_CONFIG schema defaults
  └── Sets normal run date (YYYYMMDD)
        │
        ▼
Tier 1 Spec: WorkflowSpec
  ├── mesh_type: str
  ├── case_index: int
  ├── region: dict (bbox, wkt, vector)
  ├── resolution: dict (degree, meter, ocean_km, land_km, etc.)
  ├── paths: dict
  ├── execution: dict (debug, model, run_on_hpc)
  └── features: dict (jigsaw flags, tiling, parquet, etc.)
        │
        ▼
Compiler Boundary: compile_workflow(spec_or_config)
  ├── Looks up registered workflow class in _WORKFLOWS
  └── Calls WorkflowClass.compile_from_spec(spec)
        │
        ▼
Tier 2 Compiled Representation (Backend-Native Config)
  ├── latlon: dResolution_degree, nrow, ncolumn, bbox
  ├── healpix / rhealpix / isea: dResolution_meter, sISEA_type, tile_mode
  └── mpas: dResolution_ocean, dResolution_land, sWorkspace_output, jigsaw flags
        │
        ▼
Runtime Case Adapter: BaseCase (MPASCase, LatLonCase, HealpixCase, etc.)
  ├── case.output_workspace -> resolved output directory
  ├── case.mesh_file        -> target GeoJSON/NetCDF mesh path
  ├── case.setup()          -> prepares directory structure and inputs
  ├── case.run()            -> executes generator, returns MeshCell list / result
  └── case.create_hpc_job() -> emits run script and SLURM submit.job
```

---

## 5. Commented Workflow YAML Examples

Below are reference commented Tier 1 YAML configurations for each implemented mesh family.

### 5.1 Regular Lat-Lon Grid (`examples/latlon/workflow_config.yaml`)

```yaml
# ==============================================================================
# Tier 1 Specification: Regular Lat-Lon Mesh Workflow
# ==============================================================================
workflow:
  date: null                         # Run date (YYYYMMDD); null defaults to today
  mesh_type: latlon                  # Mesh family
  case_index: 1                      # Case index
  model: latlon                      # Generator model
  debug: false                       # Diagnostic logging
  run_on_hpc: false                  # Local execution

resolutions:
  degree: 0.16666666666666666        # Angular resolution (~20 km cell spacing)

paths:
  output_workspace: ../../data/qinghaihu   # Relative to config file
  output_mesh: lake_latlon_20km.geojson     # Output filename
  region_geometry: ""                      # Vector boundary mask

region:
  bbox:
    left: 98.0                       # Min longitude
    right: 101.0                     # Max longitude
    bottom: 36.0                     # Min latitude
    top: 37.5                        # Max latitude
  wkt: ""
```

### 5.2 Variable-Resolution MPAS / JIGSAW Mesh (`examples/mpas/workflow_config.yaml`)

```yaml
# ==============================================================================
# Tier 1 Specification: Variable-Resolution MPAS Unstructured Mesh Workflow
# ==============================================================================
workflow:
  date: "20260901"                                # Run date
  mesh_type: mpas                                 # Target mesh family
  case_index: 1
  model: jigsaw                                   # Backend generator
  simplify_hydrosheds_river_network: false        # River geometry simplification
  process_coastline: false                        # Coastline masking
  debug: true
  largest_outlets: 10                             # Basin outlets to retain
  run_on_hpc: false                               # Set true for SLURM batch scripts

resolutions:
  ocean_km: 100                                   # Coarse ocean cells
  land_km: 50.0                                   # Regional land background
  river_network_km: 5                             # River corridor refinement
  coastline_km: 10                                # Coastal buffer refinement
  lake_boundary_km: 5                             # Lake perimeter refinement
  watershed_boundary_km: 5                        # Watershed divide refinement

paths:
  input_workspace: /data/global/input
  output_workspace: /scratch/earthmesh/mpas
  river_network_vector: /data/vector/flowlines.geojson
  river_network_raster: /data/raster/river_networks.tif
  lake_boundary_vector: /data/vector/lakes.geojson
  lake_boundary_raster: /data/raster/lakes.tif
  watershed_boundary_raster: /data/raster/basins.tif
  coastline_raster: /data/raster/coastline.tif
  region_geometry: /data/vector/region.geojson
  dam_vector: ""

jigsaw:
  standalone: true                                # Run standalone generator
  create_directory: true                          # Auto-create case subfolder
  hours: 3                                        # SLURM walltime allocation
  slurm: cpu_single                               # SLURM partition
  ncolumn_space: 3600                             # Spacing grid lon resolution
  nrow_space: 1800                                # Spacing grid lat resolution
  iFlag_geom: true                                # Build boundary geometry (.msh)
  iFlag_spac: true                                # Build spacing field (.msh)
  iFlag_spac_ocean: true
  iFlag_spac_land: true
  iFlag_geom_river_network: true
  iFlag_spac_river_network: true
  iFlag_geom_lake_boundary: true
  iFlag_spac_lake_boundary: true
  iFlag_spac_watershed_boundary: true
  iFlag_spac_coastline: true
```

### 5.3 HEALPix Spherical Mesh (`examples/healpix/workflow_config.yaml`)

```yaml
# ==============================================================================
# Tier 1 Specification: Equal-Area HEALPix Spherical Mesh Workflow
# ==============================================================================
workflow:
  date: null
  mesh_type: healpix
  case_index: 1
  model: healpix
  debug: false
  run_on_hpc: false

resolutions:
  meter: 5000                        # Target cell diameter (5 km)

paths:
  output_workspace: ../../data/healpix/5km
  output_mesh: healpix_mesh.geojson

features:
  use_tiles: true                    # Tiled multi-file output
  tile_mode: base                    # 'base' (12 base diamonds) or 'simple'
```

### 5.4 ISEA3H Hexagonal DGGS Mesh (`examples/isea/workflow_config.yaml`)

```yaml
# ==============================================================================
# Tier 1 Specification: ISEA3H Discrete Global Grid System (DGGS) Workflow
# ==============================================================================
workflow:
  date: null
  mesh_type: isea
  case_index: 1
  model: isea
  debug: false
  run_on_hpc: false

resolutions:
  meter: 1000.0                      # Target hexagon resolution (1 km)

paths:
  output_workspace:
    Linux: /compyfs/user/earthmesh/isea3h/1km     # HPC cluster path
    default: ./output/isea3h/1km                 # Local fallback
  output_mesh: isea3h_mesh.geojson

features:
  isea_type: ISEA3H                  # Aperture 3 Hexagon
  use_tiles: false
  tile_mode: simple
  convert_to_parquet: true           # Export attributes to Parquet
  check_mesh_quality: true           # Topological & distortion checks
```

### 5.5 rHEALPix DGGS Mesh (`examples/rhealpix/workflow_config.yaml`)

```yaml
# ==============================================================================
# Tier 1 Specification: rHEALPix Discrete Global Grid System (DGGS) Workflow
# ==============================================================================
workflow:
  date: null
  mesh_type: rhealpix
  case_index: 1
  model: rhealpix
  debug: false
  run_on_hpc: false

resolutions:
  meter: 10000.0                     # Cell width (10 km)

paths:
  output_workspace:
    Linux: /compyfs/user/earthmesh/rhealpix/10km
    default: ./output/rhealpix/10km
  output_mesh: rhealpix_mesh.geojson

features:
  use_tiles: true                    # Partition across 6 base square facets
  tile_mode: base
  convert_to_parquet: true
```

---

## 6. Implementation Architecture

### 6.1 Configuration Manager (`pyearthmesh/config/config_manager.py`)

- `load_workflow_config(filename: str | Path) -> dict[str, Any]`:
  - Resolves path: if directory, automatically discovers `workflow_config.yaml`, `workflow_config.yml`, or `workflow_config.json`. If extensionless stem, checks `.yaml`, `.yml`, `.json`.
  - Parses YAML with comments using `yaml.safe_load()`.
  - Parses JSON using `json.load()`.
  - Merges loaded dictionary with `DEFAULT_CONFIG`.
  - Resolves platform-specific paths (`Linux`, `Windows`, `Darwin`, `default`).
  - Normalizes run date.
- `dump_workflow_config(config: dict, filename: str | Path, fmt: str | None = None) -> None`:
  - Exports workflow dictionary to `.yaml` or `.json`.
  - Excludes internal non-serializable fields (e.g. `config_file` object).
- `compile_workflow(spec_or_config) -> dict[str, Any]`:
  - Dispatches to registered mesh-family compiler.
- `create_workflow(config_or_path) -> BaseWorkflow`:
  - Factory function instantiating the mesh workflow with in-memory case adapter.

### 6.2 Base Classes & Adapters (`pyearthmesh/config/workflows/`)

- `WorkflowSpec`: Dataclass capturing frontend domain intent.
- `BaseWorkflow`: Base class defining `compile_from_spec()`, `build_case()`, `setup()`, `run()`, `create_hpc_job()`.
- `BaseCase`: Abstract case adapter providing `.output_workspace`, `.mesh_file`, `.setup()`, `.run()`, `.create_hpc_job()`.

---

## 7. Acceptance & Verification Criteria

1. **Comment Support:** Users can freely add `# comments` throughout workflow files without triggering parser errors.
2. **Dual-Format Parity:** Equivalent `.yaml` and `.json` workflow specifications compile to identical backend-native configs (verified via unit test `test_yaml_and_json_parity_for_examples`).
3. **Directory & Stem Auto-Discovery:** `create_workflow(dir_path)` and `create_workflow(stem_without_ext)` seamlessly resolve the configuration file.
4. **Clean Serialization:** `dump_workflow_config()` writes clean YAML or JSON files that reload identically.
5. **No Regressions:** All existing workflows (`MPASWorkflow`, `LatLonWorkflow`, `HealpixWorkflow`, `ISEAWorkflow`, `RHEALPixWorkflow`) maintain full backward compatibility.
