# PyEarthMesh Architecture Plans & Implementation Registry

This directory contains architectural specifications, implementation plans, and design documents for `pyearthmesh`. 

---

## Plan Status Matrix

| Plan / Document | Type | Implementation Status | Implementation Location | Test Verification |
| :--- | :--- | :--- | :--- | :--- |
| [**`two_tier_compiler_design.md`**](two_tier_compiler_design.md) | Architecture Spec | **Core Implemented (5 mesh families)** | [`pyearthmesh/config/config_manager.py`](../pyearthmesh/config/config_manager.py), [`pyearthmesh/config/workflows/`](../pyearthmesh/config/workflows/) | [`tests/test_compile_workflow.py`](../tests/test_compile_workflow.py), [`tests/test_workflow_spec.py`](../tests/test_workflow_spec.py) (21 tests passing) |
| [**`zarr_implementation_plan.md`**](zarr_implementation_plan.md) | Feature Roadmap | **Phases 1 & 2 Complete; Phases 3–5 Pending** | [`pyearthmesh/utility/zarr_config.py`](../pyearthmesh/utility/zarr_config.py), [`pyearthmesh/utility/zarr_io.py`](../pyearthmesh/utility/zarr_io.py), [`pyearthmesh/utility/convert_mesh_formats.py`](../pyearthmesh/utility/convert_mesh_formats.py) | Verified with [`examples/zarr/example_convert_to_zarr.py`](../examples/zarr/example_convert_to_zarr.py) |
| [**`archive/two_tier_staged_plan.md`**](archive/two_tier_staged_plan.md) | Rollout Plan | **Completed** | Full staged rollout of the two-tier compiler boundary (Steps 1–6) | Fully covered by unit tests |
| [**`archive/mpas_workflow_redesign.md`**](archive/mpas_workflow_redesign.md) | Redesign Spec | **Completed & Superseded** | JSON workflow config loader merged into `config_manager.py`, `MPASWorkflow`, `mpas_example.py` | Verified in MPAS example and tests |
| [**`archive/buffer_simplification_plan.md`**](archive/buffer_simplification_plan.md) | Feature Plan | **Completed** | [`create_coastline_buffer`](../pyearthmesh/utility/create_coastline_buffer.py) (`iFlag_simplify_in=1`) | Used in coastline buffer workflows |

---

## Active Documents

### 1. [`two_tier_compiler_design.md`](two_tier_compiler_design.md)
- **Role:** Master architecture document for the configuration and workflow subsystem.
- **Key Concepts:**
  - **Tier 1 (Frontend Spec):** [`WorkflowSpec`](../pyearthmesh/config/workflows/base.py) defines high-level domain intent (region, resolution, features, paths) independent of mesh backends.
  - **Compiler Boundary:** [`compile_workflow()`](../pyearthmesh/config/config_manager.py) compiles the generic spec into backend-native configurations for specific generators.
  - **Tier 2 (Execution Model):** Mesh-native Case adapters (`MPASCase`, `LatLonCase`, `HealpixCase`, `ISEACase`, `RHEALPixCase`) orchestrate workspace preparation (`setup()`) and execution (`run()`).
- **Current Support:**
  - Implemented workflow adapters: `mpas`, `latlon`, `healpix`, `isea`, `rhealpix`.
  - Future extensions: `square`, `hexagon`, `tin`, `dggrid` adapters.

### 2. [`zarr_implementation_plan.md`](zarr_implementation_plan.md)
- **Role:** Roadmap for high-resolution mesh storage (100M+ cells) using chunked, compressed Zarr format.
- **Implemented Components:**
  - Configuration & profiling: [`zarr_config.py`](../pyearthmesh/utility/zarr_config.py)
  - Chunked I/O: [`zarr_io.py`](../pyearthmesh/utility/zarr_io.py) (`ZarrMeshWriter`, `ZarrMeshReader`, `write_mesh_to_zarr`, `read_mesh_from_zarr`)
  - Format conversions: [`convert_mesh_formats.py`](../pyearthmesh/utility/convert_mesh_formats.py) (`convert_vector_to_zarr`, `convert_zarr_to_vector`)
- **Remaining Roadmap (Phases 3–5):**
  - Direct Zarr emission in `create_dggrid_mesh`, `create_mpas_mesh`, `create_tin_mesh`.
  - Parallel chunk writes and spatial indexing.
  - Formal test suite and scaling benchmarks.

---

## Archive Directory (`plans/archive/`)

Historical and completed plans are preserved in [`plans/archive/`](archive/):

- [`archive/buffer_simplification_plan.md`](archive/buffer_simplification_plan.md): Original design for merging overlapping coastline buffer zones (implemented via `iFlag_simplify_in`).
- [`archive/mpas_workflow_redesign.md`](archive/mpas_workflow_redesign.md): Design that unified the config system and introduced the single `workflow_config.json` loader; superseded by the multi-mesh Two-Tier Compiler architecture.
- [`archive/two_tier_staged_plan.md`](archive/two_tier_staged_plan.md): Step-by-step rollout log for the Two-Tier Compiler architecture (all primary steps completed).

---

## Cleaned / Retired Artifacts

- **`zarr_implementation_code_scaffolds.md`**: Removed. Scaffolding code has been fully integrated into real Python modules ([`zarr_config.py`](../pyearthmesh/utility/zarr_config.py), [`zarr_io.py`](../pyearthmesh/utility/zarr_io.py), [`convert_mesh_formats.py`](../pyearthmesh/utility/convert_mesh_formats.py)).
- **`gdal_geojson_migration_design.md`**: Removed. OGR migration for JIGSAW boundary loading was completed in [`pyearthmesh/meshes/unstructured/jigsaw/loadgeo.py`](../pyearthmesh/meshes/unstructured/jigsaw/loadgeo.py).
- **`pyearthmesh/utility/zarr.py`**: Removed 0-byte orphan module.
