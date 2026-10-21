# Staged plan for the two-tier compiler architecture

> **Status: COMPLETED (ARCHIVED)**
> - **Execution Result:** All rollout steps (Steps 1–6) were successfully implemented and verified with tests (`tests/test_compile_workflow.py`, `tests/test_workflow_spec.py`).
> - **Mesh Coverage Implemented:** MPAS, LatLon, Healpix, ISEA, RHEALPix workflows and cases are operational.
> - **Active Design Spec:** See [`two_tier_compiler_design.md`](../two_tier_compiler_design.md).

## Step 1: define the generic front-end workflow contract

Create a mesh-agnostic workflow specification that describes what the user wants, without backend-specific keys.

Goals:
- define the high-level API for all mesh types
- keep mesh-specific behavior out of the user-facing contract
- establish the common fields shared by regular-grid, spherical, and unstructured meshes

Minimal fields:
- `mesh_type`
- `case_index`
- `date`
- `region`
- `resolution`
- `paths`
- `execution`
- `features`

This is the front-end layer.

---

## Step 2: keep the native backend config contract

Use the existing mesh-native config classes in `pyearthmesh/classes/meshconfig.py` as the backend contract.

Examples:
- `MPASConfig`
- `LatLonConfig`
- `HealpixConfig`
- `ISEAConfig`
- etc.

This is the back-end layer.

---

## Step 3: add the compiler boundary

Status: complete.

Add a function that converts the generic workflow spec into the mesh-native config dict expected by the chosen mesh backend.

Pseudo-shape:

```python
def compile_workflow(spec: dict, mesh_type: str) -> dict:
    native_cfg = ConfigManager(mesh_type).get_default_config()
    translated = translate_frontend_to_backend(mesh_type, spec)
    native_cfg.update(translated)
    return native_cfg
```

This is the actual two-tier compiler boundary.

---

## Step 4: refactor one mesh family first

Status: complete for MPAS.

Pick one mesh family and implement the compiler pattern end-to-end.

Recommended first target: MPAS, because the workflow adapter already exists in `pyearthmesh/config/workflows/mpas.py`.

Goal:
- parse generic workflow spec
- compile to native MPAS config
- pass native config into the runtime case

---

## Step 5: apply the same pattern to a second mesh type

Status: complete for Lat-Lon.

Choose a simpler mesh family such as `latlon` or `healpix`.

Goal:
- confirm the compiler interface works across different mesh families
- validate that the same generic workflow contract can generate different native configs

---

## Step 6: remove duplicated source-of-truth fields

Status: complete for the implemented MPAS and Lat-Lon workflow paths.

Once the compiler is stable, remove or deprecate duplicated fields that currently exist in both the workflow layer and the runtime config layer.

Goal:
- one generic workflow source
- one native backend config source
- no repeated scientific metadata in multiple shapes

---

## Step 7: keep compatibility shims

Do not break existing user scripts immediately.

Goal:
- keep old API entry points working briefly
- route them internally through the compiler pipeline
- deprecate them gradually

---

## Scope for the first implementation

For the first patch, keep the work narrow:

- define the front-end workflow contract
- implement the compiler function contract
- refactor MPAS to use the compiler
- leave the other mesh families alone initially

This gives a concrete reference implementation without a broad refactor.
