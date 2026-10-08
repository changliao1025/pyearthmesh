# GDAL-Based Vector Loader Migration Design

## Purpose

Migrate the JIGSAW vector boundary loader from the Python `geojson` parser to GDAL/OGR while keeping one production module and preserving the existing public API.

The target module is:

```text
pyearthmesh/meshes/unstructured/jigsaw/loadgeo.py
```

The current `loadgeo2.py` implementation should be treated as a prototype. It should not become a second supported loader because the two implementations can otherwise diverge in geometry support, empty-input behavior, and coordinate handling.

## Current State

`loadgeo.py` currently:

- parses the complete file with `geojson.load()`;
- supports `Point`, `LineString`, `MultiLineString`, `Polygon`, `MultiPolygon`, and `GeometryCollection`;
- converts every line or polygon ring into JIGSAW `vert2` and `edge2` arrays;
- discards the third coordinate when present;
- assigns sequential `IDtag` values to line and polygon objects;
- sets `mesh.edge2` to `None` when the input contains no edges.

The former `loadgeo2.py` prototype:

- opens the input through the OGR GeoJSON driver;
- iterates features instead of loading the complete document;
- supports lines and polygons, including multi-geometries;
- does not support points or geometry collections;
- unconditionally concatenates `eset`, which fails for point-only or empty input;
- contains duplicated conversion logic.

Neither implementation currently transforms coordinates. For this migration, GeoJSON coordinates are assumed to be WGS84 longitude/latitude.

## Goals

1. Have one production implementation for vector loading.
2. Use OGR as the input and geometry access layer.
3. Preserve the existing `loadgeo(name, mesh)` call signature.
4. Preserve current supported geometry semantics.
5. Support GDAL vector formats where the selected driver can expose a layer, while retaining normal GeoJSON behavior.
6. Preserve two-dimensional JIGSAW output by explicitly taking XY coordinates.
7. Make empty, point-only, and geometry-collection inputs deterministic.
8. Assume GeoJSON coordinates use WGS84 and do not perform CRS transformation.
9. Make failures actionable and consistent.
10. Add focused tests before removing the duplicate implementation.

## Non-Goals

- Reprojecting coordinates.
- Validating or repairing invalid geometries.
- Converting polygons into filled JIGSAW triangles. The loader creates boundary edges only.
- Changing JIGSAW mesh conventions or the meaning of `IDtag`.
- Adding a new abstraction layer for every possible OGR geometry type.

## Target Public API

Keep the existing function:

```python
def loadgeo(name, mesh):
    """Load vector geometry into a JIGSAW mesh."""
```

The first migration should continue to accept a string path and a `jigsaw_msh_t` instance. Avoid changing callers during the parser migration.

Optional keyword arguments should be considered only after the compatibility migration is complete. A future extension could be:

```python
def loadgeo(name, mesh, *, source_srs=None, target_srs=None):
```

This should not be added until the repository has an explicit CRS policy and tests for it.

## Target Module Design

`loadgeo.py` should contain four small responsibilities:

### 1. Open the data source

Use OGR's normal driver discovery rather than hard-coding only the GeoJSON driver when broader vector support is desired:

```python
datasource = ogr.Open(name, 0)
```

If the project intentionally wants GeoJSON-only behavior at first, use the GeoJSON driver explicitly and document that restriction. The broader `ogr.Open()` route is preferable for a GDAL-based loader because it permits supported vector formats without changing the mesh conversion code.

Validate the result and raise `FileNotFoundError` or `ValueError` with the input path and the reason for failure.

### 2. Traverse OGR geometry

Use one recursive helper that receives an OGR geometry and yields coordinate sequences to the mesh conversion routines.

Required mappings:

| OGR geometry | Output behavior |
|---|---|
| `Point` | Add one JIGSAW vertex and no edge |
| `LineString` | Add vertices and consecutive open edges |
| `MultiLineString` | Process each child line |
| `Polygon` | Process each exterior and interior ring as a closed edge loop |
| `MultiPolygon` | Process every ring in every child polygon |
| `GeometryCollection` | Recursively process each child geometry |

Use geometry names or normalized base geometry types so Z/M variants such as `wkbLineString25D` are handled as their two-dimensional base type. The output contract is explicitly XY, regardless of whether the source contains Z coordinates.

Unsupported geometry types should be skipped with a clear policy, or rejected with a clear error. The initial migration should skip unsupported types consistently and avoid silently treating them as lines or polygons.

### 3. Convert coordinates to JIGSAW arrays

Keep a single implementation of the existing conversion rules:

- `Point`: one row in `vert2`, no `edge2` entry;
- open line: one edge between each pair of consecutive points;
- ring: omit the repeated closing coordinate from `vert2`, then connect the final vertex back to the first;
- all coordinate arrays use only `x` and `y`;
- `IDtag` increments per processed line or ring, preserving the current sequential behavior.

The conversion helpers should receive plain coordinate sequences. They should not know whether the coordinates came from GeoJSON dictionaries or OGR objects.

### 4. Finalize the mesh

After iteration:

- concatenate all vertex arrays if any exist;
- otherwise assign an empty `vert2` array with the correct JIGSAW dtype, or raise a clear empty-input error according to the chosen API policy;
- concatenate edges when any exist;
- assign `mesh.edge2 = None` when no edges exist, preserving current behavior;
- release the datasource after reading, preferably through a local scope and explicit cleanup where needed by the GDAL bindings.

The point-only case must be valid. The empty file case must not fail with an opaque NumPy `ValueError`.

## CRS Assumption

CRS transformation is out of scope for this migration. GeoJSON input is assumed to use WGS84 longitude/latitude coordinates, consistent with the GeoJSON default convention.

The loader should pass coordinates through unchanged and should not inspect, infer, or transform a source spatial reference. Inputs in another CRS must be converted before loading.

## Attribute and ID Policy

The current loader ignores feature properties and assigns sequential object IDs. Preserve that behavior during migration to minimize downstream changes.

A later enhancement may allow an attribute field to control `IDtag`, but that should be explicit, for example:

```python
loadgeo(name, mesh, id_field="region_id")
```

Do not infer an ID field automatically from arbitrary source data.

## Error Handling

The unified module should provide errors at the input boundary:

- invalid `name` type: `TypeError`;
- invalid `mesh` type: `TypeError`;
- missing or unreadable file: `FileNotFoundError` or an OGR-backed `ValueError`;
- no readable layer: `ValueError`;
- malformed or unsupported geometry: a clear `ValueError`, unless the documented policy is to skip it.

Do not catch every OGR exception and return an empty mesh. An empty result must be distinguishable from a failed read.

## Dependency Changes

The package already imports `osgeo` in this JIGSAW module, and other repository code uses GDAL. The migration should still make the dependency explicit and reproducible:

- remove the unused `geojson` import from the production loader;
- verify how GDAL is provided by the supported environment or by `pyearth`;
- document the required GDAL/Python binding installation path for development and CI;
- avoid adding a second optional parser dependency solely for this loader.

The packaging decision should be made separately from the code change because GDAL wheels and system libraries vary by operating system and Python version.

## Migration Steps

### Phase 1: Characterize current behavior

Add small fixtures covering:

- one point;
- one open line;
- one polygon with a closed ring;
- polygon with an interior ring;
- multi-line and multi-polygon;
- geometry collection;
- 3D coordinates;
- point-only input;
- empty or geometry-free input.

For each fixture, record vertex count, edge count, edge indices, and `IDtag` values from the current loader.

### Phase 2: Build the unified OGR implementation

This phase is implemented in `loadgeo.py`. The former `loadgeo2.py` prototype is removed after the merged implementation is validated.

1. Replace the file-reading portion of `loadgeo.py` with OGR datasource/layer iteration.
2. Move or retain only one copy of the line and polygon conversion helpers.
3. Add point and recursive geometry-collection handling.
4. Normalize OGR geometry types to handle Z variants.
5. Preserve XY-only output and current ring closure behavior.
6. Preserve `mesh.edge2 = None` when appropriate.
7. Add explicit validation for empty and unsupported input.

### Phase 3: Compare outputs

Run the old parser and the new parser against the characterization fixtures. Differences should be intentional and documented. For equivalent inputs, the following should match:

- vertex coordinates;
- edge connectivity;
- number and order of generated objects;
- `IDtag` values;
- treatment of Z coordinates;
- `None` versus array representation for no edges.

### Phase 4: Remove the duplicate implementation

The duplicate `loadgeo2.py` implementation is removed. Production callers use `loadgeo.py` only, and no independent compatibility module is retained.

## Test Plan

Tests should focus on observable mesh output rather than GDAL internals.

### Unit tests

- `Point` creates one vertex and `edge2 is None`.
- `LineString` creates `n - 1` edges.
- A closed polygon ring creates one edge per unique ring vertex.
- Interior rings are preserved as separate closed loops.
- Multi-geometries produce the same ordering as the current loader.
- Geometry collections are recursively traversed.
- Z coordinates are ignored and do not change XY output.
- Point-only input does not call `np.concatenate()` on an empty edge list.
- Empty input produces the documented result.
- Missing paths and invalid arguments raise useful exceptions.

### Integration tests

- Load a real repository GeoJSON fixture through the unified module.
- Confirm the resulting mesh can be passed to the existing JIGSAW workflow.
- If non-GeoJSON support is enabled, load one representative supported format in a separate test.

### Optional performance check

For a large fixture, compare peak memory and elapsed time between the old parser and the OGR implementation. Treat this as evidence rather than a hard regression gate because GDAL driver and platform behavior can vary.

## Acceptance Criteria

The migration is complete when:

- `loadgeo.py` is the only production implementation;
- no production code needs the `geojson` package for this loader;
- all previously supported geometry types still work;
- point-only and geometry-collection inputs work;
- XY output and JIGSAW connectivity are unchanged for equivalent fixtures;
- WGS84 input is documented as the required coordinate convention and no CRS transformation is attempted;
- tests cover empty edges, 3D coordinates, and nested geometries;
- `loadgeo2.py` is deleted and `loadgeo.py` is the only production loader;
- the documented GDAL environment can run the loader and its tests.

## Risks and Decisions

### GDAL installation complexity

GDAL is more capable but heavier than a JSON parser. The project should document a tested conda-forge environment or another supported installation method instead of relying on an unspecified system GDAL.

### OGR coordinate extraction overhead

Calling `GetPoints()` and converting each geometry to Python tuples may not be faster than parsing simple GeoJSON. The primary benefit is format and GIS interoperability, not guaranteed speed.

### Geometry validity

OGR can read geometries that are invalid for meshing. Validation and repair should remain outside this loader unless the JIGSAW workflow establishes a specific requirement.
