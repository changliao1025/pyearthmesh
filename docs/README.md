# PyEarthMesh Documentation

Welcome to the **PyEarthMesh** documentation repository. This directory is structured to mirror the standard layout of **PyEarthSuite** packages (such as [`pyearth/docs`](../../pyearth/docs)) for unified deployment on **Read the Docs**.

---

## Directory Layout

```text
docs/
├── Makefile                     # UNIX build targets (e.g. `make html`)
├── make.bat                     # Windows build script
├── requirements.txt             # Documentation build dependencies
├── README.md                    # This overview file
└── source/                      # Sphinx source directory
    ├── conf.py                  # Sphinx build configuration
    ├── index.rst                # Documentation master table of contents
    ├── getting-started.rst      # Installation and quickstart guide
    ├── mpas_mesh_generation.md  # Detailed MPAS mesh generation guide
    ├── architecture.md          # Complete package architecture reference
    ├── config_manager_architecture.md # Configuration system reference
    ├── faq.rst                  # Frequently asked questions & troubleshooting
    ├── support.rst              # Support channels & PyEarthSuite links
    ├── authors.rst              # Author credits
    ├── contribution.rst         # Contribution guidelines
    └── _static/                 # Custom static assets (CSS/images)
```

---

## Quickstart: How to Generate an MPAS Mesh

Generating an MPAS mesh requires two simple steps:

### 1. Create a Configuration File (`workflow_config.json`)

```json
{
  "workflow": {
    "mesh_type": "mpas",
    "case_index": 1,
    "model": "jigsaw",
    "debug": true,
    "run_on_hpc": false
  },
  "resolutions": {
    "ocean_km": 100.0,
    "land_km": 50.0,
    "river_network_km": 5.0,
    "coastline_km": 10.0,
    "lake_boundary_km": 5.0
  },
  "paths": {
    "region_geometry": "path/to/region_boundary.geojson",
    "output_workspace": "./output/my_mpas_mesh",
    "river_network_raster": "path/to/river_network.tif"
  },
  "jigsaw": {
    "standalone": true,
    "create_directory": true,
    "ncolumn_space": 3600,
    "nrow_space": 1800,
    "iFlag_geom": true,
    "iFlag_spac": true,
    "iFlag_spac_ocean": true,
    "iFlag_spac_land": true,
    "iFlag_geom_river_network": true,
    "iFlag_spac_river_network": true
  }
}
```

### 2. Run Mesh Generation in Python

```python
from pathlib import Path
from pyearthmesh.config.config_manager import create_workflow

# 1. Initialize workflow from config file
workflow = create_workflow(Path("workflow_config.json"))

# 2. Prepare output workspace
workflow.setup()

# 3. Execute mesh generation
cells = workflow.run()

print(f"Generated MPAS mesh with {len(cells):,} cells!")
print(f"GeoJSON output saved to: {workflow.mesh_file}")
```

For large cluster runs, set `"run_on_hpc": true` and call `workflow.create_hpc_job()` to generate SLURM submission scripts (`submit.job`, `run_mpas.py`).

For full details, see [`source/mpas_mesh_generation.md`](source/mpas_mesh_generation.md).

---

## Building the Documentation Locally

To build HTML documentation locally:

```bash
# 1. Install documentation dependencies
pip install -r docs/requirements.txt

# 2. Build HTML documentation
cd docs
make html

# 3. View the generated documentation
open _build/html/index.html   # On macOS
# xdg-open _build/html/index.html  # On Linux
```

When pushed to GitHub, Read the Docs will automatically detect [`.readthedocs.yaml`](../.readthedocs.yaml) and build the documentation site.
