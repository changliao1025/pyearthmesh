### PyEarthMesh

PyEarthMesh is a lightweight mesh generation and manipulation package for Earth science applications.

Every hiker/camper would love to have a leatherman pocket knife around because it is lightweight, versatile and reliable, sometimes life saving.

This is why I developed PyEarth, a lightweight Python package to support various Earth science tasks.
I use it in my daily work, and nearly all my research papers use it in some way.

**Note:** PyEarth has been restructured and is now part of the **PyEarthSuite** ecosystem. To keep each package lightweight and focused, PyEarth has been split into several specialized packages:

- **pyearth** (this package) - Core GIS operations, spatial toolbox, and system utilities
- **pyearthviz** - 2D visualization utilities
- **pyearthviz3d** - 3D visualization with GeoVista
- **pyearthriver** - River network graph algorithms and data structures
- **pyearthmesh** - Mesh generation and manipulation tools
- **pyearthhelp** - Helper utilities for data access (NWIS, NLDI, GSIM) and HPC operations

This modular approach allows you to install only what you need while maintaining the ability to use all packages together.


### Dependency

PyEarthMesh depends on the following packages:

**Required:**

Because PyEarthMesh relies on many APIs from PyEarth, it requires PyEarth to be installed.
Other than that, additonal dependencies maybe required based on the specific functionalities you intend to use:

1. `dggrid` - For DGGRID mesh generation
2. `dggal` - For DGGS mesh manipulation
3. `jigsaw` and `MPAS-tools` - For MPAS mesh generation and manipulation


### Documentation

Please refer to the [documentation](https://pyearthmesh.readthedocs.io) for details on how to get started using the PyEarth package.

### Installation

The most easy way to install PyEarthMesh is via `conda`.

```bash
# Install from conda (recommended)
conda install pyearthmesh -c conda-forge
```

### Content

PyEarthMesh provides general-purpose functions organized into several categories:

#### 1. Structured Meshes

- **Square**: Projected coordinate handling, grid generation, coordinate reprojection
- **LatLon**: Longitude/latitude grid generation, coordinate reprojection
- **Hexagon**: Projected hexagon grid generation, coordinate reprojection
- **Triangular**: Projected triangular grid generation, coordinate reprojection
- **DGGRID**: DGGS grid generation, coordinate reprojection
- **Healpix**: Healpix grid generation, coordinate reprojection


#### 2. Unstructured Meshes

- **TIN**: Traiangular Irregular Network generation, manipulation, and analysis
- **MPAS**: Model for Prediction Across Scales mesh generation, manipulation, and analysis



### Related Packages

PyEarthMesh is part of the PyEarthSuite ecosystem, which includes several specialized packages:

- **[pyearthviz](../pyearthviz)** - 2D plotting and visualization
- **[pyearthviz3d](../pyearthviz3d)** - 3D globe visualization with GeoVista
- **[pyearthriver](../pyearthriver)** - River network topology and graph algorithms
- **[pyearthmesh](../pyearthmesh)** - Advanced mesh generation tools
- **[pyearthhelp](../pyearthhelp)** - Data retrieval and HPC job management

### Acknowledgment

This research was supported as part of the Next Generation Ecosystem Experiments-Tropics, funded by the U.S. Department of Energy, Office of Science, Office of Biological and Environmental Research at Pacific Northwest National Laboratory. The study was also partly supported by U.S. Department of Energy Office of Science Biological and Environmental Research through the Earth and Environmental System Modeling program as part of the Energy Exascale Earth System Model (E3SM) project.

### License

Copyright © 2022, Battelle Memorial Institute

1. Battelle Memorial Institute (hereinafter Battelle) hereby grants permission to any person or entity lawfully obtaining a copy of this software and associated documentation files (hereinafter "the Software") to redistribute and use the Software in source and binary forms, with or without modification. Such person or entity may use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software and may permit others to do so, subject to the following conditions:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.

* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

* Other than as used herein, neither the name Battelle Memorial Institute or Battelle may be used in any form whatsoever without the express written consent of Battelle.

2. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

### References

Several publications describe the algorithms used in `PyEarth` in detail. If you make use of `PyEarth` in your work, please consider including a reference to the following:

* Chang Liao. (2022). PyEarth: A lightweight Python package for Earth science (Software). Zenodo. https://doi.org/10.5281/zenodo.6109987

PyEarth is also supporting several other Python packages/projects, including:

* Liao et al., (2023). pyflowline: a mesh-independent river network generator for hydrologic models. Journal of Open Source Software, 8(91), 5446, https://doi.org/10.21105/joss.05446

* Liao. C. (2022). HexWatershed: a mesh independent flow direction model for hydrologic models (0.1.1). Zenodo. https://doi.org/10.5281/zenodo.6425881
