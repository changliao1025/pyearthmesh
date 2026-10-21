# Read the Docs configuration file for pyearthmesh
# Matches the conventions and structure of pyearth/docs

import os
import sys

# Try to import tomli for Python < 3.11, tomllib for Python >= 3.11
try:
    import tomllib as toml_lib
except ImportError:
    try:
        import tomli as toml_lib
    except ImportError:
        toml_lib = None

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath("../../"))


# Read version from pyproject.toml
def get_version():
    """Read version from pyproject.toml"""
    if toml_lib is None:
        return "0.1.1"

    try:
        pyproject_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "pyproject.toml"
        )
        with open(pyproject_path, "rb") as f:
            pyproject_data = toml_lib.load(f)
        return pyproject_data["project"]["version"]
    except Exception:
        return "0.1.1"


# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.intersphinx",  # Link to other project documentation
    "myst_parser",  # Markdown support (.md)
]

# Mock heavy C/C++ dependencies so Read the Docs builds without compiling GDAL/JIGSAW
autodoc_mock_imports = [
    "gdal",
    "osgeo",
    "ogr",
    "osr",
    "pyearth",
    "pyearthbuffer",
    "jigsawpy",
    "netCDF4",
    "shapely",
    "cartopy",
    "scipy",
    "dggal",
    "zarr",
    "numcodecs",
]

# Intersphinx mapping to link to other projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "gdal": ("https://gdal.org/", None),
    "pyearth": ("https://pyearth.readthedocs.io/en/latest/", None),
}

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Support both ReStructuredText (.rst) and Markdown (.md)
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_image",
    "tasklist",
]
myst_heading_anchors = 3

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "pyearthmesh"
copyright = "2026, Battelle Memorial Institute"
author = "Chang Liao"

# The version info
try:
    _version = get_version()
    version = _version
    release = _version
except Exception:
    version = "0.1.1"
    release = "0.1.1"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
pygments_style = "sphinx"

# -- Options for HTML output ----------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
