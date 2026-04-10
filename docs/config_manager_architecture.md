# ConfigManager Architecture

## Overview

The `ConfigManager` has been redesigned as a general configuration class that delegates to specific mesh configuration classes. This architecture provides a flexible, extensible system for managing configurations across different mesh types.

## Architecture

### Class Hierarchy

```
BaseMeshConfig (Abstract Base Class)
├── JigsawConfig
├── MPASConfig
├── HealpixConfig
├── ISEAConfig
└── LatLonConfig

ConfigManager (Orchestrator)
```

### Key Components

1. **`BaseMeshConfig`**: Abstract base class defining the interface for all mesh configurations
   - `get_default_config()`: Returns default configuration dictionary
   - `validate_config(config)`: Validates configuration parameters
   - `get_mesh_type()`: Returns mesh type identifier

2. **Specific Mesh Config Classes**: Implement `BaseMeshConfig` for each mesh type
   - `JigsawConfig`: JIGSAW unstructured mesh generation
   - `MPASConfig`: MPAS mesh configuration
   - `HealpixConfig`: HEALPix DGGS configuration
   - `ISEAConfig`: ISEA DGGS configuration
   - `LatLonConfig`: Regular lat-lon grid configuration

3. **`ConfigManager`**: General configuration manager that orchestrates mesh-specific configs
   - Maintains registry of available mesh types
   - Delegates operations to appropriate mesh config class
   - Provides unified interface for all mesh types

## Usage

### Basic Usage

```python
from pyearthmesh.classes.config import ConfigManager

# Create a configuration manager for a specific mesh type
manager = ConfigManager("jigsaw")

# Get default configuration
config = manager.get_default_config()

# Create a configuration file with custom values
config = manager.create_template_config(
    "config.json",
    custom_values={
        "ncolumn_space": 720,
        "nrow_space": 360,
        "dResolution_land": 30.0,
    }
)
```

### Working with Different Mesh Types

```python
# JIGSAW mesh
jigsaw_manager = ConfigManager("jigsaw")
jigsaw_config = jigsaw_manager.get_default_config()

# MPAS mesh
mpas_manager = ConfigManager("mpas")
mpas_config = mpas_manager.get_default_config()

# HEALPix mesh
healpix_manager = ConfigManager("healpix")
healpix_config = healpix_manager.get_default_config("healpix")
```

### Dynamic Mesh Type Selection

```python
# Create manager without specifying mesh type
manager = ConfigManager()

# Get configuration for any mesh type
jigsaw_config = manager.get_default_config("jigsaw")
mpas_config = manager.get_default_config("mpas")
healpix_config = manager.get_default_config("healpix")
```

### Loading and Saving Configurations

```python
# Load configuration from file
config = ConfigManager.load_config("config.json")

# Save configuration to file
ConfigManager.save_config(config, "output_config.json")
```

### Configuration Validation

```python
manager = ConfigManager("healpix")

# This will validate the configuration
try:
    manager.validate_config({"nside": 7, "mesh_type": "healpix"})
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

## Extending the System

### Adding a New Mesh Type

To add support for a new mesh type:

1. Create a new config class inheriting from `BaseMeshConfig`:

```python
from pyearthmesh.classes.config import BaseMeshConfig, ConfigManager

class MyMeshConfig(BaseMeshConfig):
    def get_default_config(self):
        return {
            "param1": "value1",
            "param2": "value2",
            "mesh_type": "mymesh",
        }

    def validate_config(self, config):
        # Add validation logic
        if "param1" not in config:
            raise ValueError("param1 is required")
        return True
```

2. Register the new config class:

```python
ConfigManager.register_mesh_config("mymesh", MyMeshConfig)
```

3. Use it like any other mesh type:

```python
manager = ConfigManager("mymesh")
config = manager.get_default_config()
```

## Available Mesh Types

Use `ConfigManager.get_available_mesh_types()` to get a list of all registered mesh types:

```python
>>> ConfigManager.get_available_mesh_types()
['jigsaw', 'mpas', 'healpix', 'isea', 'latlon']
```

## Configuration Parameters

### Common Parameters (All Mesh Types)

- `mesh_type`: Identifier for the mesh type
- `iCase_index`: Case index number
- `iFlag_standalone`: Standalone execution flag
- `iFlag_global`: Global mesh flag
- `sDate`: Date string for the run
- `sWorkspace_output`: Output workspace directory
- `sFilename_model_configuration`: Path to configuration file

### JIGSAW-Specific Parameters

- `ncolumn_space`, `nrow_space`: Spacing grid dimensions
- `dSpac_value`: Default spacing value
- `iFlag_geom`, `iFlag_spac`: Feature flags
- `dResolution_*`: Resolution parameters for different features
- `mesh_kern`, `optm_kern`: Algorithm selection
- `hfun_hmax`, `hfun_hmin`: Mesh sizing bounds
- And many more (see `JigsawConfig.get_default_config()`)

### MPAS-Specific Parameters

- `sFilename_mpas_mesh_netcdf`: MPAS mesh NetCDF file
- `sFilename_jigsaw_mesh_netcdf`: JIGSAW mesh NetCDF file
- `sFilename_land_ocean_mask`: Land-ocean mask file
- `pBoundary_wkt`: Boundary in WKT format
- `iFlag_run_jigsaw`: Flag to run JIGSAW

### HEALPix-Specific Parameters

- `nside`: HEALPix resolution parameter (must be power of 2)
- `nest`: Nested (True) or ring (False) pixel ordering

### ISEA-Specific Parameters

- `resolution`: ISEA resolution level
- `aperture`: Aperture (3 for ISEA3H, 4 for ISEA4H)

### Lat-Lon-Specific Parameters

- `dLongitude_degree`, `dLatitude_degree`: Grid resolution
- `dLongitude_left`, `dLongitude_right`: Longitude bounds
- `dLatitude_bot`, `dLatitude_top`: Latitude bounds

## Backward Compatibility

The old `JigsawConfigManager` class is still available as an alias to `JigsawConfig` for backward compatibility:

```python
from pyearthmesh.classes.config import JigsawConfigManager

# This still works
config = JigsawConfigManager.get_default_config()
```

## Integration with config_manager.py

The `pyearthmesh/config/config_manager.py` module provides high-level functions that use the new `ConfigManager`:

```python
from pyearthmesh.config.config_manager import (
    create_template_configuration_file,
    read_configuration_file
)

# Create a configuration file
config = create_template_configuration_file(
    "config.json",
    mesh_type="jigsaw",
    ncolumn_space=720,
    nrow_space=360
)

# Read configuration and create case object
case = read_configuration_file(
    "config.json",
    sWorkspace_output_in="./output"
)
```

## Example

See [`examples/config_example.py`](../examples/config_example.py) for a complete working example demonstrating all features of the new ConfigManager.

## Benefits

1. **Extensibility**: Easy to add new mesh types without modifying existing code
2. **Type Safety**: Each mesh type has its own validation logic
3. **Consistency**: Unified interface across all mesh types
4. **Maintainability**: Mesh-specific logic is encapsulated in dedicated classes
5. **Flexibility**: Can work with any mesh type through a single manager
6. **Backward Compatibility**: Existing code continues to work
