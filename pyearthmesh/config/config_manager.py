import os
import json
from pathlib import Path
from pyearthmesh.classes.config import ConfigManager


def create_template_configuration_file(sFilename_configuration_json, mesh_type="jigsaw", **kwargs):
    """Generate mesh config template file using parameter keywords.

    Looks up the mesh-specific config class registered in ``ConfigManager``
    (e.g. ``JigsawConfig``, ``MPASConfig``, …) and calls its
    ``create_template_config`` method.  The method merges the class defaults
    with any keyword arguments supplied by the caller, so both default and
    user-provided values are honoured.

    Args:
        sFilename_configuration_json (str or Path): Path to save the configuration file.
        mesh_type (str): Type of mesh configuration (jigsaw, mpas, healpix, isea, latlon).
        **kwargs: Additional configuration parameters to override defaults.

    Returns:
        dict: The final merged configuration that was written to disk.

    Raises:
        ValueError: If ``mesh_type`` is not registered in ``ConfigManager``.
    """
    mesh_type_lower = mesh_type.lower()

    # Resolve the concrete config class for this mesh type
    if mesh_type_lower not in ConfigManager._mesh_configs:
        raise ValueError(
            f"Unknown mesh type: {mesh_type!r}. "
            f"Available types: {', '.join(ConfigManager.get_available_mesh_types())}"
        )

    # Instantiate the mesh-specific config class and delegate to its
    # create_template_config, passing user kwargs as custom_values so that
    # defaults are preserved for any key the caller did not supply.
    config_instance = ConfigManager._mesh_configs[mesh_type_lower]()
    config = config_instance.create_template_config(
        sFilename_configuration_json,
        custom_values=kwargs if kwargs else None,
    )

    return config


def read_configuration_file(
    sFilename_configuration_in,
    iFlag_standalone_in=1,
    iFlag_create_directory_in=None,
    iCase_index_in=None,
    sModel_in=None,
    sDate_in=None,
    sWorkspace_output_in=None,
):
    """Read a mesh configuration from a JSON file and create appropriate case object

    Args:
        sFilename_configuration_in (str or Path): Path to the configuration file
        iFlag_standalone_in (int): Standalone flag (default: 1)
        iFlag_create_directory_in (int): Flag to create directories
        iCase_index_in (int): Case index
        sModel_in (str): Model/mesh type
        sDate_in (str): Date string
        sWorkspace_output_in (str or Path): Output workspace directory

    Returns:
        object: Case object (jigsawcase, mpascase, or meshcase depending on mesh type)
    """
    # Ensure input filenames are strings
    if isinstance(sFilename_configuration_in, Path):
        sFilename_configuration_in = str(sFilename_configuration_in)
    if isinstance(sWorkspace_output_in, Path):
        sWorkspace_output_in = str(sWorkspace_output_in)

    if not os.path.isfile(sFilename_configuration_in):
        print(sFilename_configuration_in + " does not exist")
        return

    # Load configuration using ConfigManager
    aConfig = ConfigManager.load_config(sFilename_configuration_in)

    # Extract or set parameters
    if iCase_index_in is not None:
        iCase_index = iCase_index_in
    else:
        iCase_index = int(aConfig.get("iCase_index", 1))

    if iFlag_standalone_in is not None:
        iFlag_standalone = iFlag_standalone_in
    else:
        iFlag_standalone = int(aConfig.get("iFlag_standalone", 1))

    if sModel_in is not None:
        sModel = sModel_in
    else:
        sModel = aConfig.get("sModel", aConfig.get("mesh_type", "jigsaw"))

    if sDate_in is not None:
        sDate = sDate_in
    else:
        sDate = aConfig.get("sDate")

    if sWorkspace_output_in is not None:
        sWorkspace_output = sWorkspace_output_in
    else:
        sWorkspace_output = aConfig.get("sWorkspace_output")

    # Create output workspace if needed
    try:
        print(
            "Creating the specified output workspace (if it does not exist): \n",
            sWorkspace_output,
        )
        Path(sWorkspace_output).mkdir(parents=True, exist_ok=True)
        print("The specified output workspace is: \n", sWorkspace_output)
    except (ValueError, TypeError) as e:
        print(f"The specified output workspace cannot be created: {e}")
        return None

    # Update configuration with runtime parameters
    aConfig["iCase_index"] = iCase_index
    aConfig["iFlag_standalone"] = iFlag_standalone
    aConfig["sDate"] = sDate
    aConfig["sModel"] = sModel
    aConfig["sWorkspace_output"] = sWorkspace_output
    aConfig["sFilename_model_configuration"] = sFilename_configuration_in

    # Create appropriate case object based on mesh type
    mesh_type = aConfig.get("mesh_type", sModel).lower()

    if mesh_type == "jigsaw":
        from pyearthmesh.meshes.unstructured.jigsaw.jigsawcls import jigsawcase
        oCase = jigsawcase(
            aConfig, iFlag_create_directory_in=iFlag_create_directory_in
        )
    elif mesh_type == "mpas":
        from pyearthmesh.meshes.unstructured.mpas.mpascase import mpascase
        oCase = mpascase(
            aConfig, sWorkspace_output_in=sWorkspace_output
        )
    else:
        # For other mesh types, use generic meshcase
        from pyearthmesh.classes.meshcase import meshcase
        oCase = meshcase(
            aConfig,
            iFlag_standalone_in=iFlag_standalone_in,
            iFlag_create_directory_in=iFlag_create_directory_in,
            sModel_in=sModel,
            sDate_in=sDate,
            sWorkspace_output_in=sWorkspace_output,
        )

    return oCase



