import pytest

from pyearthmesh.config.config_manager import (
    compile_workflow,
    create_workflow,
    dump_workflow_config,
    load_workflow_config,
)


def test_compile_workflow_mpas_from_generic_spec():
    spec = {
        "mesh_type": "mpas",
        "case_index": 2,
        "date": "20260916",
        "region": {"geometry": "region.geojson", "global": False},
        "resolution": {"kind": "km", "ocean_km": 100.0, "land_km": 12.5},
        "paths": {"output_workspace": "./output", "region_geometry": "region.geojson"},
        "execution": {"debug": True},
        "features": {"coastline": True},
    }

    config = compile_workflow(spec)

    assert config["mesh_type"] == "mpas"
    assert config["iCase_index"] == 2
    assert config["sWorkspace_output"] == "./output"


def test_compile_workflow_raises_for_unknown_mesh_type():
    try:
        compile_workflow({"mesh_type": "not_real"})
        assert False, "Expected ValueError for unsupported mesh type"
    except ValueError:
        pass


def test_compile_workflow_accepts_sectioned_config_without_optional_sections():
    config = compile_workflow({"workflow": {"mesh_type": "mpas"}})

    assert config["mesh_type"] == "mpas"
    assert config["iCase_index"] == 1
    assert config["iFlag_global"] == 1


def test_compile_workflow_passes_jigsaw_input_files():
    config = compile_workflow(
        {
            "workflow": {"mesh_type": "mpas"},
            "paths": {
                "output_workspace": "./output",
                "river_network_raster": "river_network.tif",
                "lake_boundary_vector": "lake_boundary.geojson",
                "watershed_boundary_raster": "watershed_boundary.tif",
            },
        }
    )

    assert config["sFilename_river_network_raster"] == "river_network.tif"
    assert config["sFilename_lake_boundary_vector"] == "lake_boundary.geojson"
    assert config["sFilename_watershed_boundary_raster"] == "watershed_boundary.tif"


def test_compile_workflow_passes_jigsaw_flags_and_controls():
    config = compile_workflow(
        {
            "workflow": {"mesh_type": "mpas"},
            "resolutions": {
                "lake_boundary_km": 3,
                "watershed_boundary_km": 3,
            },
            "paths": {"output_workspace": "./output"},
            "jigsaw": {
                "iFlag_geom": True,
                "iFlag_spac_lake_boundary": True,
                "ncolumn_space": 720,
            },
        }
    )

    assert config["iFlag_geom"] is True
    assert config["iFlag_spac_lake_boundary"] is True
    assert config["ncolumn_space"] == 720
    assert config["dResolution_lake_boundary"] == 3.0
    assert config["dResolution_watershed_boundary"] == 3.0


def test_mpas_workflow_resolution_mapping_from_json():
    config = compile_workflow(
        {
            "workflow": {"mesh_type": "mpas"},
            "resolutions": {
                "ocean_km": 100,
                "land_km": 50.0,
                "river_network_km": 5,
                "coastline_km": 10,
                "lake_boundary_km": 5,
                "watershed_boundary_km": 5,
                "dam_km": 4.0,
                "city_km": 4.0,
            },
            "paths": {"output_workspace": "./output"},
        }
    )

    assert config["dResolution_ocean"] == 100.0
    assert config["dSpac_value"] == 100.0
    assert config["dResolution_land"] == 50.0
    assert config["dResolution_river_network"] == 5.0
    assert config["dResolution_coastline"] == 10.0
    assert config["dResolution_lake_boundary"] == 5.0
    assert config["dResolution_watershed_boundary"] == 5.0
    assert config["dResolution_dam"] == 4.0
    assert config["dResolution_city"] == 4.0


def test_mpas_workflow_passes_compiled_config_to_case(monkeypatch):
    captured = {}

    class FakeMPASCase:
        def __init__(self, config, **kwargs):
            captured["config"] = config
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "pyearthmesh.meshes.unstructured.mpas.mpascase.mpascase",
        FakeMPASCase,
    )

    workflow = create_workflow(
        {
            "workflow": {
                "mesh_type": "mpas",
                "case_index": 7,
                "model": "jigsaw",
                "debug": True,
            },
            "resolutions": {"ocean_km": 50},
            "paths": {"output_workspace": "./output"},
            "jigsaw": {"standalone": True, "create_directory": False},
        }
    )

    assert workflow.case is not None
    assert workflow.aConfig_mesh is captured["config"]
    assert captured["config"]["iCase_index"] == 7
    assert captured["config"]["dSpac_value"] == 50.0
    assert captured["config"]["dResolution_ocean"] == 50.0
    assert captured["config"]["iFlag_standalone"] == 1
    assert captured["kwargs"]["iFlag_create_directory_in"] == 0


def test_compile_workflow_latlon_produces_structured_native_config():
    config = compile_workflow(
        {
            "mesh_type": "latlon",
            "region": {"bbox": {"left": -10, "right": 10, "bottom": 20, "top": 30}},
            "resolution": {"degree": 0.5},
            "paths": {"output_workspace": "./output"},
        }
    )

    assert config["mesh_type"] == "latlon"
    assert config["dResolution_degree"] == 0.5
    assert config["ncolumn"] == 40
    assert config["nrow"] == 20
    assert config["dLongitude_left"] == -10
    assert config["dLatitude_top"] == 30


def test_latlon_workflow_builds_runtime_adapter():
    workflow = create_workflow(
        {
            "workflow": {"mesh_type": "latlon"},
            "resolutions": {"degree": 2},
            "paths": {"output_workspace": "./output"},
        }
    )

    assert workflow.case.config["mesh_type"] == "latlon"
    assert workflow.case.config["dResolution_degree"] == 2
    assert workflow.case.config["ncolumn"] == 180


def test_compile_workflow_isea_produces_native_config():
    config = compile_workflow(
        {
            "mesh_type": "isea",
            "resolution": {"meter": 1000.0},
            "paths": {
                "output_workspace": "./output",
                "output_mesh": "isea3h_mesh.geojson",
            },
            "features": {
                "isea_type": "ISEA3H",
                "use_tiles": False,
                "tile_mode": "simple",
            },
        }
    )

    assert config["mesh_type"] == "isea"
    assert config["dResolution_meter"] == 1000.0
    assert config["sISEA_type"] == "ISEA3H"
    assert config["sFilename_mesh"] == "isea3h_mesh.geojson"
    assert config["sWorkspace_output"] == "./output"
    assert config["iFlag_global"] == 1


def test_isea_workflow_builds_runtime_adapter(monkeypatch, tmp_path):
    captured = {}

    def fake_create_isea_mesh(
        dResolution_meter_in, sISEA_type, sFilename_mesh, sWorkspace_output, **kwargs
    ):
        captured["resolution"] = dResolution_meter_in
        captured["isea_type"] = sISEA_type
        captured["filename"] = sFilename_mesh
        captured["workspace"] = sWorkspace_output
        captured["kwargs"] = kwargs
        return "fake_mesh_result"

    import sys
    from unittest.mock import MagicMock

    if "dggal" not in sys.modules:
        monkeypatch.setitem(sys.modules, "dggal", MagicMock())

    monkeypatch.setattr(
        "pyearthmesh.meshes.structured.dggs.isea.create_isea_mesh.create_isea_mesh",
        fake_create_isea_mesh,
    )

    out_dir = str(tmp_path / "output")
    workflow = create_workflow(
        {
            "workflow": {"mesh_type": "isea"},
            "resolutions": {"meter": 5000.0},
            "paths": {
                "output_workspace": out_dir,
                "output_mesh": "isea3h_mesh.geojson",
            },
            "features": {"isea_type": "ISEA3H"},
        }
    )

    assert workflow.case.config["mesh_type"] == "isea"
    assert workflow.case.config["dResolution_meter"] == 5000.0
    assert workflow.case.config["sISEA_type"] == "ISEA3H"

    workflow.setup()
    result = workflow.run()
    assert result == "fake_mesh_result"
    assert captured["resolution"] == 5000.0
    assert captured["isea_type"] == "ISEA3H"


def test_compile_workflow_rhealpix_produces_native_config():
    config = compile_workflow(
        {
            "mesh_type": "rhealpix",
            "resolution": {"meter": 10000.0},
            "paths": {
                "output_workspace": "./output",
                "output_mesh": "rhealpix_mesh.geojson",
            },
            "features": {
                "use_tiles": True,
                "tile_mode": "base",
            },
        }
    )

    assert config["mesh_type"] == "rhealpix"
    assert config["dResolution_meter"] == 10000.0
    assert config["sFilename_mesh"] == "rhealpix_mesh.geojson"
    assert config["sWorkspace_output"] == "./output"
    assert config["iFlag_use_tiles"] == 1
    assert config["tile_mode"] == "base"
    assert config["iFlag_global"] == 1


def test_rhealpix_workflow_builds_runtime_adapter(monkeypatch, tmp_path):
    captured = {}

    def fake_create_rhealpix_mesh(
        dResolution_meter_in, sFilename_mesh, sWorkspace_output, **kwargs
    ):
        captured["resolution"] = dResolution_meter_in
        captured["filename"] = sFilename_mesh
        captured["workspace"] = sWorkspace_output
        captured["kwargs"] = kwargs
        return "fake_rhealpix_result"

    import sys
    from unittest.mock import MagicMock

    if "dggal" not in sys.modules:
        monkeypatch.setitem(sys.modules, "dggal", MagicMock())

    monkeypatch.setattr(
        "pyearthmesh.meshes.structured.dggs.healpix.create_rhealpix_mesh.create_rhealpix_mesh",
        fake_create_rhealpix_mesh,
    )

    out_dir = str(tmp_path / "output")
    workflow = create_workflow(
        {
            "workflow": {"mesh_type": "rhealpix"},
            "resolutions": {"meter": 10000.0},
            "paths": {
                "output_workspace": out_dir,
                "output_mesh": "rhealpix_mesh.geojson",
            },
            "features": {"use_tiles": True, "tile_mode": "base"},
        }
    )

    assert workflow.case.config["mesh_type"] == "rhealpix"
    assert workflow.case.config["dResolution_meter"] == 10000.0

    workflow.setup()
    result = workflow.run()
    assert result == "fake_rhealpix_result"
    assert captured["resolution"] == 10000.0
    assert captured["kwargs"]["tile_mode"] == "base"


def test_workflow_property_accessors(tmp_path):
    out_dir = str(tmp_path / "output")
    workflow = create_workflow(
        {
            "workflow": {"mesh_type": "healpix"},
            "resolutions": {"meter": 50000.0},
            "paths": {
                "output_workspace": out_dir,
                "output_mesh": "custom_mesh.geojson",
            },
        }
    )

    assert workflow.output_workspace == out_dir
    assert workflow.mesh_file == "custom_mesh.geojson"


def test_mpas_workflow_create_hpc_job_defaults(monkeypatch):
    captured = {}

    class FakeMPASCase:
        def __init__(self, config, **kwargs):
            self.sWorkspace_output = config.get("sWorkspace_output", "./output")
            self.sFilename_mesh = "mpas.geojson"

        def _mpas_create_hpc_job(self, sSlurm_in=None, hours_in=10):
            captured["slurm"] = sSlurm_in
            captured["hours"] = hours_in

    monkeypatch.setattr(
        "pyearthmesh.meshes.unstructured.mpas.mpascase.mpascase",
        FakeMPASCase,
    )

    workflow = create_workflow(
        {
            "workflow": {"mesh_type": "mpas"},
            "paths": {"output_workspace": "./output"},
            "jigsaw": {
                "slurm": "custom_partition",
                "hours": 8,
            },
        }
    )

    # Calling create_hpc_job with no arguments picks up configured jigsaw values
    workflow.create_hpc_job()
    assert captured["slurm"] == "custom_partition"
    assert captured["hours"] == 8
    assert workflow.output_workspace == "./output"
    assert workflow.mesh_file == "mpas.geojson"


def test_workflow_package_exports():
    import pyearthmesh.config.workflows as wf

    expected_exports = [
        "BaseCase",
        "BaseWorkflow",
        "WorkflowSpec",
        "HealpixCase",
        "HealpixWorkflow",
        "ISEACase",
        "ISEAWorkflow",
        "LatLonCase",
        "LatLonWorkflow",
        "MPASCase",
        "MPASWorkflow",
        "RHEALPixCase",
        "RHEALPixWorkflow",
    ]
    for name in expected_exports:
        assert hasattr(
            wf, name
        ), f"pyearthmesh.config.workflows is missing export: {name}"


def test_case_hierarchy_and_api_uniformity():
    from pyearthmesh.config.workflows import (
        BaseCase,
        HealpixCase,
        ISEACase,
        LatLonCase,
        MPASCase,
        RHEALPixCase,
    )
    from pyearthmesh.meshes.unstructured.mpas.mpascase import mpascase

    # MPASCase is the modern alias for mpascase
    assert MPASCase is mpascase

    cases = [HealpixCase, ISEACase, LatLonCase, RHEALPixCase, MPASCase]
    required_methods = ["setup", "run", "create_hpc_job"]
    required_properties = ["output_workspace", "mesh_file", "config"]

    for case_cls in cases:
        assert issubclass(
            case_cls, BaseCase
        ), f"{case_cls.__name__} must inherit from BaseCase"
        for method in required_methods:
            assert hasattr(case_cls, method) and callable(
                getattr(case_cls, method)
            ), f"{case_cls.__name__} missing callable {method}"
        for prop in required_properties:
            assert hasattr(
                case_cls, prop
            ), f"{case_cls.__name__} missing property {prop}"


def test_workflows_case_adapter_consistency(tmp_path):
    import os

    from pyearthmesh.config.workflows import BaseCase

    out_dir = str(tmp_path / "test_out")
    test_specs = [
        (
            "healpix",
            {
                "workflow": {"mesh_type": "healpix"},
                "resolutions": {"meter": 1000.0},
                "paths": {"output_workspace": out_dir},
            },
        ),
        (
            "latlon",
            {
                "workflow": {"mesh_type": "latlon"},
                "resolutions": {"degree": 1.0},
                "paths": {"output_workspace": out_dir},
            },
        ),
        (
            "isea",
            {
                "workflow": {"mesh_type": "isea"},
                "resolutions": {"meter": 1000.0},
                "paths": {"output_workspace": out_dir},
            },
        ),
        (
            "rhealpix",
            {
                "workflow": {"mesh_type": "rhealpix"},
                "resolutions": {"meter": 1000.0},
                "paths": {"output_workspace": out_dir},
            },
        ),
    ]

    for mesh_type, config in test_specs:
        wf = create_workflow(config)
        assert isinstance(
            wf.case, BaseCase
        ), f"{mesh_type} workflow case is not a BaseCase"
        assert wf.output_workspace == os.path.abspath(out_dir)
        assert wf.case.output_workspace == os.path.abspath(out_dir)
        assert isinstance(wf.mesh_file, str)
        assert isinstance(wf.case.mesh_file, str)

        # Structured cases raise NotImplementedError for HPC jobs
        with pytest.raises(NotImplementedError):
            wf.create_hpc_job()
        with pytest.raises(NotImplementedError):
            wf.case.create_hpc_job()


def test_mpas_case_hpc_job_preserves_cwd(tmp_path):
    import os

    from pyearthmesh.config.workflows import MPASCase

    out_dir = str(tmp_path / "mpas_out")
    config = {
        "sWorkspace_output": out_dir,
        "sFilename_mesh": "mpas_mesh.geojson",
        "iCase_index": 1,
        "sDate": "20261013",
        "sModel": "jigsaw",
    }

    initial_cwd = os.getcwd()
    case = MPASCase(config, iFlag_create_directory_in=1, sWorkspace_output_in=out_dir)

    expected_output_dir = os.path.join(out_dir, case.sCase)
    assert case.output_workspace == expected_output_dir
    assert case.sWorkspace_output == expected_output_dir
    assert case.mesh_file == os.path.join(expected_output_dir, "mpas.geojson")
    assert case.config["sWorkspace_output"] == expected_output_dir

    # Running create_hpc_job must NOT change current working directory
    case.create_hpc_job(slurm="slurm", hours=5)
    assert os.getcwd() == initial_cwd, "create_hpc_job must not mutate os.getcwd()"

    # Backward compatibility alias
    case._mpas_create_hpc_job(sSlurm_in="slurm", hours_in=5)
    assert (
        os.getcwd() == initial_cwd
    ), "_mpas_create_hpc_job must not mutate os.getcwd()"


def test_yaml_workflow_loading_with_comments(tmp_path):
    yaml_content = """# Top-level comment
workflow:
  date: "20261031" # Inline comment for date
  mesh_type: latlon # Target regular mesh
  case_index: 3
  model: latlon
  debug: true

# Resolution block
resolutions:
  degree: 0.25 # quarter degree

# Output path block
paths:
  output_workspace: ./test_latlon_out # Workspace directory
  output_mesh: latlon_mesh.geojson

region:
  bbox:
    left: 100.0
    right: 110.0
    bottom: 20.0
    top: 30.0
"""
    yaml_file = tmp_path / "custom_workflow.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = load_workflow_config(yaml_file)
    assert config["workflow"]["mesh_type"] == "latlon"
    assert config["workflow"]["date"] == "20261031"
    assert config["workflow"]["case_index"] == 3
    assert config["resolutions"]["degree"] == 0.25
    assert config["region"]["bbox"]["left"] == 100.0

    compiled = compile_workflow(yaml_file)
    assert compiled["mesh_type"] == "latlon"
    assert compiled["dResolution_degree"] == 0.25
    assert compiled["ncolumn"] == 40
    assert compiled["nrow"] == 40


def test_yaml_and_json_parity_for_examples():
    from pathlib import Path

    base_examples = Path(__file__).resolve().parent.parent / "examples"
    mesh_families = ["latlon", "mpas", "healpix", "isea", "rhealpix"]

    for mesh in mesh_families:
        yaml_path = base_examples / mesh / "workflow_config.yaml"
        json_path = base_examples / mesh / "workflow_config.json"

        assert yaml_path.is_file(), f"Missing {yaml_path}"
        assert json_path.is_file(), f"Missing {json_path}"

        config_yaml = load_workflow_config(yaml_path)
        config_json = load_workflow_config(json_path)

        compiled_yaml = compile_workflow(config_yaml)
        compiled_json = compile_workflow(config_json)

        # Both YAML and JSON should produce identical compiled keys and values
        # (excluding sFilename_model_configuration which records the specific config file path)
        for k in compiled_yaml:
            if k == "sFilename_model_configuration":
                continue
            assert (
                k in compiled_json
            ), f"Key {k} missing in JSON compiled output for {mesh}"
            assert compiled_yaml[k] == compiled_json[k], (
                f"Value mismatch for key {k} in mesh {mesh}: "
                f"{compiled_yaml[k]} (YAML) != {compiled_json[k]} (JSON)"
            )


def test_directory_and_stem_resolution(tmp_path):
    # Test directory containing workflow_config.yaml
    sub_dir = tmp_path / "mesh_case"
    sub_dir.mkdir()
    yaml_file = sub_dir / "workflow_config.yaml"
    yaml_file.write_text(
        "workflow:\n  mesh_type: latlon\nresolutions:\n  degree: 1.0\n",
        encoding="utf-8",
    )

    # Pass directory directly
    cfg1 = load_workflow_config(sub_dir)
    assert cfg1["workflow"]["mesh_type"] == "latlon"
    assert cfg1["config_file"] == yaml_file.resolve()

    # Pass stem without suffix
    stem_path = sub_dir / "workflow_config"
    cfg2 = load_workflow_config(stem_path)
    assert cfg2["workflow"]["mesh_type"] == "latlon"

    # Non-existent directory raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_workflow_config(tmp_path / "empty_dir_that_does_not_exist")


def test_dump_workflow_config_roundtrip(tmp_path):
    cfg_initial = (
        load_workflow_config(
            {
                "workflow": {"mesh_type": "healpix", "case_index": 5},
                "resolutions": {"meter": 25000.0},
                "paths": {"output_workspace": str(tmp_path / "out")},
            }
            if False
            else tmp_path / "nonexistent"
        )
        if False
        else None
    )

    # Load from minimal dict via dump
    test_dict = {
        "workflow": {
            "date": "20261031",
            "mesh_type": "healpix",
            "case_index": 2,
            "model": "healpix",
            "debug": False,
            "run_on_hpc": False,
        },
        "resolutions": {"meter": 25000.0},
        "paths": {
            "output_workspace": str(tmp_path / "out"),
            "output_mesh": "healpix.geojson",
        },
        "features": {"use_tiles": True, "tile_mode": "base"},
    }

    yaml_target = tmp_path / "dumped.yaml"
    dump_workflow_config(test_dict, yaml_target, fmt="yaml")
    assert yaml_target.is_file()

    loaded_yaml = load_workflow_config(yaml_target)
    assert loaded_yaml["workflow"]["mesh_type"] == "healpix"
    assert loaded_yaml["resolutions"]["meter"] == 25000.0
    assert loaded_yaml["features"]["use_tiles"] is True

    json_target = tmp_path / "dumped.json"
    dump_workflow_config(test_dict, json_target, fmt="json")
    assert json_target.is_file()

    loaded_json = load_workflow_config(json_target)
    assert loaded_json["workflow"]["mesh_type"] == "healpix"
    assert loaded_json["resolutions"]["meter"] == 25000.0


def test_load_workflow_config_invalid_structure(tmp_path):
    invalid_file = tmp_path / "invalid.yaml"
    invalid_file.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping/dictionary object"):
        load_workflow_config(invalid_file)
