import pytest

from pyearthmesh.config.config_manager import (
    compile_workflow,
    create_workflow,
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
            "region": {
                "bbox": {"left": -10, "right": 10, "bottom": 20, "top": 30}
            },
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

    def fake_create_isea_mesh(dResolution_meter_in, sISEA_type, sFilename_mesh, sWorkspace_output, **kwargs):
        captured["resolution"] = dResolution_meter_in
        captured["isea_type"] = sISEA_type
        captured["filename"] = sFilename_mesh
        captured["workspace"] = sWorkspace_output
        captured["kwargs"] = kwargs
        return "fake_mesh_result"

    monkeypatch.setattr(
        "pyearthmesh.meshes.structured.dggs.isea.create_isea_mesh.create_isea_mesh",
        fake_create_isea_mesh,
    )

    out_dir = str(tmp_path / "output")
    workflow = create_workflow(
        {
            "workflow": {"mesh_type": "isea"},
            "resolutions": {"meter": 5000.0},
            "paths": {"output_workspace": out_dir, "output_mesh": "isea3h_mesh.geojson"},
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

    def fake_create_rhealpix_mesh(dResolution_meter_in, sFilename_mesh, sWorkspace_output, **kwargs):
        captured["resolution"] = dResolution_meter_in
        captured["filename"] = sFilename_mesh
        captured["workspace"] = sWorkspace_output
        captured["kwargs"] = kwargs
        return "fake_rhealpix_result"

    monkeypatch.setattr(
        "pyearthmesh.meshes.structured.dggs.healpix.create_rhealpix_mesh.create_rhealpix_mesh",
        fake_create_rhealpix_mesh,
    )

    out_dir = str(tmp_path / "output")
    workflow = create_workflow(
        {
            "workflow": {"mesh_type": "rhealpix"},
            "resolutions": {"meter": 10000.0},
            "paths": {"output_workspace": out_dir, "output_mesh": "rhealpix_mesh.geojson"},
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
            "paths": {"output_workspace": out_dir, "output_mesh": "custom_mesh.geojson"},
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



