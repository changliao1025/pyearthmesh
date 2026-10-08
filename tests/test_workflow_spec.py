from pyearthmesh.config.workflows import WorkflowSpec


def test_workflow_spec_from_dict_and_round_trip():
    spec = WorkflowSpec.from_dict(
        {
            "mesh_type": "mpas",
            "case_index": 3,
            "date": "20260916",
            "region": {"geometry": "region.geojson", "global": False},
            "resolution": {"kind": "km", "ocean_km": 100.0},
            "paths": {"output_workspace": "./output"},
            "execution": {"debug": True},
            "features": {"coastline": True},
        }
    )

    assert spec.mesh_type == "mpas"
    assert spec.case_index == 3
    assert spec.region["geometry"] == "region.geojson"
    assert spec.to_dict()["mesh_type"] == "mpas"


def test_workflow_spec_rejects_unknown_mesh_type():
    try:
        WorkflowSpec.from_dict({"mesh_type": "unknown_mesh"})
        assert False, "Expected ValueError for unsupported mesh type"
    except ValueError:
        pass
