import tempfile
import unittest
from pathlib import Path

from pyearthmesh.meshes.generate_mesh import (
    determine_mesh_boundary,
    generate_mesh,
)


class TestGenerateMesh(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_determine_mesh_boundary_wkt(self):
        explicit = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
        wkt, extent = determine_mesh_boundary(pBoundary_wkt=explicit)
        self.assertEqual(wkt, explicit)
        self.assertIsNone(extent)

    def test_determine_mesh_boundary_bbox(self):
        wkt, extent = determine_mesh_boundary(
            dLongitude_left=10.0,
            dLongitude_right=20.0,
            dLatitude_bot=30.0,
            dLatitude_top=40.0,
            iFlag_global=0,
        )
        self.assertIsNotNone(wkt)
        self.assertTrue(wkt.startswith("POLYGON"))
        self.assertEqual(extent, (10.0, 20.0, 30.0, 40.0))

    def test_determine_mesh_boundary_global(self):
        wkt, extent = determine_mesh_boundary(iFlag_global=1)
        self.assertIsNone(wkt)
        self.assertIsNone(extent)

    def test_generate_latlon_mesh_from_bbox(self):
        mesh_file = str(self.workspace / "latlon.geojson")
        cells = generate_mesh(
            sMesh_type="latlon",
            sFilename_mesh=mesh_file,
            dLongitude_left=100.0,
            dLongitude_right=102.0,
            dLatitude_bot=30.0,
            dLatitude_top=32.0,
            dResolution_degree=1.0,
            iFlag_global=0,
        )
        self.assertIsInstance(cells, list)
        self.assertEqual(len(cells), 4)  # 2x2 grid
        self.assertTrue(Path(mesh_file).is_file())

    def test_generate_mesh_from_case_object(self):
        class DummyCase:
            iMesh_type = 3  # latlon
            sMesh_type = "latlon"
            sFilename_mesh = str(self.workspace / "case_latlon.geojson")
            dLongitude_left = 100.0
            dLongitude_right = 101.0
            dLatitude_bot = 30.0
            dLatitude_top = 31.0
            dLatitude_mean = 30.5
            dLongitude_mean = 100.5
            dResolution_degree = 0.5
            dResolution_meter = None
            iFlag_global = 0
            iFlag_save_mesh = 1
            iFlag_mesh_boundary = 0
            iFlag_rotation = 0

        case = DummyCase()
        cells = generate_mesh(case)
        self.assertIsInstance(cells, list)
        self.assertEqual(len(cells), 4)
        self.assertTrue(Path(case.sFilename_mesh).is_file())


if __name__ == "__main__":
    unittest.main()
