import tempfile
import unittest
from pathlib import Path

from pyearthmesh.meshes.unstructured.mpas.mpascase import mpascase


class TestMPASCaseBoundary(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_explicit_boundary_wkt(self):
        wkt = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"
        case = mpascase(
            {
                "sWorkspace_output": str(self.workspace),
                "pBoundary_wkt": wkt,
                "iFlag_global": 0,
            },
            iFlag_create_directory_in=0,
        )
        self.assertEqual(case.get_boundary_wkt(), wkt)

    def test_bbox_boundary_resolution(self):
        case = mpascase(
            {
                "sWorkspace_output": str(self.workspace),
                "iFlag_global": 0,
                "dLongitude_left": 95.0,
                "dLongitude_right": 120.0,
                "dLatitude_bot": 30.0,
                "dLatitude_top": 42.0,
            },
            iFlag_create_directory_in=0,
        )
        wkt = case.get_boundary_wkt()
        self.assertIsNotNone(wkt)
        self.assertTrue(wkt.startswith("POLYGON"))
        self.assertAlmostEqual(case.dLongitude_mean, 107.5)
        self.assertAlmostEqual(case.dLatitude_mean, 36.0)

    def test_global_mesh_returns_none(self):
        case = mpascase(
            {
                "sWorkspace_output": str(self.workspace),
                "iFlag_global": 1,
            },
            iFlag_create_directory_in=0,
        )
        self.assertIsNone(case.get_boundary_wkt())

    def test_resolution_conversion(self):
        case = mpascase(
            {
                "sWorkspace_output": str(self.workspace),
                "dResolution_degree": 0.05,
                "dLatitude_bot": 30.0,
                "dLatitude_top": 40.0,
            },
            iFlag_create_directory_in=0,
        )
        self.assertIsNotNone(case.dResolution_meter)
        self.assertGreater(case.dResolution_meter, 0.0)

    def test_dem_flag(self):
        case = mpascase(
            {
                "sWorkspace_output": str(self.workspace),
                "iFlag_use_mesh_dem": 1,
            },
            iFlag_create_directory_in=0,
        )
        self.assertEqual(case.iFlag_use_mesh_dem, 1)


if __name__ == "__main__":
    unittest.main()
