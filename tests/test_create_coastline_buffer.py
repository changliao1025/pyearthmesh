import os
from unittest.mock import MagicMock, patch

from pyearthmesh.utility.create_coastline_buffer import create_coastline_buffer


def test_create_coastline_buffer_mocked(tmp_path):
    output_file = str(tmp_path / "coastline_buffer.parquet")

    with patch(
        "pyearthmesh.utility.create_coastline_buffer.create_land_ocean_mask_from_naturalearth",
        return_value=(None, output_file),
    ) as mock_shoreline, patch(
        "pyearthmesh.utility.create_coastline_buffer.ogr.Open"
    ) as mock_open:
        mock_ds = MagicMock()
        mock_layer = MagicMock()
        mock_layer.GetFeatureCount.return_value = 8
        mock_ds.GetLayer.return_value = mock_layer
        mock_open.return_value = mock_ds

        result = create_coastline_buffer(
            dThreshold_area_island=1000.0,
            dDistance_buffer_meter=5000.0,
            sFilename_out=output_file,
            iFlag_antarctic_in=0,
            sResolution_nature_earth_coastline_in="110m",
            iFlag_simplify_in=1,
            iFlag_simplify_coastline_in=1,
            dTolerance_simplify_in=0.005,
            iFlag_verbose_in=0,
        )

        assert result == 8
        mock_shoreline.assert_called_once_with(
            sWorkspace_output=str(tmp_path),
            sFilename_land_vector_in=None,
            sResolution_nature_earth="110m",
            iFlag_antarctic=0,
            dThreshold_area_island=1000.0,
            iFlag_simplify_coastline=True,
            dTolerance_simplify=0.005,
            dDistance_buffer_meter=5000.0,
            iFlag_simplify_buffer=1,
            sFilename_vector_out=output_file,
            iFlag_convert_raster=False,
            verbose=False,
        )
