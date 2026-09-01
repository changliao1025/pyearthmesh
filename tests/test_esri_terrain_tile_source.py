import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from pyearthviz.map.raster_map_servers import RasterTileServer


def test_esri_terrain_tile_source_draws_without_broadcast_error():
    server = RasterTileServer('Esri.Terrain')
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([99.5, 101.0, 36.5, 37.5], crs=ccrs.PlateCarree())

    ax.add_image(server.get_cartopy_source(), 9)
    fig.canvas.draw()

    assert fig.canvas is not None
