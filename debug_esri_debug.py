import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pyearthviz.map.raster_map_servers import RasterTileServer

server = RasterTileServer('Esri.Terrain')
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([99.5, 101.0, 36.5, 37.5], crs=ccrs.PlateCarree())
try:
    ax.add_image(server.get_cartopy_source(), 9)
    print('after add_image')
    fig.canvas.draw()
    print('after draw')
    fig.savefig('esri_test.png', dpi=100)
    print('saved')
except Exception as e:
    print('ERROR:', type(e), e)
    traceback.print_exc()
