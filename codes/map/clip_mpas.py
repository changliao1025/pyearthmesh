
import os

from pyearth.toolbox.analysis.extract.clip_vector_by_bounding_box import clip_vector_by_bounding_box

sWorkspace_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sWorkspace_output = os.path.join(sWorkspace_project, 'data', 'qinghaihu')

sFilename_mesh = "C:\\Users\\chang\\scratch\\04model\\jigsaw\\qinghaohu\\jigsaw20260801003\\mpas.geojson"

sFilename_basin_boundary = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'qinghaihu', 'basin_boundary_qinghaihu.geojson')

sFilename_out = os.path.join(sWorkspace_output, 'mpas_clipped.geojson')

clip_vector_by_bounding_box(sFilename_mesh, sFilename_basin_boundary, sFilename_out)

print('finished')