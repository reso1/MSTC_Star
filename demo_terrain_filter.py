import os

from utils.nx_graph import nx_graph_write
from terrainFilter.terrain_filter import TerrainFilter
from terrainFilter.tif_processing import readDEMTif, readSatMapTifSegNet

depot = (10, 10)
target_graph_shape = (256, 256)

DEM_tif = readDEMTif(os.path.join('data', 'geo_data', 'DEM', 'terrain1.tif'))
filtered_DEM = TerrainFilter.filterDEM(DEM_tif)

satmap_tif = readSatMapTifSegNet(os.path.join('data', 'geo_data', 'satmap', 'terrain1.tif'))
pklPath = os.path.join('data', 'models', 'args.pkl')
modelPath = os.path.join('data', 'models', 'checkpoint_step_10000.pth')
satmap_seg = TerrainFilter.filterSatMap(satmap_tif, pklPath, modelPath)

fused_map = TerrainFilter.fuseFilteredMap(filtered_DEM, satmap_seg, depot)
G = TerrainFilter.generateSpanningGraph(fused_map, depot, DEM_tif, target_graph_shape)
TerrainFilter.display(G, 128, 128)
# nx_graph_write(G, 'terrain.graph')
