import networkx as nx

from mcpp.mstc_star_planner import test_MSTC_STAR
from utils.nx_graph import nx_graph_read


prefix = 'GRID_5x10_UNWEIGHTED'
# R = [(2, 0), (3, 0), (4, 0)]
R = [(1, 0), (2, 0), (3, 0), (4, 0)]
# R = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
G = nx_graph_read(f'data/nx_graph/{prefix}.graph')
obs_graph = nx.grid_2d_graph(5, 10)
for node in G.nodes():
    obs_graph.remove_node(node)

test_MSTC_STAR(prefix, R, float('inf'), obs_graph, False, True)
