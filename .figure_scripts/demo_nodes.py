import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.append(os.path.abspath('.'))

from utils.nx_graph import nx_graph_read
from stc_planner import STCPlanner


R = [(1, 1), (2, 1)]
H = nx_graph_read(f'data/nx_graph/demo_4x3.graph')
stc_planner = STCPlanner(H)

fig = plt.figure()
ax = plt.axes()

fig.set_size_inches(4, 3)
fig.tight_layout()

major_ticks = np.arange(0, 4, 2)
minor_ticks = np.arange(0, 4, 1)

ax.set_xlim(-1, 4)
ax.set_ylim(-1, 3)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

plt.grid(True, which='both', axis='both')

# spanning nodes and covering nodes
G = nx.Graph()
for node in H.nodes():
    # ax.plot(node[0], node[1], 'o', mfc='r', mec='k', ms=8)
    direction = ['SE', 'SW', 'NE', 'NW']
    covering_nodes = [
        stc_planner.__get_subnode_coords__(node, d) for d in direction]
    for x, y in covering_nodes:
        ax.plot(x, y, 'o', mec='k', mfc='w', alpha=0.8, ms=5)
        G.add_node((x, y))

min_x = min(G.nodes(), key=lambda x: x[0])[0]
min_y = min(G.nodes(), key=lambda x: x[1])[1]
max_x = max(G.nodes(), key=lambda x: x[0])[0]
max_y = max(G.nodes(), key=lambda x: x[1])[1]
for i, j in G.nodes():
    inc = [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5),
           (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (0.5, 0.5)]
    for ci, cj in inc:
        ni, nj = i + ci, j + cj
        if min_x <= ni <= max_x and min_y <= nj <= max_y:
            G.add_edge((i, j), (ni, nj))

# covering graph
for s, t in G.edges():
    x1, y1 = s
    x2, y2 = t
    ax.plot([x1, x2], [y1, y2], '--k', alpha=0.5, lw=0.5)

# spanning graph
# for s, t in H.edges():
#     x1, y1 = s
#     x2, y2 = t
#     ax.plot([x1, x2], [y1, y2], '-ok', mec='k', mfc='r', ms=8)

# depots
# for x, y in R:
#     ax.plot(x, y, '*', mec='k', mfc='r', ms=18)

plt.show()
