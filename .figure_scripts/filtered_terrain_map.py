import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.append(os.path.abspath('.'))

from stc_planner import STCPlanner
from mstc_star_planner import MSTCStarPlanner
from mstc_planner import MSTCPlanner
from mfc_planner import MFCPlanner
from utils.nx_graph import nx_graph_read, mst


color = ['r', 'm', 'b', 'k', 'c', 'g']

prefix = 'demo_4x3'
R = [(1, 1), (2, 1)]
H = nx_graph_read(f'data/nx_graph/{prefix}.graph')
nodes = [(i, j) for i in range(5) for j in range(10)]


stc_planner = STCPlanner(H)
planner = MSTCPlanner(H, len(R), R, float('inf'))

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

# plt.title('Filtered Terrain Map', fontdict={'fontname':'Times New Roman', 'size': 20})
plt.grid(True, which='both', axis='both')
# plt.axis('off')

G = nx.Graph()
node_set = set(nodes)
for node in H.nodes():
    node_set.remove(node)
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

# TODO: use gradient color as edge color
colors = cm.get_cmap('autumn')

# spanning graph
for s, t in H.edges():
    # w = H[s][t]['weight']
    x1, y1 = s
    x2, y2 = t
    ax.plot([x1, x2], [y1, y2], '-ok', mec='k', mfc='r', ms=8)

plans = planner.allocate()
for idx, val in enumerate(plans.items()):
    c = color[idx % len(color)]
    ### for roots plot
    # ax.plot(R[idx][0], R[idx][1], '*', mec='k', mfc=c, ms=18)
    ## for covering nodes plot
    depot, serv_pts = val
    xs, ys = zip(*serv_pts)
    ax.plot(xs, ys, '-o'+c, mec='k', mfc='w', alpha=0.5, ms=5, lw=10)

obs_nodes = list(node_set)
obs_graph = nx.Graph()
obs_graph.add_nodes_from(obs_nodes)
for x, y in obs_graph.nodes():
    for ix, iy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ngb = (x + ix, y + iy)
        if ngb in node_set:
            obs_graph.add_edge((x, y), ngb)

    ax.plot(x, y, 'xk', ms=10, mew=3)

for s, t in obs_graph.edges():
    x1, y1 = s
    x2, y2 = t
    ax.plot([x1, x2], [y1, y2], '-xk', ms=10, mew=3)

# plt.savefig('.figure_scripts/stc_node_demo.pdf')
plt.show()
