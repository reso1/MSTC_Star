import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('.'))

from stc_planner import STCPlanner
from utils.nx_graph import nx_graph_read, trajectory_plot


prefix = 'GRID_10x10_WEIGHTED'
G = nx_graph_read(f'data/nx_graph/{prefix}.graph')

fig = plt.figure()
ax = plt.axes()

fig.set_size_inches(8, 8)
fig.tight_layout(pad=3.0)

font = {'fontname': 'Times New Roman', 'size': 20}
plt.title('CCW Spiral-STC', font)
plt.grid(True, which='both', axis='both')
plt.gcf().canvas.mpl_connect(
    'key_release_event',
    lambda event: [exit(0) if event.key == 'escape' else None])

major_ticks = np.arange(0, 10, 2)
minor_ticks = np.arange(0, 10, 1)

ax.set_xlim(-1, 10)
ax.set_ylim(-1, 10)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

S, dummy_parent = (5, 5), (4, 5)
stc_planner = STCPlanner(G)
route = stc_planner.spiral_route(S, dummy_parent, G)
trajectory = stc_planner.generate_cover_trajectory(S, G) \
             + [stc_planner.__get_subnode_coords__(S, 'NW')]

# spanning nodes
N, weights_total = len(route), 0
for i in range(N-1):
    weights_total += G[route[i]][route[i+1]]['weight']
    if route[i] == dummy_parent and route[i+1] == S:
        continue
    x1, y1 = route[i]
    x2, y2 = route[i+1]
    ax.plot([x1, x2], [y1, y2], '-ok', mec='k', mfc='r', ms=8)
ax.plot(S[0], S[1], '-*k', mec='k', mfc='r', ms=18)
print(f'weights_total: {weights_total}')

# covering nodes
nodes = [(i, j) for i in range(10) for j in range(10)]
node_set = set(nodes)
for node in G.nodes():
    node_set.remove(node)
    dir = ['SE', 'SW', 'NE', 'NW']
    covering_nodes = [
        stc_planner.__get_subnode_coords__(node, d) for d in dir]
    for x, y in covering_nodes:
        ax.plot(x, y, 'o', mec='k', mfc='w', alpha=0.5, ms=5)

L = len(trajectory)
xs, ys = zip(*trajectory)
ax.plot(xs, ys, '-oc', mec='k', mfc='w', alpha=0.5, ms=5, lw=10)

# obstacle nodes
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

# plt.savefig('.figure_scripts/ccw_spiral_stc.pdf')
plt.show()
