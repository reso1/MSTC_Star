import math

import networkx as nx
import matplotlib.animation
import matplotlib.pyplot as plt

from utils.robot import Robot
from utils.disjoint_set import DisjointSet


def nx_graph_write(G: nx.Graph, filepath):
    f = open(filepath, 'w')
    f.writelines('#'.join([str(node) for node in G.nodes()]) + '\n')
    for s, t in G.edges():
        f.writelines('#'.join([str(s), str(t), str(G[s][t]['weight'])])+'\n')
    f.close()


def nx_graph_read(filepath):
    G = nx.Graph()
    f = open(filepath, 'r')
    lines = f.readlines()
    for node_str in lines[0].split('#'):
        node_x, node_y = node_str.strip().split(', ')
        # G.add_node((int(node_x[1:]), int(node_y[:-1])))
        G.add_node((float(node_x[1:]), float(node_y[:-1])))
    for edge_str in lines[1:]:
        s, t, w = edge_str.strip().split('#')
        sx, sy = s.split(', ')
        # s = (int(sx[1:]), int(sy[:-1]))
        s = (float(sx[1:]), float(sy[:-1]))
        tx, ty = t.split(', ')
        # t = (int(tx[1:]), int(ty[:-1]))
        t = (float(tx[1:]), float(ty[:-1]))
        G.add_edge(s, t, weight=float(w))

    return G


def mst(G: nx.Graph):
    M = G.copy()

    # init edges
    costs = {}
    for s, t in list(M.edges()):
        costs[(s, t)] = M[s][t]['weight']
        M.remove_edge(s, t)

    # construct MST
    node_map = {}
    disjoint_set = DisjointSet()
    sorted_edges = sorted(costs.items(), key=lambda x: x[1])

    for n in M.nodes():
        node_map[n] = disjoint_set.make(n)

    for edge, cost in sorted_edges:
        s, t = node_map[edge[0]], node_map[edge[1]]
        root_s, root_t = disjoint_set.find(s), disjoint_set.find(t)

        if root_s != root_t:
            M.add_edge(s.data, t.data, weight=cost)
            disjoint_set.union(root_s, root_t)

    return M


def navigate(G, start, goal):
    def dist(p, q):
        # return abs(p[0]-q[0]) + abs(p[1]-q[1])
        return math.hypot(p[0]-q[0], p[1]-q[1])

    return nx.astar_path(G, start, goal, heuristic=dist, weight='weight')


def graph_plot(G: nx.Graph, is_show=False):
    # plot nodes
    plt.subplot(1, 2, 1)
    for i, j in G.nodes():
        plt.scatter(i, j, c='k')

    # plot edges
    plt.subplot(1, 2, 2)
    for s, t in G.edges():
        x1, y1 = s
        x2, y2 = t
        plt.plot([x1, x2], [y1, y2], '-ok')

    if is_show:
        plt.show()


def trajectory_plot(G: nx.Graph, traj, is_anim=False):
    fig = plt.figure()
    ax = plt.axes()
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    if G.number_of_nodes() == 1:
        for x, y in G.nodes():
            plt.plot(x, y, 'ok')
    else:
        for s, t in G.edges():
            x1, y1 = s
            x2, y2 = t
            plt.plot([x1, x2], [y1, y2], '-ok')

    L = len(traj)
    xs, ys = zip(*traj)

    if is_anim:
        line, = ax.plot([], [], '-c', lw=1.5)
        marker, = ax.plot([], [], 'or', ms=5)
        for i in range(L):
            line.set_data(xs[:i+1], ys[:i+1])
            marker.set_data([xs[i], ys[i]])
            plt.pause(0.25)
    else:
        for i in range(L-1):
            plt.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], '-or', ms=3, lw=1.5)

    plt.show()


def animation(G: nx.Graph, traj_vec, k):
    color = ['r', 'm', 'b', 'k', 'c', 'g']

    fig = plt.figure()
    ax = plt.axes()
    ax.margins(x=0.1, y=0.1)
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    for s, t in G.edges():
        x1, y1 = s
        x2, y2 = t
        plt.plot([x1, x2], [y1, y2], '-ok')

    # init
    lines, markers, xs_vec, ys_vec = {}, {}, {}, {}

    for i in range(k):
        c = color[i % len(color)]
        line, = ax.plot([], [], '-'+c, lw=1.5)
        marker, = ax.plot([], [], 'o'+c, ms=5)
        lines[i], markers[i] = line, marker
        xs_vec[i], ys_vec[i] = zip(*traj_vec[i])

    num_of_frames = [len(traj) for traj in traj_vec]
    max_num_frames = max(num_of_frames)

    for i_frame in range(max_num_frames):
        for i in range(k):
            j = min(i_frame, num_of_frames[i]-1)
            lines[i].set_data(xs_vec[i][:j+1], ys_vec[i][:j+1])
            markers[i].set_data([xs_vec[i][j], ys_vec[i][j]])
        plt.pause(0.25)

    plt.show()


def record(G: nx.Graph, R, traj_vec, weights, k, prefix, is_show=False):
    color = ['r', 'm', 'b', 'k', 'c', 'g']
    fig = plt.figure()
    ax = plt.axes()
    ax.margins(x=0.1, y=0.1)
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    lines, markers, xs_vec, ys_vec = [None]*k, [None]*k, [None]*k, [None]*k
    frames = [len(traj) for traj in traj_vec]

    def init():
        plt.title(f'{prefix}:(W = {weights: .3f})')

        for s, t in G.edges():
            x1, y1 = s
            x2, y2 = t
            plt.plot([x1, x2], [y1, y2], '-ok')

        for i in range(k):
            c = color[i % len(color)]
            line, = ax.plot([], [], '-'+c, lw=1.5)
            marker, = ax.plot([], [], 'o'+c, ms=5)
            lines[i], markers[i] = line, marker
            xs_vec[i], ys_vec[i] = zip(*traj_vec[i])
            plt.plot(R[i][0], R[i][1], '*k', ms=10, markerfacecolor='w')
            plt.text(R[i][0]+0.1, R[i][1]+0.1, f'R{i}')

        return lines + markers

    def animate(i_frame):
        for i in range(k):
            j = min(i_frame, frames[i]-1)
            lines[i].set_data(xs_vec[i][:j+1], ys_vec[i][:j+1])
            markers[i].set_data([xs_vec[i][j], ys_vec[i][j]])

        return lines + markers

    anim = matplotlib.animation.FuncAnimation(
        fig, animate, max(frames), init, interval=30, blit=True, repeat=False)

    anim.save(f'results/gif/{prefix}.gif', fps=30)

    if is_show:
        plt.show()


def simulation(
        planner, P, W, prefix, dt, OG=nx.Graph(), is_write=False, is_show=False):

    k, R = planner.k, planner.R
    color = ['r', 'm', 'b', 'k', 'c', 'g']
    fig = plt.figure()
    fig.set_size_inches(8, 8)
    fig.tight_layout()
    ax = plt.axes()
    # ax.margins(x=0.15, y=0.15)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.grid(True)
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    robots = [Robot(P[i], planner.H) for i in range(k)]
    t_finish = [robots[i].T[-1] for i in range(k)]
    t_max = max(t_finish)

    if not is_write and not is_show:
        print(f'Final Max Weights: {max(W)}')
        return

    lines, markers, texts = [None]*k, [None]*k, [None]*(k+1)
    xs_vec, ys_vec = [None]*k, [None]*k

    def init():
        plt.title(f'{prefix} (Max Weights={max(W): .2f})')
        texts[-1] = ax.text(
            1, 1, '', va='top', ha='right', transform=ax.transAxes,
            font={'size': 8})

        # MST of spanning graph
        M = mst(planner.G)
        for s, t in M.edges():
            x1, y1 = s
            x2, y2 = t
            ax.plot([x1, x2], [y1, y2], '-ok', mfc='r')
        # covering nodes
        rho = planner.generate_cover_trajectory(R[0], mst(planner.G))
        for cn_x, cn_y in rho:
            ax.plot(cn_x, cn_y, 'o', mec='k', mfc='w', ms=5)
        # obstacle graph
        for s, t in OG.edges():
            x1, y1 = s
            x2, y2 = t
            ax.plot([x1, x2], [y1, y2], '-xk', ms=10, mew=3)

        for i in range(k):
            c = color[i % len(color)]
            line, = ax.plot([], [], '-'+c, alpha=0.35, lw=8)
            marker, = ax.plot([], [], 'o'+c, ms=8)
            # changable texts
            texts[i] = ax.text(
                1, 0.975-i*0.025, '', va='top', ha='right',
                transform=ax.transAxes, font={'size': 8})
            # trajectories and robots
            lines[i], markers[i] = line, marker
            xs_vec[i], ys_vec[i] = zip(*P[i])
            # depots
            ax.plot(R[i][0], R[i][1], '*k', mfc=c, ms=10)
            # ax.text(R[i][0]+0.1, R[i][1]+0.1, f'R{i}')

        return lines + markers + texts

    # record remaining uncovered nodes
    uncovered = set()
    direction = ['SE', 'NE', 'NW', 'SW']
    for node in planner.G.nodes:
        for sn in [planner.__get_subnode_coords__(node, d) for d in direction]:
            uncovered.add(sn)

    def animate(ti):
        ts = ti * dt
        for i in range(k):
            last_coord_idx, cur_state = robots[i].get_cur_state(ts)
            xs = xs_vec[i][:last_coord_idx+1] + (cur_state.x, )
            ys = ys_vec[i][:last_coord_idx+1] + (cur_state.y, )
            # texts[i].set_text(f'R{i}: ')
            lines[i].set_data(xs, ys)
            markers[i].set_data(cur_state.x, cur_state.y)
            node = (xs_vec[i][last_coord_idx], ys_vec[i][last_coord_idx])
            if node in uncovered:
                uncovered.remove(node)
        texts[-1].set_text(f'T[s]={ts: .2f}, # of uncovered={len(uncovered)}')

        return lines + markers + texts

    anim = matplotlib.animation.FuncAnimation(
        fig, animate, int(t_max/dt), init, interval=5,
        blit=True, repeat=False, cache_frame_data=False)

    def func(i, n):
        print(f'Saving frame {i} of {n}')

    if is_write:
        anim.save(f'results/gif/{prefix}.mp4', fps=300, dpi=200, progress_callback=func)

    if is_show:
        plt.show()


def calc_overlapping_ratio(traj_vec, rho):
    num_of_overlapped = 0
    rho_pt_visited = {pt: False for pt in rho}
    for traj in traj_vec:
        for pt in traj:
            if pt in rho_pt_visited.keys():
                if rho_pt_visited[pt]:
                    num_of_overlapped += 1
                else:
                    rho_pt_visited[pt] = True

    return num_of_overlapped / len(rho)


def show_result(G: nx.Graph, traj_vec, k):
    color = ['r', 'm', 'b', 'k', 'c', 'g']

    fig = plt.figure()
    ax = plt.axes()
    ax.margins(x=0.1, y=0.1)
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    for s, t in G.edges():
        x1, y1 = s
        x2, y2 = t
        plt.plot([x1, x2], [y1, y2], '-ok')

    for i in range(k):
        c = color[i % len(color)]
        xs, ys = zip(*traj_vec[i])
        ax.plot(xs, ys, '-o'+c, mec='k', mfc='w', alpha=0.5, ms=5, lw=10)

    plt.show()