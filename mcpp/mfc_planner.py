import time
import networkx as nx

from mcpp.stc_planner import STCPlanner
from mcpp.rtc_planner import RTCPlanner
from utils.nx_graph import nx_graph_read, mst, navigate, show_result
from utils.nx_graph import graph_plot, trajectory_plot, calc_overlapping_ratio
from utils.nx_graph import animation, record, simulation


class MFCPlanner(STCPlanner):
    def __init__(self, G: nx.Graph, k: int, R: list, cap):
        self.G = G
        self.k = k
        self.R = R
        self.capacity = cap

        self.H = self.generate_decomposed_graph(self.G, self.R)

    def allocate(self, epsilon=1.0, debug=False):
        """ using Rooted-Tree-Cover to find k rooted-tree cover """
        res = {}

        rtc_planner = RTCPlanner(self.G, self.R, self.k)
        match_tuple, max_weights, opt_B = rtc_planner.k_tree_cover(epsilon)
        if debug:
            rtc_planner.plot_rtc_result(match_tuple)

        for r, val in match_tuple.items():
            L, S, P = val
            rho_l = self.generate_cover_trajectory(r, L)[1:]
            rho_s, i = [], 0
            if P:
                rho_s = self.generate_cover_trajectory(P[-1], S)[1:]
                while i < len(rho_s) and rho_s[i] == rho_l[-1]:
                    i += 1

            res[r] = rho_l + rho_s[i:]

        return res

    def simulate(self, plans):
        paths, weights = [[] for _ in range(self.k)], [0] * self.k
        for idx, val in enumerate(plans.items()):
            depot, serv_pts = val
            paths[idx].extend(navigate(self.H, depot, serv_pts[0]))
            L, num_of_served = len(serv_pts), 1

            for i in range(L-1):
                if num_of_served == self.capacity:
                    num_of_served = 0
                    beta = navigate(self.H, paths[idx][-1], depot)
                    alpha = navigate(self.H, depot, serv_pts[i])
                    paths[idx].extend(beta[1:-1] + alpha)

                dx = serv_pts[i+1][0] - serv_pts[i][0]
                dy = serv_pts[i+1][1] - serv_pts[i][1]
                # extend nav path
                if abs(dx) + abs(dy) != 0.5:
                    gamma = navigate(self.H, serv_pts[i], serv_pts[i+1])
                    paths[idx].extend(gamma[1:-1])

                paths[idx].append(serv_pts[i+1])
                num_of_served += 1

            if paths[idx][-1] != depot:
                paths[idx].extend(navigate(self.H, paths[idx][-1], depot)[1:])

            weights[idx] = self.__get_travel_weights__(paths[idx])

            print(f'#{idx} Total Weights: {weights[idx]}')

        print(f'---\nFinal Max Weights: {max(weights)}')

        return paths, weights


def test_MFC(prefix, R, cap, eps, obs_graph=nx.Graph(), is_write=False, is_show=False):
    k = len(R)
    G = nx_graph_read(f'data/nx_graph/{prefix}.graph')
    print(f'==== {test_MFC.__name__} on {prefix}, k={k}, cap={cap} ====')

    ts = time.time()
    planner = MFCPlanner(G, k, R, cap)
    plans = planner.allocate(epsilon=eps, debug=True)
    paths, weights = planner.simulate(plans)
    # print(f'MFC planning time elapsed: {time.time()-ts}')
    show_result(G, paths, k)
    # simulation(
    #     planner, paths, weights, 'Grid Map #1 - MFC', 0.03,
    #     obs_graph, is_write, is_show)
    # rho = planner.generate_cover_trajectory(R[0], mst(G))
    # print(f'MFC overlapping ratio: {calc_overlapping_ratio(paths, rho)}')
    print('\n')


if __name__ == '__main__':
    prefix = 'GRID_5x10_UNWEIGHTED'
    R = [(1, 0), (2, 0), (3, 0), (4, 0)]
    G = nx_graph_read(f'data/nx_graph/{prefix}.graph')
    obs_graph = nx.grid_2d_graph(5, 10)
    for node in G.nodes():
        obs_graph.remove_node(node)

    # prefix = 'GRID_10x10_WEIGHTED'
    # R = [(4, 5), (4, 6), (5, 5), (6, 5), (6, 6), (4, 7), (5, 7), (6, 7)]

    # prefix = 'TERRAIN#1'
    # R = [(16, 16), (16, 17), (17, 16), (17, 17)]
    cap = float('inf')
    test_MFC(prefix, R, cap, 1, obs_graph, True, False)
