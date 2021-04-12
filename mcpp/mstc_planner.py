import time
import math
import networkx as nx

from mcpp.stc_planner import STCPlanner
from utils.nx_graph import nx_graph_read, mst, navigate, show_result, simulation
from utils.nx_graph import graph_plot, trajectory_plot
from utils.nx_graph import animation, record, calc_overlapping_ratio


class MSTCPlanner(STCPlanner):
    def __init__(self, G: nx.Graph, k, R, cap, is_backtracking_opt=False):
        self.G = G
        self.k = k
        self.R = R
        self.capacity = cap

        self.H = self.generate_decomposed_graph(self.G, self.R)
        self.rho = self.generate_cover_trajectory(R[0], mst(G))

        self.is_backtracking_opt = is_backtracking_opt

    def allocate(self):
        """ for each robot_i, move along STC path until it reaches
        the starting position of next robot (NW subnode of spanning node)
            ref: Redundancy, Efficiency and Robustness in Multi-Robot Coverage
        """
        res = {}
        start_pos = {r: self.__get_subnode_coords__(r, 'NW') for r in self.R}

        # the last point in rho is identical to the first, so find in [0, -1)
        p_idx = {pos: self.rho.index(pos, 0, -1) for pos in start_pos.values()}
        num_of_pts = len(self.rho) - 1

        # sort starting pos by index in STC path to make seamless connecting
        sorted_s_pos = sorted(start_pos.items(), key=lambda x: p_idx[x[1]])

        h = -1
        for idx, val in enumerate(sorted_s_pos):
            trajectory = []
            r, s_pos = val
            _, t_pos = sorted_s_pos[(idx+1) % self.k]

            cur_pos_idx = p_idx[s_pos]
            tar_pos_idx = (p_idx[t_pos]) % num_of_pts
            while cur_pos_idx != tar_pos_idx:
                trajectory.append(self.rho[cur_pos_idx])
                cur_pos_idx = (cur_pos_idx + 1) % num_of_pts
            res[r] = trajectory

            if len(trajectory) > num_of_pts / 2:
                h = idx

        if self.is_backtracking_opt and h != -1:
            return self.__backtracking_opt(h, res)

        return res

    def simulate(self):
        plans = self.allocate()
        paths, weights = [[] for _ in range(self.k)], [0] * self.k
        for idx, val in enumerate(plans.items()):
            depot, serv_pts = val
            path, weight = self.__sim(depot, serv_pts)
            paths[idx], weights[idx] = path, weight

        for idx in range(len(plans)):
            print(f'#{idx} Total Weights: {weights[idx]}')
        print(f'---\nFinal Max Weights: {max(weights)}')

        return paths, weights

    def __sim(self, depot, serv_pts):
        path = []

        path.extend(navigate(self.H, depot, serv_pts[0]))
        L, num_of_served = len(serv_pts), 1

        for i in range(L-1):
            if num_of_served == self.capacity:
                num_of_served = 0
                beta = navigate(self.H, path[-1], depot)
                alpha = navigate(self.H, depot, serv_pts[i])
                path.extend(beta[1:-1] + alpha)

            l1 = abs(serv_pts[i+1][0] - serv_pts[i][0]) + \
                abs(serv_pts[i+1][1] - serv_pts[i][1])

            if l1 != 0.5:
                gamma = navigate(self.H, serv_pts[i], serv_pts[i+1])
                path.extend(gamma[1:-1])

            path.append(serv_pts[i+1])
            num_of_served += 1

        if path[-1] != depot:
            path.extend(navigate(self.H, path[-1], depot)[1:])

        return path, self.__get_travel_weights__(path)

    def __backtracking_opt(self, h, sec):
        i = (h + 1) % self.k
        j = (i + 1) % self.k
        f = (j + 1) % self.k

        s_list = list(sec.items())
        st_h, sec_h = s_list[h]
        st_i, sec_i = s_list[i]
        st_j, sec_j = s_list[j]
        st_f, sec_f = s_list[f]

        sec_h_sep = math.ceil((len(sec_h) - len(sec_i)) / 2)
        sec_i_sep = math.ceil(len(sec_i) / 2)
        sec_j_sep = math.ceil(len(sec_j) / 2)
        if len(sec_i) < len(sec_j):
            sec[st_h] = sec_h[:sec_h_sep]
            sec[st_i] = sec_i[:sec_i_sep] + sec_h[sec_h_sep:][::-1]
            sec[st_j] = sec_j + sec_i[sec_i_sep:][::-1]
        elif h == f:
            sec[st_h] = sec_h[:sec_h_sep] + sec_j[sec_j_sep:][::-1]
            sec[st_i] = sec_h[sec_h_sep:][::-1]
            sec[st_j] = sec_j[:sec_j_sep] + sec_i[::-1]
        else:
            sec[st_h] = sec_h[:sec_h_sep]
            sec[st_i] = sec_h[sec_h_sep:][::-1]
            sec[st_j] = sec_j[:sec_j_sep] + sec_i[::-1]
            sec[st_f] = sec_f + sec_j[sec_j_sep:][::-1]

        return sec


def test_MSTC(prefix, R, cap, obs_graph=nx.Graph(), is_write=False, is_show=False):
    k = len(R)
    G = nx_graph_read(f'data/nx_graph/{prefix}.graph')
    print(f'==== {test_MSTC.__name__} on {prefix}, k={k}, cap={cap} ====')

    ts = time.time()
    planner = MSTCPlanner(G, k, R, cap)
    paths, weights = planner.simulate()
    # print(f'MSTC planning time elapsed: {time.time()-ts}')
    show_result(mst(G), paths, k)
    # simulation(
    #     planner, paths, weights, 'Grid Map #1 - MSTC-NB', 0.03,
    #     obs_graph, is_write, is_show)
    # print(f'MSTC overlapping ratio: {calc_overlapping_ratio(paths, planner.rho)}')
    print('\n')


def test_MSTC_BT_OPT(prefix, R, cap, obs_graph=nx.Graph(), is_write=False, is_show=False):
    k = len(R)
    G = nx_graph_read(f'data/nx_graph/{prefix}.graph')
    print(f'==== {test_MSTC_BT_OPT.__name__} on {prefix}, k={k}, cap={cap} ====')

    ts = time.time()
    planner = MSTCPlanner(G, k, R, cap, True)
    paths, weights = planner.simulate()
    # print(f'MSTC planning time elapsed: {time.time()-ts}')
    show_result(mst(G), paths, k)
    # simulation(
    #     planner, paths, weights, 'Grid Map #1 - MSTC-BO', 0.03,
    #     obs_graph, is_write, is_show)
    # print(f'MSTC overlapping ratio: {calc_overlapping_ratio(paths, planner.rho)}')
    print('\n')


if __name__ == '__main__':
    prefix = 'GRID_5x10_UNWEIGHTED'
    # R = [(2, 0), (3, 0), (4, 0)]
    R = [(1, 0), (2, 0), (3, 0), (4, 0)]
    # R = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    G = nx_graph_read(f'data/nx_graph/{prefix}.graph')
    obs_graph = nx.grid_2d_graph(5, 10)
    for node in G.nodes():
        obs_graph.remove_node(node)

    test_MSTC_BT_OPT(prefix, R, float('inf'), obs_graph, True, False)

    # prefix = 'GRID_10x10_WEIGHTED'
    # R = [(4, 5), (4, 6), (5, 5), (6, 5), (6, 6), (4, 7), (5, 7), (6, 7)]
    # prefix = 'TERRAIN#1'
    # R = [(16, 16), (16, 17), (17, 16), (17, 17)]  # k = 4
    # R = [(16, 16), (16, 17), (16, 18), (17, 18),  # k = 8
    #      (18, 18), (18, 17), (18, 16), (17, 16)]

    # cap = 200
    # # test_MSTC(prefix, R, cap, False, False)
    # test_MSTC_BT_OPT(prefix, R, cap, False, False)
