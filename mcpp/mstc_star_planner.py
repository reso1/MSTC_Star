import time
import networkx as nx

from mcpp.stc_planner import STCPlanner
from utils.nx_graph import nx_graph_read, mst, navigate
from utils.nx_graph import graph_plot, trajectory_plot, calc_overlapping_ratio
from utils.nx_graph import animation, record, simulation


class MSTCStarPlanner(STCPlanner):
    def __init__(self, G: nx.Graph, k, R, cap, cut_off_opt=True):
        self.G = G
        self.k = k
        self.R = R
        self.capacity = cap

        self.H = self.generate_decomposed_graph(self.G, self.R)
        self.rho = self.generate_cover_trajectory(R[0], mst(G))

        self.cut_off_opt = cut_off_opt

    def allocate(self, alloc_filename=None):
        """ recursively split to keep balanced,
            ref: MSP Algorithm: Multi-Robot Patrolling based on Territory
                 Allocation using Balanced Graph Partitioning
        """
        num_of_nodes = self.__split(len(self.rho)-1, self.k, {})

        start, plans = 0, {}
        for i, n in enumerate(num_of_nodes):
            end = start + n
            plans[self.R[i]] = self.rho[start:end]
            start = end

        if self.cut_off_opt:
            _, weights = self.simulate(plans)
            self.__optimal_cut_opt(weights, plans, debug=True)

        self.__write_alloc_file(plans, alloc_filename)

        return plans

    def simulate(self, plans, is_print=True):
        paths, weights = [[] for _ in range(self.k)], [0] * self.k
        for idx, val in enumerate(plans.items()):
            depot, serv_pts = val
            path, weight = self.__sim(depot, serv_pts)
            paths[idx], weights[idx] = path, weight

            if is_print:
                print(f'#{idx} Total Weights: {weights[idx]}')
        if is_print:
            print(f'---\nFinal Max Weights: {max(weights)}')

        return paths, weights

    def __write_alloc_file(self, plans, alloc_filename=None):
        if not alloc_filename:
            return

        f = open(alloc_filename, 'w')
        for idx, val in enumerate(plans.items()):
            depot, serv_pts = val
            xs, ys = zip(*serv_pts)
            ns = len(serv_pts)
            f.writelines(
                ' '.join([str(xs[i])+','+str(ys[i]) for i in range(ns)])+'\n')
        f.close()

    def __split(self, N, K, res):
        if (N, K) in res:
            return res[(N, K)]

        if K == 1:
            return [N]

        left = K // 2
        left_N = round(N * left / K)
        left_res = self.__split(left_N, left, res)
        res[(left_N, left)] = left_res

        right = K - left
        right_N = N - left_N
        right_res = self.__split(right_N, right, res)
        res[(right_N, right)] = right_res

        return left_res + right_res

    def __optimal_cut_opt(self, weights: list, plans: dict, debug=False):
        opt = max(weights)
        cur_iter, num_of_iters = 0, 1e3
        while cur_iter < num_of_iters:
            r_min = min(list(range(self.k)), key=lambda x: weights[x])
            r_max = max(list(range(self.k)), key=lambda x: weights[x])
            print(f'iter #{cur_iter}: rmin={r_min}, rmax={r_max}, max weight={opt: .3f}', end=' ')
            # clockwise cutoff opt
            clw = self.__get_intermediate_r_index(r_min, r_max, -1)
            # counter-clockwise cutoff opt
            ccw = self.__get_intermediate_r_index(r_min, r_max, 1)
            # select smaller loop
            r_index = clw if len(clw) < len(ccw) else ccw
            self.__find_optimial_cut(r_index, weights, plans, debug)

            for i in sorted(list(range(self.k)), key=lambda x: weights[x]):
                print(f', {i}: {weights[i]: .3f}', end=' ')
            print(',')

            if max(weights) >= opt:
                print('MSTC-Star Cutoff OPT Finished')
                break
            else:
                opt = max(weights)
                cur_iter += 1

    def __sim(self, depot, serv_pts):
        path = []
        depot_small = self.__get_subnode_coords__(depot, "SE")
        path.extend([depot] + navigate(self.H, depot_small, serv_pts[0]))
        L, num_of_served = len(serv_pts), 1

        for i in range(L-1):
            if num_of_served == self.capacity:
                num_of_served = 0
                beta = navigate(self.H, path[-1], depot_small)
                alpha = navigate(self.H, depot_small, serv_pts[i])
                path.extend(beta[1:-1] + [depot] + alpha)

            l1 = abs(serv_pts[i+1][0] - serv_pts[i][0]) + \
                abs(serv_pts[i+1][1] - serv_pts[i][1])

            if l1 != 0.5:
                gamma = navigate(self.H, serv_pts[i], serv_pts[i+1])
                path.extend(gamma[1:-1])

            path.append(serv_pts[i+1])
            num_of_served += 1

        if path[-1] != depot:
            path.extend(navigate(self.H, path[-1], depot_small)[1:] + [depot])

        return path, self.__get_travel_weights__(path)

    def __find_optimial_cut(self, r_index, weights, plans, debug=True):
        """ find optimal-cut point of U{P_cutoff_index} using binary search """

        plan, N = [], {}
        r_first, r_last = r_index[0], r_index[-1]
        for ri in r_index:
            plan += plans[self.R[ri]]
            N[ri] = len(plans[self.R[ri]])

        old_weight_max = max(weights)
        old_weight_sum = sum([weights[ri] for ri in r_index])
        opt = (-1, old_weight_max, old_weight_sum, {}, weights)
        first, last = 0, N[r_last] + N[r_first] - 1

        if debug:
            print(f'--- Cutoff point={N[r_first]}', end='\t')
            for ri in r_index:
                print(f'{ri}: {weights[ri]: .3f}', end='\t')
            print(f'Weight Max: {old_weight_max: .3f}, Weight Sum: {old_weight_sum: .3f}')

        old_N_r_first, old_N_r_last = N[r_first], N[r_last]
        while first < last:
            c = (first + last) // 2
            N[r_first] = c
            N[r_last] = old_N_r_first + old_N_r_last - c

            plan_moved, weight_moved = {}, weights.copy()
            start, max_weight, sum_weight = 0, 0, 0

            if debug:
                print(f'--- Cutoff point={c}', end='\t')

            for ri, ni in N.items():
                end = start + ni
                _, weight = self.__sim(self.R[ri], plan[start:end])
                plan_moved[self.R[ri]] = plan[start:end]
                weight_moved[ri] = weight
                sum_weight += weight
                max_weight = max(max_weight, weight)
                start = end
                if debug:
                    print(f'{ri}: {weight: .3f}', end='\t')
            if debug:
                print(f'Weight Max: {max_weight: .3f}, Weight Sum: {sum_weight: .3f}')

            if max_weight < opt[1]:
                opt = (c, max_weight, sum_weight, plan_moved, weight_moved)
            elif max_weight == opt[1] and sum_weight < opt[2]:
                opt = (c, max_weight, sum_weight, plan_moved, weight_moved)

            if weight_moved[r_first] < weight_moved[r_last]:
                first = c + 1
            elif weight_moved[r_first] > weight_moved[r_last]:
                last = c - 1
            else:
                break

        if opt[0] != -1:
            for ri in r_index:
                weights[ri] = opt[4][ri]
                plans[self.R[ri]] = opt[3][self.R[ri]]
            output_str = f'--- Found OPT-CUT: c={opt[0]}, max weight={opt[1]}({old_weight_max}), weight sum={opt[2]}({old_weight_sum})'
        else:
            output_str = '--- Did not found OPT-CUT'

        if debug:
            print(output_str)

    def __get_intermediate_r_index(self, r_min, r_max, d_ri):
        r_mid, ri = [r_min], r_min
        while ri != r_max:
            ri = (ri + d_ri) % self.k
            r_mid.append(ri)

        return r_mid if d_ri == 1 else list(reversed(r_mid))


def test_MSTC_STAR(prefix, R, cap, obs_graph=nx.Graph(), is_write=False, is_show=False):
    k = len(R)
    G = nx_graph_read(f'data/nx_graph/{prefix}.graph')
    print(f'==== {test_MSTC_STAR.__name__} on {prefix}, k={k}, cap={cap} ====')

    ts = time.time()
    planner = MSTCStarPlanner(G, k, R, cap, False)
    plans = planner.allocate()
    paths, weights = planner.simulate(plans)
    # print(f'MSTC-STAR planning time elapsed: {time.time()-ts}')
    # animation(mst(G), paths, k)
    simulation(
        planner, paths, weights, 'Grid Map #1 - Naive-MSTC*', 0.03,
        obs_graph, is_write, is_show)
    # print(f'MSTC-STAR overlapping ratio: {calc_overlapping_ratio(paths, planner.rho)}')
    print('\n')


def test_MSTC_STAR_CUT_OPT(prefix, R, cap, obs_graph=nx.Graph(), is_write=False, is_show=False):
    k = len(R)
    G = nx_graph_read(f'data/nx_graph/{prefix}.graph')
    print(f'==== {test_MSTC_STAR_CUT_OPT.__name__} on {prefix}, k={k}, cap={cap} ====')

    ts = time.time()
    planner = MSTCStarPlanner(G, k, R, cap, True)
    plans = planner.allocate()
    # plans = planner.allocate(
    #     f'data/{str.lower(prefix)}/ALLOC_k_{k}_c_{cap}_MSTC_STAR_CUT_OPT.cover')

    paths, weights = planner.simulate(plans)
    # print(f'MSTC-STAR-CUT-OPT planning time elapsed: {time.time()-ts}')
    # animation(mst(G), paths, k)
    simulation(
        planner, paths, weights, 'Grid Map #1 - Balanced-MSTC*', 0.03,
        obs_graph, is_write, is_show)
    # print(f'MSTC-STAR-CUT-OPT overlapping ratio: {calc_overlapping_ratio(paths, planner.rho)}')
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

    test_MSTC_STAR(prefix, R, float('inf'), obs_graph, False, True)

    # prefix = 'TERRAIN#1'
    R = [(16, 16), (16, 17), (17, 16), (17, 17)]  # k = 4
    # R = [(16, 16), (16, 17), (16, 18), (17, 18),  # k = 8
    #      (18, 18), (18, 17), (18, 16), (17, 16)]

    # prefix = 'TERRAIN#3'
    # R = [(88, 40), (88, 39), (89, 39), (89, 40)]  # k = 4
    # R = [(88, 40), (88, 39), (88, 38), (87, 38),  # k = 8
    #      (86, 38), (86, 39), (86, 40), (87, 40)]

    # cap = 200
    # test_MSTC_STAR(prefix, R, cap, False, False)
    # test_MSTC_STAR_CUT_OPT(prefix, R, cap, False, False)
