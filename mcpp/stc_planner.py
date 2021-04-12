import sys
import math
import networkx as nx

from collections import defaultdict, deque
from utils.nx_graph import nx_graph_read, mst


class STCPlanner():
    def __init__(self, M: nx.Graph):
        pass

    def spiral_route(self, start: tuple, dummy_parent: tuple, G: nx.Graph):
        """ using DFS with step-wise backtracing """
        # to avoid reach max recursion depth of 980 in python
        sys.setrecursionlimit(10**6)

        route = []
        visited_nodes = set([start])
        visited_edges = defaultdict(list)

        def ccw_traverse(node: tuple, parent: tuple, is_backtracing):
            route.append(node)
            visited_nodes.add(node)

            if not is_backtracing and (parent, node) != (dummy_parent, start):
                visited_edges[parent].append(node)

            # get original counter-clockwise ordered neighbors
            ccw_ordered_ngbs = deque(self.__get_ccw_neighbors__(G, node))
            # circular shift, see common.py for details
            ccw_ordered_ngbs.rotate(1 - self.__get_motion_dir__(parent, node))

            for ngb in ccw_ordered_ngbs:
                if ngb and ngb not in visited_nodes:
                    ccw_traverse(ngb, node, False)

            # backtracing
            for node in visited_edges[parent]:
                visited_edges[parent].remove(node)
                ccw_traverse(parent, node, True)

        ccw_traverse(start, dummy_parent, False)

        return route  # used for ccw_stc
        # return route[:-1]

    def spiral_route_NC(self, start: tuple, dummy_parent: tuple, G: nx.Graph):
        """ using DFS with step-wise backtracing """
        route = []
        visited_nodes = set([start])
        visited_edges = defaultdict(list)

        stack = [(start, dummy_parent, False)]

        while stack:
            node, parent, is_backtracking = stack.pop()
            route.append(node)
            visited_nodes.add(node)

            if not is_backtracking and (parent, node) != (dummy_parent, start):
                visited_edges[parent].append(node)

            # get original counter-clockwise ordered neighbors
            ccw_ordered_ngbs = deque(self.__get_ccw_neighbors__(G, node))
            # circular shift, see common.py for details
            ccw_ordered_ngbs.rotate(1 - self.__get_motion_dir__(parent, node))

            is_ngb_empty = True
            for ngb in reversed(ccw_ordered_ngbs):
                if ngb and ngb not in visited_nodes:
                    is_ngb_empty = False
                    stack.append((ngb, node, False))

            if is_ngb_empty:
                # backtracking
                for node in reversed(visited_edges[parent]):
                    visited_edges[parent].remove(node)
                    stack.append((parent, node, True))

        return route  # used for ccw_stc
        # return route[:-1]

    def generate_cover_trajectory(self, start: tuple, G: nx.Graph):
        if G.number_of_nodes() == 1:
            direction = ['SE', 'NE', 'NW', 'SW', 'SE']
            return [self.__get_subnode_coords__(start, d) for d in direction]

        # select a neighbor of start as dummy pre-node to generate spiral cover
        dummy_parent = next(G.neighbors(start))
        route = self.spiral_route(start, dummy_parent, G)

        trajectory = []
        L = len(route)

        last = dummy_parent
        for i, cur in enumerate(route):
            motion = self.__get_motion_coords__(last, cur)
            # interpolate a round trip when travsersing into leafnode
            if i <= L-2 and last == route[i+1]:
                motion += self.__get_round_trip_coords__(last, cur)

            trajectory.extend(motion)
            last = cur

        i = 0
        while i < len(trajectory)-1:
            dx = trajectory[i+1][0] - trajectory[i][0]
            dy = trajectory[i+1][1] - trajectory[i][1]

            if abs(dx) + abs(dy) == 0:
                trajectory.pop(i)
            elif abs(dx) + abs(dy) == 1:
                if dx * dy > 0:  # SW/NE
                    trajectory.insert(
                        i+1, (trajectory[i+1][0], trajectory[i][1]))
                else:
                    trajectory.insert(
                        i+1, (trajectory[i][0], trajectory[i+1][1]))
                i += 2
            else:
                i += 1

        # return trajectory  # used for ccw_stc
        return trajectory[:-1]

    def generate_decomposed_graph(self, G: nx.Graph, R):
        decomped_G = nx.Graph()

        # 1)edge within subnodes that belongs to one node
        direction = ['SE', 'NE', 'NW', 'SW']
        for node in G.nodes:
            subnode = [self.__get_subnode_coords__(node, d) for d in direction]
            for i in range(4):
                decomped_G.add_edge(
                    subnode[i], subnode[(i+1) % 4], weight=0.5)
            # diagonal edges between subnodes
            decomped_G.add_edge(subnode[0], subnode[2], weight=0.5)
            decomped_G.add_edge(subnode[1], subnode[3], weight=0.5)

        sqrt_2 = math.sqrt(2)
        # 2) edges connecting R into decomposed G
        for r in R:
            for d in direction:
                subnode = self.__get_subnode_coords__(r, d)
                decomped_G.add_edge(r, subnode, weight=sqrt_2/4)

        # 3) edge between subnodes that belongs to different nodes
        for s, t in G.edges():
            dire_edges, diag_edges = [], []
            direction = self.__get_motion_dir__(s, t)
            if direction == 0:
                s_sw_node = self.__get_subnode_coords__(s, 'SW')
                s_se_node = self.__get_subnode_coords__(s, 'SE')
                t_nw_node = self.__get_subnode_coords__(t, 'NW')
                t_ne_node = self.__get_subnode_coords__(t, 'NE')
                dire_edges = [(s_sw_node, t_nw_node), (s_se_node, t_ne_node)]
                diag_edges = [(s_sw_node, t_ne_node), (s_se_node, t_nw_node)]
            elif direction == 1:
                s_se_node = self.__get_subnode_coords__(s, 'SE')
                s_ne_node = self.__get_subnode_coords__(s, 'NE')
                t_sw_node = self.__get_subnode_coords__(t, 'SW')
                t_nw_node = self.__get_subnode_coords__(t, 'NW')
                dire_edges = [(s_se_node, t_sw_node), (s_ne_node, t_nw_node)]
                diag_edges = [(s_se_node, t_nw_node), (s_ne_node, t_sw_node)]
            elif direction == 2:
                s_ne_node = self.__get_subnode_coords__(s, 'NE')
                s_nw_node = self.__get_subnode_coords__(s, 'NW')
                t_se_node = self.__get_subnode_coords__(t, 'SE')
                t_sw_node = self.__get_subnode_coords__(t, 'SW')
                dire_edges = [(s_ne_node, t_se_node), (s_nw_node, t_sw_node)]
                diag_edges = [(s_ne_node, t_sw_node), (s_nw_node, t_se_node)]
            elif direction == 3:
                s_nw_node = self.__get_subnode_coords__(s, 'NW')
                s_sw_node = self.__get_subnode_coords__(s, 'SW')
                t_ne_node = self.__get_subnode_coords__(t, 'NE')
                t_se_node = self.__get_subnode_coords__(t, 'SE')
                dire_edges = [(s_nw_node, t_ne_node), (s_sw_node, t_se_node)]
                diag_edges = [(s_nw_node, t_se_node), (s_sw_node, t_ne_node)]
            else:
                self.__exit_error__(self.generate_decomposed_graph)

            # w = G[s][t]['weight']
            decomped_G.add_edges_from(dire_edges, weight=0.5)
            decomped_G.add_edges_from(diag_edges, weight=0.5 * sqrt_2)

        return decomped_G

    def __get_ccw_neighbors__(self, G: nx.Graph, node):
        ccw_ordered_nodes = [None] * 4
        for ngb in G.neighbors(node):
            prio = self.__get_motion_dir__(node, ngb)
            ccw_ordered_nodes[prio] = ngb
        return ccw_ordered_nodes

    def __get_motion_dir__(self, p: tuple, q: tuple):
        # direction from node p to node q
        if q[1] < p[1]:    # S
            return 0
        elif q[0] > p[0]:  # E
            return 1
        elif q[1] > p[1]:  # N
            return 2
        elif q[0] < p[0]:  # W
            return 3
        else:
            print(p, q)
            self.__exit_error__(self.__get_motion_dir__)

    def __get_subnode_coords__(self, node, direction):
        x, y = node
        if direction == 'SE':
            return (x+0.25, y-0.25)
        elif direction == 'SW':
            return (x-0.25, y-0.25)
        elif direction == 'NE':
            return (x+0.25, y+0.25)
        elif direction == 'NW':
            return (x-0.25, y+0.25)
        else:
            self.__exit_error__(self.__get_subnode_coords__)

    def __get_motion_coords__(self, p: tuple, q: tuple):
        motion_direction = self.__get_motion_dir__(p, q)
        # move east
        if motion_direction == 1:    # E
            p = self.__get_subnode_coords__(p, 'SE')
            q = self.__get_subnode_coords__(q, 'SW')
        # move west
        elif motion_direction == 3:  # W
            p = self.__get_subnode_coords__(p, 'NW')
            q = self.__get_subnode_coords__(q, 'NE')
        # move south
        elif motion_direction == 0:  # S
            p = self.__get_subnode_coords__(p, 'SW')
            q = self.__get_subnode_coords__(q, 'NW')
        # move north
        elif motion_direction == 2:  # N
            p = self.__get_subnode_coords__(p, 'NE')
            q = self.__get_subnode_coords__(q, 'SE')
        else:
            self.__exit_error__(self.__get_motion_coords__)
        return [p, q]

    def __get_round_trip_coords__(self, last: tuple, pivot: tuple):
        motion_direction = self.__get_motion_dir__(last, pivot)
        if motion_direction == 1:    # E
            return [self.__get_subnode_coords__(pivot, 'SE'),
                    self.__get_subnode_coords__(pivot, 'NE')]
        elif motion_direction == 0:  # S
            return [self.__get_subnode_coords__(pivot, 'SW'),
                    self.__get_subnode_coords__(pivot, 'SE')]
        elif motion_direction == 3:  # W
            return [self.__get_subnode_coords__(pivot, 'NW'),
                    self.__get_subnode_coords__(pivot, 'SW')]
        elif motion_direction == 2:  # N
            return [self.__get_subnode_coords__(pivot, 'NE'),
                    self.__get_subnode_coords__(pivot, 'NW')]
        else:
            self.__exit_error__(self.__get_round_trip_coords__)

    def __exit_error__(self, func):
        sys.exit(f'{self.__class__.__name__}.{func.__name__}: ERROR')

    def __get_travel_weights__(self, traj):
        weights, N = 0, len(traj)
        for i in range(N-1):
            weights += self.H[traj[i]][traj[i+1]]['weight']

        return weights


if __name__ == '__main__':
    prefix = 'GRID_10x10_WEIGHTED'
    R = [(4, 5), (4, 6), (5, 5), (6, 5), (6, 6), (4, 7), (5, 7), (6, 7)]

    G = nx_graph_read(f'data/nx_graph/{prefix}.graph')

    stc_planner = STCPlanner(G)
    route = stc_planner.spiral_route(R[0], R[1], mst(G))
    route_NC = stc_planner.spiral_route_NC(R[0], R[1], mst(G))

    print(route[20:30])
    print(route_NC[20:30])

    # for i, r in enumerate(route):
    #     if i < len(route)-1:
    #         l1 = abs(r[0]-route[i+1][0]) + abs(r[1]-route[i+1][1])
    #         if l1 != 1:
    #             print(l1, r, route[i+1])
