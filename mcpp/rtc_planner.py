import sys
import heapq
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

from utils.nx_graph import graph_plot, mst

color = ['r', 'm', 'b', 'k', 'c', 'g']


class RTCPlanner:
    def __init__(self, G: nx.Graph, R, k, debug=False):
        self.G = G
        self.R = R
        self.k = k

        self.debug = debug

    def k_tree_cover(self, eps=1.0):
        n = self.G.number_of_nodes()
        m = self.G.number_of_edges()
        # w1 <  w2 < ... < wm
        sorted_edges = sorted(
            list(self.G.edges.data()), key=lambda x: x[2]['weight'])
        w_m = sorted_edges[m-1][2]['weight']  # wm
        res = self.rooted_tree_cover(w_m)

        if res:
            # if B = wm succeded, find i that:
            # when B = wi rtc fails but when B = w_i+1 rtc succeeded
            i_found, w_ip1_res = -1, res
            for i in range(m-2, -1, -1):
                w_i_res = self.rooted_tree_cover(sorted_edges[i][2]['weight'])
                if not w_i_res and w_ip1_res:
                    i_found = i
                    break
                else:
                    w_ip1_res = w_i_res
            # different handles for wi+1/wi result
            assert(i_found != -1)
            w_i = sorted_edges[i_found][2]['weight']
            w_ip1 = sorted_edges[i_found+1][2]['weight']
            if w_ip1 / w_i <= n * n / eps:
                # search_space = np.linspace(w_i, w_ip1 + eps, eps)
                search_space_param = (w_i, w_ip1)
            else:
                w_q = n * n / eps * w_i
                if self.rooted_tree_cover(w_q):
                    # search_space = np.linspace(w_i, w_q + eps, eps)
                    search_space_param = (w_i, w_q)
                else:
                    # search_space = np.linspace(w_ip1, 4*w_ip1 + eps, eps)
                    search_space_param = (w_ip1, 4 * w_ip1)
        else:
            # search_space = eps*w_m/(n*n) * np.arange(1, int((n**3+1)/eps))
            search_space_param = (eps * w_m / (n*n), n * w_m)

        opt = (None, float('inf'), -1)
        # first, last = 0, len(search_space)-1
        first, last = search_space_param
        while first <= last:
            mid = (first + last) // 2
            # res = self.rooted_tree_cover(search_space[mid])
            res = self.rooted_tree_cover(mid)
            if res:
                """ record best solution with minimal max_weights
                    rather than smallest B """
                # opt = (res[0], search_space[mid])
                forest, max_weights = res
                if not opt or max_weights < opt[1]:
                    # opt = (forest, max_weights, search_space[mid])
                    opt = (forest, max_weights, mid)
                # last = mid-1
                last = mid - eps
            else:
                # first = mid+1  # B is too low
                first = mid + eps

        self.__log('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        self.__log(f'RTC succeeded, max_weights: {opt[1]}, B*: {opt[2]}')

        return opt

    def rooted_tree_cover(self, B):
        self.__log('========================================')
        self.__log(f'- Rooted-Tree-Cover, R={self.R}, B={B}')

        if nx.number_connected_components(self.G) != 1:
            sys.exit(f'original graph is not connected piece')

        # 1) remove all edges of weight greater than B
        del_count = 0
        for s, t in list(self.G.edges()):
            if self.G[s][t]['weight'] > B:
                self.G.remove_edge(s, t)
                del_count += 1
                # if not connected to G, (B <= B*)
                if nx.number_connected_components(self.G) != 1:
                    self.__log(f'--- current B: {B: .3f} value is too low.')
                    return None

        self.__log(f'--- # of deleted edges of weight > {B}: {del_count}')
        if self.debug:
            self.__plot(1, self.G, self.R)
            plt.title('T after deleting heavy edges')

        # 2) MST of G by contracting roots in R to a single node uR
        contracted_G = nx.Graph(self.G)
        R_set = set(self.R)
        uR = (sum([r[0] for r in self.R]) // self.k + 0.5,
              sum([r[1] for r in self.R]) // self.k + 0.5)
        uR_edges = {}
        uR_edges_map = {}
        for r in self.R:
            # iterate neighbors of ri
            for ngb in contracted_G.neighbors(r):
                if ngb in R_set:
                    continue
                if ngb not in uR_edges:
                    # ii) edges(v, ri) induces edge(v, uR)
                    uR_edges[ngb] = contracted_G[r][ngb]['weight']
                    uR_edges_map[ngb] = r
                else:
                    # iii) w(v, uR) = min(w(v, ri))
                    if contracted_G[r][ngb]['weight'] < uR_edges[ngb]:
                        uR_edges[ngb] = contracted_G[r][ngb]['weight']
                        uR_edges_map[ngb] = r
            contracted_G.remove_node(r)  # i) remove nodes in R

        contracted_G.add_node(uR)  # i) introducing uR
        for t, w in uR_edges.items():
            contracted_G.add_edge(uR, t, weight=w)  # add edges of uR

        if self.debug:
            self.__plot(2, contracted_G, [uR])
            plt.title('contracted G')

        M = mst(contracted_G)  # get mst
        if self.debug:
            self.__plot(4, M, [uR])
            plt.title('MST of contracted G')

        # 3) obtain forest {T_i}_i from M by un-contracting roots in R
        M.add_nodes_from(self.R)
        for ngb in M.neighbors(uR):
            r = uR_edges_map[ngb]
            M.add_edge(r, ngb, weight=uR_edges[ngb])

        M.remove_node(uR)

        Forest = [None] * self.k
        for cc in nx.connected_components(M):
            T = M.subgraph(cc)
            T_node_set = set(T.nodes())
            for i, r in enumerate(self.R):
                if r in T_node_set:
                    Forest[i] = T
                    break

        if self.debug:
            plt.subplot(2, 3, 5)
            plt.title('Forest after uncontracting roots')
            for i, T in enumerate(Forest):
                for s, t in T.edges():
                    x1, y1 = s
                    x2, y2 = t
                    plt.plot([x1, x2], [y1, y2], '-o'+color[i % len(color)])
            for x, y in self.R:
                plt.plot([x], [y], 'or')

        subtrees, leftover_trees = [], {}
        subtree_nodes = {}
        count = 0
        # 4) edge-decompose each tree Ti into {Si_j)} + Li
        for i, T in enumerate(Forest):
            S, L = self.edge_decomposition(T, self.R[i], B)

            r_weight = L.nodes[self.R[i]]['weight']
            assert(r_weight == 0 or r_weight < B)
            L.graph['match_node'] = None
            leftover_trees[self.R[i]] = L

            for s in S:
                root = s.graph['root']
                assert(len(list(nx.connected_components(s))) == 1)
                assert(B <= s.nodes[root]['weight'] < 2*B)
                # append to subtrees
                subtrees.append(s)
                subtree_nodes[count] = set(s.nodes())
                count += 1

        # 5) find a maximum matching of {Si_j} and {R} bipartite graph
        # to finish matching, |{Si_j}| <= |{R}| is required
        if len(subtrees) > self.k:
            self.__log(
                f'> B is too low, # of subtrees: {len(subtrees)} > {self.k}')
            return None
        # i) create a bipartite graph of {Si_j} and {R}, s.t. d(Si_j, R) <= B
        # ii) find maximum matching of BG using Ford-Fulkerson max-flow method
        BG = nx.Graph()
        BG.add_nodes_from(self.R + list(range(len(subtrees))))
        for r in self.R:
            P = nx.single_source_dijkstra_path(self.G, r, B, 'weight')
            for target, s_path in P.items():
                weight = 0
                for idx in range(len(s_path)-1):
                    weight += self.G[s_path[idx]][s_path[idx+1]]['weight']
                # iterate each subtree
                for si in range(len(subtrees)):
                    if target in subtree_nodes[si]:
                        if (r, si) in BG.edges:
                            if weight < BG[r][si]['weight']:
                                BG[r][si]['weight'] = weight
                                BG[r][si]['path'] = s_path
                        else:
                            BG.add_edge(r, si, path=s_path, weight=weight)

        """ apply Ford-Fulkerson method to BG
            ref: https://www.geeksforgeeks.org/maximum-bipartite-matching/ """
        match: dict = {}  # {r} -> {Si_j}
        for si in range(len(subtrees)):
            # found one subtree not matched to roots
            if not self.__bipartite_match(BG, si, match, set()):
                self.__log(
                    f'> B is too low, not all subtrees matched to roots')
                return None

        max_weights = 0
        for r in self.R:
            L, S, P = leftover_trees[r], nx.Graph(), []
            weights = 0
            for s, t in L.edges():
                weights += L[s][t]['weight']

            if r in match:
                si = match[r]
                S = subtrees[si]
                for s, t in S.edges():
                    weights += S[s][t]['weight']
                P = BG[r][si]['path']
                self.__log(
                    f'--- matched subtree ({r}=>{P[-1]}):', end=' ')
            else:
                self.__log(f'--- leftover subtree:', end=' ')

            match[r] = [L, S, P]
            max_weights = max(max_weights, weights)
            num_nodes = L.number_of_nodes() + (S.number_of_nodes() if S else 0)
            num_edges = L.number_of_edges() + \
                (S.number_of_edges() if S else 0) + (len(P)-1 if P else 0)
            self.__log(
                f'{num_nodes} nodes, {num_edges} edges, {weights} weights')

        if self.debug:
            plt.subplot(1, 3, 3)
            plt.title('Final Matching Result')
            self.plot_rtc_result(match)
            plt.show()

        self.__log(f'> B: {B}, max weights: {max_weights}')

        return match, max_weights

    def edge_decomposition(self, T: nx.Graph, r: tuple, B):
        # i) initialize the weights of subtrees

        # to avoid reach max recursion depth of 980 in python
        sys.setrecursionlimit(10**6)

        def init(cur, visited):
            weight = 0
            for ngb in T.neighbors(cur):
                if ngb not in visited:
                    visited.add(ngb)
                    T.nodes[ngb]['parent'] = cur
                    weight += init(ngb, visited) + T[cur][ngb]['weight']
            T.nodes[cur]['weight'] = weight
            return weight

        visited = set([r])
        init(r, visited)
        T.nodes[r]['parent'] = None
        T.graph['root'] = r

        # ii) edge decomposition to medium subtrees
        subtrees, leftover_tree = [], nx.Graph(root=r)
        while T.nodes[r]['weight'] >= 2*B:
            # a) split medium subtrees
            candidates: list = []
            candidate_set = set()
            # initialize the candidate subtree
            for node, data in T.nodes.data():
                if data['parent'] and data['weight'] < 2*B:
                    heapq.heappush(candidates, (data['weight'], node))
                    candidate_set.add(node)
            # process all medium subtree candidates
            while candidates:
                w_Tv, v = heapq.heappop(candidates)
                candidate_set.remove(v)
                u = T.nodes[v]['parent']
                w_Te = w_Tv + T[u][v]['weight']
                # Te medium subtree consisted of {Te, e(u, v), Tv}
                if B <= w_Te < 2*B:
                    S = nx.Graph()
                    S, visited = self.__build_subtree(
                        T, u, v, w_Te, candidates, candidate_set, S)
                    # update Ti
                    T = self.__update_weight(
                        T, v, w_Te, candidates, candidate_set, B)
                    visited.remove(u)  # Ti will keep u after Te subtree split
                    m_T = nx.Graph(T)
                    m_T.remove_nodes_from(visited)
                    T = m_T  # copy to unfreeze old Ti, somehow it is frozen
                    # append to subtrees list
                    subtrees.append(S)
                elif B <= w_Tv < 2*B:  # Tv medium rooted subtree
                    # split Ti into Si_j and update remaining Ti
                    S = nx.Graph()
                    S, visited = self.__build_subtree(
                        T, u, v, w_Te, candidates, candidate_set, S)
                    S.graph['root'] = v
                    S.remove_node(u)
                    # update Ti
                    T.nodes[v]['weight'] = 0
                    T = self.__update_weight(
                        T, v, w_Tv, candidates, candidate_set, B)
                    visited.remove(u)  # Ti will keep u after Tv subtree split
                    visited.remove(v)  # Ti will keep v after Tv subtree split
                    m_T = nx.Graph(T)
                    m_T.remove_nodes_from(visited)
                    T = m_T
                    # append to subtrees list
                    subtrees.append(S)

            # b) pick the heavy rooted subtree with smallest weight to split
            sorted_nodes = sorted(T.nodes.data(), key=lambda x: x[1]['weight'])
            for node, data in sorted_nodes:
                if data['weight'] >= 2*B:
                    # bounching edges emanating from heavy_node until 2B
                    S, w_Te = nx.Graph(), 0
                    N = [(n, T.nodes[n]['weight']) for n in T.neighbors(node)]
                    for ngb, ngb_weight in sorted(N, key=lambda x: x[1]):
                        dw = ngb_weight + T[node][ngb]['weight']
                        if w_Te + dw < 2*B:
                            w_Te += dw
                            # split light subtrees from Ti into S
                            S, visited = self.__build_subtree(
                                T, node, ngb, 0, candidates, candidate_set, S)
                            # update Ti
                            T = self.__update_weight(
                                T, node, dw, candidates, candidate_set, B)
                            visited.remove(node)
                            m_T = nx.Graph(T)
                            m_T.remove_nodes_from(visited)
                            T = m_T
                            subtrees.append(S)
                        else:
                            # break if the first time w_Te >= B, now w_Te < 2*B
                            assert(w_Te < 2*B)  # because w(e)<B for any e(u,v)
                            if node not in S.nodes:
                                S.add_node(node)
                            S.nodes[node]['weight'] = w_Te
                            T.nodes[node]['weight'] -= w_Te
                            break
                    break  # only process one heavy rooted subtree, then break

        # iii) decide whether there is a leftover tree
        if T.nodes[r]['weight'] >= B:
            subtrees.append(T)
            leftover_tree.add_node(r, weight=0)
        else:
            leftover_tree = T

        return subtrees, leftover_tree

    def plot_rtc_result(self, match, is_progressive=False, is_show=False):
        R = match.keys()
        match_pair = match.values()

        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        for i, r in enumerate(R):
            plt.text(r[0]+0.1, r[1]+0.1, f'R{i}', c='k')

        for i, val in enumerate(match_pair):
            c = color[i % len(color)]
            L, S, P = val

            self.__edge_plot(L, c)
            self.__edge_plot(S, c)

            if P:
                plt.plot([P[0][0], P[-1][0]], [P[0][1], P[-1][1]], '--o'+c)

            if is_progressive:
                plt.draw()
                plt.waitforbuttonpress(0)

        if is_show:
            plt.show()

    def __update_weight(
            self, T: nx.Graph, node, d_weight, candidates, candidate_set, B):
        parent = T.nodes[node]['parent']
        while parent:
            if parent in candidate_set:
                candidate_set.remove(parent)
                candidates.remove((T.nodes[parent]['weight'], parent))
            T.nodes[parent]['weight'] -= d_weight
            if T.nodes[parent]['parent'] and T.nodes[parent]['weight'] < 2*B:
                candidate_set.add(parent)
                heapq.heappush(
                    candidates, (T.nodes[parent]['weight'], parent))
            parent = T.nodes[parent]['parent']

        return T

    def __build_subtree(self, T, u, v, w_Te, candidates, candidate_set, S):
        S.graph['root'] = u
        S.add_node(u, weight=w_Te, parent=None)
        S.add_edge(u, v, weight=T[u][v]['weight'])
        # dfs to get Te subtree nodes
        stack, visited = [v], set([u, v])
        while stack:
            node = stack.pop()
            data = T.nodes[node]
            S.add_node(node, weight=data['weight'], parent=data['parent'])
            if node in candidate_set:
                candidate_set.remove(node)
                candidates.remove((T.nodes[node]['weight'], node))
            for ngb in T.neighbors(node):
                if ngb not in visited:
                    visited.add(ngb)
                    stack.append(ngb)
                    S.add_edge(node, ngb, weight=T[node][ngb]['weight'])

        return S, visited

    def __bipartite_match(self, BG: nx.Graph, u, match: dict, visited: set):
        """ ref: https://www.geeksforgeeks.org/maximum-bipartite-matching/ """
        for v in BG.neighbors(u):
            if v not in visited:
                visited.add(v)
                if v not in match or \
                        self.__bipartite_match(BG, match[v], match, visited):
                    match[v] = u
                    return True
        return False

    def __log(self, s, end='\n'):
        # if self.debug:
        print(s, end=end)

    def __plot(self, i, G, R):
        plt.subplot(2, 3, i)
        for s, t in G.edges():
            x1, y1 = s
            x2, y2 = t
            plt.plot([x1, x2], [y1, y2], '-ok')
        for x, y in R:
            plt.plot([x], [y], 'or')

    def __edge_plot(self, G: nx.Graph, c):
        for s, t in G.edges():
            x1, y1 = s
            x2, y2 = t
            weight = G[s][t]['weight']
            plt.plot([x1, x2], [y1, y2], '-o'+c)
            plt.text((x1+x2)/2, (y1+y2)/2+0.1, f'{weight: .3f}', c='k')


if __name__ == '__main__':
    from utils.nx_graph import nx_graph_read

    # G = nx_graph_read('data/nx_graph/GRID_10x10_WEIGHTED.graph')
    # R = [(5, 5), (5, 6), (6, 5), (6, 6)]

    G = nx_graph_read('data/nx_graph/GRID_5x10_UNWEIGHTED.graph')
    R = [(1, 0), (2, 0), (3, 0), (4, 0)]

    k = len(R)

    rtc = RTCPlanner(G, R, k, False)
    forest, max_weights, opt_B = rtc.k_tree_cover(1)

    weights = 0
    for s, t in G.edges:
        weights += G[s][t]['weight']

    print(f'max weights: {max_weights}, B: {opt_B}')
    rtc.plot_rtc_result(forest, False, True)
