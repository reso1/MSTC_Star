import networkx as nx

from mcpp.mfc_planner import test_MFC
from mcpp.mstc_planner import test_MSTC, test_MSTC_BT_OPT
from mcpp.mstc_star_planner import test_MSTC_STAR, test_MSTC_STAR_CUT_OPT
from utils.nx_graph import nx_graph_read


def test_grid_5x10_unweighted(cap=float('inf'), is_write=False, is_show=False):
    prefix = 'GRID_5x10_UNWEIGHTED'
    # R = [(2, 0), (3, 0), (4, 0)]
    R = [(1, 0), (2, 0), (3, 0), (4, 0)]
    # R = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    G = nx_graph_read(f'data/nx_graph/{prefix}.graph')
    obs_graph = nx.grid_2d_graph(5, 10)
    for node in G.nodes():
        obs_graph.remove_node(node)

    test_MSTC(prefix, R, cap, obs_graph, is_write, is_show)
    test_MSTC_BT_OPT(prefix, R, cap, is_write, is_show)
    # test_MFC(prefix, R, cap, 1.0, is_write, is_show)
    test_MSTC_STAR(prefix, R, cap, is_write, is_show)
    test_MSTC_STAR_CUT_OPT(prefix, R, cap, is_write, is_show)


def test_grid_10x10_weighted(cap=float('inf'), is_write=False, is_show=False):
    prefix = 'GRID_10x10_WEIGHTED'
    # R = [(5, 5), (5, 6), (6, 6)]
    R = [(5, 5), (5, 6), (6, 5), (6, 6)]
    # R = [(4, 5), (4, 6), (5, 5), (6, 5), (6, 6), (4, 7), (5, 7), (6, 7)]

    test_MSTC(prefix, R, cap, is_write, is_show)
    test_MSTC_BT_OPT(prefix, R, cap, is_write, is_show)
    test_MFC(prefix, R, cap, 1.0, is_write, is_show)
    test_MSTC_STAR(prefix, R, cap, is_write, is_show)
    test_MSTC_STAR_CUT_OPT(prefix, R, cap, is_write, is_show)


def test_grid_20x20_unweighted_free(cap=float('inf'), is_write=False, is_show=False):
    prefix = 'GRID_20x20_UNWEIGHTED_FREE'
    # R = [(5, 5), (6, 6)]
    # R = [(5, 5), (5, 6), (6, 5), (6, 6)]
    # R = [(4, 5), (4, 6), (5, 5), (5, 6), (6, 5), (6, 6)]
    R = [(4, 5), (4, 6), (5, 5), (6, 5), (6, 6), (4, 7), (5, 7), (6, 7)]

    test_MSTC(prefix, R, cap, is_write, is_show)
    test_MSTC_BT_OPT(prefix, R, cap, is_write, is_show)
    test_MFC(prefix, R, cap, 1.0, is_write, is_show)
    test_MSTC_STAR(prefix, R, cap, is_write, is_show)
    test_MSTC_STAR_CUT_OPT(prefix, R, cap, is_write, is_show)


def test_terrain_1(cap=float('inf'), is_write=False, is_show=False):
    prefix = 'TERRAIN#1'
    # R = [(16, 16), (16, 17), (17, 16), (17, 17)]  # k = 4
    # R = [(16, 16), (16, 17), (16, 18), (17, 18),  # k = 8
    #      (18, 18), (18, 17), (18, 16), (17, 16)]
    R = [(16, 16), (16, 17), (16, 18), (17, 18),  # k = 12
         (18, 18), (18, 17), (18, 16), (17, 16),
         (12, 16), (12, 17), (12, 18), (13, 18)]
    # R = [(16, 16), (16, 17), (16, 18), (17, 18),  # k = 16
    #      (18, 18), (18, 17), (18, 16), (17, 16),
    #      (12, 16), (12, 17), (12, 18), (13, 18),
    #      (14, 18), (14, 17), (14, 16), (13, 16)]

    # test_MSTC(prefix, R, cap, is_write, is_show)
    # test_MSTC_BT_OPT(prefix, R, cap, is_write, is_show)
    # test_MFC(prefix, R, cap, 10.0, is_write, is_show)
    test_MSTC_STAR(prefix, R, cap, is_write, is_show)
    test_MSTC_STAR_CUT_OPT(prefix, R, cap, is_write, is_show)


def test_terrain_3(cap=float('inf'), is_write=False, is_show=False):
    prefix = 'TERRAIN#3'
    R = [(88, 40), (88, 39), (89, 39), (89, 40)]  # k = 4
    # R = [(88, 40), (88, 39), (88, 38), (87, 38),  # k = 8
    #      (86, 38), (86, 39), (86, 40), (87, 40)]
    R = [(88, 40), (88, 39), (88, 38), (87, 38),  # k = 12
         (86, 38), (86, 39), (86, 40), (87, 40),
         (84, 40), (84, 39), (84, 38), (83, 38)]
    # R = [(88, 40), (88, 39), (88, 38), (87, 38),  # k = 16
    #      (86, 38), (86, 39), (86, 40), (87, 40),
    #      (84, 40), (84, 39), (84, 38), (83, 38),
    #      (82, 38), (82, 39), (82, 40), (83, 40)]

    test_MSTC(prefix, R, cap, is_write, is_show)
    test_MSTC_BT_OPT(prefix, R, cap, is_write, is_show)
    test_MFC(prefix, R, cap, 10.0, is_write, is_show)
    test_MSTC_STAR(prefix, R, cap, is_write, is_show)
    test_MSTC_STAR_CUT_OPT(prefix, R, cap, is_write, is_show)


if __name__ == '__main__':
    test_grid_5x10_unweighted()
