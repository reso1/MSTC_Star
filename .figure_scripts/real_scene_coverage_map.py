import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.append(os.path.abspath('.'))

from utils.nx_graph import nx_graph_read


color_RGB = {'y': [1, 1, 0], 'r': [1, 0, 0],
             'm': [0.2, 1, 0.6], 'a': [0.502, 0.502, 0.502],
             'b': [0, 0, 1], 'p': [1, 0, 1],
             'i': [1, 0.2, 0.6], 'g': [0, 0.6, 0]}
color = list(color_RGB.values())

# prefix = 'TERRAIN#1'
# R = [(16, 16), (16, 17), (16, 18), (17, 18),
#      (18, 18), (18, 17), (18, 16), (17, 16)]

prefix = 'TERRAIN#3'
R = [(88, 40), (88, 39), (88, 38), (87, 38),
     (86, 38), (86, 39), (86, 40), (87, 40)]

G = nx_graph_read(f'data/nx_graph/{prefix}.graph')

img = np.zeros((256, 256, 3))

f = open(f'data/{str.lower(prefix)}/ALLOC_k_8_c_400_MSTC_STAR_CUT_OPT.cover', 'r')
idx = 0
for line in f.readlines():
    c = color[idx % len(color)]
    plt.plot(2*(128-R[idx][0]), 2*(128-R[idx][1]), '*', mfc=c, mec='k', ms=16)
    for coords in line.split(' '):
        x, y = [float(val) for val in coords.split(',')]
        img[int(2*x+0.5)][int(2*y+0.5)] = c
    idx += 1


plt.imshow(img)
plt.show()
