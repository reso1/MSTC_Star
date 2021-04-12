import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

color = ['r', 'm', 'b', 'y', 'g', 'k']


def subplot(ax, sub_data, sub_labels):
    ax.grid(True, which='both', axis='both')
    ax.axes.xaxis.set_ticklabels([])
    ax.set_xticks(np.arange(0, 5))

    major_ticks = np.linspace(
        np.floor(np.min(sub_data)/10)*10,
        np.ceil(np.max(sub_data)/10)*10, 5)
    minor_ticks = np.linspace(
        np.floor(np.min(sub_data)/10)*10,
        np.ceil(np.max(sub_data)/10)*10, 5)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    for i, data in enumerate(sub_data):
        line, = ax.plot(data, '-o'+color[i], mec='k', label=sub_labels[i])

    ax.legend(loc='upper right', fontsize=9, fancybox=True,
              borderpad=0.5, labelspacing=0.25, borderaxespad=0.1)


fig = plt.figure()
fig.set_size_inches(15, 10)
fig.tight_layout()

# artificial terrain #1
_GRID_5x10_k_3_cap_inf = [185.83, 114.31, 110.07, 100.73, 90.66]
_GRID_5x10_k_3_cap_30 = [362.74, 264.74, 213.87, 180.53, 148.70]
_GRID_5x10_k_4_cap_inf = [185.83, 112.90, 88.66, 88.14, 77.56]
_GRID_5x10_k_4_cap_20 = [413.99, 301.95, 168.94, 173.94, 135.36]
data = [_GRID_5x10_k_3_cap_inf, _GRID_5x10_k_3_cap_30,
        _GRID_5x10_k_4_cap_inf, _GRID_5x10_k_4_cap_20]
labels = [r'$k=3, c=\infty$', r'$k=3, c=30$',
          r'$k=4, c=\infty$', r'$k=4, c=20$']
ax = plt.subplot(3, 1, 1)
plt.title('Artificial Terrain #1: Blocked (unweighted)',
          fontdict={'fontname': 'serif', 'weight': 'bold', 'size': 11})
subplot(ax, data, labels)

# artificial terrain #2
_GRID_10x10_k_4_cap_inf = [173.83, 158.24, 111.07, 100.07, 99.9]
_GRID_10x10_k_4_cap_30 = [285.91, 243.36, 170.04, 147.53, 140.7]
_GRID_10x10_k_8_cap_inf = [173.83, 153.31, 92.24, 69.38, 57.9]
_GRID_10x10_k_8_cap_20 = [326.4, 283.88, 164.38, 88.21, 83.31]
data = [_GRID_10x10_k_4_cap_inf, _GRID_10x10_k_4_cap_30,
        _GRID_10x10_k_8_cap_inf, _GRID_10x10_k_8_cap_20]
labels = [r'$k=4, c=\infty$', r'$k=4, c=30$',
          r'$k=8, c=\infty$', r'$k=8, c=20$']
ax = plt.subplot(3, 1, 2)
plt.title('Artificial Terrain #2: Random (weighted)',
          fontdict={'fontname': 'serif', 'weight': 'bold', 'size': 11})
subplot(ax, data, labels)

# artificial terrain #2
_GRID_20x20_k_4_cap_inf = [1524.24, 793.28, 644.66, 442.77, 442.77]
_GRID_20x20_k_4_cap_40 = [3043.2, 1566.72, 1099.13, 987.21, 807.69]
_GRID_20x20_k_8_cap_inf = [1447.07, 746.66, 416.66, 248.7, 248.7]
_GRID_20x20_k_8_cap_30 = [3437.81, 1747.49, 817.33, 562.06, 498.37]
data = [_GRID_20x20_k_4_cap_inf, _GRID_20x20_k_4_cap_40,
        _GRID_20x20_k_8_cap_inf, _GRID_20x20_k_8_cap_30]
labels = [r'$k=4, c=\infty$', r'$k=4, c=40$',
          r'$k=8, c=\infty$', r'$k=8, c=30$']
ax = plt.subplot(3, 1, 3)
plt.title('Artificial Terrain #3: Free (unweighted)',
          fontdict={'fontname': 'serif', 'weight': 'bold', 'size': 11})
subplot(ax, data, labels)
ax.axes.xaxis.set_ticklabels(
    ['MSTC-NB', 'MSTC-BO', 'MFC', 'Naïve-MSTC*', 'Balanced-MSTC*'],
    fontstyle='italic', family='serif', weight='bold', fontsize=8)

plt.show()
