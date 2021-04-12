import math

import cv2
import torch
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy import interpolate
from matplotlib import colors
from skimage.transform import resize

from terrainFilter.tif_processing import SATMAP_BAND_SEGNET
from terrainFilter.segNet.deeplab import DeepLab


class TerrainFilter:
    BARREN_TYPE = 8
    TIF_IMAGE_SHAPE = (512, 512)
    FILTERED_MAP_SHAPE = (256, 256)
    segClassNames = ["Forest", "Shrubland", "Savanna", "Grassland",
                     "Wetlands", "Croplands", "Urban/Built-up",
                     "Snow/Ice", "Barren", "Water"]
    segColorMap = colors.ListedColormap(
        ['#009900', '#c6b044', '#fbff13', '#b6ff05', '#27ff87', '#1c0dff',
         '#a5a5a5', '#69fff8', '#f9ffa4', '#1c0dff', '#ffffff'])

    @staticmethod
    def filterDEM(DEM, res=0.35, slopeThreshold=20.0):
        filtered = 255 * np.ones(
            TerrainFilter.FILTERED_MAP_SHAPE, dtype=np.uint8)
        mHeight, mWidth = TerrainFilter.FILTERED_MAP_SHAPE
        DEM = TerrainFilter.__downUpSamplingDEM(DEM)

        for i in range(mHeight):
            for j in range(mWidth):
                # use 4 neighbors
                sum_h, count_h = 0, 0
                for ci, cj in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                    ni, nj = i + ci, j + cj
                    if 0 <= ni < mHeight and 0 <= nj < mWidth:
                        sum_h += DEM[ni][nj]
                        count_h += 1
                de = abs(sum_h / count_h - DEM[i][j])
                slope = math.degrees(math.atan2(de, res))
                if slope < slopeThreshold:
                    filtered[i][j] = 0

        return filtered

    @staticmethod
    def filterSatMap(satMap, pklPath, modelPath):
        """ inference using DeepLab neural network """

        use_gpu = False
        if torch.cuda.is_available():
            use_gpu = True
            if torch.cuda.device_count() > 1:
                raise NotImplementedError("multi GPU Error")

        # Setup network
        train_args = pickle.load(open(pklPath, "rb"))
        n_classes = len(TerrainFilter.segClassNames) - 1
        n_inputs = len(SATMAP_BAND_SEGNET)
        model = DeepLab(num_classes=n_classes, backbone='resnet',
                        pretrained_backbone=False,
                        output_stride=train_args.out_stride,
                        sync_bn=False, freeze_bn=False, n_in=n_inputs)

        model = model.cuda() if use_gpu else model

        # Restore weights from file
        state = torch.load(modelPath)
        step = state["step"]
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        # print("loaded checkpoint from step", step)

        # Load image
        img_input = torch.from_numpy(satMap)
        img_input = img_input.repeat(16, 1, 1, 1)
        img_input = img_input.cuda() if use_gpu else img_input

        # Predict
        with torch.no_grad():
            prediction = model(img_input)

        prediction = prediction.cpu().numpy()
        prediction = np.argmax(prediction, axis=1)

        # Savanna data
        pre_copy = prediction.copy()
        prediction[pre_copy > 2] += 1

        return prediction[0]

    @staticmethod
    def fuseFilteredMap(filteredDEM, filteredSatMap, depotG):
        mHeight, mWidth = TerrainFilter.FILTERED_MAP_SHAPE

        # generate barren mask
        fSatMapHeight, fSatMapWidth = TerrainFilter.TIF_IMAGE_SHAPE
        barrenMask = np.zeros_like(filteredSatMap, dtype=np.uint8)
        for r in range(fSatMapHeight):
            for c in range(fSatMapWidth):
                if filteredSatMap[r, c] == TerrainFilter.BARREN_TYPE:
                    barrenMask[r, c] = 1

        # resize barren mask into shape of filteredmap
        resizedBarrenMask = TerrainFilter.downscaleBinMap(
            barrenMask, fSatMapHeight // mHeight, fSatMapWidth // mWidth)

        fusedMap = ~(~filteredDEM & resizedBarrenMask)
        # remove unreachable nodes from depot of covering graph
        mask = np.zeros((mHeight+2, mWidth+2), np.uint8)
        floodfill_map = fusedMap.copy()
        cv2.floodFill(floodfill_map, mask, depotG, 255)
        fusedMap = (fusedMap & floodfill_map) | ~ floodfill_map

        # fig, ax = plt.subplots(1)
        # ax.imshow(~fusedMap, 'gray')
        # fig.show()

        return ~fusedMap

    @staticmethod
    def generateSpanningGraph(binMap, depotH, DEM, graphShape, res=0.35):
        # first resize to dst graph shape
        mHeight, mWidth = binMap.shape
        rScaling, cScaling = mHeight // graphShape[0], mWidth // graphShape[1]
        binMap = TerrainFilter.downscaleBinMap(binMap, rScaling, cScaling)

        # make sure depot of covering graph free
        depotH = (depotH[0] // rScaling, depotH[1] // cScaling)
        depotGx, depotGy = depotH[0] * 2, depotH[1] * 2
        for ci, cj in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            binMap[depotGx+ci][depotGy+cj] = 1

        H = nx.Graph()
        # construct spanning nodes
        for r in range(0, graphShape[0], 2):
            for c in range(0, graphShape[1], 2):
                if binMap[r][c] == 1 and binMap[r+1][c] == 1 and \
                        binMap[r][c+1] == 1 and binMap[r+1][c+1] == 1:
                    H.add_node((r // 2, c // 2))  # free spanning node
        # construct spanning edges
        min_slope, max_slope, H = \
            TerrainFilter.__constructSpanningEdges(depotH, binMap, DEM, H, res)
        for s, t in H.edges():
            normalized_slope = (H[s][t]['weight'] - min_slope) / max_slope
            H[s][t]['weight'] = 1 + normalized_slope

        return nx.subgraph(H, nx.node_connected_component(H, depotH))

    @staticmethod
    def downscaleBinMap(
            binMap, rScalingFactor, cScalingFactor, barrenRatioThreshold=1/3):
        """ downscale a binary map by integer scaling factor """
        if rScalingFactor == 1 and cScalingFactor == 1:
            return binMap

        height, width = binMap.shape
        numBarrenThreshold = rScalingFactor*cScalingFactor*barrenRatioThreshold
        mHeight, mWidth = height // rScalingFactor, width // cScalingFactor
        rescaled = np.zeros((mHeight, mWidth), dtype=np.uint8)

        for r in range(mHeight):
            for c in range(mWidth):
                barrenPixCount = 0
                for i in range(rScalingFactor):
                    for j in range(cScalingFactor):
                        if binMap[rScalingFactor*r + i, cScalingFactor*c + j]:
                            barrenPixCount += 1
                if barrenPixCount > numBarrenThreshold:
                    rescaled[r, c] = 1

        return rescaled

    @staticmethod
    def display(G, mWidth, mHeight):
        img = np.zeros((mWidth, mHeight), dtype=np.uint8)
        for s, t in G.edges():
            for x, y in [s, t]:
                img[int(x)][int(y)] = 1
        plt.imshow(img, 'gray')
        plt.show()

    @staticmethod
    def __constructSpanningEdges(depotH, binMap, DEM, H: nx.Graph, res=0.35):
        GHeight, GWidth = binMap.shape
        HHeight, HWidth = GHeight // 2, GWidth // 2
        stack, visited = [depotH], set()
        min_slope, max_slope = float('inf'), float('-inf')

        def __getValidSpanningNodeNgb(node):
            ngbs = []
            for ci, cj in [[1, 0], [0, 1], [-1, 0], [0, -1]]:
                ni, nj = node[0] + ci, node[1] + cj
                if 0 <= ni < HHeight and 0 <= nj < HWidth \
                        and (ni, nj) in H.nodes():
                    ngbs.append((ni, nj))

            return ngbs

        def __sumSlopeDegs(n):
            return DEM[n[0]*2, n[1]*2] + DEM[n[0]*2+1, n[1]*2] + \
                   DEM[n[0]*2, n[1]*2+1] + DEM[n[0]*2+1, n[1]*2+1]

        # slopes = []
        while stack:
            cur = stack.pop()
            visited.add(cur)

            for ngb in __getValidSpanningNodeNgb(cur):
                if ngb not in visited:
                    # calculate average slope from cur to ngb
                    de = abs(__sumSlopeDegs(cur) - __sumSlopeDegs(ngb))
                    slope = math.degrees(math.atan2(de / 4, 2 * res))
                    # slopes.append(slope)
                    stack.append(ngb)
                    H.add_edge(cur, ngb, weight=slope)
                    min_slope = min(slope, min_slope)
                    max_slope = max(slope, max_slope)

        # fig, ax = plt.subplots(1)
        # ax.hist(slopes, bins=20)
        # fig.show()

        return min_slope, max_slope, H

    @staticmethod
    def __downUpSamplingDEM(DEM: np.array):
        # downsampling
        downsampledShape = (64, 64)
        downsampledDEM = resize(DEM, downsampledShape, anti_aliasing=True)
        # upsampling by quintic interp2d
        funcInterpDEM = interpolate.interp2d(
            list(range(downsampledShape[0])), list(range(downsampledShape[1])),
            downsampledDEM, kind='quintic')
        interpK = TerrainFilter.FILTERED_MAP_SHAPE[0] / downsampledShape[0]
        interpedDEM = funcInterpDEM(
            np.arange(0, downsampledShape[0], 1 / interpK),
            np.arange(0, downsampledShape[1], 1 / interpK))
        return interpedDEM
