import rasterio
import numpy as np

DEM_BAND_COUNT = 1
DEM_BAND_GRAY = [1]
SATMAP_BAND_COUNT = 23
SATMAP_BAND_RGB = [4, 3, 2]
SATMAP_BAND_SEGNET = [2, 3, 4, 8]


def readSatMapTif(fileName, brightnessFactor=3.0):
    with rasterio.open(fileName) as satMapTifFile:
        if satMapTifFile.count != SATMAP_BAND_COUNT:
            return None
        satMapTif = satMapTifFile.read(SATMAP_BAND_RGB)
    satMapTif = satMapTif.astype(np.float32)
    satMapTif = np.clip(satMapTif, 0, 10000)
    satMapTif /= 10000
    satMapTif = np.rollaxis(satMapTif, 0, 3)
    return np.clip(satMapTif * brightnessFactor, 0, 1)


def readDEMTif(fileName):
    with rasterio.open(fileName) as DEMTifFile:
        if DEMTifFile.count != DEM_BAND_COUNT:
            return None
        DEMTif = DEMTifFile.read(DEM_BAND_GRAY)
        DEMTif = np.rollaxis(DEMTif, 0, 3)
    return DEMTif


def readSatMapTifSegNet(fileName):
    with rasterio.open(fileName) as satMapTifFile:
        if satMapTifFile.count != SATMAP_BAND_COUNT:
            return None
        satMapTif = satMapTifFile.read(SATMAP_BAND_SEGNET)
    satMapTif = satMapTif.astype(np.float32)
    satMapTif = np.clip(satMapTif, 0, 10000)
    satMapTif /= 10000
    return satMapTif
