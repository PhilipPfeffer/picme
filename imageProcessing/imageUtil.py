import skimage
import os
from skimage import io
import numpy as np

class Image:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.imageArray = []
        self.imageShape = (0,0,0)
        self.imageProcess()

    def getImageData(self):
        return (self.imageArray, self.imageShape)

    def getImageArray(self):
        return self.imageShape

    def getImageShape(self):
        return self.imageShape

    # ================== IMPLEMENTATION ================== #
    def imageProcess(self):
        """ Takes image location, returns (image ndarray, array shape)
            e.g. array shape = (1080, 1080, 3)."""
        filename = os.path.join(self.path, self.name)
        imageArray = io.imread(filename)
        self.imageArray, self.imageShape = (imageArray, imageArray.shape)
        return (self.imageArray, self.imageShape)
    
    def getAverageRGB(self):
        """Returns average (r,g,b) over all image."""
        rTotal = 0
        gTotal = 0
        bTotal = 0
        numPix = self.imageShape[0]*self.imageShape[1]
        for x in range(self.imageShape[0]):
            for y in range(self.imageShape[1]):
                rTotal += self.imageArray[x][y][0]
                gTotal += self.imageArray[x][y][1]
                bTotal += self.imageArray[x][y][2]
        return (rTotal/numPix, gTotal/numPix, bTotal/numPix)

    def sectorColours(self, numSectorsX, numSectorsY):
        """Returns an imageArray with the """
        self.imageArray + self.imageShape


def saveNewAverageImage(name, avgPixs, shape):
    """Saves new image with all pixels avaraged, provided the average (r,g,b) pixel values"""
    newImageArray = np.zeros(shape, dtype = np.uint8)
    for x in range(shape[0]):
        for y in range(shape[1]):
            newImageArray[x,y,0] = int(avgPixs[0])
            newImageArray[x,y,1] = int(avgPixs[1])
            newImageArray[x,y,2] = int(avgPixs[2])
    # print(newImageArray)
    outputPath = os.path.abspath(".") + "/" + name
    io.imsave(outputPath, newImageArray)

def saveNewImage(name, sectorCols, shape):
    pass