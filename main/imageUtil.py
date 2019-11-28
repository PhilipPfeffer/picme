import skimage
import os
from skimage import io
import numpy as np
from collections import defaultdict
import urllib
import cv2

class Image:
    def __init__(self, filePath, isURL=False):
        self.filePath = filePath
        self.isURL = isURL
        self.skimageImage = None
        self.imageArray = []
        self.imageShape = (0,0,0)
        processed = self.imageProcess()
        if (not processed):
            raise ValueError("Invalid URL for image. Image object could not be created")
        self.distanceCache = defaultdict(int)
        if isURL: self.imageOpenCV = self.convertURLtoImage(filePath)

    def getImageData(self):
        return (self.imageArray, self.imageShape)

    def getImageArray(self):
        return self.imageArray

    def getImageShape(self):
        return self.imageShape
    
    def getOpenCVImage(self):
        return self.imageOpenCV

    # ================== IMPLEMENTATION ================== #
    def imageProcess(self):
        """ Takes image location, returns (image ndarray, array shape)
            e.g. array shape = (1080, 1080, 3)."""
        filename = self.filePath if self.isURL else os.path.join(self.filePath[0], self.filePath[1])
        try:
            self.skimageImage = io.imread(filename)
            self.imageArray, self.imageShape = (self.skimageImage, self.skimageImage.shape)
            return True
        except Exception as e:
            print(e)
            return False
    
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

    def sectorColours(self, numSectorsX, numSectorsY, verbose = 0):
        """Returns an imageArray and compact array with each sector's colour averaged out."""
        sectorArray = np.zeros(self.imageShape, dtype = np.uint8)
        sectorWidth = self.imageShape[0]//numSectorsX
        sectorHeight = self.imageShape[1]//numSectorsY
        numSectors = numSectorsX*numSectorsY
        compactArray = np.zeros((numSectors, 3), dtype = np.uint8)
        for n in range(numSectors):
            if verbose != 0: print(f"Sector {n+1}/{numSectors}")
            sectorR = 0
            sectorG = 0
            sectorB = 0
            for x in range((n%numSectorsX) * sectorWidth, (n%numSectorsX + 1) * sectorWidth, 10):
                for y in range((n//numSectorsY) * sectorHeight, (n//numSectorsY + 1) * sectorHeight, 10):
                    sectorR += self.imageArray[x][y][0]
                    sectorG += self.imageArray[x][y][1]
                    sectorB += self.imageArray[x][y][2]
            sectorR /= (sectorWidth*sectorHeight)
            sectorG /= (sectorWidth*sectorHeight)
            sectorB /= (sectorWidth*sectorHeight)
            compactArray[n][0] = sectorR
            compactArray[n][1] = sectorG
            compactArray[n][2] = sectorB
            for x in range((n%numSectorsX) * sectorWidth, (n%numSectorsX + 1) * sectorWidth):
                for y in range((n//numSectorsY) * sectorHeight, (n//numSectorsY + 1) * sectorHeight):
                    sectorArray[x][y][0] = sectorR
                    sectorArray[x][y][1] = sectorG
                    sectorArray[x][y][2] = sectorB
        
        return (sectorArray, compactArray)

    def calculateColourDistance(self, compactArr, numSectorsX, numSectorsY, sector1, sector2):
        if self.distanceCache[(numSectorsX, numSectorsY, sector1, sector2)] != 0:
            return self.distanceCache[(numSectorsX, numSectorsY, sector1, sector2)]
        if self.distanceCache[(numSectorsX, numSectorsY, sector2, sector1)] != 0:
            return self.distanceCache[(numSectorsX, numSectorsY, sector2, sector1)]
        distance = 0
        distance += abs(int(compactArr[sector1][0]) - compactArr[sector2][0])
        distance += abs(int(compactArr[sector1][1]) - compactArr[sector2][1])
        distance += abs(int(compactArr[sector1][2]) - compactArr[sector2][2])
        self.distanceCache[(numSectorsX, numSectorsY, sector1, sector2)] = distance
        return distance
 
    def convertURLtoImage(self, URL): # TODO why is URL a parameter here?
        """Converts a URL to the image format used by OpenCV, stores it in self.image."""
        return cv2.cvtColor(self.skimageImage, cv2.COLOR_BGR2RGB)


# ================== OTHER FUNCTIONS ================== #
def saveNewAverageImage(name, avgPixs, shape):
    """Saves new image with all pixels avaraged, provided the average (r,g,b) pixel values."""
    newImageArray = np.zeros(shape, dtype = np.uint8)
    for x in range(shape[0]):
        for y in range(shape[1]):
            newImageArray[x,y,0] = int(avgPixs[0])
            newImageArray[x,y,1] = int(avgPixs[1])
            newImageArray[x,y,2] = int(avgPixs[2])
    # print(newImageArray)
    outputPath = os.path.abspath(".") + "/" + name
    io.imsave(outputPath, newImageArray)

def saveNewImage(name, sectorArray):
    outputPath = os.path.abspath(".") + "/" + name
    io.imsave(outputPath, sectorArray)