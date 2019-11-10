import skimage
import os
from skimage import io
import numpy as np

def imageProcess(imageLocation, imageName):
    """ Takes image location, returns (image ndarray, array shape)
        e.g. array shape = (1080, 1080, 3)."""
    filename = os.path.join(imageLocation, imageName)
    imageArray = io.imread(filename)
    return (imageArray, imageArray.shape)

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

def averageColour(imageArray, imageArrayShape):
    """Returns average (r,g,b) over all image."""
    rTotal = 0
    gTotal = 0
    bTotal = 0
    numPix = imageArrayShape[0]*imageArrayShape[1]
    for x in range(imageArrayShape[0]):
        for y in range(imageArrayShape[1]):
            rTotal += imageArray[x][y][0]
            gTotal += imageArray[x][y][1]
            bTotal += imageArray[x][y][2]
    return (rTotal/numPix, gTotal/numPix, bTotal/numPix)