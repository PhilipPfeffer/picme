import skimage
import os
from skimage import io
import numpy as np

# Main for now
def dostuff():
    imageData = imageProcess('./', 'testimage.jpg')
    averagePixs = averageColour(imageData[0], imageData[1])
    # saveNewImage('averageImage.jpg', averagePixs, imageData[1]) # avg image
    


# Takes image location, returns (image ndarray, array shape)
# e.g. array shape = (1080, 1080, 3)
def imageProcess(imageLocation, imageName):
    filename = os.path.join(imageLocation, imageName)
    imageArray = io.imread(filename)
    return (imageArray, imageArray.shape)

# Returns average (r,g,b) over all image
def averageColour(imageArray, imageArrayShape):
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

# New Image
def saveNewImage(name, avgPixs, shape):
    newImageArray = np.zeros(shape, dtype = np.uint8)
    for x in range(shape[0]):
        for y in range(shape[1]):
            newImageArray[x,y,0] = int(avgPixs[0])
            newImageArray[x,y,1] = int(avgPixs[1])
            newImageArray[x,y,2] = int(avgPixs[2])
    # print(newImageArray)
    outputPath = os.path.abspath(".") + "/" + name
    io.imsave(outputPath, newImageArray)

if __name__ == "__main__":
    dostuff()
    
