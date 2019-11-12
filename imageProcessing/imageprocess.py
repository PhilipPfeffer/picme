import imageUtil
from imageUtil import Image
import numpy as np

imagesCreated = []

def averageTest(image):
    """Average out all pixel colours."""
    newImageName = 'averageImage.jpg'
    averagePixs = image.getAverageRGB()
    imageUtil.saveNewAverageImage(newImageName, averagePixs, image.imageShape)
    imagesCreated.append(newImageName)

def derezzed(image, numSectorsX, numSectorsY, verbose=0):
    """Averages out pixel over numSectorsX*numSectorsY sectors."""
    newImageName = 'sectorCols.jpg'
    sectorCols, compactArr = image.sectorColours(numSectorsX, numSectorsY, verbose)
    imageUtil.saveNewImage(newImageName, sectorCols)
    imagesCreated.append(newImageName)
    return compactArr

def extractSectorsFeature(image, numSectorsX=30, numSectorsY=30):
    """A list of the colour distance of each sector."""
    compactArr = derezzed(image, numSectorsX, numSectorsY, 1)
    numSectors = numSectorsX*numSectorsY
    sectorColourDistance = np.zeros(numSectors, dtype = np.int64)
    for n in range(numSectors):
        colourDistance = 0
        colourDistance += image.calculateColourDistance(compactArr, numSectorsX, numSectorsY, n, n-1) if n%numSectorsX != 0 else 0
        colourDistance += image.calculateColourDistance(compactArr, numSectorsX, numSectorsY, n, n+1) if n%numSectorsX != numSectorsX-1 else 0
        colourDistance += image.calculateColourDistance(compactArr, numSectorsX, numSectorsY, n, n-numSectorsX) if n//numSectorsY != 0 else 0
        colourDistance += image.calculateColourDistance(compactArr, numSectorsX, numSectorsY, n, n+numSectorsX) if n//numSectorsY != numSectorsY-1 else 0
        colourDistance /= 4
        sectorColourDistance[n] = colourDistance
    return sectorColourDistance

if __name__ == "__main__":
    # filePath = ('./', 'testimage.jpg')
    filePath = "https://scontent-lax3-1.cdninstagram.com/vp/01dd9b31a97c8d49d774f2af7f22e6a9/5E56FE87/t51.2885-15/e35/s1080x1080/72668674_673939036467328_7431812160719464417_n.jpg?_nc_ht=scontent-lax3-1.cdninstagram.com&_nc_cat=104"
    image = Image(filePath, True)
    extractSectorsFeature(image, 30, 30)