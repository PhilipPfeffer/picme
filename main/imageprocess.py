import imageUtil
from imageUtil import Image
import numpy as np
import statistics
import argparse
import cv2

imagesCreated = []

def averageTest(image):
    """Average out all pixel colours."""
    newImageName = 'averageImage.jpg'
    averagePixs = image.getAverageRGB()
    imageUtil.saveNewAverageImage(newImageName, averagePixs, image.imageShape)
    imagesCreated.append(newImageName)

def derezzed(image, numSectorsX, numSectorsY, saveImage=False, verbose=0):
    """Averages out pixel over numSectorsX*numSectorsY sectors."""
    newImageName = 'sectorCols.jpg'
    sectorCols, compactArr = image.sectorColours(numSectorsX, numSectorsY, verbose)
    if saveImage:
        imageUtil.saveNewImage(newImageName, sectorCols)
        imagesCreated.append(newImageName)
    return compactArr

def extractSectorsFeature(image, numSectorsX=30, numSectorsY=30, saveImage=False):
    """A list of the colour distance of each sector."""
    compactArr = derezzed(image, numSectorsX, numSectorsY, saveImage, 1)
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
    return statistics.mean(sectorColourDistance)
    
def extractNumFaces(faceInfo):
    return len(faceInfo.keys())

def extractTotalPercentAreaFaces(faceInfo):
    totalArea = 0
    for face in faceInfo:
        totalArea += faceInfo[face][1]
    return totalArea

def runFaceDetectDNN():
    ptxt = 'deploy.prototxt.txt'
    dnnModel = 'res10_300x300_ssd_iter_140000.caffemodel'
    return cv2.dnn.readNetFromCaffe(ptxt, dnnModel)


def extractFaceInfo(image, confidenceThreshold=0.5):
    """Returns a dict of the {index of the face detected: (area of the image that image takes up, % area that face takes up)}."""
    net = runFaceDetectDNN()
    openCVImage = image.getOpenCVImage()
    # Below code structure inspired by https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
    (imageHeight, imageWidth) = openCVImage.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(openCVImage, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    facesDetected = net.forward()
    
    faceInfo = dict()
    for i in range(facesDetected.shape[2]):
        if facesDetected[0, 0, i, 2] > confidenceThreshold:
            imageScaling = facesDetected[0, 0, i, 3:7] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
            (faceX, faceY, faceEndX, faceEndY) = imageScaling.astype("int")
            faceArea = abs(faceX-faceEndX)*abs(faceY-faceEndY)
            faceInfo[i] = (faceArea, 100*faceArea/(imageHeight*imageWidth))
    return faceInfo

if __name__ == "__main__":
    # filePath = ('./', 'testimage.jpg')
    # filePath = "https://scontent-lax3-1.cdninstagram.com/vp/01dd9b31a97c8d49d774f2af7f22e6a9/5E56FE87/t51.2885-15/e35/s1080x1080/72668674_673939036467328_7431812160719464417_n.jpg?_nc_ht=scontent-lax3-1.cdninstagram.com&_nc_cat=104"
    filePath = "https://scontent-sjc3-1.cdninstagram.com/vp/096b4dd04709198a8d5bfe1f65d25254/5E4D3773/t51.2885-15/e35/49394327_933306223546224_3139891258521358385_n.jpg?_nc_ht=scontent-sjc3-1.cdninstagram.com&amp;_nc_cat=100"
    image = Image(filePath, True)
    # extractSectorsFeature(image, 30, 30)
    faceInfo = extractFaceInfo(image)
    print(faceInfo)
    print(extractNumFaces(faceInfo))
    print(extractTotalPercentAreaFaces(faceInfo))