import imageUtil
from imageUtil import Image

imagesCreated = []

# Average out all pixel colours
def averageTest(image):
    newImageName = 'averageImage.jpg'
    averagePixs = image.getAverageRGB()
    imageUtil.saveNewAverageImage(newImageName, averagePixs, image.imageShape)
    imagesCreated.append(newImageName)

def colourful(path, name, numSectorsX=30, numSectorsY=30):
    newImageName = 'sectorCols.jpg'
    sectorCols = image.sectorColours(numSectorsX, numSectorsY)
    imageUtil.saveNewImage(newImageName, sectorCols)
    imagesCreated.append(newImageName)

if __name__ == "__main__":
    path = './'
    name = 'testimage.jpg'
    image = Image(path, name)
    # averageTest(image)
    colourful(path, name)