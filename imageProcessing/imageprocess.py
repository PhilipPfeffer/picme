import imageUtil
from imageUtil import Image

imagesCreated = []

# Average out all pixel colours
def averageTest(image):
    # imageData = imageProcess(path, name)
    newImageName = 'averageImage.jpg'
    averagePixs = image.getAverageRGB()
    imageUtil.saveNewAverageImage(newImageName, averagePixs, image.imageShape)
    imagesCreated.append(newImageName)

def colourful(path, name, numSectorsX, numSectorsY):
    # imageData = imageProcess(path, name)
    newImageName = 'sectorCols.jpg'
    sectorCols = image.sectorColours(numSectorsX, numSectorsY)
    imageUtil.saveNewImage(newImageName, sectorCols, image.imageShape)
    imagesCreated.append(newImageName)

if __name__ == "__main__":
    path = './'
    name = 'testimage.jpg'
    image = Image(path, name)
    averageTest(image)
    # colourful(path, name, 8, 8)