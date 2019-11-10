import imageUtil
from imageUtil import imageProcess

# Average out all pixel colours
def averageTest(path, name):
    imageData = imageProcess(path, name)
    averagePixs = imageUtil.averageColour(imageData[0], imageData[1])
    imageUtil.saveNewAverageImage('averageImage.jpg', averagePixs, imageData[1]) # avg image

def colourful(path, name):
    imageData = imageProcess(path, name)

if __name__ == "__main__":
    path = './'
    name = 'testimage.jpg'
    averageTest(path, name)
    # colourful(path, name)