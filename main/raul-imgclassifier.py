import sys
import numpy as np
import imageprocess as imageProcess
import csv
from skimage.transform import rescale, resize, downscale_local_mean
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt

TARGET_X = 270
TARGET_Y = 270

def downloadImages(dataset):
    print('Start reading features')
    with open(dataset) as f:
        allImgs = []
        allResults = []
        notProcessed = 0
        totalImgs = 0
        correctShape = 0
        for row in csv.DictReader(f):
            totalImgs += 1
            try:
                image = imageProcess.Image(row["imgUrl"], True)
                imageShape = image.getImageShape()
                # squaredImage = imageShape[0] == imageShape[1]
                # isRgb = imageShape[2] == 3;
                # if (not squaredImage) or (not isRgb):    
                #     continue
                # image_rescaled = rescale(image.skimageImage, RESIZE_FACTOR, anti_aliasing=False, multichannel=True)
                image_rescaled = resize(image.skimageImage, (TARGET_X, TARGET_Y),anti_aliasing=False)
                correctShape += 1
            except Exception as e:
                notProcessed += 1
                continue
            allImgs.append(image_rescaled)
            allResults.append(float(row["likeRatio"]))
    print(f"not processed: {notProcessed/totalImgs}")
    print(f"correct shape total: {correctShape}")
    print(f"correct shape ratio: {correctShape/totalImgs}")
    np.save("allImgs.npy", allImgs)
    np.save("allResults.npy", allResults)
    return (allImgs, allResults)

def loadFromFiles():
    allImgs = np.load("allImgs.npy")
    allResults = np.load("allResults.npy")
    return allImgs, allResults

def trainModel(allImgs, allResults):
    

    # TODO DEVSET
    (x_train, x_dev, x_test), (y_train, y_dev, y_test) = splitDataset(allImgs, allResults)


    #reshape data to fit model
    # NO NEED TO CALL RESHAPE BECAUSE ALL IMAGES HAVE BEEN PREVIOUSLY RESIZED
    # x_train = x_train.reshape(len(x_train),int(1080*RESIZE_FACTOR),int(1080*RESIZE_FACTOR),3)
    # x_test = x_test.reshape(len(x_test),int(1080*RESIZE_FACTOR),int(1080*RESIZE_FACTOR),3)


    #one-hot encode target column
    oneHots = []
    for y in y_train:
        # 10 buckets
        index = int(y*10)%10
        oneHot = []
        for i in range(10):
            oneHot.append(0 if i != index else 1)
        oneHots.append(oneHot)
    y_train = np.array(oneHots)

    oneHots = []
    for y in y_test:
        # 10 buckets
        index = int(y*10)%10
        oneHot = []
        for i in range(10):
            oneHot.append(0 if i != index else 1)
        oneHots.append(oneHot)
    y_test = np.array(oneHots)

    #create model
    model = Sequential()

    #add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(TARGET_X, TARGET_Y,3)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)
    return model, x_dev, y_dev

def predict(model, x_dev, y_dev):
    # display x_dev bc humans cant see arrays of pixels
    # for image in x_dev:
    #     plt.figure()
    #     plt.imshow(image)
    # plt.show()
    # print(y_dev)
    theoreticalBestImageIndex = np.argmax(y_dev)
    print(f"theoretical best image index: {theoreticalBestImageIndex}")

    predictions = model.predict(x_dev)
    print(predictions)

    maxBucket = 0
    maxProbsForMaxBucket = 0.0
    idealImageIndex = 0;
    for i in range(len(predictions)):
        prediction = predictions[i]
        maxResultIndex = np.argmax(prediction)
        maxResult = prediction[maxResultIndex]
        if maxBucket < maxResultIndex:
            maxBucket = maxResultIndex
            maxProbsForMaxBucket = maxResult
            idealImageIndex = i
        elif maxBucket == maxResultIndex:
            if maxProbsForMaxBucket < maxResult:
                maxProbsForMaxBucket = maxResult
                idealImageIndex = i

    plt.figure()
    print(f"idealImageIndex: {idealImageIndex}")
    plt.imshow(x_dev[idealImageIndex])
    plt.show()


def splitDataset(allImgs, allResults):
    limitTrain = 6*int(len(allImgs)/10)
    x_train = np.array(allImgs[:limitTrain])
    y_train = np.array(allResults[:limitTrain])
    prevx_test = np.array(allImgs[limitTrain:])
    prevy_test = np.array(allResults[limitTrain:])
    limitTest = int(len(prevx_test)/2)
    x_dev = np.array(prevx_test[limitTest:])
    y_dev = np.array(prevy_test[:limitTest])
    x_test = np.array(prevx_test[:limitTest])
    y_test = np.array(prevy_test[:limitTest])
    return (x_train, x_dev, x_test), (y_train, y_dev, y_test)

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Don't forget the flag!")
        sys.exit(0)
    if (sys.argv[1] == "-d"):
        allImgs, allResults = downloadImages('datasets/neuralnet-firstdataset.csv')
    else: #load from file
        allImgs, allResults = loadFromFiles()
    model, x_dev, y_dev = trainModel(allImgs, allResults)
    predict(model, x_dev, y_dev)
