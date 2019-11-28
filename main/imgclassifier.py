import numpy as np
import imageprocess as imageProcess
import csv
from skimage.transform import rescale, resize, downscale_local_mean

RESIZE_FACTOR = 0.25
print('Start reading features')
with open('datasets/neuralnet-firstdataset.csv') as f:
    allImgs = []
    allResults = []
    notProcessed = 0
    totalImgs = 0
    correctShape = 0
    for row in csv.DictReader(f):
        totalImgs += 1
        try:
            image = imageProcess.Image(row["imgUrl"], True)
            if image.getImageShape() != (1080, 1080, 3):
                continue
            image_rescaled = rescale(image.skimageImage, RESIZE_FACTOR, anti_aliasing=False, multichannel=True)
            correctShape += 1
        except Exception as e:
            notProcessed += 1
            continue
        allImgs.append(image_rescaled)
        allResults.append(float(row["likeRatio"]))
print(f"not processed: {notProcessed/totalImgs}")
print(f"correct shape total: {correctShape}")
print(f"correct shape ratio: {correctShape/totalImgs}")


limitLen = 2*int(len(allImgs)/3)
x_train = np.array(allImgs[:limitLen])
y_train = np.array(allResults[:limitLen])
x_test = np.array(allImgs[limitLen:])
y_test = np.array(allResults[limitLen:])

#reshape data to fit model
x_train = x_train.reshape(len(x_train),int(1080*RESIZE_FACTOR),int(1080*RESIZE_FACTOR),3)
x_test = x_test.reshape(len(x_test),int(1080*RESIZE_FACTOR),int(1080*RESIZE_FACTOR),3)


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

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

#create model
model = Sequential()

#add model layers
model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(int(1080*RESIZE_FACTOR),int(1080*RESIZE_FACTOR),3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)