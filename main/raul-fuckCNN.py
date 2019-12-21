import sys
import numpy as np
import imageprocess as imageProcess
import csv
from skimage.transform import rescale, resize, downscale_local_mean
from keras.models import Sequential, load_model, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv1D, MaxPooling1D, AveragePooling2D, BatchNormalization, concatenate
import matplotlib.pyplot as plt
import math
from keras import regularizers
import random
from datetime import datetime 
import textdistance
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from utils import *

TARGET_X = 135
TARGET_Y = 135

def downloadImages(dataset):
    print('Start reading features')
    with open(dataset) as f:
        allImgs = []
        shapes = []
        allResults = []
        notProcessed = 0
        totalImgs = 0
        correctShape = 0
        for row in csv.DictReader(f):
            if (float(row["likeRatio"]) > 1.):
                continue
            print(totalImgs)
            totalImgs += 1
            try:
                image = imageProcess.Image(row["imgUrl"], True)
                imageShape = image.getImageShape()
                shapes.append(imageShape)
                # squaredImage = imageShape[0] == imageShape[1]
                # isRgb = imageShape[2] == 3;
                # if (not squaredImage) or (not isRgb):    
                #     continue
                # image_rescaled = rescale(image.skimageImage, RESIZE_FACTOR, anti_aliasing=False, multichannel=True)
                image_rescaled = resize(image.skimageImage, (TARGET_X, TARGET_Y),anti_aliasing=False)
            except Exception as e:
                notProcessed += 1
                print(e)
                continue
            allImgs.append(image_rescaled)
            allResults.append(float(row["likeRatio"]))
    
    print("not processed: " + str(notProcessed/totalImgs))
    slashIndex = dataset.find("/")
    slashIndex += 1
    plt.figure()
    plt.plot(shapes)
    plt.title('image shape distribution')
    plt.ylabel('width')
    plt.xlabel('height')
    plt.savefig(f"datasets/{dataset[slashIndex:-4]}_distribution.png")
    np.save(f"allImgs_{dataset[slashIndex:-4]}.npy", allImgs)
    np.save(f"allResults_{dataset[slashIndex:-4]}.npy", allResults)
    return allImgs, allResults

############################################################
# Feature extraction
def extractFeaturesFromDataset(filename):
    print("PELE MEJOR QUE MARADONA!")
    net = imageProcess.runFaceDetectDNN()
    print('Start reading features')
    with open(filename) as f:
        featureVectors = []
        results = []
        allImgs = []
        allResults = []
        shapes = []
        notProcessed = 0
        totalImgs = 0
        correctShape = 0
        for row in csv.DictReader(f):
            if (float(row["likeRatio"]) > 1.):
                continue
            print(totalImgs)
            totalImgs += 1
            featureVector = []
            somethingFailed = False
            for key in row: #  each row is a dict
                try:
                    if (key == "timestamp"): 
                        hourOfDay = datetime.fromtimestamp(int(row[key])).hour
                        between2and6 = (hourOfDay >= 2 and hourOfDay < 6)
                        between6and10 = (hourOfDay >= 6 and hourOfDay < 10)
                        between10and14 = (hourOfDay >= 10 and hourOfDay < 14)
                        between14and18 = (hourOfDay >= 14 and hourOfDay < 18)
                        between18and22 = (hourOfDay >= 18 and hourOfDay < 22)
                        between22and2 = (hourOfDay >= 22) or (hourOfDay < 2)
                        featureVector.append(int(between2and6))
                        featureVector.append(int(between6and10))
                        # featureVector['between10and14'] = int(between10and14)
                        featureVector.append(int(between14and18)) 
                        featureVector.append(int(between18and22))
                        featureVector.append(int(between22and2))
                
                    
                    elif (key == "caption"):
                        # featureVector["captionLength"] = (len(row[key]))
                        featureVector.append(1 if "food" in row[key].lower() else 0)
                        featureVector.append(1 if "follow" in row[key].lower() else 0)
                        featureVector.append(1 if "ad" in row[key].lower() else 0)
                    
                    # if key == "hashtags":
                    #     hashtags = ast.literal_eval(row[key])
                    #     hashtags = [n.strip() for n in hashtags]
                        # featureVector["numHash"] = 1 if len(hashtags) == 0 else 1./len(hashtags)

                    elif key == "imgUrl":
                        image = imageProcess.Image(row[key], True)
                        imageShape = image.getImageShape()
                        shapes.append((imageShape[0], imageShape[1]))
                        print(f"shape: ({imageShape[0]}, {imageShape[1]})")
                        # squaredImage = imageShape[0] == imageShape[1]
                        # isRgb = imageShape[2] == 3;
                        # if (not squaredImage) or (not isRgb):    
                        #     continue
                        # image_rescaled = rescale(image.skimageImage, RESIZE_FACTOR, anti_aliasing=False, multichannel=True)
                        image_rescaled = resize(image.skimageImage, (TARGET_X, TARGET_Y),anti_aliasing=False)
                        # featureVector.append(imageProcess.extractSectorsFeature(image, 20, 20))
                        # faceInfo = imageProcess.extractFaceInfo(image, net)
                        # featureVector.append(imageProcess.extractNumFaces(faceInfo))
                        # featureVector.append(imageProcess.extractTotalPercentAreaFaces(faceInfo))
                    elif key == "likeRatio": # we will append the result at the end
                        continue #allResults.append(float(row[key]))
                    elif (key == "likeCount" or key == "commentCount" or key == "timestamp"):
                        featureVector.append(row[key])
                    # this should fail all the time we have a string as the value feature
                    # probably bad style but  python has no better way to check if 
                    # a string contains a float or not
                    else:
                        try:
                            val = float(row[key])
                            featureVector[key] = val
                        except Exception as e:
                            continue
                except Exception as e:
                    somethingFailed = True
                    notProcessed += 1
                    print(e)
                    break
            if (somethingFailed):
                continue
            label = float(row["likeRatio"])
            allResults.append(label)
            allImgs.append(image_rescaled)
            featureVectors.append(featureVector)
        slashIndex = filename.find("/")
        slashIndex += 1
        featureVectors = np.array(featureVectors)
        allResults = np.array(allResults)
        allImgs = np.array(allImgs)

        plt.figure()    
        plt.title('image shape distribution')
        plt.ylabel('width')
        plt.xlabel('height')
        plt.scatter(*zip(*shapes))
        plt.savefig(f"datasets/{filename[slashIndex:-4]}_distribution.png")
        np.save(f"allImgs_{filename[slashIndex:-4]}.npy", allImgs)
        np.save(f"allResults_{filename[slashIndex:-4]}.npy", allResults)
        np.save(f"featureVectors_{filename[slashIndex:-4]}.npy", featureVectors)
        return allImgs, featureVectors, allResults



def trainModel(modelfilename, x_train, y_train, x_test, y_test, concat=False):
    #create model
    model = Sequential()

    # #add model layers
    # model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(TARGET_X, TARGET_Y,3)))
    # # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(Dropout(0.3))
    # # model.add(Conv2D(32, kernel_size=3, activation='relu'))
    # # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(Dropout(0.3))
    # # model.add(Conv2D(64, kernel_size=3, activation='relu'))
    # # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(Dropout(0.3))
    # # model.add(Conv2D(128, kernel_size=3, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(100, activation='softmax'))
    #             # kernel_regularizer=regularizers.l2(0.01),
    #             # activity_regularizer=regularizers.l1(0.01)))

    model.add(Conv2D(16, kernel_size=(3,3), input_shape=(TARGET_X, TARGET_Y, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(BatchNormalization())
    # model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(BatchNormalization())
    # model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1250, activation='relu'))
    model.add(Dropout(.5))
    # model.add(BatchNormalization())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(.25))
    # model.add(BatchNormalization())
    model.add(Dense(500, activation='relu'))
    # model.add(BatchNormalization())

    if concat:
        model.add(Dense(100, activation = 'relu'))
        return model
    else:
        model.add(Dense(100, activation = 'softmax'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    #compile model using accuracy to measure model performance
    model.compile(optimizer=Adam(lr=1e-4, decay=1e-4 / 200), loss='categorical_crossentropy', metrics=['accuracy', 'cosine_proximity'])

    #train the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

    print(model.summary())

    plot_model(model, to_file=f"models/{modelfilename}_plot.png", show_shapes=True, show_layer_names=True)

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"models/{modelfilename}_accuracy.png")

    plt.figure()
    plt.plot(history.history['cosine_proximity'])
    plt.plot(history.history['val_cosine_proximity'])
    plt.title('cosine proximity')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"models/{modelfilename}_cosineproximity.png")

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"models/{modelfilename}_loss.png")

    model.save(f"models/{modelfilename}.h5")
    return model

def trainMdModel(modelfilename, x_train, y_train, x_test, y_test, concat=False):
    #create model
    model = Sequential()

    model = Sequential()
    model.add(Dense(400, input_dim=10, activation="relu"))
    model.add(Dense(300, activation="relu"))
    model.add(Dense(300, activation="relu"))
    model.add(Dense(200, activation="relu"))
    if concat:
        model.add(Dense(100, activation = 'relu'))
        return model
    else:
        model.add(Dense(100, activation = 'softmax'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    #compile model using accuracy to measure model performance
    model.compile(optimizer=Adam(lr=1e-4, decay=1e-4 / 200), loss='categorical_crossentropy', metrics=['accuracy', 'cosine_proximity'])

    #train the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

    print(model.summary())

    plot_model(model, to_file=f"models/{modelfilename}_metadata_plot.png", show_shapes=True, show_layer_names=True)

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"models/{modelfilename}_metadata_accuracy.png")

    plt.figure()
    plt.plot(history.history['cosine_proximity'])
    plt.plot(history.history['val_cosine_proximity'])
    plt.title('cosine proximity')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"models/{modelfilename}_metadata_cosineproximity.png")

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"models/{modelfilename}_metadata_loss.png")

    model.save(f"models/{modelfilename}.h5")
    return model

def test_model(model, x, y):
    indexed_y = list(enumerate(y))
    indexed_y.sort(key=lambda tup: tup[1])
    ideal_ranking = []
    for elem in indexed_y:
        ideal_ranking.append(elem[0])

    predictions = model.predict(x)

    scores = []
    for prediction in predictions:
        scores.append(np.dot(prediction, [i for i in range(1,101)]))
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda tup: tup[1])
    predicted_ranking = []
    for elem in indexed_scores:
        predicted_ranking.append(elem[0])
    if (ideal_ranking == predicted_ranking):
        print("Success! Prediction matched!")
    else:
        mistakes = 0
        for i in range(len(ideal_ranking)):
            if ideal_ranking[i] != predicted_ranking[i]:
                mistakes += 1 
        print(f"We made {mistakes} mistakes. Error ratio: {mistakes/len(ideal_ranking)}")
        print(f"similarity: {textdistance.levenshtein.similarity(ideal_ranking,predicted_ranking)}")
        print(f"distance: {textdistance.levenshtein.distance(ideal_ranking,predicted_ranking)}")
    print(f"ideal_ranking:     {ideal_ranking}")
    print(f"predicted_ranking: {predicted_ranking}\n\n")

def predict(model, x_dev, y_dev):
    print(f"y_dev: {y_dev}")
    actualBestImageIndex = np.argmax(y_dev)

    print(f"theoretical best image index: {actualBestImageIndex}")

    predictions = model.predict(x_dev)

    scores = []
    for prediction in predictions:
        scores.append(np.dot(prediction, [i for i in range(1,101)]))
    scores = np.array(scores)
    predictedImageIndex = np.argmax(scores)
    print(f"scores: {scores}")
    print(f"predictedImageIndex: {predictedImageIndex}")
    plt.figure()
    plt.imshow(x_dev[predictedImageIndex])
    
    #  # WRONG - USE PHIL'S VERSION ABOVE
    # maxBucket = 0 
    # maxProbsForMaxBucket = 0.0
    # predictedImageIndex = 0
    # for i in range(len(predictions)):
    #     prediction = predictions[i]
    #     maxResultIndex = np.argmax(prediction)
    #     maxResult = prediction[maxResultIndex]
    #     if maxBucket <= maxResultIndex:
    #         maxBucket = maxResultIndex
    #         maxProbsForMaxBucket = maxResult
    #         predictedImageIndex = i
    #     elif maxBucket == maxResultIndex:
    #         if maxProbsForMaxBucket < maxResult:
    #             maxProbsForMaxBucket = maxResult
    #             predictedImageIndex = i
    # print(f"predictedImageIndex: {predictedImageIndex}")
    # plt.figure()
    # plt.imshow(x_dev[predictedImageIndex])

    plt.figure()
    plt.imshow(x_dev[actualBestImageIndex])
    plt.show()


def concatenatedModelMain():
    if (len(sys.argv) < 2):
        print("Don't forget the flag!")
        sys.exit(0)
    if (sys.argv[1] == "-d"):
        allImgs, featureVectors, allResults = extractFeaturesFromDataset(sys.argv[2])
        imgX_train, imgX_dev, imgX_test, imgY_train, imgY_dev, imgY_test, imgY_train_one_hot, imgY_test_one_hot = splitAndPrep(allImgs, allResults)
        mdX_train, mdX_dev, mdX_test, mdY_train, mdY_dev, mdY_test, mdY_train_one_hot, mdY_test_one_hot = splitAndPrep(featureVectors, allResults)
        # assert(imgY_train, mdY_train) # raul-imgclassifier.py:537: SyntaxWarning: assertion is always true, perhaps remove parentheses?
        # imgX_train = imgX_train.reshape(imgX_train.shape[0], TARGET_X, TARGET_Y, 3)
        # imgX_test = imgX_test.reshape(imgX_test.shape[0], TARGET_X, TARGET_Y, 3)
        imageModel = trainModel(extractDatasetNameCSV(sys.argv[2]), imgX_train, imgY_train_one_hot, imgX_test, imgY_test_one_hot, True)
        mdModel = trainMdModel(extractDatasetNameCSV(sys.argv[2]), mdX_train, mdY_train_one_hot, mdX_test, mdY_test_one_hot, True)

        combinedInput = concatenate([mdModel.output, imageModel.output])
        x = Dense(100, activation="relu")(combinedInput)
        x = Dense(100, activation="softmax")(x)
        concatModel = Model(inputs=[mdModel.input, imageModel.input], outputs=x)
        concatModel.compile(optimizer=Adam(lr=1e-4, decay=1e-4 / 200), loss='categorical_crossentropy', metrics=['accuracy', 'cosine_proximity'])
        concatModel.fit([mdX_train, imgX_train], mdY_train,validation_data=([mdX_test, imgX_test], mdY_test), epochs=100, batch_size=8)


    elif (sys.argv[1] == "-f"):
        allImgs, allResults, featureVectors = loadFromFile([sys.argv[2], sys.argv[3], sys.argv[4]])
        # allImgs = grayscaleResize(allImgs)
        imgX_train, imgX_dev, imgX_test, imgY_train, imgY_dev, imgY_test, imgY_train_one_hot, imgY_test_one_hot = splitAndPrep(allImgs, allResults)
        mdX_train, mdX_dev, mdX_test, mdY_train, mdY_dev, mdY_test, mdY_train_one_hot, mdY_test_one_hot = splitAndPrep(featureVectors, allResults)
        # assert(imgY_train, mdY_train) #raul-imgclassifier.py:537: SyntaxWarning: assertion is always true, perhaps remove parentheses?
        # imgX_train = imgX_train.reshape(imgX_train.shape[0], TARGET_X, TARGET_Y, 3)
        # imgX_test = imgX_test.reshape(imgX_test.shape[0], TARGET_X, TARGET_Y, 3)
        imageModel = trainModel(extractDatasetNameCSV(sys.argv[2]), imgX_train, imgY_train_one_hot, imgX_test, imgY_test_one_hot, True)
        mdModel = trainMdModel(extractDatasetNameCSV(sys.argv[2]), mdX_train, mdY_train_one_hot, mdX_test, mdY_test_one_hot, True)

        combinedInput = concatenate([mdModel.output, imageModel.output])
        x = Dense(250, activation="relu")(combinedInput)
        x = Dense(100, activation="softmax")(x)
        concatModel = Model(inputs=[mdModel.input, imageModel.input], outputs=x)
        plot_model(concatModel, to_file=f"models/concatModel_plot.png", show_shapes=True, show_layer_names=True)
        concatModel.compile(optimizer=Adam(lr=1e-4, decay=1e-4 / 200), loss='categorical_crossentropy', metrics=['accuracy', 'cosine_proximity'])
        concatModel.fit([mdX_train, imgX_train], mdY_train,validation_data=([mdX_test, imgX_test], mdY_test), epochs=100, batch_size=8)

        # try:
        #     if(sys.argv[4] == "-m"):
        #         model = load_model(sys.argv[5])
        # except:
        #     model = trainModel(extractDatasetNameNPY(sys.argv[2]), x_train, y_train_one_hot, x_test, y_test_one_hot)    
    else:
        print("Invalid flag, mate!")
        sys.exit(0)
    # TODO uncomment below later    
    # split_test_set = splitList(x_test, 10)
    # split_result_test = splitList(y_test, 10)   
    # for i in range(len(split_result_test)):
    #     test_model(model, split_test_set[i], split_result_test[i])

def oldmain():
    if (len(sys.argv) < 2):
        print("Don't forget the flag!")
        sys.exit(0)
    if (sys.argv[1] == "--full"):
        # featureVectors, results = extractFeaturesFromDataset(sys.argv[2])
        featureVectors, results = loadFromFile([sys.argv[2], sys.argv[3]])
        x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot = splitAndPrep(featureVectors, results)
        model = trainModelFeatureVec("featuresvectormodel", x_train, y_train_one_hot, x_test, y_test_one_hot)
    elif (sys.argv[1] == "-d"):
        allImgs, allResults = downloadImages(sys.argv[2])
        x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot = splitAndPrep(allImgs, allResults)
        # x_train = x_train.reshape(x_train.shape[0], TARGET_X, TARGET_Y, 1)
        # x_test = x_test.reshape(x_test.shape[0], TARGET_X, TARGET_Y, 3)
        model = trainModel(extractDatasetNameCSV(sys.argv[2]), x_train, y_train_one_hot, x_test, y_test_one_hot)
    elif (sys.argv[1] == "-f"):
        allImgs, allResults = loadFromFile([sys.argv[2], sys.argv[3]])
        # allImgs = grayscaleResize(allImgs)
        x_train, x_dev, x_test, y_train, y_dev, y_test, y_train_one_hot, y_test_one_hot = splitAndPrep(allImgs, allResults)        
        try:
            if(sys.argv[4] == "-m"):
                model = load_model(sys.argv[5])
        except:
            model = trainModel(extractDatasetNameNPY(sys.argv[2]), x_train, y_train_one_hot, x_test, y_test_one_hot)    
    else:
        print("Invalid flag, mate!")
        sys.exit(0)
    # split_test_set = splitList(x_test, 10)
    # split_result_test = splitList(y_test, 10)   
    # for i in range(len(split_result_test)):
    #     test_model(model, split_test_set[i], split_result_test[i])

if __name__ == "__main__":
    oldmain()