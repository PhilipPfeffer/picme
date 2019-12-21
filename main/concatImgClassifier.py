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
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from utils import *
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from statistics import mean
from matplotlib.ticker import FormatStrFormatter


TARGET_X = 135
TARGET_Y = 135
BUCKET_NUM = 10
EPOCH_NUM = 6

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

def extractFeaturesFromDataset(filename):
    print("PELE MEJOR QUE MARADONA!")
    # net = imageProcess.runFaceDetectDNN()
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
        a = True
        for row in csv.DictReader(f):
            if a:
                print(row.keys())
                a = False
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
                        featureVector.append(int(between10and14))
                        featureVector.append(int(between14and18))
                        featureVector.append(int(between18and22))
                        featureVector.append(int(between22and2))

                        dayOfWeek = ep_to_day(int(row[key]))
                        if dayOfWeek == "Sunday":
                            featureVector.append(1)
                        else:
                            featureVector.append(0)                        
                        if dayOfWeek == "Monday":
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if dayOfWeek == "Tuesday":
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if dayOfWeek == "Wednesday":
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if dayOfWeek == "Thursday":
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if dayOfWeek == "Friday":
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if dayOfWeek == "Saturday":
                            featureVector.append(1)
                        else:
                            featureVector.append(0) 
                
                    
                    elif (key == "accessibilityCaption"):

                        accessibilityCaption = row[key]
                        if "people" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "and" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "one" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "or" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "more" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "standing" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "nature" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "closeup" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "sitting" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "tree" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "photo" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "no" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "description" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "available" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "cloud" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "beard" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "mountain" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "child" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "playing" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "sports" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "sunglasses" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "on" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "grass" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "suit" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "selfie" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "crowd" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "1" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "person" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "wedding" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)
                        if "baby" in accessibilityCaption:
                            featureVector.append(1)
                        else:
                            featureVector.append(0)

                    elif key == "imgUrl":

                        image = imageProcess.Image(row[key], True)
                        imageShape = image.getImageShape()
                        shapes.append((imageShape[0], imageShape[1]))
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
                    elif (key == "likeCount" or key == "commentCount"):

                        continue
                        featureVector.append(row[key])
                    elif (key == "isBusinessAcc"):
                        featureVector.append(int(row[key]=="True"))
                    elif (key == "isVerified"):
                        featureVector.append(int(row[key]=="True"))
                    elif (key == "hasChannel"):
                        featureVector.append(int(row[key]=="True"))

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
            label = float(row["likeCount"])/float(row["userAverageLikes"])
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
    model.add(Conv2D(16, kernel_size=(3,3), input_shape=(TARGET_X, TARGET_Y, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1250, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(.25))

    if concat:
        model.add(Dense(BUCKET_NUM, activation='relu'))
        return model
    else:
        model.add(Dense(BUCKET_NUM, activation = 'softmax'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    #compile model using accuracy to measure model performance
    model.compile(optimizer=Adam(lr=1e-4, decay=1e-4 / 200), loss='categorical_crossentropy', metrics=['accuracy'])

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

    model.add(Dense(200, input_dim=46, activation="relu"))
    model.add(Dense(150, activation="relu"))
    model.add(Dense(125, activation="relu"))
    model.add(Dense(110, activation="relu"))
    model.add(Dense(105, activation="relu"))
    if concat:
        model.add(Dense(BUCKET_NUM, activation = "relu"))
        return model
    else:
        model.add(Dense(BUCKET_NUM, activation = 'softmax'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    #compile model using accuracy to measure model performance
    model.compile(optimizer=Adam(lr=1e-4, decay=1e-4 / 200), loss='categorical_crossentropy', metrics=['accuracy'])

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

    model.save(f"models/{modelfilename}_metadata_plot.h5")
    return model

def test_model(model, x, y, dataname):
    predictions = model.predict(x)
    hist_scatter(predictions, y, dataname)
    spearman(predictions, y, dataname)

def hist_scatter(predictions, y, dataname):
    absdiff = 0
    diffList = []
    numThresh1 = 0
    numThresh2 = 0
    for i in range(len(y)):
        err = abs(predictions[i]-y[i])
        absdiff += err
        diffList.append(err)
        if err <= 0.4:
            numThresh2 += 1
            if err <= 0.2:
                numThresh1 += 1
    meanAbsPercentDiff = (100*absdiff)/(len(y)*sum(y))
    print(f"Less than 0.2: {numThresh1}, Less than 0.4: {numThresh2}")
    fig, axes = plt.subplots()
    counts, bins, patches = axes.hist(np.array(diffList), 20, edgecolor='black')
    bins = np.linspace(0,5,26)
    print(bins)
    axes.set_xticks(bins)
    axes.set_xticklabels(bins, rotation=45)
    axes.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    axes.set_title('Histogram of |Predicted ratio - Actual ratio|')
    axes.set_xlabel('Abs diff between prediction and true')
    axes.set_ylabel('Count')
    # plt.show()
    plt.savefig(f"models/concatModel_{dataname}_hist.png", bbox_inches='tight')

    plt.figure()
    np_y = np.array(y)
    np_predictions = np.array(predictions)
    r2 = r2_score(np_y, np_predictions)
    m, b =  np.polyfit(np_y, np_predictions, 1)
    print(np.polyfit(np_y, np_predictions, 1))
    print(np.polyfit(np_y, np_predictions, 1, full=True))
    plt.scatter(np_y, np_predictions)
    plt.plot(np_y, b + m*np_y, 'r-')
    plt.title(f"{dataname}set Predicted vs True like:avg")
    print(f"MAPE: {meanAbsPercentDiff}, r2: {r2}")
    plt.ylabel('Predicted like:avg')
    plt.xlabel('True like:avg')
    plt.savefig(f"models/concatModel_{dataname}_comparison.png", bbox_inches='tight')

def spearman(predictions, y, dataname):
    avg_spearman = 0
    spearmanList = []
    for i in range(0,len(predictions),10):
        if len(y[i:i+10]) > 1:
            spearmanList.append(spearmanr(predictions[i:i+10], y[i:i+10]).correlation)
    avg_spearman = mean(spearmanList)
    print(f"Average Spearman RCC for 10-image batches: {avg_spearman}")

    print(f"Spearman RCC for whole dataset: {spearmanr(predictions, y).correlation}")

    # print(spearmanList)

def concatenatedModelMain():
    modelfilename = "concatModel"
    if (len(sys.argv) < 2):
        print("Don't forget the flag!")
        sys.exit(0)
    if (sys.argv[1] == "-d"):
        allImgs, featureVectors, allResults = extractFeaturesFromDataset(sys.argv[2])
        imgX_train, imgX_dev, imgX_test, imgY_train, imgY_dev, imgY_test, imgY_train_one_hot, imgY_dev_one_hot = splitAndPrep(allImgs, allResults)
        mdX_train, mdX_dev, mdX_test, mdY_train, mdY_dev, mdY_test, mdY_train_one_hot, mdY_dev_one_hot = splitAndPrep(featureVectors, allResults)
        # assert(imgY_train, mdY_train) # raul-imgclassifier.py:537: SyntaxWarning: assertion is always true, perhaps remove parentheses?
        # imgX_train = imgX_train.reshape(imgX_train.shape[0], TARGET_X, TARGET_Y, 3)
        # imgX_test = imgX_test.reshape(imgX_test.shape[0], TARGET_X, TARGET_Y, 3)
        
        imageModel = trainModel(extractDatasetNameCSV(sys.argv[2]), imgX_train, imgY_train, imgX_dev, imgY_dev, True)
        mdModel = trainMdModel(extractDatasetNameCSV(sys.argv[2]), mdX_train, mdY_train, mdX_dev, mdY_dev, True)

        combinedInput = concatenate([mdModel.output, imageModel.output])

        x = Dense(BUCKET_NUM, activation="relu")(combinedInput)
        x = Dense(1, activation="linear")(x)
        concatModel = Model(inputs=[mdModel.input, imageModel.input], outputs=x)
        concatModel.compile(optimizer=Adam(lr=1e-4, decay=1e-4 / 200), loss='mean_squared_error')
        history  = concatModel.fit([mdX_train, imgX_train], imgY_train,validation_data=([mdX_test, imgX_test], imgY_test), epochs=EPOCH_NUM, batch_size=8)
        # plt.figure()
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.savefig(f"models/concat_{modelfilename}_accuracy.png")

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f"models/concat_{modelfilename}_loss.png")

        concatModel.save(f"models/concat_{modelfilename}.h5")


    elif (sys.argv[1] == "-f"):
        allImgs, allResults, featureVectors = loadFromFile([sys.argv[2], sys.argv[3], sys.argv[4]])

        # allImgs = grayscaleResize(allImgs)
        imgX_train, imgX_dev, imgX_test, imgY_train, imgY_dev, imgY_test, imgY_train_one_hot, imgY_dev_one_hot = splitAndPrep(allImgs, allResults)
        mdX_train, mdX_dev, mdX_test, mdY_train, mdY_dev, mdY_test, mdY_train_one_hot, mdY_dev_one_hot = splitAndPrep(featureVectors, allResults)
        # assert(imgY_train, mdY_train) #raul-imgclassifier.py:537: SyntaxWarning: assertion is always true, perhaps remove parentheses?
        # imgX_train = imgX_train.reshape(imgX_train.shape[0], TARGET_X, TARGET_Y, 3)
        # imgX_test = imgX_test.reshape(imgX_test.shape[0], TARGET_X, TARGET_Y, 3)
        imageModel = trainModel(extractDatasetNameCSV(sys.argv[2]), imgX_train, imgY_train, imgX_dev, imgY_dev, True)
        mdModel = trainMdModel(extractDatasetNameCSV(sys.argv[2]), mdX_train, mdY_train, mdX_dev, mdY_dev, True)

        combinedInput = concatenate([mdModel.output, imageModel.output])

        x = Dense(BUCKET_NUM, activation="relu")(combinedInput)
        x = Dense(1, activation="linear")(x)
        concatModel = Model(inputs=[mdModel.input, imageModel.input], outputs=x)
        concatModel.compile(optimizer=Adam(lr=1e-4, decay=1e-4 / 200), loss='mean_squared_error')
        history = concatModel.fit([mdX_train, imgX_train], imgY_train, validation_data=([mdX_dev, imgX_dev], imgY_dev), epochs=EPOCH_NUM, batch_size=8)
        # plt.figure()
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.savefig(f"models/concat_{modelfilename}_accuracy.png")

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f"models/concat_{modelfilename}_loss.png")

        concatModel.save(f"models/concat_{modelfilename}.h5")
        # Train accuracy
        test_model(concatModel, [mdX_train, imgX_train], imgY_train, 'Train')

        # Val Accuracy
        test_model(concatModel, [mdX_dev, imgX_dev], imgY_dev, 'Dev')
        
        # Test Accuracy
        test_model(concatModel, [mdX_test, imgX_test], imgY_test, 'Test')
    
    elif(sys.argv[1] == "-m"):
        model = load_model(sys.argv[5])
        allImgs, allResults, featureVectors = loadFromFile([sys.argv[2], sys.argv[3], sys.argv[4]])
        imgX_train, imgX_dev, imgX_test, imgY_train, imgY_dev, imgY_test, imgY_train_one_hot, imgY_dev_one_hot = splitAndPrep(allImgs, allResults)
        mdX_train, mdX_dev, mdX_test, mdY_train, mdY_dev, mdY_test, mdY_train_one_hot, mdY_dev_one_hot = splitAndPrep(featureVectors, allResults)
        test_model(model, [mdX_train, imgX_train], imgY_train, 'Train')
        test_model(model, [mdX_dev, imgX_dev], imgY_dev, 'Dev')
        test_model(model, [mdX_test, imgX_test], imgY_test, 'Test')
    else:
        print("Invalid flag, mate!")
        sys.exit(0)  

    

if __name__ == "__main__":
    concatenatedModelMain()