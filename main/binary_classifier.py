#!/usr/bin/python

import random
from collections  import defaultdict
import math
import sys
import csv
import numpy as np

############################################################
# Binary classifier
############################################################

def main():
    trainData, testData = extractFeaturesFromDataset(sys.argv[1])
    print(trainData)
    numIters = 10000
    stepSz = 0.01
    learnPredictor(trainData, testData, numIters, stepSz)


############################################################
# Feature extraction
def extractFeaturesFromDataset(filename):
    with open(filename) as f:
        listFeatureVectorsWithResult  = []
        for row in csv.DictReader(f):
            featureVector = defaultdict(float)
            for key in row: #  each row is a dict
                if (key == "timestamp"): 
                    continue
                # this should fail all the time we have a string as the value feature
                # probably bad style but  python has no better way to check if 
                # a string contains a float or not
                try:
                    val = float(row[key])
                    featureVector[key] = val
                except:
                    continue
            label = 1 if float(row["likeRatio"]) > 0.05 else -1
            listFeatureVectorsWithResult.append((featureVector, label))
        limitLen = int(len(listFeatureVectorsWithResult)/3)
        trainData = listFeatureVectorsWithResult[:limitLen]
        testData = listFeatureVectorsWithResult[limitLen:]
        return (trainData, testData)


def learnPredictor(trainExamples, testExamples, numIters, eta):
    '''
    Given |trainExamples| and |testExamples|, a |featureExtractor|, 
    and the number of iterations to train |numIters|, the step size |eta|, 
    this function returns the weight vector (sparse feature vector) learned.

    We'll be using stochastic gradient descent for this implementation.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = defaultdict(lambda: 3)  # feature => weight

    for i in range(numIters):
        for example in trainExamples: # trainExamples is a a list of feature vectors
            result = example[1]
            featureVector = example[0]
            # take step
            # print(f"weights {weights}\nfeaturevector: {featureVector}\nresult: {result}")
            gradient = hingeLossGradient(weights, featureVector, result)
            # print(f"gradient {gradient}")
            if gradient  != 0:
                increment(weights,-eta,gradient)

        def predictor(featureVectorInput): 
            if featureVectorInput == defaultdict(float):
                return True
            # feature_vector = extractWordFeatures(input_text)
            return np.sign(dotProduct(weights, featureVectorInput))
        
        print("evaluatingPredictor with trainExamples: " + str(evaluatePredictor(trainExamples, predictor)))
        print("evaluatingPredictor with testExamples: " + str(evaluatePredictor(testExamples, predictor)))
    outputWeights(weights, "weights/weights.txt")
    return weights

def hingeLossGradient(weights, features, result):
    if dotProduct(weights, features)*result > 1:
        return 0
    else:
        gradient = features
        for f in features:
            gradient[f] *= result*(-1)
        return  gradient 

##########################################
#           MOVE TO UTIL FILE            #
##########################################
def evaluatePredictor(examples, predictor):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassiied examples.
    '''
    error = 0
    for ex in examples:
        if predictor(ex[0]) != ex[1]:
            error += 1
    return 1.0 * error / len(examples)

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in list(d2.items()))

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in list(d2.items()):
        d1[f] = d1.get(f, 0) + v * scale

def outputWeights(weights, path):
    print("%d weights" % len(weights))
    out = open(path, 'w')
    for f, v in sorted(list(weights.items()), key=lambda f_v : -f_v[1]):
        print('\t'.join([f, str(v)]), file=out)
    out.close()
#########################################

# MAIN
if __name__ == "__main__":
    main()