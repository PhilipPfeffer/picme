#!/usr/bin/python
import random
import numpy as np
import math
import sys
from collections import defaultdict
import csv


def main():
    #stepSize =  0.000001
    #intercept = some sht
    #extractShortCodesFromCsv() to create features
    #target set (partition)
    #weights = lr(features, )
    #test 
    features = parseCsv(sys.argv[1])
    
    print(logistic_regression(features, 0.35, 10, 0.1))

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def parseCsv(filename):
    with open(filename) as f:
        myCsv = []
        for row in csv.DictReader(f):
            row["likeRatio"] = 1 if float(row["likeRatio"]) > 0.35 else 0
            myCsv.append(row)
        limitLen = int(len(myCsv)/3)
        trainCsv = myCsv[:limitLen]
        return trainCsv

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

############################################################
# Logistic Regression
############################################################

#def lr(features, target, steps, learningRate, intercept or not):

    #weights = np.zeros with the features

    #for step in range(steps):
        #dot features and weights
        #predict = sigmoid(scores)
        #error = target - predict (this we will do with if above 35% of ratio)
        #gradient will be dot of features and error || and use stochastic gradient descent
        #update weights by multiplying learning rate and gradient

#return weights

def multiplyByScalar(d1, scalar):
    for key in d1:
        d1[key] *= scalar
    return d1


def incrementByScalar(d1, scale):
    for key in d1:
        d1[key] += scalar
    return d1

def logistic_regression(features, target, num_steps, learning_rate):
    weights = defaultdict(float)
    
    for step in range(num_steps):
        print("------------------------------")
        print(f"weights: {weights}")
        scores = dotProduct(features, weights)
        print(f"scores: {scores}")
        predictions = sigmoid(scores)
        print(f"predictions: {predictions}")

        # Update weights with log likelihood gradient
        output_error_signal = target - predictions
        print(f"error: {output_error_signal}")
        gradient = dotProduct(features, defaultdict(lambda:output_error_signal))
        print(f"gradient: {gradient}")
        weights = incrementByScalar(weights, learning_rate * gradient)

        # Print log-likelihood every so often
        # if step % 10000 == 0:
        #     print (log_likelihood(features, target, weights))
        
    return weights

if __name__ == "__main__":
    main()

