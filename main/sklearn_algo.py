from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from binary_classifier import *

class SKLearnPredictor:
    def __init__(self):
        self.trainData = []
        self.testData = []

    def loadDataset(self):
        # Load dataset
        self.trainData, self.testData = extractFeaturesFromDataset('datasets/thegreatdataset.csv')

    def run(self):
        label_names = ["instagram worthy", "shit"]
        labels = [ex[0] for ex in trainData]
        feature_names = list(trainData[0][0].keys())
        # Initialize our classifier
        gnb = GaussianNB()
        # Train our classifier
        model = gnb.fit(trainData, train_labels)

        # Make predictions
        preds = gnb.predict(testData)
        print(preds)

# Evaluate accuracy
print(accuracy_score(test_labels, preds))