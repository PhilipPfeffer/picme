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
        # general for both test and train
        label_names = ["instagram worthy", "shit"]
        feature_names = list(self.trainData[0][0].keys())

        # train
        train_labels = [ex[1] for ex in self.trainData]
        train = [list(ex[0].values()) for ex in self.trainData]

        # test
        test_labels =  [ex[1] for ex in self.testData]
        test = [list(ex[0].values()) for ex in self.testData]

        # Initialize our classifier
        gnb = GaussianNB()

        # Train our classifier
        model = gnb.fit(train, train_labels)

        # Make predictions
        preds = gnb.predict(test)
        print(preds)
        print(f"weights: {gnb.get_params()}")

        # Evaluate accuracy
        print("GaussianNB Prediction Accuracy: " + str(accuracy_score(test_labels, preds)))

    def runWithoutFeature(self, delIndex):
        # general for both test and train
        label_names = ["instagram worthy", "shit"]

        feature_names = list(self.trainData[0][0].keys())
        print(f"deleting feature: {feature_names[delIndex]}")
        del feature_names[delIndex]

        # train
        train_labels = [ex[1] for ex in self.trainData]
        train = []
        for ex in self.trainData:
            train.append(list(ex[0].values())
            for t in train:
                del t[delIndex]
                print()

        # test
        test_labels =  [ex[1] for ex in self.testData]
        test = [list(ex[0].values()) for ex in self.testData]
        for tst in test:
            del tst[delIndex]

        # Initialize our classifier
        gnb = GaussianNB()

        # Train our classifier
        model = gnb.fit(train, train_labels)

        # Make predictions
        preds = gnb.predict(test)
        print(preds)

        # Evaluate accuracy
        print("GaussianNB Prediction Accuracy: " + str(accuracy_score(test_labels, preds)))

if __name__ == "__main__":
    sk = SKLearnPredictor()
    sk.loadDataset()
    sk.run()