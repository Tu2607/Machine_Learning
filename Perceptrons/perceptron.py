import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import struct

class Perceptron(object):
    def __init__(self, trainingSetFile, testSetFile, scaleValue, outputCount):
        self.trainingSetData, self.trainingSetTarget, self.testSetData, self.testSetTarget = \
            self.csvToNumpy(trainingSetFile, testSetFile, scaleValue)
        self.inputCount = np.size(self.trainingSetData, 1)
        self.outputCount = outputCount
        self.weights = np.random.uniform(-0.05, 0.05, (self.inputCount, outputCount))  # assign +-0.5 randomly to the weight array
        self.finalOutput = np.zeros(10000)

    # Keep data outside of Perceptron class to make it scalable
    def csvToNumpy(self, trainingSetFile, testSetFile, scaleValue):
        # Training Set initialization
        tempTraining = pd.read_csv(trainingSetFile, header=None)  # using pandas since loadtxt takes a long time
        trainingSetTargets = tempTraining.to_numpy()[:, 0]
        trainingSetData = np.insert(((tempTraining.to_numpy()) / scaleValue), 0, 1, axis=1)  # scale data then add 1 to each row

        # Test Set initialization
        tempTest = pd.read_csv(testSetFile, header=None)
        testSetTargets = tempTest.to_numpy()[:, 0]
        testSetData = np.insert(((tempTest.to_numpy()) / scaleValue), 0, 1, axis=1)
        return trainingSetData, trainingSetTargets, testSetData, testSetTargets

    def runEpochs(self, learningRate, epochCount):
        trainingAccuracyArray = []
        trainingAccuracy = 0
        testAccuracyArray = []

        for epoch in range(epochCount):
            print('epoch ' + str(epoch))
            lastAccuracy = trainingAccuracy
            correct = 0
            correctTest = 0
            self.finalOutput.fill(0)

            for i in range(np.size(self.trainingSetData, 0)):
                correct += self.train(learningRate, i)

                if i < np.size(self.testSetData, 0):
                    correctTest += self.test(i)

            #Calculate accuracy of one epoch
            testAccuracy = (correctTest / np.size(self.testSetData, 0)) * 100
            testAccuracyArray.append(testAccuracy)
            trainingAccuracy = (correct / np.size(self.trainingSetData, 0)) * 100
            trainingAccuracyArray.append(trainingAccuracy)
            if(abs(trainingAccuracy - lastAccuracy) < 0.01):
                break

        return np.array(trainingAccuracyArray), np.array(testAccuracyArray)

    def train(self, learningRate, i):
        t = self.computeTarget(i)
        y, prediction = self.computeOutputAndPredict(i)
        if prediction != self.trainingSetTarget[i]:
            for j in range(self.inputCount):
                error = t - y
                self.weights[j] += learningRate * error * self.trainingSetData[i, j]
            return 0
        else:
            return 1

    def test(self, i):
        #Sending row ith of the testSetData to dot with weights to find the output
        #i.e dotting (0,785) with (785,10) but with python notation, it is (785,) with (785,10)
        output = np.dot(self.testSetData[i, :], self.weights) #This output dimesion should be (0,10)
        prediction = np.argmax(output) #Return the label i.e number with that largest percentage
        self.finalOutput[i] = prediction
        if(prediction == self.testSetTarget[i]):
            return 1
        else:
            return 0

    def computeTarget(self, i):
        t = np.zeros(self.outputCount)
        t[self.trainingSetTarget[i]] = 1  # Set the index matching target value to 1
        return t

    def computeOutputAndPredict(self, i):
        output = np.dot(self.trainingSetData[i, :], self.weights)
        y = np.where(output >= 0, 1, 0)  # Encoded output
        prediction = np.argmax(output)
        return y, prediction

def plotAccuracy(trainingAccuracy, testAccuracy, learningRate):
    # Plot accuracy
    plt.suptitle('Learning Rate: ' + str(learningRate))
    train, = plt.plot(trainingAccuracy, label='Training Set Accuracy')
    test, = plt.plot(testAccuracy, label='Test Set Accuracy')
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

epochCount = 70
a = Perceptron('mnist_train.csv', 'mnist_test.csv', 255, 10)
trainingAccuracy, testAccuracy = a.runEpochs(0.001, epochCount)
print(confusion_matrix(a.testSetTarget, np.array(a.finalOutput)))
plotAccuracy(trainingAccuracy, testAccuracy, 0.001)
