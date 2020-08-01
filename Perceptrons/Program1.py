#Tu Vu
#Program 1 MLP
#CS445
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class MLP:
    def __init__(self, trainingFile, testFile, hiddenLayerCount, momentum):
        self.train_data, self.train_target, self.test_data, self.test_target = self.readFile(trainingFile, testFile)
        self.inputCount = np.size(self.train_data, 1)  #Should be 785 with the bias added
        self.outputCount = 10
        self.hiddenLayerCount = hiddenLayerCount 
        self.momentum = momentum

        #The weight for input to hidden (785,hiddenLayerCount)
        self.weight1 = np.random.uniform(-0.05,0.05, (self.inputCount, self.hiddenLayerCount)) 
        # The weight for hidden layer to output layer (hidden + 1,10)
        self.weight2 = np.random.uniform(-0.05,0.05, (self.hiddenLayerCount + 1,self.outputCount)) 

        #For storing old deltas
        self.delta1 = np.zeros(np.shape(self.weight1))
        self.delta2 = np.zeros(np.shape(self.weight2))

        self.finalOutput = np.zeros(10000) 

    def readFile(self, trainingFile, testFile):
        #Train Set
        temp_train = pd.read_csv(trainingFile, header=None)

        #Comment the line below out for reading full 60000 samples of training data
        temp_train = temp_train.sample(frac= 0.5) #Randomly select a quarter of data 

        trainingTarget = temp_train.to_numpy()[:, 0]  #This is geting the correct label for each image
        trainingData = np.delete(temp_train.to_numpy(),0, axis = 1) #Remove the first column which is the label
        trainingData = np.insert((trainingData / 255), 0, 1, axis = 1)

        #Test set
        temp_test = pd.read_csv(testFile, header = None)
        testTarget = temp_test.to_numpy()[:,0] #Getting the correct label for reach row of input
        testData = np.delete(temp_test.to_numpy(), 0, axis = 1) # Again, removing the first column which is the label
        testData = np.insert((testData / 255), 0, 1, axis = 1)# Adding the bias and scaled the test input

        return trainingData, trainingTarget, testData, testTarget

    def sigmoidActivation(self, x):
        return (1 / (1 + np.exp(-1 * x))) 

    def Epochs(self, epochCounts, learningRate):
        accuracyTrainSet = []
        trainAccuracy = 0
        accuracyTestSet = []

        for epoch in range(epochCounts):
            print('epoch ' + str(epoch))
            trainCorrect = 0
            testCorrect = 0
            self.finalOutput.fill(0) 

            #Sending each 785 inputs vector in xxxxx amout of total images
            for index in range(np.size(self.train_data,0)):
                trainCorrect += self.train(index, learningRate, epoch)

                if index < np.size(self.test_data,0):
                    testCorrect += self.test(index)
            
            testAccuracy = (testCorrect / np.size(self.test_data, 0)) * 100
            accuracyTestSet.append(testAccuracy)
            trainAccuracy = (trainCorrect / np.size(self.train_data,0)) * 100 
            accuracyTrainSet.append(trainAccuracy)

        return np.array(accuracyTrainSet), np.array(accuracyTestSet)

    #Feed forward then backprop immediately
    def train(self,index, learningRate, currEpoch):
        target = self.getTarget(index) 
        output, hiddenLayer, prediction = self.feedForward(index) 
        self.backprop(target,output,hiddenLayer,learningRate, index, currEpoch)
        if prediction != self.train_target[index]:
            return 0
        else:
            return 1

    def getTarget(self, index):
        target = np.zeros(self.outputCount)
        target[self.train_target[index]] = 0.9  # Setting the target to 0.9
        for i in range(self.outputCount):
            if target[i] != 0.9:
                target[i] = 0.1
        return target

    #Feeding forward, going from input to hidden then hidden to ouput
    def feedForward(self, index):
        #The math : (0,785) dot (785, hiddenLayerCount) = (0, hiddenLayerCount)
        hiddenLayer = np.dot(self.train_data[index,:], self.weight1) 
        hiddenLayer = self.sigmoidActivation(hiddenLayer) 
        hiddenLayer = np.insert(hiddenLayer, 0, 1) #Insert the bias to the first column

        #The math is  = (0, hiddenLayerCount + 1) dot (hiddenLayerCount, outputCount) = (0, 10)
        output = self.sigmoidActivation(np.dot(hiddenLayer,self.weight2))

        prediction = np.argmax(output)
        return output, hiddenLayer, prediction

    def backprop(self, target, output, hiddenLayer, learningRate, index, currEpoch):
        updateWeight1 = np.zeros(np.shape(self.weight1)) #temp array for delta
        updateWeight2 = np.zeros(np.shape(self.weight2)) #temp array for delta

        #Matrix (0,10)
        outputWeightUpdate = output * (1 - output) * (target - output)

        #Updating the weight from hidden to output
        #The dimension of updateWeight 2 is (hiddenLayerCount + 1, outputCount)
        updateWeight2 = np.outer(hiddenLayer,outputWeightUpdate)

        if(currEpoch == 0):
            updateWeight2 = (learningRate*updateWeight2) + (self.momentum* np.zeros(np.shape(self.weight2)))
        else:
            updateWeight2 = (learningRate*updateWeight2) + (self.momentum* self.delta2)

        #Matrix for updating weight of input to hidden
        #The math: (hiddenLayerCount + 1,0) * (1 - (hiddenLayerCount + 1,0)) * np.dot((0,10),(10,hiddenLayerCount + 1))
        #The resulting dimension is: (hiddenLayerCount + 1, 0)
        hiddenWeightUpdate =  hiddenLayer * (1 - hiddenLayer) * np.dot(outputWeightUpdate,np.transpose(self.weight2)) 

        #Dimension of weight1 is (785,hiddenLayerCount)
        #Train_Data dimension is (785,0) and dimension of hiddenWeightUpdate is (hiddenLayerCount,0)
        #Remember that the weight1 has less row than hiddenWeightUpdate due to adding in the bias
        #The output dimesion is (785, hiddenLayerCount)
        updateWeight1 = np.outer(self.train_data[index,:], hiddenWeightUpdate[1:])
        
        if(currEpoch == 0):
            updateWeight1 = (learningRate*updateWeight1) + (self.momentum* np.zeros(np.shape(self.weight1)))
        else:
            updateWeight1 = (learningRate*updateWeight1) + (self.momentum*self.delta1)

        #Storing the deltas for next set of inputs
        self.delta1 = updateWeight1
        self.delta2 = updateWeight2

        #Updating the weight
        self.weight1 += updateWeight1
        self.weight2 += updateWeight2

    def test(self,index):
        hiddenLayer = np.dot(self.test_data[index,:], self.weight1) 
        hiddenLayer = self.sigmoidActivation(hiddenLayer)
        hiddenLayer = np.insert(hiddenLayer, 0, 1) 
        output = self.sigmoidActivation(np.dot(hiddenLayer,self.weight2))
        prediction = np.argmax(output)

        self.finalOutput[index] = prediction
        if(prediction == self.test_target[index]):
            return 1
        else:
            return 0

#Argument("train file", "test file", hidden input count, momentum)
a = MLP("mnist_train.csv","mnist_test.csv", 100, 0.9)

print(a.train_data.shape)
print(a.test_data.shape)
trainAccuracy, testAccuracy = a.Epochs(50,0.1)

plt.suptitle('Training with ' + str(30000) + ' samples')
train = plt.plot(trainAccuracy, label = 'Training set accuracy')
test = plt.plot(testAccuracy, label = 'Test set accuracy')
plt.ylabel("Accuracy (%)")
plt.xlabel("Epoch")
plt.legend()
plt.show()
print(confusion_matrix(a.test_target, np.array(a.finalOutput)))
