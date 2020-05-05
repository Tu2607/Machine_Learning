import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

class Bayes(object):
    def __init__(self, filename):
        self.trainData, self.testData, self.trainTarget, self.testTarget = self.splitData(filename)
        self.TrainingpriorProb, self.trainingmeanSD = self.probModel()

    def splitData(self,filename):
        database = np.loadtxt(filename, delimiter= ",")
        train, test = train_test_split(database, test_size = 0.50)
        trainData, trainTarget = train[:, :-1], train[:, -1]
        testData, testTarget = test[:, :-1], test[:, -1]
        return trainData, testData, trainTarget, testTarget

    #Return a tuple of spam% and nonspam% for each class of training data
    def priorProb(self,array):
        spam = np.count_nonzero(array) #Count all the non zeroes 
        spam_percent = (spam / len(array)) * 100
        nonspam_percent = 100 - spam_percent
        return spam_percent, nonspam_percent

    #Return the mean and standard deviation of each class
    def mean_SD(self,array):
        std = np.std(array)
        if(std == 0):
            std == 0.0001 # hard code value
            return np.mean(array), np.std(array)
        else:
            return np.mean(array), std

    #The last column of the spambase data denote whether the email is spam(1) or not spam(0)
    #That last column could be the target that we use.
    #Check the document on the database online
    def probModel(self):
        prior = []
        mean_SD = []
        for i in range(2300):
            spam, nonspam = self.priorProb(self.trainData[i,:])
            prior.append((spam,nonspam)) 
            mean_SD.append(self.mean_SD(self.trainData[i,:]))
        return np.asarray(prior), np.asarray(mean_SD)
    
    
a = Bayes("spambase.data")
print(a.trainData.shape)
print(a.testData.shape)