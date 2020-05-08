import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

class Bayes(object):
    def __init__(self, filename):
        self.trainData, self.testData, self.trainTarget, self.testTarget = self.splitData(filename)

    def splitData(self,filename):
        database = np.loadtxt(filename, delimiter= ",")
        X = database[:, :-1]
        y = database[:, -1]
        return train_test_split(X, y, test_size = 0.5)

    #Might be using a tuple here
    def mean_std(self):
        features = self.trainData.shape[1]
        m = np.ones((2, features))
        std = np.ones((2, features))

        spam = []  #2d array of all the spam mail
        nonspam =  []   #2d array of all the nonspam email

        for i in range(self.trainData.shape[0]):
            if self.trainTarget[i] == 0:
                nonspam.append(self.trainData[i])
            else:
                spam.append(self.trainData[i])

        #To make sure that it is an array
        spam = np.asarray(spam)        
        nonspam = np.asarray(nonspam)


        for i in range(features):
            m[0,i] = np.mean(nonspam.T[i]) #Row 0 is the non spam
            m[1,i] = np.mean(spam.T[i]) #Row 1 is the spam
            std[0,i] = np.std(nonspam.T[i])
            std[1,i] = np.std(spam.T[i])

        #Special case checking
        for j in range(features):
            if std[0,j] == 0:
                std[0,j] = 0.0001
            if std[1,j] == 0:
                std[1,j] = 0.0001

        return m,std

    def probModel(self):
        trainSpam_percent = (np.count_nonzero(self.trainTarget) / len(self.trainTarget)) * 100
        trainNonspam_percent = 100 - trainSpam_percent

        train_mean_tuple, train_std_tuple = self.mean_std()
        return trainSpam_percent, trainNonspam_percent, train_mean_tuple, train_std_tuple

    #Reminder that row 0 is non spam, and row 1 is spam
    #Work in progress, not sure of the input beside mean and std
    def PFC(self, mean, std):
        rows = self.testData.shape[0]
        P = []
        for i in range(2):
            product = 1
            for j in range(rows):
                exponent = np.exp(-1 * ((self.testData[j] - mean[i])**2) / (2 * std[i]**2))
                product = product * (1 / (np.sqrt(2*np.pi) * std[i]) * exponent)
            P.append(product)
        return np.asarray(P)

a = Bayes("spambase.data")
spam_percent, nspam_percent, mean, std = a.probModel()
print(mean.shape)
print(std.shape)
b = a.PFC(mean,std)
print(b.shape)