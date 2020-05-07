import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

class Bayes(object):
    def __init__(self, filename):
        self.trainData, self.testData, self.trainTarget, self.testTarget = self.splitData(filename)
        self.trainSpam, self.trainNonspam, self.trainMeanArray, self.trainStdArray = self.probModel()

    def splitData(self,filename):
        database = np.loadtxt(filename, delimiter= ",")
        X = database[:, :-1]
        y = database[:, -1]
        return train_test_split(X, y, test_size = 0.5)

    #The last column of the spambase data denote whether the email is spam(1) or not spam(0)
    #That last column could be the target that we use.
    #Check the document on the database online
    def probModel(self):
        trainSpam_percent = (np.count_nonzero(self.trainTarget) / len(self.trainTarget)) * 100
        trainNonspam_percent = 100 - trainSpam_percent

        train_mean = [] 
        train_std = []
        for i in range(self.trainData.shape[1]):  #Go through each feature (i.e column)
           train_mean.append(np.mean(self.trainData[:,i])) 
           train_std.append(np.std(self.trainData[:,i]))
        
        for index in range(len(train_std)):
            if train_std[index] == 0:
                train_std[index] = 0.0001
        
        return trainSpam_percent, trainNonspam_percent, np.asarray(train_mean), np.asarray(train_std)
    
    def NaiveBayes(self):
       return 0 

    
a = Bayes("spambase.data")
print(a.trainMeanArray.shape)