import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

class Bayes(object):
    def __init__(self, filename):
        self.trainData, self.testData = self.splitData(filename)
        self.TrainingpriorProb, self.trainingmeanSD = self.probModel()

    def splitData(self,filename):
        data = np.loadtxt(filename, delimiter= ",")
        trainData = data[:2300,:] #Selecting 2300 rows for the trainData set, 1st half of the data
        testData = data[2301:,:] #Selecting 2300 rows for the testData set, 2nd half of the data
        return trainData, testData

    #Return a tuple of spam% and nonspam% for each class of training data
    def priorProb(self,array):
        spam = np.count_nonzero(array) #Count all the non zeroes 
        spam_percent = (spam / 57) * 100
        nonspam_percent = 100 - spam_percent
        return spam_percent, nonspam_percent

    #Return the mean and standard deviation of each class
    def mean_SD(self,array):
        return np.mean(array), np.std(array)

    #The last column of the spambase data denote whether the email is spam(1) or not spam(0)
    def probModel(self):
        prior = []
        mean_SD = []
        for i in range(2300):
            spam, nonspam = self.priorProb(self.trainData[i,:])
            prior.append((spam,nonspam)) 
            mean_SD.append(self.mean_SD(self.trainData[i,:]))
        return np.asarray(prior), np.asarray(mean_SD)
    
    
a = Bayes("spambase.data")

