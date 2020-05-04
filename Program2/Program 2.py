import numpy as np
from sklearn.metrics import confusion_matrix

class Bayes(object):
    def __init__(self, filename):
        self.trainData, self.testData = self.splitData(filename)
    
    def splitData(self,filename):
        data = np.loadtxt(filename, delimiter= ",")
        trainData = data[:2300,:] #This is hard coding
        testData = data[2301:,:]
        return trainData, testData

    #Template for later
    def probModel(self):
        return 0    

a = Bayes("spambase.data")
print(a.trainData.shape)
print(a.testData.shape)
