import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

class Bayes(object):
    def __init__(self, filename):
        self.trainData, self.testData, self.trainTarget, self.testTarget = self.splitData(filename)

    def splitData(self,filename):
        database = np.loadtxt(filename, delimiter= ",")
        X = database[:, :-1]
        y = database[:, -1]
        return train_test_split(X, y, test_size = 0.5)

    def mean_std(self):
        features = self.trainData.shape[1]
        m = np.ones((2, features))
        std = np.ones((2, features))

        spam = []  #2d array of all the spam mail
        nonspam =  []   #2d array of all the nonspam email

        #Going throught the training set to append the row that is deemed
        #non spam to the non spam list and the same for the spam list
        for i in range(self.trainData.shape[0]):
            if self.trainTarget[i] == 0:
                nonspam.append(self.trainData[i])
            else:
                spam.append(self.trainData[i])

        #To make sure that it is an array
        spam = np.asarray(spam)        
        nonspam = np.asarray(nonspam)

        #Find the mean and std of each column within the spam and non spam array
        #For all 57 features
        for i in range(features):
            m[0,i] = np.mean(nonspam.T[i]) #Row 0 is the non spam
            m[1,i] = np.mean(spam.T[i]) #Row 1 is the spam
            std[0,i] = np.std(nonspam.T[i]) #Row 0 is non spam
            std[1,i] = np.std(spam.T[i]) #Row 1 is spam

        #Special case checking for std. Get it??
        for j in range(features):
            if std[0,j] == 0:
                std[0,j] = 0.0001
            if std[1,j] == 0:
                std[1,j] = 0.0001

        return m.astype(np.float64),std.astype(np.float64)

    def probModel(self):
        trainSpam_percent = (np.count_nonzero(self.trainTarget) / len(self.trainTarget)) * 100
        trainNonspam_percent = 100 - trainSpam_percent

        train_mean_tuple, train_std_tuple = self.mean_std()
        return trainSpam_percent, trainNonspam_percent, train_mean_tuple, train_std_tuple

    #Reminder that row 0 is non spam, and row 1 is spam
    #Work in progress, not sure of the input beside mean and std
    #Mean and Std are 2d arrays of dimension (2,57)
    def posteriorProb(self, x, mean, std, spam, nspam):
        pProb = np.ones(2)
        for i in range(2):
            p = 0
            for j in range(len(x)):
                exponent = np.exp(-(((x[j] - (mean[i][j])**2)) / (2 * std[i][j])))
                p += np.log((1 / (np.sqrt(2 * np.pi) * std[i][j])) * exponent)
            if i == 0:
                p += spam
            elif i == 1:
                p += nspam
            pProb[i] = p
        
        return pProb

    def predict(self):
        spam_percent, nspam_percent, mean, std = self.probModel()

        prediction = []
        for rows in range(self.testData.shape[0]):
            p = self.posteriorProb(self.testData[rows], mean, std, spam_percent, nspam_percent)
            prediction.append(np.argmax(p))
        
        return np.asarray(prediction)

    def accuracy(self, prediction):
        correct = 0
        for i in range(len(self.testTarget)):
            if prediction[i] == self.testTarget[i]:
                correct += 1
        
        accuracy = correct / len(self.testTarget) * 100
        return accuracy

a = Bayes("spambase.data")
spam,nospam,mean,std = a.probModel()
#b = a.predict()
#c = a.accuracy(b)
print(min(std[0]))
print(min(std[1]))
print(min(mean[0]))
print(min(mean[0]))