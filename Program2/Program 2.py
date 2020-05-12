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
        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.5)
        return X_train, X_test, y_train, y_test

    def mean_std(self):
        features = self.trainData.shape[1]
        m = np.ones((2, features)) #2 rows, one for spam(1) class, and one for nonspam(0) class
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
        for i in range(0,features):
            if np.count_nonzero(nonspam.T[i]) == 0:
                print("We got a zero column in non spam")
            if np.count_nonzero(spam.T[i]) == 0:
                print("We got a zero column in spam")
            m[0,i] = np.nanmean(nonspam.T[i]) #Row 0 is the non spam
            m[1,i] = np.nanmean(spam.T[i]) #Row 1 is the spam
            std[0,i] = np.std(nonspam.T[i]) #Row 0 is non spam
            std[1,i] = np.std(spam.T[i]) #Row 1 is spam

        #Special case checking for std. Get it??
        for j in range(features):
            if std[0,j] == 0:
                std[0,j] = 0.0001
            if std[1,j] == 0:
                std[1,j] = 0.0001

        print("Minimum value in mean rows:")
        print(min(m[0]))
        print(min(m[1])) 

        return m,std

    def probModel(self):
        trainSpam_percent = (np.count_nonzero(self.trainTarget) / len(self.trainTarget))
        trainNonspam_percent = 1 - trainSpam_percent

        train_mean_tuple, train_std_tuple = self.mean_std()

        return trainSpam_percent, trainNonspam_percent, train_mean_tuple, train_std_tuple

    #Reminder that row 0 is non spam, and row 1 is spam
    #This function return the following:
    #1. The probability of x given that it is not spam
    #2. The probability of x given that it is spam
    def posteriorProb(self, x, mean, std, spam, nspam):
        pProb = np.ones(2)
        
        for i in range(2):
            p = 0

            if i == 0:
                p += np.log(nspam)
            elif i == 1:
                p += np.log(spam)

            for j in range(len(x)):
                #If x[j] is 0, and then mean[i][j] is also 0, we got issue
                a = (x[j] - mean[i][j])**2 # <-- This is a problem. Apparently x[j] - mean[i][j] is 0. Relates to the mean_std
                b = 2 * ((std[i][j])**2)
                exponent = np.exp(-(a/b))
                N = 1 / (np.sqrt(2*np.pi) * std[i][j])
                p += np.log(N * exponent)

            pProb[i] = p
        return pProb

    #Predict the class based on Naive Bayes Algorithm
    def predict(self):
        spam_percent, nspam_percent, mean, std = self.probModel()

        prediction = []
        for rows in range(self.testData.shape[0]):
            #return the probability of each class (this is a tuple) per row 
            p = self.posteriorProb(self.testData[rows], mean, std, spam_percent, nspam_percent) 
            prediction.append(np.argmax(p)) #Append the index of the higher probability tuple into a list
        
        return np.asarray(prediction)

    def accuracy(self, prediction):
        correct = 0
        for i in range(len(self.testTarget)):
            if prediction[i] == self.testTarget[i]:
                correct += 1
        
        accuracy = correct / len(self.testTarget)
        return accuracy

    #For comparison
    def GB(self):
        gnb = GaussianNB()
        gnb.fit(self.trainData, self.trainTarget)
        y_pred = gnb.predict(self.testData)
        return metrics.accuracy_score(self.testTarget, y_pred)


a = Bayes("spambase.data")
b = a.predict()
c = a.accuracy(b)
d = a.GB()

print(confusion_matrix(a.testTarget, b))

print(c)
print(d)