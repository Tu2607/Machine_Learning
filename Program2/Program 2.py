#Tu Vu
#A Naive Bayes Classifier

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

        #Find the mean and std of each feature within the spam and non spam array
        for i in range(0,features):
            m[0,i] = np.nanmean(nonspam.T[i]) #Row 0 is the non spam
            m[1,i] = np.nanmean(spam.T[i]) #Row 1 is the spam
            std[0,i] = np.std(nonspam.T[i]) #Row 0 is non spam
            std[1,i] = np.std(spam.T[i]) #Row 1 is spam

        #Special case checking for std...
        for j in range(features):
            if std[0,j] == 0:
                std[0,j] = 0.0001
            if std[1,j] == 0:
                std[1,j] = 0.0001
        
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
                a = (x[j] - mean[i][j])**2 # 
                b = 2 * ((std[i][j])**2)
                exponent = np.exp(-1 *(a/b))
                N = 1 / (np.sqrt(2*np.pi) * std[i][j])
                probability = N * exponent
                if probability == 0: # <-- In case either N or exponent is 0
                    probability = 10**-320
                p += np.log(probability)

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

    def precision_recall(self, prediction):
        tp, tn, fp, fn = 0,0,0,0
        for i in range(len(self.testTarget)):
            if prediction[i]  == 1 and self.testTarget[i] == 1:
                tp += 1
            elif prediction[i] == 0 and self.testTarget[i] == 0:
                tn += 1
            elif prediction[i] == 1 and self.testTarget[i] == 0:
                fp += 1
            else:
                fn += 1
        return tp, tn, fp, fn

    #For comparison
    def GB(self):
        gnb = GaussianNB()
        gnb.fit(self.trainData, self.trainTarget)
        y_pred = gnb.predict(self.testData)
        return metrics.accuracy_score(self.testTarget, y_pred)


NaiveBayes = Bayes("spambase.data")
prediction = NaiveBayes.predict()
Accuracy = NaiveBayes.accuracy(prediction)
tp, tn, fp, fn = NaiveBayes.precision_recall(prediction)
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("Confusion Matrix:")
print(confusion_matrix(NaiveBayes.testTarget, prediction))

print("Accuracy: ")
print(Accuracy)

print("Accuracy from sklearn library for comparison:")
print(NaiveBayes.GB())

print("Precision: ")
print(precision)

print("Recall: ")
print(recall)