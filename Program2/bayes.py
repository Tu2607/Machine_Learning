# Angie McGraw
# CS 445 Programming Assignment 2
# This program uses Gaussian Naive Bayes and Logistic Regression 
# to classify the Spambase data from the UCL ML repository. 

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# part I: classification with Naive Bayes
# 1. create training and test set
# split the data into a training and test set; each has 2300 instances, 
# 40% spam and 60% not-spam
class Naive_Bayes(object):
  def __init__(self, file):
    self.train_data, self.test_data, self.target_train, self.target_test = self.split(file)
     
  def split(self, file):
    data = np.loadtxt(file, delimiter = ",")
    X, target = data[:, :-1], data[:, -1]
    train_data, test_data, target_train, target_test = train_test_split(X, target, test_size = 0.50)
    return train_data, test_data, target_train, target_test

  def probabilistic_model(self): 
    
    # PROBABILITY IN THE TRAINING SET
    # find the probability of the spam data in the training set 
    for i in range(len(target_train)): 
      if(target_train[i] == 1): 
        train_spam_data += 1
  
    # the sum of the probability of the spam and not-spam data in the training set is 1  
    probability_train_spam = train_spam_data / (len(target_train))
    probability_train_not_spam = 1 - probabilty_train_spam

    # PROBABILITY IN THE TEST SET
    # find the probability of the spam data in the test set
    for i in range(len(target_test)): 
      if(target_test[i] == 1): 
        test_spam_data += 1
  
    # the sum of the probability of the spam and not-spam data in the test set is 1  
    probability_test_spam = test_spam_data / (len(target_test))
    probability_test_not_spam = 1 - probability_test_spam  

    # MEAN AND STANDARD DEVIATION FOR FEATURES
    for a_feature in range(0, features): 
      # we need to separate our spam data from our non-spam data 
      spam_data, not_spam_data = [], []

      for row in range(len(train_data)): 
        # if we get a training input value of 1, it is spam, 0 means not-spam
        if(target_train[r] == 1): 
          spam_data.append(train_data[r][feature])
        else: 
          not_spam_data.append(train_data[r][feature])

      # find the mean of the spam data in the training set
      train_mean_spam_data.append(np.mean(spam_data))

      # find the mean of the not-spam data in the training set
      train_mean_not_spam_data.append(np.mean(not_spam_data))

      # find the standard deviation of the spam data in the training set
      train_std_spam_data.append(np.std(spam_data))

      # find the standard deviation of the not-spam data in the training set
      train_std_not_spam_data.append(np.std(not_spam_data))
       
    # SPECIAL CASE FOR STANDARD DEVIATION
    # If any of the features has zero standard deviation, assign it a "minimal" standard
    # deviation (e.g., 0.0001) to avoid a divide-by-zero error in Gaussian Naive Bayes
    for i in range(len(train_std_spam_data)): 
      if(train_std_spam_data[i] == 0): 
        train_std_spam_data[i] = std

       elif(train_std_not_spam_data[i] == 0): 
        train_std_not_spam_data[i] = std

input = Naive_Bayes("spambase.data")
print(input.train_data.shape)
print(input.test_data.shape)
print(input.target_train.shape)
print(input.target_test.shape)

# 2. create probabilistic model 
# compute the prior priority for each class, 1 (spam), 0 (not-spam) in 
# the training data. P(1) should be about 0.4
train_spam_data = test_spam_data = train_probability = test_probability = 0
train_spam_feature, train_not_spam_feature = [], []

# for each of the 57 features, compute the mean and standard deviation in
# the training set of the values given each class
# note: if any of the features has zero standard deviation, assign it a 
# "minimal" standard deviation (e.g. 0.0001) to avoid a divide by zero 
# error in Gaussian Naive Bayes
std = 0.0001
train_mean_spam_data, train_mean_not_spam_data = [], []
train_std_spam_data, train_std_not_spam_data = [], []
