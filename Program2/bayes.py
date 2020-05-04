# Angie McGraw
# CS 445 Programming Assignment 2
# This program uses Gaussian Naive Bayes and Logistic Regression 
# to classify the Spambase data from the UCL ML repository. 

import numpy as np
from sklearn.metrics import confusion_matrix

# part I: classification with Naive Bayes
# 1. create training and test set
# split the data into a training and test set; each has 2300 instances, 
# 40% spam and 60% not-spam
class Naive_Bayes(object):
  def __init__(own, file):
    own.train_data, own.test_data = own.split(file)
    
  def split(own, file):
    data = np.loadtxt(file, delimiter = ",")
    train_data = data[:2300, :] 
    test_data = data[2301:, :]
    return train_data, test_data

  def probability(own): 
    # get the probability of the training set
    for i in range(len(train_data)): 
      if(train_data[i] == 1): 
        train_spam_data += 1
	
    train_probability_spam = (train_spam / len(train_data))
    train_probability_not_spam = 1 - train_probability_spam

    # get the probability of the test set
    for i in range(len(test_data)): 
      if(test_data[i] == 1): 
        test_spam_data += 1
    
    test_probability_spam = (test_spam / len(test_data))
    test_probability_not_spam = 1 - test_probability_spam

input_data = Naive_Bayes("spambase.data")

# 2. create probabilistic model 
# compute the prior priority for each class, 1 (spam), 0 (not-spam) in 
# the training data. P(1) should be about 0.4
train_spam_data = test_spam_data = train_probability = test_probability = 0

# for each of the 57 features, compute the mean and standard deviation in
# the training set of the values given each class
# note: if any of the features has zero standard deviation, assign it a 
# "minimal" standard deviation (e.g. 0.0001) to avoid a divide by zero 
# error in Gaussian Naive Bayes
std = 0.0001
train_mean_spam_data, train_mean_not_spam_data = [], []
train_std_spam_data, train_std_not_spam_data = [], []
