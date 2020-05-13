import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

class Naive_Bayes(object): 
  def __init__(self, file): 
    self.train_data, self.test_data, self.train_target, self.test_target = self.split(file) 

  def split(self, file): 
    data = np.loadtxt(file, delimiter = ',', dtype = float)   
    np.random.shuffle(data)
    X, Target = data[:, :-1], data[:, -1]
    X_train, X_test, Target_train, Target_test = train_test_split(X, Target, test_size = 0.5, random_state = 20)
    
    return X_train, X_test, Target_train, Target_test

  def mean_stddev(self): 
    features = self.train_data.shape[1]
    m, s = np.ones((2, features)), np.ones((2, features))

    spam_data, not_spam_data = [], [] 

    for i in range(self.train_data.shape[0]): 
      if self.train_target[i] == 1: 
        spam_data.append(self.train_data[i])
      else: 
        not_spam_data.append(self.train_data[i])

    spam_data, not_spam_data = np.asarray(spam_data), np.asarray(not_spam_data)

    for i in range(0, features): 
      m[0, i] = np.mean(not_spam_data.T[i])
      m[1, i] = np.mean(spam_data.T[i])
      s[0, i] = np.std(not_spam_data.T[i])
      s[1, i] = np.std(spam_data.T[i])

    for i in range(features): 
      if s[0, i] == 0: 
        s[0, i] = 0.0001
      if s[1, i] == 0: 
        s[1, i] = 0.0001

    return m, s

  def probabilistic_model(self): 
    probability_train_spam_data = (np.count_nonzero(self.train_target) / len(self.train_target))
    probability_train_not_spam_data = 1 - probability_train_spam_data

    train_mean, train_stddev = self.mean_stddev()

    return probability_train_spam_data, probability_train_not_spam_data, train_mean, train_stddev

  def gauss_algo(self, x, mean, stddev, spam_data, not_spam_data): 
    probability = np.ones(2)

    for i in range(2): 
      prob = 0
 
      if i != 0: 
        prob += np.log(spam_data)
      else:  
        prob += np.log(not_spam_data)
     
      for j in range(len(x)): 
        if x[j] != mean[i][j]: 
          a = (x[j] - mean[i][j]) ** 2 
          b = (2 * ((stddev[i][j]) ** 2))
          e = np.exp(-(a / b))
          N = (1 / (np.sqrt(2 * np.pi) * stddev[i][j]))
          if N == 0 or e == 0:
            N = 0.000000000000000000000000000000001
            e = 0.000000000000000000000000000000001
          prob += np.log(N * e)
        else:  
          continue
          
      probability[i] = prob
    
    return probability

  def nb_algo(self): 
    probability_spam_data, probability_not_spam_data, mean, stddev = self.probabilistic_model()
    predicted = []
 
    for rows in range(self.test_data.shape[0]): 
      p = self.gauss_algo(self.test_data[rows], mean, stddev, probability_spam_data, probability_not_spam_data)
     
      # get the tuple with the higher probability 
      predicted.append(np.argmax(p))

    return predicted

  def accuracy(self, predicted): 
    total_correct = 0
    for i in range(len(self.test_target)): 
      if predicted[i] == self.test_target[i]: 
        total_correct += 1
  
    accuracy = total_correct / len(self.test_target) 
    return accuracy

  def calculation_variables(self, predicted):
    true_pos = true_neg = false_pos = false_neg = 0 
    for i in range(len(self.test_target)):  
      if predicted[i] == 1 and self.test_target[i] == 1:  
        true_pos += 1
      elif predicted[i] == 0 and self.test_target[i] == 0: 
        true_neg += 1
      elif predicted[i] == 1 and self.test_target[i] == 0: 
        false_pos += 1
      else: 
        false_neg += 1

    return true_pos, true_neg, false_pos, false_neg

input = Naive_Bayes("spambase.data")
nb = input.nb_algo()
accuracy = (input.accuracy(nb) * 100)
tp, tn, fp, fn = input.calculation_variables(nb)
precision = float(((tp) / (tp + fn)) * 100)
recall = float(((tp) / (tp + fn)) * 100)

print('\nAccuracy: ')
print(accuracy)

print('\nPrecision: ')
print(precision)

print('\nRecall: ')
print(recall)

cfm = confusion_matrix(input.test_target, nb)
print(cfm)
