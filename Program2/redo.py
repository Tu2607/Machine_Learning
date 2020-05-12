import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

data = np.loadtxt('spambase.data', delimiter = ',', dtype = float)
X, target = data[:, :-1], data[:, -1]
train_data, test_data, target_train, target_test = train_test_split(X, target, test_size = 0.5)

train_spam_data = test_spam_data = probability_train_data = probability_test_data = 0
train_mean_spam_data, train_mean_not_spam_data = [], []
train_stddev_spam_data, train_stddev_not_spam_data = [], []
train_feature_spam_data, train_feature_not_spam_data = [], []

for i in range(len(target_train)): 
  if(target_train[i] == 1): 
    train_spam_data += 1
probability_train_spam_data = (train_spam_data / len(target_train))
probability_train_not_spam_data = 1 - probability_train_spam_data

for i in range(len(target_test)): 
  if(target_test[i] == 1): 
    test_spam_data += 1
probability_test_spam_data = (test_spam_data / len(target_test))
probability_test_not_spam_data = 1 - probability_test_spam_data

features = train_data.shape[1]
for i in range(0, features): 
  spam_data, not_spam_data = [], []

  for j in range(len(train_data)): 
    if(target_train[j] != 0): 
      spam_data.append(train_data[j][i])
    else: 
      not_spam_data.append(train_data[j][i])

  train_mean_spam_data.append(np.mean(spam_data))
  train_mean_not_spam_data.append(np.mean(not_spam_data))
  train_stddev_spam_data.append(np.std(spam_data))
  train_stddev_not_spam_data.append(np.std(not_spam_data))

for i in range(len(train_stddev_spam_data)): 
  if(train_stddev_spam_data[i] == 0): 
    train_stddev_spam_data[i] = 0.0001
  elif(train_stddev_not_spam_data[i] == 0): 
    train_stddev_not_spam_data[i] = 0.0001

std = 0.0001
calculated = []
def gauss_algo(x, mean, std): 
  N = float(1 / (np.sqrt(2 * np.pi) * 0.0001)) * float(np.exp(-((x - mean)**2) / (2 * float(0.0001 * 0.0001))))
  return N

for row in range(len(test_data)): 
  probability_spam_data = np.log(probability_train_spam_data)
  probability_not_spam_data = np.log(probability_train_not_spam_data)

  for i in range(0, features): 
    x = test_data[j][i]
    probability_spam_data += np.log(gauss_algo(x, train_mean_spam_data[i], train_stddev_spam_data[i]))
    probability_not_spam_data += np.log(gauss_algo(x, train_mean_not_spam_data[i], train_stddev_not_spam_data[i]))

  class_x = np.argmax([probability_not_spam_data, probability_spam_data])
  calculated.append(class_x)

# print confusion matrix
cfm = confusion_matrix(target_test, calculated)
print(cfm)

# accuracy
TP = TN = FP = FN = 0
for i in range(len(calculated)):
  if(calculated[i] == 1 and target_test[i] == 1):
    TP += 1
  elif(calculated[i] == 0 and target_test[i] == 0): 
    TN += 1
  elif(calculated[i] == 1 and target_test[i] == 0): 
    FP += 1
  else: 
    FN += 1 

accuracy = float((TP + TN) / (TP + TN + FP + FN))
print(accuracy)

precision = float((TP) / (TP + FP))
print(precision)

recall = float(TP) / (TP + FN)
print(recall)
