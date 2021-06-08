import numpy as np
from decision_tree import Decision_Tree as DT
from naive_bayes import Naive_Bayes as NB
import weak_classifiers as wc
from itertools import combinations
from scipy import stats
import sys

filename = "./processedData/raceExcludedProcessedFeatures.csv"
# Import the data to run the classifiers on
data = np.genfromtxt(filename, delimiter=',')
data = data[1:,:]
X = data[:,:-1]
Y = data[:,-1]

# convert all the continuous columns to categorical to work with the
# decision tree
cols = X.shape[1]
cols_list = []
bins = [.25, .5, .75, 1.0]

for col in range(cols):
    if max(X[:,col]) <= 1.0:
        temp = np.nan_to_num(X[:,col])
        temp = np.digitize(temp, bins)
        cols_list.append(np.expand_dims(temp, axis=1))
    else:
        cols_list.append(np.expand_dims(X[:,col], axis=1))

newX = np.concatenate(cols_list, axis=1)
binned_data = np.c_[newX, Y]

# split the data into a training and validation set
N = binned_data.shape[0]
split_val = int(np.ceil((N*2)/3))
train = binned_data[:split_val, :]
val = binned_data[split_val:, :]
valX = val[:,:-1]
valY = val[:,-1]
valY = np.expand_dims(valY, axis=1)


########################### Logistic Regression ###########################
# split the data for logistic regression and add 1s for bias term
ones = np.ones((len(data),1), dtype=np.int8)
ones_data = np.c_[ones, data]
N = ones_data.shape[0]
split_val = int(np.ceil((N*2)/3))
train = ones_data[:split_val, :]
val = ones_data[split_val:, :]
trainX = train[:, :-1]
trainY = train[:, -1]
valX = val[:, :-1]
valY = val[:, -1]

# find all possible class options
temp_trainY = train[:, None,-1]
temp_valY = val[:, None,-1]
ys = np.vstack((temp_trainY,temp_valY))
options = np.unique(ys)

probs = np.zeros((len(valY), len(options)))   # holder for class probs

perm = combinations(options, 2)
for i in list(perm):
    f, l = i
    mask = ((trainY == f) | (trainY == l))
    sub_trainX = trainX[mask,:]
    sub_trainY = trainY[mask]
    Ymask = sub_trainY == f
    sub_trainY = Ymask.astype(int)
    sub_trainY = np.expand_dims(sub_trainY, axis=1)
    w = wc.logistic_regression(sub_trainX, sub_trainY, 10000, 2**-32, .0001)
    f_prob = wc.sigmoid(valX, w)
    l_prob = 1 - f_prob

    probs[:, None, int(f)] += f_prob
    probs[:, None, int(l)] += l_prob
    
probs = probs/(len(options)-1)

lr_preds = []
for row in probs:
    idx = np.argmax(row)
    lr_preds.append(idx)

lr_preds = np.array(lr_preds)
lr_preds = np.expand_dims(lr_preds, axis=1)

temp_valY = np.expand_dims(valY, axis=1)

lr_accuracy = sum(temp_valY==lr_preds)/len(temp_valY)
print(f'Logistic Regression accuracy: {lr_accuracy}')


# Produce confusion matrix
confusionMatrix = np.zeros((11,11))
for i in range(0,len(valY)):
    was = int(valY[i])
    predicted = int(lr_preds[i,0])
    confusionMatrix[was,predicted] += 1

print(confusionMatrix)