import numpy as np
from decision_tree import Decision_Tree as DT
from naive_bayes import Naive_Bayes as NB
import weak_classifiers as wc
from itertools import combinations
from scipy import stats
import sys


def voting(filename):
    accuracies = np.zeros((1,5))
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
    
    
    ############################## Decision Tree ##############################
    # perform perdiction using a decision tree
    dt = DT(train)
    dt_preds = np.apply_along_axis(dt.predict, axis=1, arr=valX, tree=dt.tree)
    dt_preds = np.expand_dims(dt_preds, axis=1)
    dt_accuracy = sum(valY==dt_preds)/len(valY)
    # print(f'Decision Tree accuracy: {dt_accuracy[0]}')
    accuracies[0,0] = dt_accuracy[0]
    
    
    ############################### Naive Bayes ###############################
    # perform perdiction using Naive Bayes
    nb = NB(train)
    nb_preds = np.apply_along_axis(nb.predict, axis=1, arr=valX)
    nb_preds = np.expand_dims(nb_preds, axis=1)
    nb_accuracy = sum(valY==nb_preds)/len(valY)
    # print(f'Naive Bayes accuracy: {nb_accuracy[0]}')
    accuracies[0,1] = nb_accuracy[0]
    
    ################################### KNN ###################################
    # split data for the knn algorithm
    N = data.shape[0]
    split_val = int(np.ceil((N*2)/3))
    train = data[:split_val, :]
    val = data[split_val:, :]
    trainX = train[:, :-1]
    trainY = train[:, -1]
    valX = val[:, :-1]
    valY = val[:, None,-1]
    
    # perform prediction using knn
    knn_preds = wc.knn(trainX, valX, trainY, 4)
    knn_accuracy = sum(valY==knn_preds)/len(valY)
    # print(f'KNN accuracy: {knn_accuracy[0]}')
    accuracies[0,2] = knn_accuracy[0]
    
    
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
    # print(f'Logistic Regression accuracy: {lr_accuracy}')
    accuracies[0,3] = lr_accuracy
    
    # combine predictions from all 4 classifiers and find the mode
    all_preds = np.c_[lr_preds, knn_preds, dt_preds, nb_preds]
    mode_preds = np.apply_along_axis(stats.mode, 1, all_preds)
    mode_preds = mode_preds[:,0,:]
    
    total_accuracy = sum(temp_valY==mode_preds)/len(temp_valY)
    # print(f'Voting ensemble method accuracy: {total_accuracy}')
    accuracies[0,4] = total_accuracy
    return accuracies
    
    
# print('Accuracies for all data')
# output = voting('processedData/raceIncludedProcessedFeatures.csv')
# print(output)

# print('Accuracies for race data excluded')
# output = voting('processedData/raceExcludedProcessedFeatures.csv')
# print(output)

if __name__ == "__main__":
    totalArgs = len(sys.argv)
    if totalArgs != 2:
        print("This script only takes one argument")
    else:
        #argv 1 is the first command line argument
        dataName = sys.argv[1]
        output = voting(dataName)
        print(output)