import numpy as np
from scipy import stats

############################# Logistic Regression #############################
def sigmoid(X, w):
    Z = np.dot(X, w)
    return 1/(1+np.exp(-Z))

def logistic_regression(X, Y, epochs, term_point, lr):
    
    per_change = np.Inf
    r, c = X.shape
    w = np.random.randn(c,1)*0.001
    N = len(X)

    i = 0
    costs = []

    # iterator for the given number of epochs or until the percent change
    # is less than the given termination criteria
    while i < epochs and per_change > term_point:

        Yhat = sigmoid(X, w)

        first = np.dot(Y.T, np.log(Yhat))
        second = np.dot((1-Y).T, np.log(1-Yhat))
        cost = ((first + second)/N)[0][0]

        if len(costs) > 0:
            change = costs[-1] - cost
            per_change = (change/costs[-1])*100

        error = Y - Yhat
        dw = np.dot(X.T, error)/N

        costs.append(cost)

        w = w + (lr*dw)
        i += 1

    return w

###############################################################################


############################# K-Nearest Neighbor ##############################

def euclidean(p1, p2):
    # subtracting vector
    diff = p1 - p2

    # doing dot product for finding sum of the squares
    sum_sq = np.dot(diff.T, diff)

    # Doing squareroot and printing Euclidean distance
    return sum_sq

def knn(train_X, val_X, train_Y, k):
    y_hat = []
    
    # loop over all samples that need to be classified
    for x in val_X:
        distances = []

        # loop over train examples and find distance between the samples that
        # need to be classified and the labeled samples
        for p in train_X:
            dist = euclidean(x, p)
            distances.append(dist)
            
        dists = np.expand_dims(distances, axis=1)
        
        # match the distances to the training labels and sort by distance in
        # ascending order
        nn_val = np.c_[train_Y, dists]
        sorted_nn = nn_val[nn_val[:,1].argsort()]
        
        # get the most common label the the k nearest neighbors and set as the
        # prediction
        nns = sorted_nn[:k,0]
        most_common = stats.mode(nns)
        if most_common[1][0] == 1:
            pred = nns[0]
        else:
            pred = most_common[0][0]
            
        y_hat.append(pred)
    
    y_hat = np.expand_dims(y_hat, axis=1)
    return y_hat
###############################################################################