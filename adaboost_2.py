import numpy as np
from decision_tree import Decision_Tree as DT


def bin_data(X, Y):
    cols = X.shape[1]
    cols_list = []
    bins = [.25, .5, .75, 1.0]

    for col in range(cols):
        if max(X[:,col]) <= 1.0:
            temp = np.nan_to_num(X[:,col])
    #         mu = np.mean(X[:,col])
    #         mu = np.mean(temp)
            temp = np.digitize(temp, bins)
    #         temp = temp >= mu
    #         temp = temp.astype(int)
    #         print(sum(temp))
            cols_list.append(np.expand_dims(temp, axis=1))
        else:
            cols_list.append(np.expand_dims(X[:,col], axis=1))

    newX = np.concatenate(cols_list, axis=1)

    binned_train = np.c_[newX, Y]
    return binned_train


data = np.genfromtxt('data/raceExcludedProcessedFeatures.csv', delimiter=',')
data = data[1:,:]

X = data[:,:-1]
Y = data[:,-1]
# Y = np.expand_dims(Y, axis=1)
binned_data = bin_data(X, Y)
# X = binned_data[:,:-1]
# Y = binned_data[:,-1]
# Y = np.expand_dims(Y, axis=1)
N = len(binned_data)
split_size = int(np.ceil((N*2)/3))
train = binned_data[:split_size, :]
val = binned_data[split_size:, :]
train_X = train[:,:-1]
train_Y = train[:,None,-1]
val_X = val[:,:-1]
val_Y = val[:,None,-1]


# print(train.shape)
# print(val.shape)
# print(train_X.shape)
# print(val_X.shape)
# print(train_Y.shape)
# print(val_Y.shape)


indices = np.indices((1, split_size))[1,:,:][0]

dist = np.ones(split_size)
dist *= 1/split_size

# print(f'Dist: {dist}')

models = []

#This will iterate however many times we want it to.
for i in range(1):
    kpop = np.empty((train_X.shape[0],11)) #holds the outcomes for each class for each observation
    np.random.seed(221)
    train_indices = np.random.choice(indices, size=split_size, replace=True, p=dist)
    train_indices.sort()
    sub_train = train[train_indices, :].copy()
    sub_train_X = sub_train[:,:-1]
    sub_train_Y = sub_train[:,-1]
#     sub_train_Y = np.expand_dims(sub_train_Y, axis=1)
	#Iterates across classes, creating a binary class system for each.
    for k in np.unique(train_Y):
        zTrain = sub_train.copy()
        zero = sub_train_Y==k
        zTrain[zero,-1]=1; zTrain[~zero,-1]=-1 #1 is class k, -1 is not class k
        zTrain_X = zTrain[:,:-1]
        zTrain_Y = zTrain[:,-1]
        zTrain_Y = np.expand_dims(zTrain_Y, axis=1)
    
        dt = DT(zTrain,1)

        train_preds = np.apply_along_axis(dt.predict, axis=1, arr=zTrain_X, tree=dt.tree)
#         print(np.unique(train_preds,return_counts=True),np.unique(zTrain_Y,return_counts=True))
#         train_preds = np.expand_dims(train_preds, axis=1)
        misclass = zTrain_Y!=train_preds 
        dt_accuracy = sum(~misclass)/len(zTrain_Y)
        error = sum(misclass)/len(zTrain_Y) + 2**-22222
        beta = np.log((1-error)/error) + np.log(10)
        kpop[:,int(k)] = train_preds*beta
    kpred = np.apply_along_axis(np.argmax, axis=1, arr=kpop)
    print(np.unique(kpred,return_counts=True))
    
    misclass = sub_train_Y!=kpred 
    dt_accuracy = sum(~misclass)/len(train_Y)
    error = sum(misclass)/len(train_Y) + 2**-222
    beta = np.log((1-error)/error) + np.log(10)
    print(dt_accuracy,error,beta)
	
#########################################################################################################################################	
######################## Everything below here has not been updated from the other adaboost attempt #####################################
########################  If you want to actually run this code, you'll need to comment it out or   #####################################
########################                                 remove it.                                 #####################################
#########################################################################################################################################

    misclass = sub_train_Y!=kpred 
    dt_accuracy = sum(~misclass)/len(train_Y)
    error = sum(misclass)/len(train_Y) + 2**-222
    if error < 10/11:
#         print(f'Dist: {dist}')
            print(f'Decision Tree error: {error}')
#         print(f'Decision Tree accuracy: {dt_accuracy}')
        
        beta = .5*np.log((1-error)/error)
#         beta = error/(1-error)
#         print(f'Beta: {beta}')
        
        dt_preds = np.apply_along_axis(dt.predict, axis=1, arr=train_X, tree=dt.tree)
        dt_preds = np.expand_dims(dt_preds, axis=1)

        correct = train_Y==dt_preds 
        correct = correct.flatten()
        
        dist = np.where(correct, dist*np.exp(-beta), dist*np.exp(beta))
        Z = sum(dist)
#         print(f'Z: {Z}')
        dist = dist/Z

        dt_accuracy = sum(correct)/len(train_Y)
#         print(f'Decision Tree accuracy: {dt_accuracy}')
        
        dt_preds = np.apply_along_axis(dt.predict, axis=1, arr=val_X, tree=dt.tree)
        dt_preds = np.expand_dims(dt_preds, axis=1)

        correct = val_Y==dt_preds 
        correct = correct.flatten()

        val_accuracy = sum(correct)/len(val_Y)
#         print(f'Val accuracy: {val_accuracy}')
#         print()
        
        models.append((dt, beta))
        
pred_bucket = np.empty((len(train_X), len(models)))

betas = np.empty(len(models))
i = 0
for model, beta in models:
    betas[i] = beta
    preds = np.apply_along_axis(dt.predict, axis=1, arr=train_X, tree=model.tree)
    pred_bucket[:,i] = preds
    i += 1


def whatev( array, betas ):
    votes = []
    for k in np.unique(train_Y):
        ind = array==k
        vote = sum(betas[ind])
        votes.append(vote)
    return np.argmax(votes)

test = np.apply_along_axis(whatev, axis=1, arr=pred_bucket, betas=betas)
test = np.expand_dims(test, axis=1)
correct = test==train_Y
accuracy = sum(correct)/len(train_Y)
print(f'acc: {accuracy}')
print(betas)