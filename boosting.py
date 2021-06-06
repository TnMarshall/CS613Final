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


data = np.genfromtxt('processedData/raceExcludedProcessedFeatures.csv', delimiter=',')
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

print(f'Dist: {dist}')



for i in range(10):
    train_indices = np.random.choice(indices, size=split_size, replace=True, p=dist)
    train_indices.sort()
    
    sub_train = train[train_indices, :]
    sub_train_X = sub_train[:,:-1]
    sub_train_Y = sub_train[:,-1]
    sub_train_Y = np.expand_dims(sub_train_Y, axis=1)
    
    dt = DT(sub_train, 1)

    train_preds = np.apply_along_axis(dt.predict, axis=1, arr=sub_train_X, tree=dt.tree)
    train_preds = np.expand_dims(train_preds, axis=1)
    misclass = sub_train_Y!=train_preds 
    dt_accuracy = sum(~misclass)/len(train_Y)
    error = sum(misclass)/len(train_Y)
    if error < 10/11:
        print(f'Decision Tree error: {error}')
        print(f'Decision Tree accuracy: {dt_accuracy}')
        
        beta = error/(1-error)
        print(f'Beta: {beta}')
        
        dt_preds = np.apply_along_axis(dt.predict, axis=1, arr=train_X, tree=dt.tree)
        dt_preds = np.expand_dims(dt_preds, axis=1)

        correct = train_Y==dt_preds 
        correct = correct.flatten()
        
        dist = np.where(correct, dist, dist*beta)
        Z = sum(dist)
        print(f'Z: {Z}')
        dist = dist/Z

        dt_accuracy = sum(correct)/len(train_Y)
        print(f'Decision Tree accuracy: {dt_accuracy}')
        print()