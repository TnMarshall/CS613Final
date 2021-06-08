import numpy as np
from scipy import stats
from decision_tree import Decision_Tree as DT

def bagging(fileName):
    def bin_data(X, Y):
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

        binned_train = np.c_[newX, Y]
        return binned_train


    data = np.genfromtxt(fileName, delimiter=',')
    data = data[1:,:]
    X = data[:,:-1]
    Y = data[:,-1]
    binned_data = bin_data(X, Y)
    N = len(binned_data)
    split_size = int(np.ceil((N*2)/3))
    train = binned_data[:split_size, :]
    val = binned_data[split_size:, :]
    train_X = train[:,:-1]
    train_Y = train[:,None,-1]
    val_X = val[:,:-1]
    val_Y = val[:,None,-1]


    indices = np.indices((1, split_size))[1,:,:][0]
    classifiers = []

    for i in range(10):
        train_indices = np.random.choice(indices, size=split_size, replace=True)
        train_indices.sort()
        
        sub_train = train[train_indices, :]
        sub_train_X = sub_train[:,:-1]
        sub_train_Y = sub_train[:,-1]
        sub_train_Y = np.expand_dims(sub_train_Y, axis=1)
        
        dt = DT(sub_train)
        
        classifiers.append(dt)        
            
            
    all_preds = []
    for classifier in classifiers:
        
        val_preds = np.apply_along_axis(dt.predict, axis=1, arr=val_X, tree=dt.tree)
        val_preds = np.expand_dims(val_preds, axis=1)
        all_preds.append(val_preds)
        
    all_preds = np.concatenate(all_preds, axis=1)

    mode_preds = np.apply_along_axis(stats.mode, 1, all_preds)
    mode_preds = mode_preds[:,0,:]

    total_accuracy = sum(val_Y==mode_preds)/len(val_Y)
    print(f'Bagging ensemble method accuracy: {total_accuracy[0]}')
    return total_accuracy