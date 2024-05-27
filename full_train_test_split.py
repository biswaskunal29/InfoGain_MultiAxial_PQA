import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


full_X = np.load('full_X_data_subsample.npy')
full_y = np.load('full_y_data_subsample.npy')

#print(full_X.shape)
#print(full_y.shape)


# =============================================================================
# change random_state for different values
# =============================================================================
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=4)

for train_index, test_index in sss.split(full_X, full_y):
    print("TEST:\n", full_y[test_index])
#    print("TRAIN:", labels[train_index], "TEST:", labels[test_index])
    X_train, X_test = full_X[train_index], full_X[test_index]
    y_train, y_test = full_y[train_index], full_y[test_index]

print(X_train.shape)
#print(X_train[4])

np.save('full_X_train_50split.npy', X_train)
np.save('full_X_test_50split.npy', X_test)
np.save('full_y_train_50split.npy', y_train)
np.save('full_y_test_50split.npy', y_test)
# =============================================================================
# match = np.array([0., 0., 0., 0., 1.])
# print(match)
# 
# count = 0
# for item in y_train:
#     if list(item) == list(match):
#         count += 1
# 
# print(count)
# =============================================================================














































