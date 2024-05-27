#from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import numpy as np
# =============================================================================
# X = np.array([[-2,-3,-5,-6],
#               [-7,-4,8,9]])
# =============================================================================

X= np.random.randint(5, size=(3, 4, 3))



#normalized_arr = preprocessing.normalize([x_array])
#print(normalized_arr)


# =============================================================================
# scaler = MinMaxScaler(feature_range=(0,255))
# scaled = scaler.fit_transform([[x] for x in x_array])
# print(scaled)
# =============================================================================



scaler = MinMaxScaler(feature_range=(0,255))
X_one_column = X.reshape([-1,1])
result_one_column = scaler.fit_transform(X_one_column)
result = result_one_column.reshape(X.shape)
print(result)




















