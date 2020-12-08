import numpy as np
import os
from sklearn import preprocessing

x = np.load('data/step_pre/1127_Xrow.npy')
y = np.load('data/step_pre/1127_Yrow.npy')
y = y.reshape(740, -1)
'''
print(x[1])
for ind, x_ins in enumerate(x):
    x[ind] = preprocessing.MinMaxScaler().fit_transform(x_ins)
print(x[1])
'''

print(y[:10])
print(max(y))
print(min(y))
y = preprocessing.MinMaxScaler().fit_transform(y)
print(y[:10])



print("after reshape", y.shape)
x_train = x[:600]
x_val = x[600:]
y_train = y[:600]
y_val = y[600:]

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)


dir_name = "./data/step_test_1127/"
if os.path.exists(dir_name):
    print("path already exists!!")
    exit()
else:
    print("saving ...")
    os.mkdir(dir_name)
    np.save(dir_name + "TRAIN_X_1.npy", x_train)
    np.save(dir_name + "TRAIN_Y_1.npy", y_train)
    np.save(dir_name + "DEV_X_1.npy", x_val)
    np.save(dir_name + "DEV_Y_1.npy", y_val)
print("complete!!")
