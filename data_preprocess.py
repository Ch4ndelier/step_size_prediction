import numpy as np
import os
from sklearn import preprocessing

x_path = 'data/step_cord/TRAIN_X_cordi_reg.npy'
y_path = 'data/step_cord/TRAIN_Y_cordi_reg.npy'

x = np.load(x_path)
y = np.load(y_path)

print("x.shape: ", x.shape)
print("y.shape: ", y.shape)

'''
print(x[1])
for ind, x_ins in enumerate(x):
    x[ind] = preprocessing.MinMaxScaler().fit_transform(x_ins)
print(x[1])
'''

y_1 = y[:, 0]
y_2 = y[:, 1]

y = preprocessing.MinMaxScaler().fit_transform(y)



print("after reshape", y.shape)

dis = int(len(x) * 0.85 // 1)
print("dis", dis)
x_train = x[:dis]
x_val = x[dis:]
y_train = y[:dis]
y_val = y[dis:]

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

dir_name = "./data/step_cord_run/"

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
