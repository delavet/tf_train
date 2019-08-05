import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from sklearn.preprocessing import StandardScaler #用于归一化
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow import keras
from pprint import pprint

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)


housing = fetch_california_housing()
#print the description of the california housing dataset
print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)

pprint(housing.data[:5])
pprint(housing.target[:5])
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state = 7, test_size = 0.1)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state = 11)
print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)

#nomalization
scalar = StandardScaler()
x_train_scaled = scalar.fit_transform(x_train)
x_valid_scaled = scalar.transform(x_valid)
x_test_scaled = scalar.transform(x_test)

print("x_train.shapep[1:]: ", x_train.shape[1:])
#函数式API实现
input_wide = keras.layers.Input(shape = (5,))
input_deep = keras.layers.Input(shape = (6,))
hidden1 = keras.layers.Dense(30, activation = 'relu')(input_deep)
hidden2 = keras.layers.Dense(30, activation = 'relu')(hidden1)

concat = keras.layers.concatenate([input_wide, hidden2])
output = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs = [input_wide, input_deep], outputs = [output])

model.summary()

#拆分训练数据，分别喂给wide和deep模型
x_train_scaled_wide = x_train_scaled[:, :5]
x_train_scaled_deep = x_train_scaled[:, 2:]
x_valid_scaled_wide = x_valid_scaled[:, :5]
x_valid_scaled_deep = x_valid_scaled[:, 2:]
x_test_scaled_wide = x_test_scaled[:, :5]
x_test_scaled_deep = x_test_scaled[:, 2:]

model.compile(loss = "mean_squared_error", optimizer = "sgd")
callbacks = [keras.callbacks.EarlyStopping(patience = 5, min_delta = 1e-2)]
history = model.fit([x_train_scaled_wide, x_train_scaled_deep], y_train, validation_data = ([x_valid_scaled_wide, x_valid_scaled_deep], y_valid), epochs = 100, callbacks = callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()


plot_learning_curves(history)
model.evaluate([x_test_scaled_wide, x_test_scaled_deep], y_test)
