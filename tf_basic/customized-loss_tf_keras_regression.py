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

x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state = 7, test_size = 0.1)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state = 11)
#nomalization
scalar = StandardScaler()
x_train_scaled = scalar.fit_transform(x_train)
x_valid_scaled = scalar.transform(x_valid)
x_test_scaled = scalar.transform(x_test)


def customized_mse(y_true, y_predict):
    return tf.reduce_mean(tf.square(y_predict - y_true))


model = keras.models.Sequential([
    keras.layers.Dense(30, activation = 'relu', input_shape = x_train.shape[1:]),
    keras.layers.Dense(1),
])
model.summary()
model.compile(loss = customized_mse, optimizer = "sgd", metrics = ["mean_square_error"])
callbacks = [keras.callbacks.EarlyStopping(patience = 5, min_delta = 1e-2)]
history = model.fit(x_train_scaled, y_train, validation_data = (x_valid_scaled, y_valid), epochs = 100, callbacks = callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()


plot_learning_curves(history)
model.evaluate(x_test_scaled, y_test)
