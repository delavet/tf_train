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

from tensorflow import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data() 
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(np.max(x_train), np.min(x_train))

#数据归一化 x = (x-u)/std
scalar = StandardScaler()
x_train_scaled = scalar.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_scaled = scalar.transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_scaled = scalar.transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
print(x_test_scaled)
print(np.max(x_train_scaled), np.min(x_train_scaled))


def main():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape = [28,28]))
    for _ in range(20):
        #model.add(keras.layers.Dense(100, activation = "relu"))
        #model.add(keras.layers.BatchNormalization())
        """
        #另一种选择，先做批归一化再做激活
        model.add(keras.layers.Dense(100))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
        """
        """
        #另一种选择，使用selu激活函数，这是一种自带归一化的激活函数（可以去了解一下）
        model.add(keras.layers.Dense(100, activation = "selu"))
        """
        model.add(keras.layers.Dense(100, activation = "selu"))
        
    model.add(keras.layers.Dense(10, activation = "softmax"))

    #relu: y = max(0,x)
    #softmax: x = [x1, x2, x3], y=[e^x1/sum, e^x2/sum, e^x3/sum] sum 是上面三个的和

    #sparse的原因：这里y只是一个index，要用sparse_categorical_crossentropy，这个会用onehot把y转成向量
    model.compile(loss="sparse_categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])
    model.summary()
    logdir = '.\\dnn-bn-callbacks'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    output_model_file = os.path.join(logdir, "fashion_mnist_model.h5")
    callbacks = [
        keras.callbacks.TensorBoard(logdir),
        keras.callbacks.ModelCheckpoint(output_model_file, save_best_only = True),
        keras.callbacks.EarlyStopping(patience = 5, min_delta = 1e-2)
    ]
    history = model.fit(x_train_scaled, y_train, epochs = 10, validation_data=(x_valid, y_valid), callbacks = callbacks)
    model.evaluate(x_test_scaled, y_test)


if __name__ == "__main__":
    main()
