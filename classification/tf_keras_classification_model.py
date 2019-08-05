import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

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
print(type(x_valid))


def show_single_image(img_arr):
    plt.imshow(img_arr, cmap="binary")
    plt.show()


#show_single_image(x_train[0])


def show_imgs(n_rows,n_cols,x_data,y_data,class_names):
    assert len(x_data) == len(y_data)
    assert n_rows * n_cols < len(x_data)
    plt.figure(figsize=(n_cols*1.4,n_rows*1.6))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(x_data[index], cmap="binary", interpolation="nearest")
            plt.axis('off')
            plt.title(class_names[y_data[index]])
    plt.show()


#class_names = ['T-shirts', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
#show_imgs(3,5,x_train,y_train,class_names)
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()


def main():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape = [28,28]))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(10,activation="softmax"))

    #relu: y = max(0,x)
    #softmax: x = [x1, x2, x3], y=[e^x1/sum, e^x2/sum, e^x3/sum] sum 是上面三个的和

    #sparse的原因：这里y只是一个index，要用sparse_categorical_crossentropy，这个会用onehot把y转成向量
    model.compile(loss="sparse_categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])
    model.summary()
    history = model.fit(x_train, y_train, epochs = 10, validation_data=(x_valid, y_valid))
    plot_learning_curves(history)


if __name__ == "__main__":
    main()
