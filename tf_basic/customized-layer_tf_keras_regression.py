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

pprint(housing.data[:5])
pprint(housing.target[:5])
x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state = 7, test_size = 0.1)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state = 11)


#nomalization
scalar = StandardScaler()
x_train_scaled = scalar.fit_transform(x_train)
x_valid_scaled = scalar.transform(x_valid)
x_test_scaled = scalar.transform(x_test)

"""
#简单的尝试
layer = tf.keras.layers.Dense(100, input_shape = (None, 5))
layer(tf.zeros([10, 5]))
#打印layer的所有参数
#layer: x * w + b
print(layer.variables)
#打印layer所有可以训练的变量
print(layer.trainable_variables)
#可以进一步查看怎么用layer
help(layer)
"""

#自定义layer
class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation = None, **kwargs):
        self.units = units #输出有多少单元
        self.activation = keras.layers.Activation(activation) #使用什么激活函数
        super(CustomizedDenseLayer, self).__init__(**kwargs)

    
    def build(self, input_shape):
        """
        构建需要的参数，
        这里自定义了一个全连接层
        """
        #x * w + b. input_shape: [None, a] output_shape = [None, number of units(c)]
        #w: a * c的矩阵
        #kernel就是这个层的核心参数，即w
        self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[1], self.units), initializer = 'uniform', trainable = True)
        #全连接层的偏置，就是b
        self.bias = self.add_weight(name = 'bias', shape = (self.units, ), initializer = 'zeros', trainable = True)
        super(CustomizedDenseLayer, self).build(input_shape)

    def call(self, x):
        """
        call函数完成一次layer的正向计算
        
        """
        return self.activation(x @ self.kernel + self.bias)

"""
使用lambda的形式将softplus函数定义成一个层次
要将一些简单的函数定义成层次，使用子类过于heavy，可以考虑用lambda
tf.nn.softplus: log(1 + e^x) 是一个平滑版的relu
"""
customized_softplus = keras.layers.Lambda(lambda x : tf.nn.softplus(x))
print(customized_softplus([-10., -5., 0., 5., 10.]))

model = keras.models.Sequential([
    CustomizedDenseLayer(30, activation = 'relu', input_shape = x_train.shape[1:]),
    CustomizedDenseLayer(1),
    customized_softplus #增加了一个自定义的激活函数层
])
"""
在模型最后加了一个简单的softplus激活层
这和tf.keras.layers.DenseLayer(1, activation = 'softplus')其实是一样的
这个customized_softplus层其实keras也实现了，就是keras.layers.Activation('softplus')   
"""
model.summary()
model.compile(loss = "mean_squared_error", optimizer = "sgd")
callbacks = [keras.callbacks.EarlyStopping(patience = 5, min_delta = 1e-2)]
history = model.fit(x_train_scaled, y_train, validation_data = (x_valid_scaled, y_valid), epochs = 100, callbacks = callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()


plot_learning_curves(history)
model.evaluate(x_test_scaled, y_test)
