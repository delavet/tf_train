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

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

def f(x):
    return 3. * x ** 2 + 2. * x - 1

#近似对f求导数
def approximate_derivative(f, x, eps = 1e-3):
    return (f(x+eps) - f(x-eps)) / (2. * eps)

print(approximate_derivative(f, 1.))

def  g(x1, x2):
    return(x1 + 5) * (x2 ** 2)
#二元求导
def approximate_gradient(g, x1, x2, eps = 1e-3):
    dg_x1 = approximate_derivative(lambda x : g(x, x2), x1, eps)
    dg_x2 = approximate_derivative(lambda x : g(x1, x), x2, eps)
    return dg_x1, dg_x2

print(approximate_gradient(g, 2., 3.))

"""
TensorFlow中自定义求导实现
"""
x1 = tf.Variable(2.)
x2 = tf.Variable(3.)
with tf.GradientTape() as tape:
    z = g(x1, x2)

dz_x1 = tape.gradient(z, x1)
print(dz_x1)

try:
    dz_x2 = tape.gradient(z, x2)
except RuntimeError as ex:
    print(ex)
