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

t = tf.constant([[1., 2., 3.], [4., 5., 6.]])

#index
print("index:")
print(t)
print(t[:, 1:])
#如果前面有很多冒号[:,:,:,1]，可以用...代替[..., 1]
print(t[..., 1])

#ops
print("==================")
print("ops:")
#每个元素做加法
print(t + 10)
#每个元素做乘方
print(tf.square(t))
#矩阵乘法乘上自己的转置
print(t @ tf.transpose(t))

#numpy conversion
print("==================")
print("numpy conversion:")
#TensorFlow张量向numpy数组直接转换
print(t.numpy())
#可以向numpy的函数中直接传入tf.Tensor，计算后会得到一个numpy的数组
print(np.square(t))
np_t = np.array([[1., 2., 3.], [4., 5., 6.]])
#可以用numpy数组构造TensorFlow张量
print(tf.constant(np_t))


#scalars
print("==================")
print("scalars:")
t = tf.constant(2.718)
print(t)
print(t.numpy())
print(t.shape)


#strings
print("====================")
print("strings:")
t = tf.constant("cafe")
print(t)
print(tf.strings.length(t))
print(tf.strings.length(t, unit="UTF8_CHAR"))
print(tf.strings.unicode_decode(t, "UTF8"))


#string array
print("====================")
print("string array:")
t = tf.constant(["cafe", "coffee", "咖啡"])
print(t)
print(tf.strings.length(t, unit="UTF8_CHAR"))
r = tf.strings.unicode_decode(t, "UTF8")
print(r)

#ragged tensor
print("====================")
print("ragged tensor:")
r = tf.ragged.constant([[11, 12], [21, 22, 23], [], [41]])
#ragged index
print(r)
print(r[1])
print(r[1:2])
#print(r[:, 1:]) will cause error beacuse it is ragged
#ops on ragged tensor
r2 = tf.ragged.constant([[51, 52], [], [71]])
print(tf.concat([r, r2], axis = 0))
#print(tf.concat([r, r2], axis = 1)) can't concat beacuse rag
r3 = tf.ragged.constant([[13, 14], [15], [], [42, 43, 45]])
print(tf.concat([r, r3], axis = 1))
print(r.to_tensor())
print(tf.concat([r.to_tensor(), r3.to_tensor()], axis = 0))

#sparse tensor
#这里给的indices必须是排好序的，不然to_dense会出错
s = tf.SparseTensor(indices = [[0, 1], [1, 0], [2, 3]], values = [1., 2., 3.], dense_shape = [3, 4])
print(s)
print(tf.sparse.to_dense(s))
s2 = s * 2.0
print(s2)
try:
    s3 = s + 1
except TypeError as ex:
    print(ex)

s4 = tf.constant([[10., 20.], [30., 40.], [50., 60.], [70., 80.]])
print(s4)
print(tf.sparse.sparse_dense_matmul(s, s4))

#variable
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print(v)
print(v.value())
print(v.numpy())
#变量的赋值：赋值必须使用assign而不能使用
v.assign(2 * v)
print(v.numpy())
v[0, 1].assign(42)
print(v.numpy())
v[1].assign([7., 8., 9.])
print(v.numpy())
