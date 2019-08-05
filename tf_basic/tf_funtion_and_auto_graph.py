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
from IPython.display import display, Markdown

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

#tf.funtion and autograph
"""
tf.funtion and autograph
经常有的用处：保存模型并用模型做inference
"""

#自定义激活函数Scaled elu，使用python的语法
def scaled_elu(z, scale = 1.0, alpha = 1.0):
    #z > 0 ? scale * z : scale * alpha * tf.nn.elu(z)
    #使用tf的函数去操作tf.Tensor，这样比较方便，比如说输入一个向量，tf.where直接可以分别计算
    is_positive = tf.greater_equal(z, 0.0)
    return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))


print(scaled_elu(tf.constant(-3.)))
print(scaled_elu(tf.constant([-3., -2.5])))

#tf.funtion，用普通python函数构建了一个优化后的可调用的TensorFlow图，emmm，优势是虽然调用一样，但是比普通函数快
scaled_elu_tf = tf.function(scaled_elu)
print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3., -2.5])))
#返回原先用于构建的函数
print(scaled_elu_tf.python_function is scaled_elu)

"""
接下来是基于装饰器的tf.funtion方法
"""
#1 + 1/2 + 1/2^2 + ... + 1/2 ^ n
#使用装饰器直接就变成可调用图了
@tf.function
def converge_to_2(n_iters):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iters):
        total += increment
        increment /= 2
    return total


print(converge_to_2(20))


#使用tf.Variable
#variable必须定义在函数外面
var  = tf.Variable(0.)

@tf.function
def add_21():
    return var.assign_add(21)

print(add_21())

#函数签名增加输入类型限制，进行签名才能保存为tf的SavedModel
@tf.function(input_signature=[tf.TensorSpec([None], tf.int32, name = 'x')])
def cube(z):
    return tf.pow(z, 3)


#print(cube(tf.constant([1., 2., 3.])))
print(cube(tf.constant([1, 2, 3])))

#@tf.funtion py func -> graph
#get_concrete_function -> add input signiture -> Saved Model

cube_func_int32 = cube.get_concrete_function(tf.TensorSpec([None], tf.int32))
print(cube_func_int32)
print(cube_func_int32 is cube.get_concrete_function(tf.TensorSpec([5], tf.int32)))
print(cube_func_int32 is cube.get_concrete_function(tf.constant([1, 2, 3])))

#concrete_function的graph属性可以看到图是怎么样的
print(cube_func_int32.graph.get_operations())
print(cube_func_int32.graph.as_graph_def())
pow_op = cube_func_int32.graph.get_operations()[2]
print(pow_op)
print(list(pow_op.inputs))
print(list(pow_op.outputs))

print(cube_func_int32.graph.get_operation_by_name("x"))
print(cube_func_int32.graph.get_tensor_by_name("x:0"))

