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

#从内存构建的数据集
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
print(dataset)
for item in dataset:
    print(item)
"""
dataset 上可能的操作
1. 重复遍历（epoch）
2. get batch (batch size)
"""
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)

"""
interleave: 对训练集中的数据进行一定的映射转换
case：文件dataset -> 具体数据集
参数：map_fn 映射方法, cycle_length 并行处理数量 and block_length 变换的结果每次取多少个出来，起到了均匀混合数据集的效果
"""

dataset2 = dataset.interleave(
    lambda v: tf.data.Dataset.from_tensor_slices(v),
    cycle_length = 5,
    block_length = 5
)

x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array(['cat', 'dog', 'fox'])
dataset3 = tf.data.Dataset.from_tensor_slices((x, y))
print(dataset3)
for item_x, item_y in dataset3:
    print(item_x.numpy(), item_y.numpy())

dataset4 = tf.data.Dataset.from_tensor_slices({"feature": x, "label": y})
for item in dataset4:
    print(item['feature'].numpy(), item['label'].numpy())
