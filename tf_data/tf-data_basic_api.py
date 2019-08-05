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
dataset = tf.data.Dataset.from_tensor_slice(np.arange(10))
print(dataset)
for item in dataset:
    print(item)
"""
dataset 上可能的操作
1. 重复遍历（epoch）
2. get batch
"""