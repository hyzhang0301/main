import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def get_labels(row):
    """
    转换为标签数据
    """
    one_hot_labels = pd.get_dummies(row)
    return one_hot_labels


def get_mean(row):
    """
    获取均值
    :row pandas.core.frame.DataFrame
    """
    rows= row.to_numpy()
    row_means = np.mean(rows, axis=1)

    return row_means

def get_std(row):
    """
    获取标准差
    :row pandas.core.frame.DataFrame
    """
    rows= row.to_numpy()
    row_std = np.std(rows, axis=1)
    return row_std

def max(row):
    """
    获取最大值
    """
    rows= row.to_numpy()
    rows_max = np.max(rows, axis=1)
    return rows_max

def min(row):
    """
    获取最小值
    """
    rows = row.to_numpy()
    rows_min = np.min(rows, axis=1)
    return rows_min

def labels(row):
    """
    获取标签（非one-hot向量）
    """
    rows = row.to_numpy()
    le = LabelEncoder()
    rows = le.fit_transform(rows)
    return rows







