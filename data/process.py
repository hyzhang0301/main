import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.preprocessing import LabelEncoder

def process():
    """
    数据预处理代码逻辑
    """
    pass

def extract_features(X):
    features = pd.DataFrame()
    
    # 统计特征
    features['mean'] = np.mean(X, axis=1)
    features['std'] = np.std(X, axis=1)
    features['max'] = np.max(X, axis=1)
    features['min'] = np.min(X, axis=1)
    features['skew'] = X.apply(lambda row: skew(row), axis=1)
    features['kurtosis'] = X.apply(lambda row: kurtosis(row), axis=1)
    
    # 频域特征（傅里叶变换）
    def compute_fft_features(row):
        fft_vals = np.abs(fft(row))  # 计算傅里叶变换的绝对值
        fft_power = fft_vals ** 2  # 计算频谱的功率
        return pd.Series({
            'fft_max_freq': fft_vals.argmax(),  # 最大频率
            'fft_max_val': fft_vals.max(),  # 最大频率幅值
            'fft_power_sum': np.sum(fft_power)  # 频谱能量
        })

    # Apply FFT features
    fft_features = X.apply(compute_fft_features, axis=1)
    features = pd.concat([features, fft_features], axis=1)
    
    return features