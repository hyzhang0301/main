import pandas as pd
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


def get_columns(file_path, star_coloumn_id):
    """
    返回筛选的列
    :file_path 使用文件相对路径
    """
    # file_path = '附件一（训练集）.xlsx' 
    df1 = pd.read_excel(file_path, sheet_name="材料1")
    selected_columns1 = df1.iloc[:, star_coloumn_id:]

    df2 = pd.read_excel(file_path, sheet_name="材料2")
    selected_columns2 = df2.iloc[:, star_coloumn_id:]

    df3 = pd.read_excel(file_path, sheet_name="材料3")
    selected_columns3 = df3.iloc[:, star_coloumn_id:]

    df4 = pd.read_excel(file_path, sheet_name="材料4")
    selected_columns4 = df4.iloc[:, star_coloumn_id:]

    return selected_columns1, selected_columns2, selected_columns3, selected_columns4
    
def get_labels(row):
    """
    转换为标签数据 one-hot向量形式
    """
    one_hot_labels = pd.get_dummies(row)
    return one_hot_labels

def get_onehot_labels(file_path, column_name, name):
    """
    输入文件路径，返回onehot形式标签
    """
    df = pd.read_excel(file_path, sheet_name=name)
    labels = df[column_name]
    onehot_labels = get_labels(labels)
    return onehot_labels

def get_predict_clolumns(file_path):

    # 读取数据
    df = pd.read_excel(file_path)

    # 筛选第四列为‘材料1’的行x
    filtered_df = df[df.iloc[:, 3] == '材料1']
    # 选择第五列及其后面的所有列
    result1 = filtered_df.iloc[:, 4:]

    filtered_df = df[df.iloc[:, 3] == '材料2']
    # 选择第五列及其后面的所有列
    result2 = filtered_df.iloc[:, 4:]

    filtered_df = df[df.iloc[:, 3] == '材料3']
    # 选择第五列及其后面的所有列
    result3 = filtered_df.iloc[:, 4:]

    filtered_df = df[df.iloc[:, 3] == '材料4']
    # 选择第五列及其后面的所有列
    result4 = filtered_df.iloc[:, 4:]

    return result1, result2, result3, result4



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

def peak_to_peak(row):
    """
    返回峰峰值
    """
    return max(row) - min(row)

def labels(row):
    """
    获取标签（非one-hot向量）
    """
    rows = row.to_numpy()
    le = LabelEncoder()
    rows = le.fit_transform(rows)
    return rows

def read_column_from_excel(file_path, column_name, sheet_name):
    """
    读取单列
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df[column_name]

class RandomForest():
    """
    随机森林
    """
    def __init__(self, train_file_path, predict_file_path, train_start_column_id, predict_start_column_id):
        self.train_file_path = train_file_path
        self.predict_file_path = predict_file_path

        self.train_start_column_id = train_start_column_id
        self.predict_start_column_id = predict_start_column_id

    def train(self, selected_columns, name):
        train_file_path = self.train_file_path
        start_column_id = self.train_start_column_id
        
        # 获取特征
        mean = get_mean(selected_columns)
        std = get_std(selected_columns)
        max_values = max(selected_columns)
        min_values = min(selected_columns)

        # 训练标签
        Y = get_onehot_labels(train_file_path, name)
        X = np.column_stack((max_values, min_values, std))

        # 拆分数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        # 训练模型
        rf.fit(X_train, y_train)

        # 评估模型
        accuracy = rf.score(X_test, y_test)
        print(f"附件一测试集模型准确率: {accuracy:.2f}")
        y_pred = rf.predict(X_test)
        print(classification_report(y_test, y_pred))
        return rf
    
    def predict(self, train_selected_columns, predict_columns, name):
        rf = self.train(train_selected_columns, name)
        # 获取特征
        mean = get_mean(predict_columns)
        std = get_std(predict_columns)
        max_values = max(predict_columns)
        min_values = min(predict_columns)

        X = np.column_stack((max_values, min_values, std))
        y_pred = rf.predict(X)

        return y_pred
        # 预测
    
class MultiLogisticRegression:
    """
    多元逻辑回归
    """
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        """拟合模型"""
        self.model.fit(X, y)

    def predict(self, X):
        """进行预测"""
        return self.model.predict(X)

    def score(self, X, y):
        """评估模型准确率"""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
class MultiLinearRegression:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        """拟合模型"""
        self.model.fit(X, y)

    def predict(self, X):
        """进行预测"""
        return self.model.predict(X)

    def score(self, X, y):
        """评估模型的 R² 得分"""
        return self.model.score(X, y)


class RandomForestRegressorModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None

    def train(self, X_train, y_train):
        """训练随机森林回归模型"""
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """进行预测"""
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        """评估模型性能"""
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return r2, mse


class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=self.degree)
        self.model = LinearRegression()
        
    def fit(self, X, y):
        # 生成多项式特征
        X_poly = self.poly_features.fit_transform(X)
        # 拟合模型
        self.model.fit(X_poly, y)
        
    def predict(self, X):
        # 生成多项式特征
        X_poly = self.poly_features.transform(X)
        # 进行预测
        return self.model.predict(X_poly)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, r2

