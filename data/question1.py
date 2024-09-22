import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据并处理
def load_and_preprocess_data(filepath):
    # 读取 Excel 数据
    data = pd.read_excel(filepath)
    # 重命名列
    data.columns = ['T/oC', 'f/Hz', 'P_w/m3', 'waveform'] + [f'B(t)_{i}' for i in range(0, 1024)]
    # 波形分类为数值
    le = LabelEncoder()
    data['waveform'] = le.fit_transform(data['waveform'])
    return data

# 特征提取
def extract_features(X):
    features = pd.DataFrame()
    
    # 1. 时域统计特征
    features['mean'] = np.mean(X, axis=1)
    features['std'] = np.std(X, axis=1)
    features['max'] = np.max(X, axis=1)
    features['min'] = np.min(X, axis=1)
    features['range'] = features['max'] - features['min']
    features['skew'] = skew(X, axis=1)
    features['kurtosis'] = kurtosis(X, axis=1)
    
    # 2. 斜率特征（梯度）
    def compute_gradient(row):
        gradient = np.gradient(row)
        return np.max(gradient), np.min(gradient)
    
    gradients = np.apply_along_axis(compute_gradient, 1, X)
    features['max_gradient'] = gradients[:, 0]
    features['min_gradient'] = gradients[:, 1]
    
    # 3. 频域特征（傅里叶变换）
    fft_vals = np.abs(fft(X, axis=1))
    psd = fft_vals ** 2  # Power Spectrum Density
    features['fft_power_sum'] = np.sum(psd, axis=1)
    
    psd_norm = psd / np.sum(psd, axis=1, keepdims=True)
    features['fft_entropy'] = -np.sum(psd_norm * np.log(psd_norm + 1e-8), axis=1)
    
    features['fft_kurtosis'] = kurtosis(fft_vals, axis=1)
    features['fft_max_val'] = np.max(fft_vals, axis=1)
    features['fft_max_freq'] = np.argmax(fft_vals, axis=1)
    
    return features

# 模型评估（交叉验证 + 混淆矩阵）
def evaluate_model(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # 分割数据集并训练模型
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵
    plot_confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# 特征选择
def select_important_features(X_train, y_train, X_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    selector = SelectFromModel(clf, threshold='mean', prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Reduced number of features: {X_train_selected.shape[1]}")
    
    return X_train_selected, X_test_selected

# 主流程
def main():
    # 加载和处理数据
    filepath = 'data\附件一（训练集）.xlsx'
    data = load_and_preprocess_data(filepath)
    
    # 提取特征
    X_raw = data.iloc[:, 4:].values  # 磁通密度数据
    X = extract_features(X_raw)
    y = data['waveform']
    
    # 评估模型
    evaluate_model(X, y)

    # 特征选择
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_selected, X_test_selected = select_important_features(X_train, y_train, X_test)

    # 在特征选择后的数据上训练模型
    clf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_selected.fit(X_train_selected, y_train)
    y_pred_selected = clf_selected.predict(X_test_selected)
    
    # 输出特征选择后的模型性能
    print("Classification Report after Feature Selection:")
    print(classification_report(y_test, y_pred_selected))
    plot_confusion_matrix(y_test, y_pred_selected, labels=[0, 1, 2])

if __name__ == "__main__":
    main()
