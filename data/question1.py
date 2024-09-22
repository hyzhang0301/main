# @title
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from scipy.stats import skew, kurtosis
from scipy.fft import fft

# 数据处理与加载
def rename_columns(data, is_test=False):
    """重命名训练集或测试集的列"""
    if not is_test:
        columns = ['T/oC', 'f/Hz', 'P_w/m3', 'waveform'] + [f'B(t)_{i}' for i in range(1024)]
    else:
        columns = ['serial number', 'T/oC', 'f/Hz', 'Core material'] + [f'B(t)_{i}' for i in range(1024)]
    
    data.columns = columns
    return data

# 读取并处理数据
def load_and_preprocess_data(filepath, is_test=False):
    """根据是否为测试集加载并处理数据"""
    if not is_test:
        sheet_names = ['材料1', '材料2', '材料3', '材料4']
        data_frames = [rename_columns(pd.read_excel(filepath, sheet_name=sheet)) for sheet in sheet_names]
        data = pd.concat(data_frames, axis=0, ignore_index=True)
        print(data.info())
        # 波形分类为数值
        le = LabelEncoder()
        data['waveform'] = le.fit_transform(data['waveform'])

        # 查看类别与数值映射
        label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
        print("类别到数值的映射:", label_mapping)

        # # 如果需要反向映射
        # reverse_mapping = dict(zip(range(len(le.classes_)), le.classes_))
        # print("数值到类别的映射:", reverse_mapping)

    else:
        data = rename_columns(pd.read_excel(filepath), is_test=True)
    
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

# 特征选择
def select_important_features(X_train, y_train, X_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    selector = SelectFromModel(clf, threshold='mean', prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Reduced number of features: {X_train_selected.shape[1]}")
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    print("Selected features: ", selected_features)

    return X_train_selected, X_test_selected, selector

# 预测并输出结果
def predict_on_test_data(test_filepath, selector, clf_selected):
    test_data = load_and_preprocess_data(test_filepath, is_test=True)
    X_test_raw = test_data.iloc[:, 4:].values
    X_test_features = extract_features(X_test_raw)
    X_test_selected = selector.transform(X_test_features)
    
    test_data['waveform_prediction'] = clf_selected.predict(X_test_selected)
    test_data[['serial number', 'waveform_prediction']].to_csv('test_predictions.csv', index=False)
    print(test_data[['serial number', 'waveform_prediction']].head())

# 主流程
def main():
    # 加载和处理训练集数据
    train_filepath = '/content/drive/MyDrive/loss modelling of magnetic components/train_data.xlsx'
    test_filepath = '/content/drive/MyDrive/loss modelling of magnetic components/test1.xlsx'
    data = load_and_preprocess_data(train_filepath)
    
    # 提取特征
    X_raw = data.iloc[:, 4:].values  # 磁通密度数据
    X = extract_features(X_raw)
    y = data['waveform']
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 特征选择
    X_train_selected, X_test_selected, selector = select_important_features(X_train, y_train, X_test)

    # 在特征选择后的数据上训练模型
    clf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_selected.fit(X_train_selected, y_train)
    y_pred_selected = clf_selected.predict(X_test_selected)
    
    # 输出特征选择后的模型性能
    print("Classification Report after Feature Selection:")
    print(classification_report(y_test, y_pred_selected))

    # 测试集预测
    predict_on_test_data(test_filepath, selector, clf_selected)

if __name__ == "__main__":
    main()
