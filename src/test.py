import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models


from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 假设你已经有包含温度、磁通密度、频率和磁芯损耗密度的数据
# 数据格式：DataFrame形式，列为: [温度, 磁通密度, 频率, 磁芯损耗密度]
# 请加载你的数据
# df = pd.read_csv('your_data.csv')

file_path = 'C:\Users\duanz\Desktop\数模大赛\\1.2024年中国研究生数学建模竞赛赛题\\2024年中国研究生数学建模竞赛赛题\C题\附件一（训练集）.xlsx'
data = pd.read_excel(file_path)


np.random.seed(0)
data_size = 1000
temperature = np.random.uniform(20, 90, data_size)  # 温度 (°C)
flux_density = np.random.uniform(0.01, 1.0, data_size)  # 磁通密度 (T)
frequency = np.random.uniform(50000, 500000, data_size)  # 频率 (Hz)
core_loss_density = 1e-3 * frequency * flux_density**2 * (1 - 0.001 * (temperature - 20))  # 模拟的磁芯损耗密度 (W/m^3)

# 将数据整理为DataFrame
df = pd.DataFrame({
    'Temperature': temperature,
    'FluxDensity': flux_density,
    'Frequency': frequency,
    'CoreLossDensity': core_loss_density
})

# 分离输入特征 (温度, 磁通密度, 频率) 和目标值 (磁芯损耗密度)
X = df[['Temperature', 'FluxDensity', 'Frequency']].values
y = df['CoreLossDensity'].values

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 对输入特征进行标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建神经网络模型
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  # 输出一个值，磁芯损耗密度
])

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1)

# 在测试集上进行预测
y_pred = model.predict(X_test_scaled)

# 计算均方误差（MSE）
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 将预测结果与真实结果进行比较
comparison = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred.flatten()
})
print(comparison.head())

