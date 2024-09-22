import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 假设你已经有包含温度、磁通密度、频率和磁芯损耗密度的数据
# 数据格式：DataFrame形式，列为: [温度, 磁通密度, 频率, 磁芯损耗密度]
# 请加载你的数据


file_path = 'C:\Users\duanz\Desktop\数模大赛\\1.2024年中国研究生数学建模竞赛赛题\\2024年中国研究生数学建模竞赛赛题\C题\附件一（训练集）.xlsx'
data = pd.read_excel(file_path)
flux_density = data.iloc[:, 4:] #磁通密度
flux_densitymax = flux_density.max(axis=1) #磁通密度最大值
temperature = data.iloc[:, 0]# 第一列温度 (°C)
frequency = data.iloc[:, 2] # 频率 (Hz)
core_loss_density = 1e-3 * frequency * flux_densitymax**2 * (1 - 0.001 * (temperature - 20))  # 模拟的磁芯损耗密度 (W/m^3)

# 将数据整理为DataFrame
df = pd.DataFrame({
    'Temperature': temperature,
    'FluxDensitymax': flux_densitymax,
    'Frequency': frequency,
    'CoreLossDensity': core_loss_density
})

# 分离输入特征 (温度, 磁通密度, 频率) 和目标值 (磁芯损耗密度)
X = df[['Temperature', 'FluxDensitymax', 'Frequency']].values
y = df['CoreLossDensity'].values

# 数据归一化处理
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 初始化模型、损失函数和优化器
model = LossModel()
criterion = nn.MSELoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器