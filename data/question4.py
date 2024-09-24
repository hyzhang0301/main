import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
# from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
# from lightgbm import LGBMRegressor

# Load your dataset
data = pd.read_csv('processed_train_data.csv')

# 创建标签列表
labels = ['material_1'] * 3400 + ['material_2'] * 3000 + ['material_3'] * 3200 + ['material_4'] * 2800

# 确保标签数量与 DataFrame 行数一致
if len(labels) == len(data):
   data['material'] = labels
else:
   print(f"标签数量 ({len(labels)}) 与 DataFrame 行数 ({len(data)}) 不一致")

# 计算传输磁能
data['Bm'] = np.max(data.iloc[:, 4:1028].values, axis=1)
data['transmission_energy'] = data['f/Hz'] * data['Bm']

# Define features and target
# X = data[['T/oC', 'material', 'waveform']]
X = data[[ 'waveform', 'T/oC']]

X = pd.get_dummies(X)  # One-hot encoding for 'material' and 'waveform'
print(X)
y = data['P_w/m3']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    # 'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    # 'Gradient Boosting': GradientBoostingRegressor(),
    # 'SVR': SVR(),
    # 'KNeighbors Regressor': KNeighborsRegressor(),
    # 'MLP Regressor': MLPRegressor(max_iter=500),
    # 'XGBoost': XGBRegressor(),
    # 'Polynomial Regression (degree 2)': make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    # 'LGBM': LGBMRegressor()
}

# Train models and make predictions
model_predictions = {}
model_errors = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_predictions[name] = y_pred
    model_errors[name] = (np.abs(y_test - y_pred) / y_test).mean()

print(model_errors)
# Calculate model weights (higher RMSE means lower weight)
total_error = sum(1 / np.array(list(model_errors.values())))
weights = {name: (1 / error) / total_error for name, error in model_errors.items()}

# Weighted average of predictions
final_prediction = np.zeros(len(y_test))
for name, weight in weights.items():
    final_prediction += weight * model_predictions[name]

# Calculate final RMSE of the ensemble
final_error = (np.abs(y_test - final_prediction) / y_test).mean()

print(f"Final Error of the weighted ensemble: {final_error}")
print(f"Model weights: {weights}")



from sklearn.metrics import r2_score

# 训练随机森林模型并进行预测
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)

# 计算 R² 值
r2_rf = r2_score(y_test, y_rf_pred)
print(f"Random Forest R²: {r2_rf}")

# 打印特征的重要性
importances = rf_model.feature_importances_
feature_names = X.columns

# 创建 DataFrame 以便于可视化
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)

import matplotlib.pyplot as plt

# 创建条形图
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance from Random Forest')
plt.gca().invert_yaxis()  # 反转 y 轴，使最重要特征在顶部
plt.show()

