import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 斯坦麦茨方程
def steinmetz(f, Bm, k1, alpha1, beta1):
    return k1 * (f ** alpha1) * (Bm ** beta1)

# 温度修正斯坦麦茨方程
def steinmetz_with_temp(f, Bm, T, k1, alpha1, beta1, c, T0=25):
    return k1 * (f ** alpha1) * (Bm ** beta1) * (1 + c * (T - T0))

# 数据读取与处理
data = pd.read_excel('materials_data.xlsx')  # 假设数据存储在此文件
f = data['frequency'].values
Bm = data['flux_density'].values
T = data['temperature'].values
P_real = data['loss'].values

# 1. 拟合传统斯坦麦茨方程
popt_se, _ = curve_fit(lambda f, Bm, k1, alpha1, beta1: steinmetz(f, Bm, k1, alpha1, beta1), 
                       (f, Bm), P_real)
k1, alpha1, beta1 = popt_se

# 2. 拟合温度修正斯坦麦茨方程
popt_tse, _ = curve_fit(lambda f, Bm, T, k1, alpha1, beta1, c: steinmetz_with_temp(f, Bm, T, k1, alpha1, beta1, c), 
                        (f, Bm, T), P_real)
k1, alpha1, beta1, c = popt_tse

# 3. 预测与误差分析
P_pred_se = steinmetz(f, Bm, *popt_se)  # 传统SE方程预测
P_pred_tse = steinmetz_with_temp(f, Bm, T, *popt_tse)  # 温度修正SE方程预测

error_se = np.abs(P_real - P_pred_se) / P_real * 100
error_tse = np.abs(P_real - P_pred_tse) / P_real * 100

# 4. 结果可视化
plt.figure(figsize=(10, 5))
plt.plot(T, error_se, label='传统SE方程误差')
plt.plot(T, error_tse, label='温度修正SE方程误差')
plt.xlabel('Temperature (°C)')
plt.ylabel('Error (%)')
plt.legend()
plt.show()

# 打印系数与误差
print(f"传统斯坦麦茨方程拟合系数: k1={k1}, alpha1={alpha1}, beta1={beta1}")
print(f"温度修正斯坦麦茨方程拟合系数: k1={k1}, alpha1={alpha1}, beta1={beta1}, c={c}")
