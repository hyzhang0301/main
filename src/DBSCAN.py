import pandas as pd
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录
parent_dir = os.path.dirname(current_dir)
# 添加父目录到 sys.path
sys.path.append(parent_dir)
from data.utils import get_std


# 读取 Excel 文件
file_path = '../data/附件一（训练集）.xlsx'  # 相对路径
df = pd.read_excel(file_path)
# 选择第五到第1029列（注意索引从0开始）
selected_columns = df.iloc[:, 4:]
# 显示结果
print(f"筛选特征列：\n", selected_columns)

# 计算标准差
std_values = get_std(selected_columns).reshape(-1, 1)  # 转换为二维数组

# 使用 DBSCAN 进行聚类
dbscan = DBSCAN(eps=0.01, min_samples=1100)  # 设置参数
labels = dbscan.fit_predict(std_values)

# 绘制聚类结果
plt.figure(figsize=(10, 6))
plt.scatter(labels, std_values, c=labels, cmap='viridis', s=100)
plt.title('DBSCAN Clustering: Cluster Label vs. Standard Deviation')
plt.xlabel('Cluster Label')
plt.ylabel('Standard Deviation')
plt.grid(True)
plt.colorbar(label='Cluster Label')
plt.xticks(np.unique(labels))  # 设置x轴刻度为唯一类标签
plt.show()

