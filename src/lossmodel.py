import torch
import torch.nn as nn
import torch.optim as optim

# 定义神经网络模型
class LossModel(nn.Module):
    def __init__(self):
        super(LossModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 输入层：3个输入特征（温度、频率、磁通密度）
        self.fc2 = nn.Linear(64, 64)  # 隐藏层
        self.fc3 = nn.Linear(64, 1)   # 输出层：1个输出（磁芯损耗密度）

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    