import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pyswarm import pso

# 斯坦麦茨方程模型
def steinmetz(f_Bm, k1, alpha1, beta1):
    f, Bm = f_Bm
    return k1 * (f ** alpha1) * (Bm ** beta1)

def steinmetz_with_temp(f_Bm_T, k1, alpha1, beta1, c, T0=25):
    f, Bm, T = f_Bm_T
    return k1 * (f ** alpha1) * (Bm ** beta1) * (c * (T - T0))

def steinmetz_with_nonlinear_temp(f_Bm_T, k1, alpha1, beta1, c1, c2, T0=25):
    f, Bm, T = f_Bm_T
    # return k1 * (f ** alpha1) * (Bm ** beta1) * (c1 * (T / T0) ** c2)
    return k1 * (f ** alpha1) * (Bm ** beta1) * (c1 * (T / T0) ** c2 )

# 自定义 PSO 类
class EnhancedPSO:
    def __init__(self, func, lb, ub, args, swarmsize=50, maxiter=100,
                 inertia_weight=0.9, inertia_damp=0.99, c1=2.0, c2=2.0, seed=None):
        self.func = func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.args = args
        self.swarmsize = swarmsize
        self.maxiter = maxiter
        self.inertia_weight = inertia_weight
        self.inertia_damp = inertia_damp
        self.c1 = c1
        self.c2 = c2
        self.history = []  # 记录每次迭代的全局最佳值
        self.best_positions = []  # 记录每次迭代的全局最佳位置

    def optimize(self):
        # np.random.seed(self.seed)
        dim = len(self.lb)
        swarm = np.random.uniform(self.lb, self.ub, (self.swarmsize, dim))
        velocity = np.zeros((self.swarmsize, dim))
        personal_best = np.copy(swarm)
        personal_best_value = np.array([self.func(p, *self.args) for p in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        for i in range(self.maxiter):
            for j in range(self.swarmsize):
                # 更新速度
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                velocity[j] = (self.inertia_weight * velocity[j] +
                               self.c1 * r1 * (personal_best[j] - swarm[j]) +
                               self.c2 * r2 * (global_best - swarm[j]))

                # 更新位置
                swarm[j] += velocity[j]
                swarm[j] = np.clip(swarm[j], self.lb, self.ub)

                # 计算新的目标函数值
                current_value = self.func(swarm[j], *self.args)

                # 更新个体最佳位置
                if current_value < personal_best_value[j]:
                    personal_best[j] = swarm[j]
                    personal_best_value[j] = current_value

                # 更新全局最佳位置
                if current_value < global_best_value:
                    global_best = swarm[j]
                    global_best_value = current_value

            # 动态调整惯性权重
            self.inertia_weight *= self.inertia_damp

            # 记录每次迭代的全局最佳值和位置
            self.history.append(global_best_value)
            self.best_positions.append(global_best)
            
            if i % 500 == 0:
              print(f"Iteration {i+1}/{self.maxiter}, Global Best Value: {global_best_value:.6f}")

        return global_best, global_best_value

# 目标函数，计算误差（使用向量化）
def objective_func(params, func, f, Bm, T, P_real):
    if T is not None:
        f_Bm_T = np.vstack((f, Bm, T)).T
    else:
        f_Bm_T = np.vstack((f, Bm)).T

    # 批量计算预测值
    P_pred = func(f_Bm_T.T, *params)

    # 计算误差并返回平均值
    return np.mean(np.abs(P_real - P_pred) / P_real)

# PSO 优化过程
def pso_optimize(func, lb, ub, f, Bm, T, P_real, swarmsize=100, maxiter=2000):
    # 确定是否传递 T
    if len(lb) == 3:
        T = None  # 对于传统斯坦麦茨方程，T 为 None
    pso_optimizer = EnhancedPSO(objective_func, lb, ub, args=(func, f, Bm, T, P_real), swarmsize=swarmsize, maxiter=maxiter)
    best_solution, best_value = pso_optimizer.optimize()
    return best_solution, pso_optimizer.history  # 返回最优解和优化历史

# 拟合三种斯坦麦茨方程
def fit_steinmetz_models(f, Bm, T, P_real):
    bounds = (-10, 10)
    models = {
        'steinmetz': (steinmetz, [bounds[0]] * 3, [bounds[1]] * 3),
        'steinmetz_with_temp': (steinmetz_with_temp, [bounds[0]] * 4, [bounds[1]] * 4),
        'steinmetz_with_nonlinear_temp': (steinmetz_with_nonlinear_temp, [bounds[0]] * 5, [bounds[1]] * 5)
    }

    popt = {}
    history_dict = {}
    function = {}
    for model_name, (model_func, lb, ub) in models.items():
        initial_guess, history = pso_optimize(model_func, lb, ub, f, Bm, T, P_real)
        popt[model_name] = initial_guess
        history_dict[model_name] = history
        function[model_name] = model_func

    return popt, history_dict, function

# 可视化 PSO 优化历史
def plot_optimization_history(f, Bm, T, P_real, popt, history_dict, function):
    plt.figure(figsize=(10, 6))
    for model_name, history in history_dict.items():
        print(f"{model_name} 的最优平均误差: {min(history):.4f}")
        iterations = list(range(1, len(history) + 1))
        plt.plot(iterations, history, label=f'{model_name} history')

    plt.xlabel('Iteration')
    plt.ylabel('Global Best Value')
    plt.title('PSO Optimization History')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    for model_name, position in popt.items():
        print(f"{model_name} 的最优拟合参数: {position}")
        func = function[model_name]
        # 确定是否传递 T
        if model_name == 'steinmetz':
            f_Bm_T = np.vstack((f, Bm)).T
        else:
            f_Bm_T = np.vstack((f, Bm, T)).T

        # 批量计算预测值
        P_pred = func(f_Bm_T.T, *position)

        # 计算误差并返回平均值
        errors = np.abs(P_real - P_pred) / P_real
    
        plt.plot(T, errors, label=f"Error of the {model_name} equation")

    plt.xlabel('Temperature (°C)')
    plt.ylabel('Error (%)')
    plt.legend()
    plt.title('Prediction Errors for Different Models')
    plt.grid()
    plt.show()

    # 绘制拟合结果与真实数据
    plt.figure(figsize=(10, 5))
    for model_name, position in popt.items():
        func = function[model_name]
        if model_name == 'steinmetz':
            f_Bm_T = np.vstack((f, Bm)).T
        else:
            f_Bm_T = np.vstack((f, Bm, T)).T
        P_pred = func(f_Bm_T.T, *position)
        errors = np.abs(P_real - P_pred) / P_real
    
        plt.plot(errors, label=f"Error of the {model_name} equation")

    # plt.plot(P_real, label="Real Core Loss")
    plt.xlabel('Operating conditions')
    plt.ylabel('Error Scaled Density')
    plt.legend()
    plt.title('Height Distribution of Core Loss Density Prediction Error')
    plt.grid()
    plt.show()

# 主程序调用示例
if __name__ == "__main__":
    # 数据加载
    # data = pd.read_excel('/content/drive/MyDrive/loss modelling of magnetic components/train_data.xlsx')
    # data.columns = ['T/oC', 'f/Hz', 'P_w/m3', 'waveform'] + [f'B(t)_{i}' for i in range(1024)]
    # data = data[data['waveform'] == '正弦波']
    # f = data['f/Hz'].values
    # Bm = np.max(data.iloc[:, 4:].values, axis=1)
    # T = data['T/oC'].values
    # P_real = data['P_w/m3'].values

    # 拟合模型
    popt, history_dict, function = fit_steinmetz_models(f, Bm, T, P_real)

    # 查看优化历史
    plot_optimization_history(f, Bm, T, P_real, popt, history_dict, function)
