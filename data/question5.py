import numpy as np
import pandas as pd
import pickle
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Load your dataset
data = pd.read_csv('/content/drive/MyDrive/loss modelling of magnetic components/processed_train_data.csv')
labels = ['material_1'] * 3400 + ['material_2'] * 3000 + ['material_3'] * 3200 + ['material_4'] * 2800
data['material'] = labels
data['Bm'] = np.max(data.iloc[:, 4:1028].values, axis=1)
data['transmission_energy'] = data['f/Hz'] * data['Bm']

# Define features and target
X = data[['f/Hz', 'Bm', 'T/oC', 'material', 'waveform']]
X = pd.get_dummies(X)  # One-hot encoding for 'material' and 'waveform'
y = data['P_w/m3']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'LGBM': LGBMRegressor()
}

# Train models and collect predictions
model_predictions = {}
model_errors = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_predictions[name] = y_pred
    model_errors[name] = (np.abs(y_test - y_pred) / y_test).mean()

# Calculate weights for ensemble
total_error = sum(1 / np.array(list(model_errors.values())))
weights = {name: (1 / error) / total_error for name, error in model_errors.items()}

# Weighted average of predictions
final_prediction = sum(weights[name] * model_predictions[name] for name in weights)
final_error = (np.abs(y_test - final_prediction) / y_test).mean()

# Save the models
for name, model in models.items():
    with open(f'{name}_model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Define the objective function for optimization
def objective_function(params):
    f, Bm, T_index, material_index, waveform_index = params
    T_values = [25, 50, 70, 90]
    material_values = ['material_1', 'material_2', 'material_3', 'material_4']
    waveform_values = [1, 2, 3]
    
    T = T_values[int(T_index)]
    material = material_values[int(material_index)]
    waveform = waveform_values[int(waveform_index)]
    
    # Prepare input for prediction
    X_input = pd.DataFrame([[f, Bm, T, material, waveform]], 
                            columns=['f/Hz', 'Bm', 'T/oC', 'material', 'waveform'])
    X_input = pd.get_dummies(X_input, columns=['material', 'waveform'], drop_first=True)
    X_input = X_input.reindex(columns=X_train.columns, fill_value=0)  # Ensure matching columns
    
    # Calculate final prediction
    final_prediction = sum(weights[name] * model.predict(X_input)[0] for name in weights)
    transmission_energy = f * Bm
    return abs(final_prediction - transmission_energy)

# Define bounds for the optimization
bounds = [
    (32000, 50000),        # f (frequency)
    (0.012, 0.26),         # Bm (flux density)
    (0, 3),                # T (temperature index for [25, 50, 70, 90])
    (0, 3),                # material index for ['material_1', 'material_2', 'material_3', 'material_4']
    (0, 2)                 # waveform
]

# Run the optimization
result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=1000, tol=1e-6)
optimized_params = result.x

# Decode optimized categorical values
optimized_T = [25, 50, 70, 90][int(optimized_params[2])]
optimized_material = ['material_1', 'material_2', 'material_3', 'material_4'][int(optimized_params[3])]
optimized_f = optimized_params[0]
optimized_Bm = optimized_params[1]
optimized_waveform = [1, 2, 3][int(optimized_params[4])]

print(f'Optimized f: {optimized_f}, Optimized Bm: {optimized_Bm}, Optimized T: {optimized_T}, Optimized Material: {optimized_material}, Optimized Waveform: {optimized_waveform}')
