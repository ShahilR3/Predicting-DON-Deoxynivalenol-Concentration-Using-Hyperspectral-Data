#In[]

# Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pickle
from tensorflow import keras
from tensorflow.keras import layers, regularizers #type:ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances, plot_contour
import requests
#In[]

# Load Dataset

data = pd.read_csv('C:\\Users\\shahi\\OneDrive\\Desktop\\App\\MLE-Assignment.csv')
df = pd.DataFrame(data)

#In[]

#Finding Null Value

df.isnull().sum()

#In[]

#Finding duplicate value

df.duplicated()

#In[]

# Encode Categorical Variables
le = LabelEncoder()
df['hsi_id'] = le.fit_transform(df['hsi_id'])

#In[]

df.columns

#In[]

# Automate Sensor Drift Check
def check_sensor_drift(df, threshold=0.1):
    drift = df.iloc[:, 1:].mean().diff().abs()
    drift_detected = (drift > threshold).sum()
    print(f"Number of Wavelengths with Significant Drift: {drift_detected}")
    return drift_detected

drift_count = check_sensor_drift(df)
print(df.columns)

#In[]

# Split Data into Training and Testing Sets
X = df.iloc[:, 1:-1].values
y = df["vomitoxin_ppb"].values  # Target variable

#In[]

#Creating a pickle for Scaler Function

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# In[]

# Define Neural Network Model

model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)  # Output layer (regression)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])

# Train the Model
history = model.fit(
    X_train, y_train, epochs=100, batch_size=32,
    validation_split=0.2,  verbose=1
)

# In[]

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute Regression Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print Regression Metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# In[]

# Hyperparameter Optimization with Optuna
def objective(trial):
    # Define the hyperparameter search space
    units_1 = trial.suggest_int("units_1", 32, 256)
    units_2 = trial.suggest_int("units_2", 16, 128)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)  # Search dropout rates
    
    # Define the Model
    model = keras.Sequential([
        layers.Dense(units_1, activation="relu", input_shape=(X_train.shape[1],),
                     kernel_regularizer=regularizers.l2(0.01)),  # L2 Regularization
        layers.Dropout(dropout_rate),  # Dropout layer
        layers.Dense(units_2, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(dropout_rate),
        layers.Dense(1)  # Output Layer
    ])
    
    # Compile the Model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    
    # Train the Model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    
    # Return Validation Mean Absolute Error
    val_mae = np.min(history.history["val_mae"])
    return val_mae

# Run Optuna Optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

# Print Best Hyperparameters
print("Best hyperparameters:", study.best_params)

# In[]

# Visualize the Optimization Results
plot_optimization_history(study)

plot_param_importances(study)

plot_contour(study, params=["units_1", "learning_rate"])

# In[]

# Rebuild and Retrain the Final Model with Best Parameters
best_params = study.best_params

model = keras.Sequential([
    layers.Dense(best_params['units_1'], activation="relu", input_shape=(X_train.shape[1],),
                 kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(best_params['dropout_rate']),
    layers.Dense(best_params['units_2'], activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(best_params['dropout_rate']),
    layers.Dense(1)  # Output Layer
])

# Compile the Model with Best Learning Rate
model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_params['learning_rate']), loss="mse", metrics=["mae"])

# Train the Model
history = model.fit(
    X_train, y_train, epochs=100, batch_size=32,
    validation_split=0.2, verbose=1
)

# Save the Model
model.save("optimized_don_model.h5")
print("Optimized model saved as 'optimized_don_model.h5'")

# In[]

# Evaluate the Model on the Test Set
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print Final Evaluation Metrics
print(f"Final Model Test Set MAE: {mae}")
print(f"Final Model Test Set RMSE: {rmse}")
print(f"Final Model Test Set R² Score: {r2}")