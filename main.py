import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from Models.LinearRegression import LR
from Models.NeuralNetwork import NeuralNetwork

# -----------------------------
# STEP 1: Load and Inspect Data
# -----------------------------
print("🌞 Solar Irradiance Forecasting Project")

# Load dataset
data = pd.read_csv("Data/SolarPrediction.csv")

print("\n✅ Data Loaded Successfully!")
print(f"Shape of dataset: {data.shape}")
print("Columns:", list(data.columns))

# -----------------------------
# STEP 2: Preprocessing
# -----------------------------
features = ["Temperature", "Pressure", "Humidity", "WindDirection(Degrees)", "Speed"]
target = ["Radiation"]

X = data[features].values
y = data[target].values.flatten()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n📊 Data split complete:")
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# -----------------------------
# STEP 3: Train Linear Regression Model
# -----------------------------
print("\n🚀 Training Linear Regression Model...")
lr_model = LR(learning_rate=0.01, iterations=6000)
lr_model.fit(X_train, y_train, X_test, y_test)

# Evaluate model
lr_rmse, lr_mse, lr_mae = lr_model.evaluate(X_test, y_test)
lr_preds = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_preds)

print("\n📈 Linear Regression Performance:")
print(f"RMSE: {lr_rmse:.4f}")
print(f"MSE:  {lr_mse:.4f}")
print(f"MAE:  {lr_mae:.4f}")
print(f"R²:   {lr_r2:.4f}")

# Plot LR losses
lr_model.plot_loss()

# -----------------------------
# STEP 4: Train Neural Network Model
# -----------------------------
print("\n🤖 Training Neural Network Model...")
nn_model = NeuralNetwork(
    input_size=X_train.shape[1],
    hidden_size=16,
    learning_rate=0.01,
    iterations=1000
)
nn_model.fit(X_train, y_train, X_test, y_test)

# Evaluate model
nn_rmse, nn_mse, nn_mae = nn_model.evaluate(X_test, y_test)
nn_preds = nn_model.predict(X_test)
nn_r2 = r2_score(y_test, nn_preds)

print("\n🤖 Neural Network Performance:")
print(f"RMSE: {nn_rmse:.4f}")
print(f"MSE:  {nn_mse:.4f}")
print(f"MAE:  {nn_mae:.4f}")
print(f"R²:   {nn_r2:.4f}")

# Plot NN losses
nn_model.plot_loss()

# -----------------------------
# STEP 5: Compare Models
# -----------------------------
models = ["Linear Regression", "Neural Network"]
rmse_values = [lr_rmse, nn_rmse]
mae_values = [lr_mae, nn_mae]
r2_values = [lr_r2, nn_r2]

plt.figure(figsize=(9, 5))
bar_width = 0.25
x = np.arange(len(models))

plt.bar(x - bar_width, rmse_values, bar_width, label="RMSE")
plt.bar(x, mae_values, bar_width, label="MAE")
plt.bar(x + bar_width, r2_values, bar_width, label="R²")
plt.xticks(x, models)
plt.ylabel("Metric Value")
plt.title("Model Comparison: RMSE, MAE, R²")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# STEP 6: Save Summary
# -----------------------------
summary = pd.DataFrame({
    "Model": models,
    "RMSE": rmse_values,
    "MAE": mae_values,
    "R2": r2_values
})
summary.to_csv("model_comparison_summary.csv", index=False)

print("\n💾 Model comparison summary saved as 'model_comparison_summary.csv'")
print("\n✅ All tasks completed successfully!")
