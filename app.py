import numpy as np
import pandas as pd
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from Models.NeuralNetwork import Net
from Models.LinearRegression import LR

# Load and preprocess the data
data = pd.read_csv("Solar-Irradiance-Forecasting-using-ANNs-from-Scratch\Data\SolarPrediction.csv")
X = data.drop(["UNIXTime", "Radiation"], axis=1)
Y = pd.DataFrame(data.loc[:, "Radiation"])

# Feature Engineering
X['TSR_Minute'] = pd.to_datetime(X['TimeSunRise']).dt.minute
X['TSS_Minute'] = pd.to_datetime(X['TimeSunSet']).dt.minute
X['TSS_Hour'] = np.where(pd.to_datetime(X['TimeSunSet']).dt.hour == 18, 1, 0)

X['Month'] = pd.to_datetime(X['Data']).dt.month
X['Day'] = pd.to_datetime(X['Data']).dt.day
X['Hour'] = pd.to_datetime(X['Time']).dt.hour
X['Minute'] = pd.to_datetime(X['Time']).dt.minute
X['Second'] = pd.to_datetime(X['Time']).dt.second

X = X.drop(['Data', 'Time', 'TimeSunRise', 'TimeSunSet'], axis=1)
X['WindDirection(Degrees)_bin'] = np.digitize(X['WindDirection(Degrees)'], np.arange(0.0, 1.0, 0.02).tolist())
X['TSS_Minute_bin'] = np.digitize(X['TSS_Minute'], np.arange(0.0, 288.0, 12).tolist())
X['Humidity_bin'] = np.digitize(X['Humidity'], np.arange(32, 3192, 128).tolist())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# Initialize and Train Linear Regression model
lr = LR("Batch", 0.001, 5000)
lr.fit(X_train, y_train, X_test, y_test, "Batch")

# Predictions for Linear Regression
y_pred_lr = lr.predict(X_test)

# --- Performance Metrics for Regression ---
def regression_metrics(y_true, y_pred, model_name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\nPerformance Metrics for {model_name}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    return rmse, mse, mae, r2

# Evaluate Linear Regression
lr_rmse, lr_mse, lr_mae, lr_r2 = regression_metrics(y_test, y_pred_lr, "Linear Regression")

# Plot Loss Curve for Linear Regression
lr.plot_loss()

# Initialize and Train Neural Network
net = Net(iterations=200, learning_rate=0.001)
net.fit(X_train, y_train, X_test, y_test, optimizer="SGD", batch_size=128)

# Predictions for Neural Network
y_pred_nn = net.predict(X_test)

# Evaluate Neural Network
nn_rmse, nn_mse, nn_mae, nn_r2 = regression_metrics(y_test, y_pred_nn, "Neural Network")

# Plot Loss Curve for Neural Network
net.plot_loss()

# --- Classification Metrics for NN (if applicable) ---
if len(np.unique(y_test)) <= 10:  # If solar radiation has discrete classes
    y_pred_nn_class = np.round(y_pred_nn)  # Convert predictions to nearest integer
    y_test_class = np.round(y_test)

    # Compute Confusion Matrix
    cm = confusion_matrix(y_test_class, y_pred_nn_class)
    accuracy = accuracy_score(y_test_class, y_pred_nn_class)
    report = classification_report(y_test_class, y_pred_nn_class)

    print("\nNeural Network Classification Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", report)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_test_class), yticklabels=np.unique(y_test_class))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Print the learned weights
print("W1:", net.params["W1"])
print("W2:", net.params["W2"])
print("W3:", net.params["W3"])
print("W4:", net.params["W4"])
print("b1:", net.params["b1"])
print("b2:", net.params["b2"])
print("b3:", net.params["b3"])
print("b4:", net.params["b4"])

# Save weights to HDF5 file after training
with h5py.File("OptimalWeights.h5", 'w') as f:
    f.create_dataset("W1", data=net.params["W1"])
    f.create_dataset("W2", data=net.params["W2"])
    f.create_dataset("W3", data=net.params["W3"])
    f.create_dataset("W4", data=net.params["W4"])
    f.create_dataset("b1", data=net.params["b1"])
    f.create_dataset("b2", data=net.params["b2"])
    f.create_dataset("b3", data=net.params["b3"])
    f.create_dataset("b4", data=net.params["b4"])

# Save weights to text file after training
with open("OptimalWeights.txt", 'w') as f:
    f.write("W1: ")
    np.savetxt(f, net.params["W1"])
    f.write("\n")
    f.write("W2: ")
    np.savetxt(f, net.params["W2"])
    f.write("\n")
    f.write("W3: ")
    np.savetxt(f, net.params["W3"])
    f.write("\n")
    f.write("W4: ")
    np.savetxt(f, net.params["W4"])
    f.write("\n")
    f.write("b1: ")
    np.savetxt(f, net.params["b1"])
    f.write("\n")
    f.write("b2: ")
    np.savetxt(f, net.params["b2"])
    f.write("\n")
    f.write("b3: ")
    np.savetxt(f, net.params["b3"])
    f.write("\n")
    f.write("b4: ")
    np.savetxt(f, net.params["b4"])
    f.write("\n")