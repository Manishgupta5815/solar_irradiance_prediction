import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

# === Load dataset ===
data_path = os.path.join("Solar-Irradiance-Forecasting-using-ANNs-from-Scratch\Data\SolarPrediction.csv")
data = pd.read_csv(data_path)

# === Preprocessing ===
data = data.dropna()
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S')
data['TimeSunRise'] = pd.to_datetime(data['TimeSunRise'], format='%H:%M:%S')
data['TimeSunSet'] = pd.to_datetime(data['TimeSunSet'], format='%H:%M:%S')

data['Hour'] = data['Time'].dt.hour
data['Minute'] = data['Time'].dt.minute
data['SunRiseHour'] = data['TimeSunRise'].dt.hour
data['SunSetHour'] = data['TimeSunSet'].dt.hour

# Features and Target
X = data[['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'Hour', 'Minute', 'SunRiseHour', 'SunSetHour']]
y = data[['Radiation']]

# Scale
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Save the scalers
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# === Model ===
model = Sequential([
    Dense(64, input_dim=9, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1)

# Save model weights
model.save_weights("Optimal.weights.h5")

print("✅ Training complete. Weights and scalers saved.")
