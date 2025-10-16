import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Load the full dataset for Solar Irradiance (make sure this file path is correct)
irradiance_df = pd.read_csv('Solar-Irradiance-Forecasting-using-ANNs-from-Scratch\Data\SolarPrediction.csv')

# Clean column names to remove any extra spaces
irradiance_df.columns = irradiance_df.columns.str.strip()

# Check if the necessary columns exist in the dataset
required_columns = ['Data', 'Time', 'TimeSunRise', 'TimeSunSet', 'Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'Radiation']
missing_columns = [col for col in required_columns if col not in irradiance_df.columns]

if missing_columns:
    raise ValueError(f"The following required columns are missing: {', '.join(missing_columns)}")

# Preprocess Date-Time columns (Data, Time, TimeSunRise, TimeSunSet)
irradiance_df['Data'] = pd.to_datetime(irradiance_df['Data'])
irradiance_df['Time'] = pd.to_datetime(irradiance_df['Time'], format='%H:%M:%S')
irradiance_df['TimeSunRise'] = pd.to_datetime(irradiance_df['TimeSunRise'], format='%H:%M:%S')
irradiance_df['TimeSunSet'] = pd.to_datetime(irradiance_df['TimeSunSet'], format='%H:%M:%S')

# Feature Engineering (Extract useful features)
irradiance_df['Hour'] = irradiance_df['Time'].dt.hour
irradiance_df['Minute'] = irradiance_df['Time'].dt.minute
irradiance_df['SunRiseHour'] = irradiance_df['TimeSunRise'].dt.hour
irradiance_df['SunSetHour'] = irradiance_df['TimeSunSet'].dt.hour

# Drop original datetime columns if not needed
features = irradiance_df[['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'Hour', 'Minute', 'SunRiseHour', 'SunSetHour']]
target = irradiance_df['Radiation'].values  # Extract the target column (Solar Radiation)

# Scaling features and target
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(features)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(target.reshape(-1, 1))

# Define the model
net1 = Sequential([
    Dense(64, input_dim=X_scaled.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
net1.compile(optimizer='adam', loss='mean_squared_error')

# Try to load pre-trained weights
try:
    net1.load_weights('OptimalWeights.weights.h5')  # .h5 format
    print("Loaded pre-trained weights.")
except Exception as e:
    print("Could not load weights:", e)
    # Train the model with the entire dataset
    net1.fit(X_scaled, y_scaled, epochs=200, batch_size=32)
    net1.save_weights('OptimalWeights2.weights.h5')  # Save the trained weights

# Sample input for prediction (replicate the structure of your input)
sample_input = {
    "Temperature": [25],
    "Pressure": [25.95],
    "Humidity": [95],
    "WindDirection(Degrees)": [0],
    "Speed": [40],
    "Time": ["10:18:00"],
    "TimeSunRise": ["05:46:00"],
    "TimeSunSet": ["16:54:00"]
}

# Convert the sample input to a DataFrame
sample_df = pd.DataFrame(sample_input)

# Preprocess sample input
sample_df['Time'] = pd.to_datetime(sample_df['Time'], format='%H:%M:%S')
sample_df['TimeSunRise'] = pd.to_datetime(sample_df['TimeSunRise'], format='%H:%M:%S')
sample_df['TimeSunSet'] = pd.to_datetime(sample_df['TimeSunSet'], format='%H:%M:%S')
sample_df['Hour'] = sample_df['Time'].dt.hour
sample_df['Minute'] = sample_df['Time'].dt.minute
sample_df['SunRiseHour'] = sample_df['TimeSunRise'].dt.hour
sample_df['SunSetHour'] = sample_df['TimeSunSet'].dt.hour

# Drop original datetime columns
sample_df = sample_df.drop(columns=['Time', 'TimeSunRise', 'TimeSunSet'])

# Scale the input sample using the same scaler used for training
sample_input_scaled = scaler_X.transform(sample_df)

# Use the trained model to make predictions for the sample input
prediction_scaled = net1.predict(sample_input_scaled)

# Convert the scaled prediction back to the original scale of y
prediction = scaler_y.inverse_transform(prediction_scaled)

# Print the prediction
print(f"Predicted Solar Irradiance: {prediction[0][0]}")