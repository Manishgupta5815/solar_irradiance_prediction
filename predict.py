import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Input

# Define the Neural Network class
class NeuralNetwork:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(Input(shape=(input_shape,)))  # Use Input layer
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1))  # Output layer for regression
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, X_train, y_train, epochs=100, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

# Sample input data
data = {
    "Data": ["10/11/2024 1:00:00 PM"],
    "Time": ["10:18:00"],
    "Temperature": [81],
    "Pressure": [29.95],
    "Humidity": [66],
    "WindDirection (Degrees)": [0],
    "Speed": [6],
    "TimeSunRise": ["05:46:00"],
    "TimeSunSet": ["16:54:00"]
}

# Create DataFrame
X = pd.DataFrame(data)

# Convert time columns to datetime
X['TimeSunRise'] = pd.to_datetime(X['TimeSunRise'])
X['TimeSunSet'] = pd.to_datetime(X['TimeSunSet'])
X['Data'] = pd.to_datetime(X['Data'])

# Extract features
X['TSR_Minute'] = X['TimeSunRise'].dt.minute
X['TSS_Minute'] = X['TimeSunSet'].dt.minute
X['TSS_Hour'] = np.where(X['TimeSunSet'].dt.hour - 18 > 0, 1, 0)
X['Month'] = X['Data'].dt.month
X['Day'] = X['Data'].dt.day
X['Hour'] = X['Time'].apply(lambda x: pd.to_datetime(x).hour)
X['Minute'] = X['Time'].apply(lambda x: pd.to_datetime(x).minute)
X['Second'] = X['Time'].apply(lambda x: pd.to_datetime(x).second)

# Drop unnecessary columns
X.drop(['Data', 'Time', 'TimeSunRise', 'TimeSunSet'], axis=1, inplace=True)

# Bin the wind direction
X['WindDirection_bin'] = np.digitize(X['WindDirection (Degrees)'], np.arange(0.0, 360.0, 20).tolist())
X['TSS_Minute_bin'] = np.digitize(X['TSS_Minute'], np.arange(0.0, 288.0, 12).tolist())
X['Humidity_bin'] = np.digitize(X['Humidity'], np.arange(32, 100, 10).tolist())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a synthetic target variable (solar irradiance)
y_train = np.array([200])  # target value for the single input sample

# Instantiate the neural network
input_shape = X_scaled.shape[1]
net1 = NeuralNetwork(input_shape)

# Load pre-trained weights
try:
    net1.load_weights('OptimalWeights.txt') 
    print("Loaded pre-trained weights.")
except Exception as e:
    print("Could not load weights:", e)

    net1.train(X_scaled, y_train, epochs=100, batch_size=1)

# Make predictions
predictions = net1.predict(X_scaled)
predicted_irradiance = predictions.squeeze()

print(f"Predicted Solar Irradiance: {predicted_irradiance}")