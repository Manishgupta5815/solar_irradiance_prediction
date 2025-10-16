from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import h5py
from Models.NeuralNetwork import Net  # import your Net class

# ----------------------------
# Initialize Flask app
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Set seed for reproducibility
# ----------------------------
np.random.seed(42)

# ----------------------------
# Load saved scaler
# ----------------------------
with open("scaler_X.pkl", "rb") as f:
    scaler_X = pickle.load(f)

# ----------------------------
# Load trained model
# ----------------------------
# Make sure to match the architecture used during training
net = Net(layers=[16, 64, 48, 32, 1])
net.params = {}  # initialize params dict

# Load weights
with h5py.File("NN_Weights.h5", "r") as f:
    for key in f.keys():
        net.params[key] = np.array(f[key])

# ----------------------------
# Helper function: preprocess input
# ----------------------------
def preprocess_input(form):
    """
    Convert raw form data into feature vector matching the trained model.
    Performs same feature engineering as training phase.
    """
    try:
        # Raw user input
        date = form["Date"]
        temp = float(form["Temperature"])
        pressure = float(form["Pressure"])
        humidity = float(form["Humidity"])
        wind_dir = float(form["WindDirection"])
        wind_speed = float(form["Speed"])
        time = form["Time"]
        sunrise = form["TimeSunRise"]
        sunset = form["TimeSunSet"]

        # Feature engineering
        df = pd.DataFrame([{
            "Temperature": temp,
            "Pressure": pressure,
            "Humidity": humidity,
            "WindDirection(Degrees)": wind_dir,
            "Speed": wind_speed,
            "Data": date,
            "Time": time,
            "TimeSunRise": sunrise,
            "TimeSunSet": sunset
        }])

        df["TSR_Minute"] = pd.to_datetime(df["TimeSunRise"], errors="coerce").dt.minute
        df["TSS_Minute"] = pd.to_datetime(df["TimeSunSet"], errors="coerce").dt.minute
        df["TSS_Hour"] = np.where(
            pd.to_datetime(df["TimeSunSet"], errors="coerce").dt.hour == 18, 1, 0
        )
        df["Month"] = pd.to_datetime(df["Data"], errors="coerce").dt.month
        df["Day"] = pd.to_datetime(df["Data"], errors="coerce").dt.day
        df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour
        df["Minute"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.minute
        df["Second"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.second

        df = df.drop(["Data", "Time", "TimeSunRise", "TimeSunSet"], axis=1)

        df["WindDirection(Degrees)_bin"] = np.digitize(
            df["WindDirection(Degrees)"], np.linspace(0, 360, 19)
        )
        df["TSS_Minute_bin"] = np.digitize(df["TSS_Minute"], np.arange(0, 288 + 12, 12))
        df["Humidity_bin"] = np.digitize(df["Humidity"], np.arange(32, 3200, 128))

        # Scale input
        X_scaled = scaler_X.transform(df)

        return X_scaled
    except Exception as e:
        raise ValueError(f"Input preprocessing failed: {str(e)}")

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Preprocess input
            X_scaled = preprocess_input(request.form)

            # Predict
            prediction = net.predict(X_scaled)
            predicted_value = float(prediction[0][0])  # keep continuous output

            return render_template("index.html", prediction=predicted_value)
        except Exception as e:
            return render_template("index.html", error=f"Error: {str(e)}")
    else:
        return render_template("index.html")

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
