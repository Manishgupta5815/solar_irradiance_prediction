


# Solar Irradiance Prediction

## Project Overview
It is a web application that predicts solar irradiance (Radiation) based on environmental parameters using a custom-built Neural Network. The model is trained on historical weather data, enabling accurate short-term solar radiation forecasting for a given location and time.  

This project demonstrates the integration of **machine learning with a Flask web application**, allowing users to input environmental data and receive predictions instantly.

---

## Features

- Predicts **solar irradiance** using temperature, pressure, humidity, wind speed, wind direction, and time-related features.
- Handles **feature engineering** for date, time, sunrise, sunset, and weather measurements.
- Provides a **web interface** for user input.
- Outputs predictions in **realistic units of radiation**.
- Visualizes **training and testing loss** (MSE, MAE, RMSE) for model evaluation.
- Fully **deterministic predictions** after model is trained and weights are loaded.

---
## Website Preview

![MindEase Screenshot](<img width="1920" height="2951" alt="screencapture-127-0-0-1-5000-2025-10-17-03_07_31" src="https://github.com/user-attachments/assets/3ad31089-d922-4ead-92ac-ba14b2350b2b" />
)  
*Screenshot of the MindEase web interface.*

---

## Technologies Used

- **Python**  
- **NumPy, Pandas** – Data manipulation and preprocessing  
- **Matplotlib** – Visualization of loss curves  
- **Scikit-learn** – Data scaling and train-test splitting  
- **Flask** – Web framework for serving predictions  
- **h5py** – Saving and loading neural network weights  

---

## Project Structure


```
Solar Irradiance Prediction/
│
├─ app.py                  # Flask web app
├─ Models/
    └─ LinearRegression.py #Custom LR class
│   └─ NeuralNetwork.py    # Custom Neural Network class
├─ Data/
│   └─ SolarPrediction.csv # Dataset
├─ scaler_X.pkl            # Saved input scaler
├─ NN_Weights.h5           # Saved neural network weights
├─ templates/
│   └─ index.html          # Web interface template
└─ README.md               # Project documentation
```



## Dataset

- **SolarPrediction.csv**: Contains historical weather and solar radiation data.  
- Features include:
  - Temperature, Pressure, Humidity  
  - Wind Speed, Wind Direction  
  - Date, Time, Sunrise, Sunset  

---

## Installation

1. Clone the repository:


git clone <repository_url>
cd MindEase


2. Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

3. Install required dependencies:

```
pip install -r requirements.txt
```

---

## Running the App

1. Ensure `scaler_X.pkl` and `NN_Weights.h5` are present in the project root.
2. Start the Flask server:

```bash
python app.py
```

3. Open your browser and go to:

```
http://127.0.0.1:5000/
```

4. Enter the required weather and time inputs to receive a solar irradiance prediction.

---

## Model Training (Optional)

If you want to retrain the model:

1. Run `NeuralNetwork.py` with your dataset.
2. Feature engineering, scaling, and neural network training are implemented in the script.
3. Trained weights will be saved as `NN_Weights.h5`.
4. Input scaler will be saved as `scaler_X.pkl`.

---

## Notes

* The Neural Network uses **ReLU activations** and **Mean Squared Error** as the primary loss function.
* Predictions are deterministic **after training and loading weights**, ensuring the same input always produces the same output.
* For best results, use inputs in the same scale and units as the training data.

---

## Future Improvements

* Add **real-time weather API integration** for automatic solar irradiance prediction.
* Enhance the model with **more layers or advanced architectures** (e.g., LSTM for temporal trends).
* Improve the web interface with **interactive plots** and **historical data visualization**.

---

## Author

**Manish Kumar Gupta**

* GitHub: [github.com/Manishgupta5815](https://github.com/Manishgupta5815)
* LinkedIn: [linkedin.com/in/manni2026](https://linkedin.com/in/manni2026)

```

